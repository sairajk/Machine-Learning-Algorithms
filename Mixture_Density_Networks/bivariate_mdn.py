import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math


NHIDDEN_1 = 5 * 6
NHIDDEN_2 = 5 * 6
STDEV = 0.5
KMIX = 5  # number of mixtures
NOUT = KMIX * 6  # pi, 2 for mu, 2 for stddev, rho

NSAMPLE = 2400
save_path = "./model_weights/bivar_mdn/model"


# Function to generate 2D training data
def generate_data():
    c1 = np.float32(np.random.uniform(-10, 10, (1, NSAMPLE))).T
    c2 = np.float32(np.random.uniform(-10, 10, (1, NSAMPLE))).T
    y_data = np.concatenate((c1, c2), axis=1)
    r_data = np.float32(np.random.normal(size=(NSAMPLE, 2)))  # random noise
    x_data = np.concatenate((np.sin(c1), np.cos(c2)), axis=1) + r_data

    # Plot the data 
    # f, axarr = plt.subplots(2, sharex=True)
    # f.suptitle('Generated Training Data')
    # axarr[0].scatter(x_data[:, 0], y_data[:, 0])
    # axarr[1].scatter(x_data[:, 1], y_data[:, 1])
    # plt.show()

    return x_data, y_data


# Generate train and test data
x_data, y_data = generate_data()
x_test = np.float32(np.arange(-3, 3, 0.05))
NTEST = x_test.size
x_test = x_test.reshape(NTEST, 1)  # needs to be a matrix, not a vector
x_test = np.concatenate((x_test, x_test), axis=1)


# Get mixture coefficients from the network output
def get_mixture_coef(output):
    out_pi = tf.placeholder(dtype=tf.float32, shape=[None, KMIX], name="out_pi")
    out_sigma = tf.placeholder(dtype=tf.float32, shape=[None, KMIX, 2], name="out_sigma")
    out_mu = tf.placeholder(dtype=tf.float32, shape=[None, KMIX, 2], name="out_mu")
    out_rho = tf.placeholder(dtype=tf.float32, shape=[None, KMIX], name="out_rho")

    out_pi, out_rho, mu_1, mu_2, sigma_1, sigma_2 = tf.split(output, num_or_size_splits=6, axis=1)
    out_sigma = tf.stack([sigma_1, sigma_2], axis=2)
    out_mu = tf.stack([mu_1, mu_2], axis=2)

    max_pi = tf.reduce_max(out_pi, 1, keepdims=True)
    out_pi = tf.subtract(out_pi, max_pi)
    out_pi = tf.exp(out_pi)
    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keepdims=True))
    out_pi = tf.multiply(normalize_pi, out_pi)

    out_rho = tf.nn.tanh(out_rho)

    out_sigma = tf.exp(out_sigma)

    # print("out_pi", out_pi)
    # print("out_mu", out_mu)
    # print("out_sigma", out_sigma)
    # print("out_rho", out_rho)

    return out_pi, out_mu, out_sigma, out_rho


# Compute Normal distribution for 2D data
def tf_normal(y, mu, sigma, rho):
    print(mu[:, :, 0], y[:, 0])
    z_1 = tf.divide(tf.square(tf.subtract(mu[:, :, 0], tf.expand_dims(y[:, 0], -1))),
                    tf.square(sigma[:, :, 0]))
    print(mu[:, :, 1], y[:, 1])
    z_2 = tf.divide(tf.square(tf.subtract(mu[:, :, 1], tf.expand_dims(y[:, 1], -1))),
                    tf.square(sigma[:, :, 1]))

    z_3 = tf.multiply(2 * rho, tf.multiply(tf.subtract(mu[:, :, 0], tf.expand_dims(y[:, 0], -1)),
                                           tf.subtract(mu[:, :, 1], tf.expand_dims(y[:, 1], -1))))
    z_3 = tf.divide(z_3, tf.multiply(sigma[:, :, 0], sigma[:, :, 1]))

    Z = z_1 + z_2 - z_3

    N = tf.divide(-Z, 2 * (tf.ones_like(rho) - tf.square(rho)))
    N = tf.exp(N)

    normalizer = 2 * math.pi * tf.multiply(sigma[:, :, 0], sigma[:, :, 1])
    normalizer = tf.multiply(normalizer, tf.sqrt(tf.ones_like(rho) -
                                                 tf.square(rho)))
    normalizer = tf.reciprocal(normalizer)

    N = tf.multiply(normalizer, N)

    return N


# Calculate Loss : -log(Probability)
def get_lossfunc(out_pi, out_mu, out_sigma, out_rho, y):
    result = tf_normal(y, out_mu, out_sigma, out_rho)
    result = tf.multiply(result, out_pi)
    result = tf.reduce_sum(result, 1, keepdims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)


# The model
def model():
    # Placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="y")

    # MODEL
    Wh_1 = tf.Variable(0.01 * tf.random_normal([2, NHIDDEN_1], stddev=STDEV, dtype=tf.float32))
    bh_1 = tf.constant(0.1, shape=[NHIDDEN_1])

    Wh_2 = tf.Variable(0.01 * tf.random_normal([NHIDDEN_1, NHIDDEN_2], stddev=STDEV, dtype=tf.float32))
    bh_2 = tf.constant(0.1, shape=[NHIDDEN_2])

    Wo = tf.Variable(0.01 * tf.random_normal([NHIDDEN_2, NOUT], stddev=STDEV, dtype=tf.float32))
    bo = tf.constant(0.1, shape=[NOUT])

    hidden_layer_1 = tf.nn.sigmoid(tf.matmul(x, Wh_1) + bh_1)
    hidden_layer_2 = tf.nn.sigmoid(tf.matmul(hidden_layer_1, Wh_2) + bh_2)
    output = tf.matmul(hidden_layer_2, Wo) + bo

    return x, y, output


# Trainer
def train(NEPOCH=1000, mode="new"):
    x, y, model_op = model()

    out_pi, out_mu, out_sigma, out_rho = get_mixture_coef(model_op)

    lossfunc = get_lossfunc(out_pi, out_mu, out_sigma, out_rho, y)
    train_op = tf.train.AdamOptimizer().minimize(lossfunc)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    if mode is "load":
        saver.restore(sess, save_path)
    elif mode is "new":
        sess.run(tf.global_variables_initializer())

    NEPOCH += 1
    loss = np.zeros(NEPOCH)
    for i in range(NEPOCH):
        sess.run(train_op, feed_dict={x: x_data, y: y_data})
        loss[i], pi_, mu_, sig_, ro_, op_ = sess.run([lossfunc, out_pi, out_mu, out_sigma, out_rho, model_op],
                                                     feed_dict={x: x_data, y: y_data})
        if i % 200 == 0:
            print("Loss at epoch ", i + 1, " : ", loss[i])
            # save weights only if loss is not NaN
            if not math.isnan(loss[i]):
                saver.save(sess, save_path)

    # Plot loss at the end of training
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(100, NEPOCH, 1), loss[100:], 'r-')
    plt.show()


# The next two functions are used for sampling data
# Select a Probab. Dist. randomly
def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    print('error with sampling ensemble')
    return -1


# Sample points from a selected distribution
def generate_ensemble(out_pi, out_mu, out_sigma, out_rho, M=10):
    NTEST = x_test.shape[0]
    result = np.random.rand(NTEST, 2, M)  # initially random [0, 1]
    rn = np.random.randn(NTEST, 2, M)  # normal random matrix (0.0, 1.0)

    # transforms result into random ensembles
    # FORMULA
    # u = mu1 + sigma1 * x1
    # v = mu2 + sigma2 * (rho * x1 + sqrt(1 - rho^2) * x2)

    for j in range(0, M):
        for i in range(0, NTEST):
            # generating data for axis 1
            idx = get_pi_idx(result[i, 0, j], out_pi[i])
            mu = out_mu[i, idx, 0]
            std = out_sigma[i, idx, 0]
            result[i, 0, j] = mu + rn[i, 0, j] * std

            # Generating data for axis 2
            idx = get_pi_idx(result[i, 1, j], out_pi[i])
            mu = out_mu[i, idx, 1]
            std = out_sigma[i, idx, 1]
            rho = out_rho[i, idx]
            result[i, 1, j] = mu + std * rn[i, 1, j] * (rho * rn[i, 0, j] +
                                                        math.sqrt(1 - rho * rho))

    return result


# Test and Visualize
def test():
    x, y, model_op = model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    saver.restore(sess, save_path)

    out_pi_test, out_mu_test, out_sigma_test, out_rho_test = sess.run(get_mixture_coef(model_op),
                                                                      feed_dict={x: x_test})

    y_test = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test, out_rho_test)

    # print("y_test.shape", y_test.shape)
    # print("x_test.shape", x_test.shape)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data[:, 0], y_data[:, 0], 'ro', x_test[:, 0:1], y_test[:, 0, :], 'bo', alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(x_data[:, 1], y_data[:, 1], 'ro', x_test[:, 1:], y_test[:, 0, :], 'bo', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Task can either be "train" or "test"
    # mode can either be "load" or "new"

    mode = "load"
    task = "test"

    if task is "train":
        train(NEPOCH=10000, mode=mode)
    elif task is "test":
        test()
