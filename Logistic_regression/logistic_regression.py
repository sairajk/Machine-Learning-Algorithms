import math
import matplotlib.pyplot as plt
import numpy as np


# Load, shuffle and split data
def load_data(file_name, frac_as_test=0.15):
    with open(file_name, 'r') as reader:
        lines = reader. readlines()

        x_data = [[float(i) for i in line.rstrip().split(',')[:-1]] for line in lines]
        y_data = [line.rstrip().split(',')[-1] for line in lines]

        # Load any two classes for classification
        temp_x = []
        temp_y = []
        for x, y in zip(x_data, y_data):
            if y == 'Iris-setosa':
                temp_x.append(x)
                temp_y.append(0)
            elif y == 'Iris-versicolor':
                temp_x.append(x)
                temp_y.append(1)

        x_data = temp_x
        y_data = temp_y

        # shuffle data
        idx = [i for i in range(len(y_data))]
        np.random.shuffle(idx)

        x_data = np.array([x_data[curr_idx] for curr_idx in idx])
        y_data = np.array([y_data[curr_idx] for curr_idx in idx])

        # split data
        n_test = int(len(y_data) * frac_as_test)
        x_test = x_data[:n_test]
        y_test = np.expand_dims(y_data[:n_test], axis=-1)

        x_train = x_data[n_test:]
        y_train = np.expand_dims(y_data[n_test:], axis=-1)

    return x_train, y_train, x_test, y_test


# binary cross entropy loss fn
def cost_fn(y_hat, y):
    cost = -1 * (y * np.log10(y_hat) + (1 - y) * np.log10(1 - y_hat))
    # print(y_hat)
    return np.sum(cost)


# sigmoid function for probability prediction
def sigmoid(x):
    sig = np.reciprocal(1 + np.exp(-1*x))
    return sig


# fn for updating parameters
def fit(params, x, y, alpha=0.01):
    y_hat = np.dot(x, params)
    y_hat = sigmoid(y_hat)
    params -= alpha * np.mean(y_hat - y) * np.expand_dims(np.mean(x, axis=0), axis=-1)
    return params, cost_fn(y_hat, y)


# train fn
def train(x, y, features=1, b_size=1, epochs=2000):
    global params, features_n, batch_size
    features_n = features
    batch_size = b_size

    # Add features
    if features > 1:
        new_x = x
        for i in range(2, features + 1):
            temp_x = np.power(x, i)
            new_x = np.concatenate((new_x, temp_x), axis=-1)
        x = new_x

    # Append ones for bias
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=-1)
    print("Shape of modified input data with features and bias :", x.shape, "\n")

    # Parameters
    params = np.random.rand(x.shape[-1], 1)

    n_batches = int(math.ceil(x.shape[0] / b_size))
    loss = [0]

    for epoch in range(epochs):
        start = 0
        bat_cost = 0
        for batch in range(n_batches):
            end = min(start+b_size, x.shape[0])

            temp_x = x[start: end]
            temp_y = y[start: end]
            # print(params)
            params, cost = fit(np.copy(params), temp_x, temp_y)

            bat_cost += cost
            start = end

        curr_loss = bat_cost/x.shape[0]
        if epoch % 1 == 0:
            print("Epoch :", epoch, " Loss :", curr_loss)

        loss.append(curr_loss)

    print("Training complete !!!")
    print("\nThe trained parameters :\n", params)

    # Plot the training loss
    plt.plot(range(1, len(loss)), loss[1:])
    plt.xlabel("# of iterations")
    plt.ylabel("Cost function")
    plt.show()


# test fn
def test(x, y):
    features = features_n

    # Add features
    if features > 1:
        new_x = x
        for i in range(2, features + 1):
            temp_x = np.power(x, i)
            new_x = np.concatenate((new_x, temp_x), axis=-1)
        x = new_x

    # Append ones for bias
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=-1)
    print("Shape of modified input data with features and bias :", x.shape, "\n")

    y_hat = sigmoid(np.dot(x, params))
    corrects = np.count_nonzero((y_hat > 0.5) == y)
    print("Acurracy :", corrects/float(len(y)) * 100, "%")

    # Plot for visualization
    y_hat = [float(y_h) for y, y_h in sorted(zip(y, y_hat))]
    y = sorted(y)
    plt.scatter(range(len(y)), y, label='True class')
    plt.scatter(range(len(y)), y_hat, label='Predicted class')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    # frac_as_test = fraction of total data as test data
    x_tr, y_tr, x_te, y_te = load_data("iris.txt", frac_as_test=0.15)

    # train
    train(x_tr, y_tr, features=1, b_size=2, epochs=300)

    # test on testing data
    test(x_te, y_te)
