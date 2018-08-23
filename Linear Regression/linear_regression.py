import math
import matplotlib.pyplot as plt
import numpy as np


# Function to generate data
def generate_lin_data(n_pts=100):
    # Generate x random points in the range from 0 to 1000
    x = np.random.rand(n_pts, 1) * 500
    # generate y, linearly related to x + noise
    y = 3 * x + 10 + np.random.rand(n_pts, 1) * 500

    # Plot generated data
    plt.scatter(x, y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Generated Un-normalized data")
    plt.show()

    return x, y


# Data scaling
def normalize(x, y):
    global x_norm, y_norm

    x_norm = max(x) - min(x)
    x = x / x_norm

    y_norm = max(y) - min(y)
    y = y / y_norm

    # Plot generated data
    plt.scatter(x, y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Generated normalized data")
    plt.show()

    return x, y


# Calculate cost
def cost_fn(est, y):
    # MSE
    cost = np.square(est - y) / 2
    return np.sum(cost)


# fitting the data to parameters
def fit(params, x, y, alpha=0.01):
    y_hat = np.dot(x, params)
    params -= alpha * np.mean(y_hat - y) * np.expand_dims(np.mean(x, axis=0), axis=-1)
    return params, cost_fn(y_hat, y)


# x and y are input and target values
# features is number of features to use
# b_size = batch size
def train(x, y, features=1, b_size=1, epochs=2000):
    global params

    original_x = x
    original_y = y

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

            params, cost = fit(np.copy(params), temp_x, temp_y)

            bat_cost += cost
            start = end

        curr_loss = bat_cost/x.shape[0]
        if epoch % 100 == 0:
            print("Epoch :", epoch, " Loss :", curr_loss)

        loss.append(curr_loss)

    print("\nThe trained parameters :\n", params)

    # Plot the training loss
    plt.plot(range(1, len(loss)), loss[1:])
    plt.xlabel("# of iterations")
    plt.ylabel("Cost function")
    plt.show()

    # Plot line
    plot_x = np.linspace(0, 1, 70).reshape(70, 1)
    mod_plot_x = plot_x

    # Add features
    if features > 1:
        new_x = plot_x
        for i in range(2, features + 1):
            temp_x = np.power(plot_x, i)
            new_x = np.concatenate((new_x, temp_x), axis=-1)
        # Modified plot_x
        mod_plot_x = new_x

    # Append ones for bias
    ones = np.ones((plot_x.shape[0], 1))
    mod_plot_x = np.concatenate((ones, mod_plot_x), axis=-1)

    # get Y values for plotting from mod_plot_x
    plot_y = np.dot(mod_plot_x, params)
    plt.scatter(original_x*x_norm, original_y*y_norm)
    plt.plot(plot_x*x_norm, plot_y*y_norm, 'r')
    plt.show()


if __name__ == '__main__':
    data_x, data_y = generate_lin_data()
    data_x, data_y = normalize(data_x, data_y)

    # b_size = batch size
    train(data_x, data_y, features=2, b_size=1, epochs=3000)
