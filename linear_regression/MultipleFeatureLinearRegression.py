import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def linear_regression():
    data = pd.read_csv("./linear_regression/house-prices.csv")
    data = data.astype(
        {
            "SqFt": "float64",
            "Price": "float64",
            "Bedrooms": "float64",
            "Bathrooms": "float64",
            "Offers": "float64",
        }
    )

    # Feature Re-scaling Z-Score
    price_mean = data["Price"].mean()
    price_std = data["Price"].std()
    sqft_mean = data["SqFt"].mean()
    sqft_std = data["SqFt"].std()
    data["Price"] = data["Price"].apply(lambda x: ((x - price_mean) / price_std))
    data["SqFt"] = data["SqFt"].apply(lambda x: ((x - sqft_mean) / sqft_std))

    # data["Price"] = data["Price"].div(100000).round(4)
    # data["SqFt"] = data["SqFt"].div(1000).round(2)

    actual_data = data[["SqFt", "Bedrooms", "Bathrooms", "Offers", "Price"]].to_numpy()
    # print(actual_data)
    x_train = actual_data[:, : actual_data.shape[1] - 1]
    y_train = actual_data[:, -1:]
    # print(x_train[:, 0])

    J_cost_history = []
    w = np.zeros(x_train.shape[1])
    b = 0.0
    alpha = 0.001
    epochs = 3000

    for k in range(epochs):
        if k % 100 == 0:
            print(f"Epochs : {k}")
        w, b = gradient_descent(w, b, x_train, y_train, alpha)
        J_cost_history.append(compute_cost(x_train, y_train, w, b))

    print(w, b)
    plot_cost(J_cost_history)
    # plot(x_train, y_train, w, b)


def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x_train[i], w) + b
        cost = cost + (f_wb_i - y_train[i]) ** 2
    cost = cost / (2 * m)
    return cost


"""
def plot(x_train, y_train, w, b):
    m, n = x_train.shape
    for j in range(n):
        f_x = np.zeros(m)
        for i in range(m):
            f_x[i] = w[j] * x_train[i : i + 1, j : j + 1] + b

        # print(f_x)
        plt.scatter(x_train[:, j], y_train, c="r", label="Actual Values")
        plt.plot(x_train[:, j], f_x, c="b")
        plt.show()
"""


def plot_cost(cost_arr):
    plt.plot(range(len(cost_arr)), cost_arr, c="b")
    plt.title("Cost Function")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()


def gradient_descent(w_init, b_init, x_train, y_train, alpha):  # alpha is learning rate
    m, n = x_train.shape
    w_now = np.zeros(n)
    b_now = 0.0
    f_x = 0.0
    w_sum_diff = np.zeros(n)
    b_sum_diff = 0.0
    for i in range(m):
        f_x = np.dot(w_init, x_train[i]) + b_init
        for j in range(n):
            w_sum_diff[j] += (f_x - y_train[i]) * x_train[i, j]

        b_sum_diff += f_x - y_train[i]

    w_now = w_init - (alpha * (w_sum_diff)) / m
    b_now = b_init - (alpha * (b_sum_diff)) / m

    return w_now, b_now
