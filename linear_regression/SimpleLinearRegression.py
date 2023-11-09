import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def linear_regression():
    data = pd.read_csv("./linear_regression/house-prices.csv")
    data = data.astype({"SqFt": "float64", "Price": "float64"})

    # Feature Re-scaling Z-Score
    price_mean = data["Price"].mean()
    price_std = data["Price"].std()
    sqft_mean = data["SqFt"].mean()
    sqft_std = data["SqFt"].std()
    data["Price"] = data["Price"].apply(lambda x: (((x - price_mean) / price_std) * 10))
    data["SqFt"] = data["SqFt"].apply(lambda x: (((x - sqft_mean) / sqft_std) * 10))

    J_cost_history = []
    w = 0.0
    b = 0.0
    alpha = 0.001
    epochs = 300

    for k in range(epochs):
        if k % 100 == 0:
            print(f"Epochs : {k}")
        w, b = gradient_descent(w, b, data, alpha)
        J_cost_history.append(compute_cost(data, w, b))

    print(w, b)
    plot(data, w, b)
    plot_cost(J_cost_history)


def plot(data, w, b):
    m = len(data)
    f_x = np.zeros(m)
    x = np.zeros(m)
    for i in range(m):
        x[i] = data.iloc[i]["SqFt"]
        f_x[i] = w * x[i] + b
    plt.scatter(data.SqFt.values, data.Price.values, c="r", label="Actual Values")
    plt.plot(x, f_x, c="b")
    plt.show()


def gradient_descent(w_init, b_init, points, alpha):  # alpha is learning rate
    m = len(points)
    f_x = 0.0
    w_sum_diff = 0.0
    b_sum_diff = 0.0
    for i in range(m):
        x = points.iloc[i]["SqFt"]
        y = points.iloc[i]["Price"]
        f_x = w_init * x + b_init

        w_sum_diff += (f_x - y) * x
        b_sum_diff += f_x - y

    w_now = w_init - (alpha * (w_sum_diff)) / m
    w_now = round(w_now, 2)
    b_now = b_init - (alpha * (b_sum_diff)) / m

    return w_now, b_now


def compute_cost(data, w, b):
    m = len(data)
    cost = 0.0
    sqr_diff = 0.0
    for i in range(m):
        x = data.iloc[i]["SqFt"]
        y = data.iloc[i]["Price"]
        f_x = w * x + b
        sqr_diff += (f_x - y) ** 2

    cost = sqr_diff / (2 * m)
    return cost


def plot_cost(cost_arr):
    plt.plot(range(len(cost_arr)), cost_arr, c="b")
    plt.title("Cost Function")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()
