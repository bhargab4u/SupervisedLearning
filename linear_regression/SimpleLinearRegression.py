import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def calculate_cost():
    data = pd.read_csv("house-prices.csv")
    data = data.astype({"SqFt": "float64", "Price": "float64"})
    # data_normalized = data.copy(deep=True)
    data["Price"] = data["Price"].div(100000).round(4)
    data["SqFt"] = data["SqFt"].div(1000).round(2)
    w = 0.0
    b = 0.0
    alpha = 0.001
    epochs = 3000

    for k in range(epochs):
        if k % 100 == 0:
            print(f"Epochs : {k}")
        w, b = gradient_descent(w, b, data, alpha)

    print(w, b)
    plot(data, w, b)


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
        if math.isnan(f_x) or math.isinf(f_x):
            print(f"x:{x}, y:{y}, f_x:{f_x}")
            break

        w_sum_diff += (f_x - y) * x
        b_sum_diff += f_x - y

    w_now = w_init - (alpha * (w_sum_diff)) / m
    w_now = round(w_now, 2)
    b_now = b_init - (alpha * (b_sum_diff)) / m

    if math.isnan(w_now) or math.isinf(w_now) or math.isinf(b_now) or math.isnan(b_now):
        print(f"w_init : {w_init}, w_now: {w_now}, b_init: {b_init}, b_now: {b_now}")
        return 0, 0

    return w_now, b_now
