import numpy as np
import pandas as pd
import math
import logistic_regression.LogisticRegressionPlot as lrp


def logistic_regression():
    data = pd.read_csv("./logistic_regression/bmd.csv")
    data = data.astype(
        {
            "age": "float64",
            "height_cm": "float64",
            "waiting_time": "float64",
            "bmd": "float64",
        }
    )
    decision_boundry = 0.8
    data["healthy"] = data["bmd"].apply(lambda x: 0 if x < decision_boundry else 1)

    # Feature Re-scaling
    age_mean = data["age"].mean()
    age_std = data["age"].std()
    bmd_mean = data["bmd"].mean()
    bmd_std = data["bmd"].std()
    data["age"] = data["age"].apply(lambda x: ((x - age_mean) / age_std))
    data["bmd"] = data["bmd"].apply(lambda x: ((x - bmd_mean) / bmd_std))

    # pd.set_option("max_rows", None)
    x_train = data[["age", "bmd"]].to_numpy()
    y_train = data[["healthy"]].to_numpy()
    # print(x_train)
    # print(y_train)
    # plot(x_train, y_train)

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
    lrp.plot_cost(J_cost_history)
    lrp.plot_decision_boundary(x_train, y_train, w, b)


def gradient_descent(w_init, b_init, x_train, y_train, alpha):  # alpha is learning rate
    m, n = x_train.shape
    w_now = np.zeros(n)
    b_now = 0.0
    f_x = 0.0
    w_sum_diff = np.zeros(n)
    b_sum_diff = 0.0

    for i in range(m):
        f_x = sigmoid(np.dot(x_train[i], w_init) + b_init)
        for j in range(n):
            w_sum_diff[j] += (f_x - y_train[i]) * x_train[i, j]

        b_sum_diff += f_x - y_train[i]

    w_now = w_init - (alpha * (w_sum_diff)) / m
    b_now = b_init - (alpha * (b_sum_diff)) / m

    return w_now, b_now


def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x_train[i], w) + b)
        cost += -y_train[i] * math.log(f_wb_i) - (1 - y_train[i]) * math.log(1 - f_wb_i)
    cost = cost / (1 * m)
    return cost


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
