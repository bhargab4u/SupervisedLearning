import numpy as np
import copy, math
import matplotlib.pyplot as plt


class Linearregression:
    def __init__(self):
        pass

    def compute_model_output():
        x_train = np.array([1.0, 2.0])
        y_train = np.array([300.0, 500.0])
        w = 100
        b = 100
        m = x_train.shape[0]
        f_x = np.zeros(m)
        for i in range(m):
            f_x[i] = w * x_train[i] + b
        print(f_x)
        plt.plot(x_train, f_x, c="b", label="Out Prediction")
        plt.scatter(x_train, y_train, c="r", marker="x", label="Actual Values")
        plt.title("House Price")
        plt.xlabel("Price")
        plt.ylabel("Area")
        plt.show()

    def compute_cost(x_train, y_train, w, b):
        x_train = np.array([1.0, 2.0])
        y_train = np.array([300.0, 500.0])
        w = 100
        b = 100
        cost_sum = 0

        m = x_train.shape[0]
        f_x = np.zeros(m)

        for i in range(m):
            f_x[i] = w * x_train[i] + b
            diff = f_x[i] - y_train[i]
            diff = diff**2
            cost_sum += diff
        total_cost = (1 / (2 * m)) * cost_sum
        return total_cost

    def compute_gradient(x_train, y_train, w_init, b_init):
        # x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
        # y_train = np.array([460, 232, 178])

        # b_init = 785.1811367994083
        # w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        w = copy.deepcopy(w_init)
        b = b_init

        m, n = x_train.shape
        dj_dw = np.zeros(
            n,
        )
        dj_db = 0.0

        for i in range(m):
            f_x = np.dot(w, x_train[i]) + b - y_train[i]
            for j in range(n):
                dj_dw[j] += f_x * x_train[i, j]

            dj_db += f_x

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def compute_gradient_descent():
        x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
        y_train = np.array([460, 232, 178])
        b_init = 785.1811367994083
        w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        alpha = 0.01
        num_iters = 10

        w = copy.deepcopy(w_init)
        b = b_init
