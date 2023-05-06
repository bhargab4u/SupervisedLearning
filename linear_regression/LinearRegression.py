import numpy as np
import copy, math
import matplotlib.pyplot as plt


class Linearregression:
    def __init__(self):
        pass

    """
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
    """

    def compute_cost(x_train, y_train, w, b):
        cost_sum = 0.0

        m = x_train.shape[0]

        for i in range(m):
            f_x = np.dot(x_train[i], w) + b
            diff = f_x - y_train[i]
            diff = diff**2
            cost_sum += diff
        total_cost = (1 / (2 * m)) * cost_sum
        return total_cost

    def compute_gradient(x_train, y_train, w, b):
        m, n = x_train.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.0

        for i in range(m):
            f_x = np.dot(x_train[i], w) + b - y_train[i]
            for j in range(n):
                dj_dw[j] += f_x * x_train[i, j]
            dj_db += f_x

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def compute_gradient_descent(
        x_train,
        y_train,
        w_init,
        b_init,
        cost_function,
        gradient_function,
        alpha,
        num_iters,
    ):
        J_history = []

        w = copy.deepcopy(w_init)
        b = b_init

        # print(f"num_iters : {num_iters}")
        for i in range(num_iters):
            dj_dw, dj_db = gradient_function(x_train, y_train, w, b)

            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            # print(f"w {w}: b {b}")

            if i < 10000:
                J_history.append(cost_function(x_train, y_train, w, b))

            if i % 100 == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        return w, b, J_history

    def calculate_linear_regression():
        X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
        y_train = np.array([460, 232, 178])
        b_init = 785.1811367994083
        w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        # initialize parameters
        initial_w = np.zeros_like(w_init)
        initial_b = 0.0
        # some gradient descent settings
        iterations = 10000
        alpha = 5.0e-7
        # run gradient descent
        w_final, b_final, J_hist = Linearregression.compute_gradient_descent(
            X_train,
            y_train,
            initial_w,
            initial_b,
            Linearregression.compute_cost,
            Linearregression.compute_gradient,
            alpha,
            iterations,
        )
        print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
        m, _ = X_train.shape
        for i in range(m):
            print(
                f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}"
            )
