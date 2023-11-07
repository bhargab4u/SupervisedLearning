from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import logistic_regression.LogisticRegression as lr


def plot(x_train, y_train):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_data(x_train, y_train, ax)

    # ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel("$x_1$", fontsize=12)
    ax.set_xlabel("$x_0$", fontsize=12)
    plt.show()


def plot_cost(cost_arr):
    plt.plot(range(len(cost_arr)), cost_arr, c="b")
    plt.title("Cost Function")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc="best"):
    """plots logistic data with two axis"""
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(
        -1,
    )  # work with 1D or 1D y vectors
    neg = neg.reshape(
        -1,
    )

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker="x", s=s, c="red", label=pos_label)
    ax.scatter(
        X[neg, 0],
        X[neg, 1],
        marker="o",
        s=s,
        label=neg_label,
        facecolors="none",
        edgecolors="b",
        lw=3,
    )
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False


def plot_decision_boundary(x_train, y_train, w_out, b_out):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Plot the original data
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_0$")
    plot_data(x_train, y_train, ax)

    # Plot the decision boundary
    x0 = -b_out / w_out[0]
    x1 = -b_out / w_out[1]

    plot_line([0, x0[0]], [x1[0], 0], ax)
    plt.show()


def plot_line(x, y, plt):
    # Calculate the coefficients.
    coefficients = np.polyfit(x, y, 1)

    # Let's compute the values of the line...
    polynomial = np.poly1d(coefficients)
    x_axis = range(-3, 3)
    y_axis = polynomial(x_axis)

    # ...and plot the points and the line
    plt.plot(x_axis, y_axis)
    plt.plot(x[0], y[0], "go")
    plt.plot(x[1], y[1], "go")
