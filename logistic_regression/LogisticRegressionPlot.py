from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import logistic_regression.LogisticRegression as lr

dlc = dict(
    dlblue="#0096ff",
    dlorange="#FF9300",
    dldarkred="#C00000",
    dlmagenta="#FF40FF",
    dlpurple="#7030A0",
)
dlblue = "#0096ff"
dlorange = "#FF9300"
dldarkred = "#C00000"
dlmagenta = "#FF40FF"
dlpurple = "#7030A0"
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]


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
    # plot the probability
    plt_prob(ax, w_out, b_out)

    # Plot the original data
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_0$")
    ax.axis([0, 4, 0, 3.5])
    plot_data(x_train, y_train, ax)

    # Plot the decision boundary
    x0 = -b_out / w_out[0]
    x1 = -b_out / w_out[1]
    ax.plot([0, x0], [x1, 0], c=dlc["dlblue"], lw=1)
    plt.show()


def plt_prob(ax, w_out, b_out):
    """plots a decision boundary but include shading to indicate the probability"""
    # setup useful ranges and common linspaces
    x0_space = np.linspace(0, 4, 100)
    x1_space = np.linspace(0, 4, 100)

    # get probability for x0,x1 ranges
    tmp_x0, tmp_x1 = np.meshgrid(x0_space, x1_space)
    z = np.zeros_like(tmp_x0)
    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x1.shape[1]):
            z[i, j] = lr.sigmoid(
                np.dot(w_out, np.array([tmp_x0[i, j], tmp_x1[i, j]])) + b_out
            )

    cmap = plt.get_cmap("Blues")
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(
        tmp_x0,
        tmp_x1,
        z,
        norm=cm.colors.Normalize(vmin=0, vmax=1),
        cmap=new_cmap,
        shading="nearest",
        alpha=0.9,
    )
    ax.figure.colorbar(pcm, ax=ax)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """truncates color map"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
