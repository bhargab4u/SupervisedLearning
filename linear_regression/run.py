import MultipleFeatureLinearRegression
import numpy as np
import SimpleLinearRegression


def run():
    operation = input(
        "Input number : (1. SimpleLinearRegression, 2. MultipleFeatureLinearRegression) : "
    )

    match operation:
        case "1":
            SimpleLinearRegression.linear_regression()

        case "2":
            MultipleFeatureLinearRegression.linear_regression()


if __name__ == "__main__":
    run()
