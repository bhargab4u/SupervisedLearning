import linear_regression.MultipleFeatureLinearRegression as MultipleFeatureLinearRegression
import numpy as np
import linear_regression.SimpleLinearRegression as SimpleLinearRegression
import logistic_regression.LogisticRegression as LogisticRegression


def run():
    operation = input(
        "Input number : \n1. SimpleLinearRegression, \n2. MultipleFeatureLinearRegression, \n3. LogisticRegression\n -> "
    )

    match operation:
        case "1":
            SimpleLinearRegression.linear_regression()

        case "2":
            MultipleFeatureLinearRegression.linear_regression()

        case "3":
            LogisticRegression.logistic_regression()


if __name__ == "__main__":
    run()
