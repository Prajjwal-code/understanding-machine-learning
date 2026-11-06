"""Fit a straight line using gradient descent.

This is a small implementation for learning purposes. It uses only Python's
standard library and keeps the maths visible rather than hiding it in a library.
"""

from math import isfinite


def predict(x, weight, bias):
    return weight * x + bias


def mean_squared_error(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must have the same non-zero length")
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)


def fit(x_values, y_values, learning_rate=0.01, epochs=1000):
    """Return the learned weight, bias, and loss recorded after each epoch."""
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("x_values and y_values must have the same non-zero length")
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("learning_rate must be positive and epochs must be >= 1")
    if not all(isfinite(float(v)) for v in [*x_values, *y_values]):
        raise ValueError("all input values must be finite")

    weight = 0.0
    bias = 0.0
    count = len(x_values)
    history = []

    for _ in range(epochs):
        predictions = [predict(x, weight, bias) for x in x_values]
        weight_gradient = (2 / count) * sum(
            x * (prediction - actual)
            for x, prediction, actual in zip(x_values, predictions, y_values)
        )
        bias_gradient = (2 / count) * sum(
            prediction - actual
            for prediction, actual in zip(predictions, y_values)
        )

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

        updated = [predict(x, weight, bias) for x in x_values]
        history.append(mean_squared_error(y_values, updated))

    return weight, bias, history


if __name__ == "__main__":
    # A simple dataset with a little variation around y = 2x + 1.
    x = [1, 2, 3, 4, 5, 6]
    y = [3.1, 4.9, 7.2, 8.8, 11.1, 12.9]

    weight, bias, losses = fit(x, y, learning_rate=0.01, epochs=2000)
    print(f"Learned line: y = {weight:.3f}x + {bias:.3f}")
    print(f"Final MSE: {losses[-1]:.4f}")
