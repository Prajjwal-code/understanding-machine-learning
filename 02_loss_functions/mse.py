"""Mean squared error, implemented without NumPy."""


def mse(actual, predicted):
    """Calculate the average squared difference between two sequences."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if len(actual) == 0:
        raise ValueError("cannot calculate MSE for empty sequences")

    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    return sum(squared_errors) / len(squared_errors)


def main():
    actual = [3, 5, 7, 9]
    predicted = [2.5, 5.5, 6, 10]
    print("Actual:   ", actual)
    print("Predicted:", predicted)
    print(f"MSE: {mse(actual, predicted):.3f}")


if __name__ == "__main__":
    main()
