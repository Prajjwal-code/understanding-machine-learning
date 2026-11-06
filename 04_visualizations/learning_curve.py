"""Plot the loss recorded during gradient descent.

Install matplotlib with `python -m pip install matplotlib` to use this file.
The training code itself does not depend on matplotlib.
"""

from pathlib import Path
import sys

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from _linear_regression_import import load_fit  # noqa: E402


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Matplotlib is required for this plot. Run: python -m pip install matplotlib"
        ) from error

    fit = load_fit()
    x = [1, 2, 3, 4, 5, 6]
    y = [3.1, 4.9, 7.2, 8.8, 11.1, 12.9]
    _, _, losses = fit(x, y, learning_rate=0.01, epochs=2000)

    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("Gradient descent: loss over time")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
