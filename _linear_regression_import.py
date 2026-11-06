"""Small import helper so the visualisation can run from the project root."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parent / "01_linear_regression" / "gradient_descent.py"


def load_fit():
    spec = importlib.util.spec_from_file_location("gradient_descent", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit
