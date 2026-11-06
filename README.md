# Understanding Machine Learning

A small, from-scratch collection of Python examples covering a few ideas that
show up in introductory machine learning:

- fitting a line with gradient descent
- measuring prediction error with mean squared error
- following gradients through a simple computational graph
- plotting how loss changes during training

The goal is to make the calculations easy to inspect. The examples use plain
Python so you can follow the steps without relying on a machine-learning
framework.

## Project structure

```text
understanding-machine-learning/
│
├── 01_linear_regression/
│   └── gradient_descent.py
│
├── 02_loss_functions/
│   └── mse.py
│
├── 03_backpropagation/
│   └── computational_graph.py
│
├── 04_visualizations/
│   └── learning_curve.py
│
├── _linear_regression_import.py
└── README.md
```

## Requirements

- Python 3.9 or newer
- Matplotlib only for the learning-curve plot

No NumPy, pandas, scikit-learn, TensorFlow, or PyTorch is needed for the
examples themselves.

## Run the examples

Open a terminal in the project folder.

### 1. Linear regression with gradient descent

```bash
python 01_linear_regression/gradient_descent.py
```

The program starts with a weight and bias of zero. Each epoch calculates the
gradients of the mean squared error and adjusts both parameters.

### 2. Mean squared error

```bash
python 02_loss_functions/mse.py
```

MSE squares each prediction error and takes the average. Larger errors have a
larger effect because the difference is squared.

### 3. A small computational graph

```bash
python 03_backpropagation/computational_graph.py
```

This example evaluates `f = (a * b) + c`, then works backwards to calculate
the derivative of the output with respect to each input.

### 4. Plot the learning curve

First install Matplotlib:

```bash
python -m pip install matplotlib
```

Then run:

```bash
python 04_visualizations/learning_curve.py
```

The plot shows the training loss after each epoch. A downward curve means the
model is getting a smaller error on this training example.

## The main idea

For a straight-line model,

```text
prediction = weight * input + bias
```

gradient descent repeatedly does three things:

1. Make predictions using the current weight and bias.
2. Calculate how much the loss changes when each parameter changes.
3. Move the parameters in the direction that reduces the loss.

The learning rate controls the size of each update. If it is too large, the
updates can jump around or diverge. If it is too small, learning can be slow.

## A note about the examples

These are teaching examples, not production machine-learning tools. The
training data is deliberately tiny so the calculations stay readable. Try
changing the data, learning rate, or number of epochs and see what happens.

## License

MIT
