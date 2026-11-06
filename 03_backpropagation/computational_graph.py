"""A tiny computational graph showing how backpropagation works.

The Value class stores a number and the local relationships that produced it.
Calling backward() applies the chain rule from the output back to each input.
"""


class Value:
    def __init__(self, data, _children=(), _operation=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._operation = _operation
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("power must be a number")
        result = Value(self.data ** power, (self,), f"**{power}")

        def backward():
            self.grad += power * (self.data ** (power - 1)) * result.grad

        result._backward = backward
        return result

    def backward(self):
        """Compute gradients for every node that contributes to this value."""
        order = []
        visited = set()

        def build_order(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_order(parent)
                order.append(node)

        build_order(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == "__main__":
    # f = (a * b) + c
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    result = (a * b) + c
    result.backward()

    print(f"Output: {result.data:.2f}")
    print(f"df/da:  {a.grad:.2f}")
    print(f"df/db:  {b.grad:.2f}")
    print(f"df/dc:  {c.grad:.2f}")
