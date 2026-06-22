"""
engine.py - A tiny scalar-valued automatic differentiation engine.

The Value class wraps a single number and records how it was computed, so that
calling .backward() can walk the computation graph in reverse and fill in the
gradient of the final output with respect to every Value that contributed to it.

The one idea behind every operation below:

    each operation's _backward does:  input.grad += (local derivative) * out.grad

That single pattern - local derivative times the incoming gradient - is the
chain rule, and it is the entire mechanism of backpropagation.
"""

import math


class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=()):
        self.data = data                  # the actual number this node holds
        self.grad = 0.0                   # d(final output) / d(this value); starts at 0
        self._children = set(_children)   # the Values that produced this one (graph edges)
        self._backward = lambda: None     # how this node pushes gradient to its children
                                          # (default does nothing - correct for leaf nodes)

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # Allow `Value + raw_number` by wrapping the raw number in a Value.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other))

        def _backward():
            # Addition passes gradient straight through (local derivative = 1).
            # += accumulates, so a value used in multiple places sums its contributions.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other))

        def _backward():
            # For out = self * other, the local derivative wrt self is other's VALUE.
            # (derivative of x*k wrt x is k) - so we use other.data, not other.grad.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Compute the gradient of this value with respect to every node behind it."""
        # 1. Build a topological ordering: every node appears after its children,
        #    so reversing it processes the output first and inputs last.
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # 2. Seed the output gradient: d(output)/d(output) = 1.
        self.grad = 1.0

        # 3. Apply the chain rule backward through the graph.
        for node in reversed(topo):
            node._backward()

    # ---- Extended operations ----

    def __pow__(self, other):
        # Only constant (int/float) exponents are supported - the exponent is a
        # fixed part of the formula, never a trainable value, so it gets no gradient.
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, _children=(self,))

        def _backward():
            # Power rule: d/dx (x**n) = n * x**(n-1), then * out.grad for the chain rule.
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        # -self is just self * -1, reusing __mul__ (gradient handled for free).
        return self * -1

    def __sub__(self, other):
        # a - b is a + (-b), reusing __add__ and __neg__.
        return self + (-other)

    # Reverse-operator hooks so raw numbers on the LEFT also work,
    # e.g. `1 - Value(2)` calls __rsub__, `2 * Value(3)` calls __rmul__.
    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        out = Value(math.tanh(self.data), _children=(self,))

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)**2. Since out.data IS tanh(x),
            # the whole local derivative is computable from the output alone.
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), _children=(self,))

        def _backward():
            # ReLU derivative: 1 if the input was positive, else 0.
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = _backward
        return out
