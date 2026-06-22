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
        return self * -1

    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        out = Value(math.tanh(self.data), _children=(self,))

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)**2. 
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
