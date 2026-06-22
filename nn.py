import random
from engine import Value


class Module:
    """Base class. Gives every network piece a uniform way to zero grads and
    list its parameters - the same interface PyTorch's nn.Module exposes."""

    def zero_grad(self):
        # Reset every parameter's gradient to 0 before each backward pass.
        # Required because Value._backward uses += and would otherwise accumulate
        # gradients across training steps.
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron: computes activation(w . x + b).

    activation can be "tanh", "relu", or None (linear). tanh is the safe default
    for small networks - ReLU can suffer "dead units" that output 0 forever once
    their input goes negative, which can stall learning on tiny problems like XOR.
    """

    def __init__(self, n_inputs, activation="tanh"):
        # One weight Value per input, randomly initialized in [-1, 1].
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        # One bias Value.
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        # Weighted sum w . x + b. sum(..., self.b) starts the running total at
        # the bias, then adds each wi * xi on top.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "tanh":
            return act.tanh()
        elif self.activation == "relu":
            return act.relu()
        else:
            return act  # linear (no activation)

    def parameters(self):
        # All trainable Values in this neuron: its weights plus its bias.
        return self.w + [self.b]


class Layer(Module):
    """A layer: several neurons that all receive the same input."""

    def __init__(self, n_inputs, n_neurons, activation="tanh"):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def __call__(self, x):
        # Run x through every neuron, collecting the outputs.
        out = [n(x) for n in self.neurons]
        # If the layer has a single neuron, return that Value directly
        # rather than a one-element list (cleaner to chain).
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Flatten the parameters of every neuron into one list.
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """A multi-layer perceptron: a stack of layers.

    Hidden layers use the given activation (tanh by default); the final layer is
    linear so its outputs are not squashed into a fixed range.
    """

    def __init__(self, n_inputs, layer_sizes, activation="tanh"):
        # Full list of sizes: input count followed by each layer's neuron count.
        sizes = [n_inputs] + layer_sizes

        # One layer between each consecutive pair of sizes. The last layer is
        # linear (activation=None) so predictions aren't squashed.
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation=(activation if i != len(layer_sizes) - 1 else None)
            )
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        # Forward pass: each layer's output becomes the next layer's input.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Flatten the parameters of every layer into one list.
        return [p for layer in self.layers for p in layer.parameters()]
