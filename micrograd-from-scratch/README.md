# micrograd-from-scratch

A tiny scalar-valued automatic differentiation engine and a small neural network
library built on top of it - implemented from scratch to understand how
backpropagation actually works under the hood of libraries like PyTorch.

This was built as a learning project, following along with and then re-deriving
the ideas from [Andrej Karpathy](https://github.com/karpathy)'s excellent
[micrograd](https://github.com/karpathy/micrograd) and his
[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
series. Every line here was typed and reasoned through by hand, not copy-pasted -
the goal was understanding, not just a working result. Huge thanks to Andrej for
making this material so clear and freely available.

## What's inside

| File | What it does |
|------|--------------|
| `engine.py` | The `Value` class - scalar autograd with a full backward pass |
| `nn.py` | `Module`, `Neuron`, `Layer`, `MLP` built on `Value` |
| `train_xor.py` | Trains a network to solve XOR |
| `test_engine.py` | Gradient checks: analytical vs numerical |

## The core idea

Every math expression is a **computational graph**. Each `Value` remembers how it
was computed, so calling `.backward()` can walk the graph in reverse and fill in
the gradient of the output with respect to every input.

The entire engine rests on one pattern. Every operation's backward step is:

```
input.grad += (local derivative) * out.grad
```

That is the chain rule, applied mechanically backward through the graph. Once you
see that `+`, `*`, `**`, `tanh`, and `relu` are all just different "local
derivatives" plugged into that same shape, backpropagation stops being magic.

| Operation | Local derivative | Backward line |
|-----------|------------------|---------------|
| `a + b` | 1 | `self.grad += out.grad` |
| `a * b` | the other operand's value | `self.grad += other.data * out.grad` |
| `a ** n` | `n * a**(n-1)` | `self.grad += n * self.data**(n-1) * out.grad` |
| `tanh(a)` | `1 - tanh(a)**2` | `self.grad += (1 - out.data**2) * out.grad` |
| `relu(a)` | 1 if a > 0 else 0 | `self.grad += (1 if self.data>0 else 0) * out.grad` |

## Quick start

```python
from engine import Value

# Build an expression - the graph is recorded automatically
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)
f = (a + b) * c        # f.data == 20

f.backward()           # fill in gradients
print(a.grad, b.grad, c.grad)   # 4.0 4.0 5.0
```

## Training a network

```python
from nn import MLP

model = MLP(2, [4, 4, 1])   # 2 inputs -> hidden 4 -> hidden 4 -> 1 output

# standard loop: forward -> loss -> zero_grad -> backward -> update
ypred = [model(x) for x in xs]
loss = sum((p - y)**2 for y, p in zip(ys, ypred))
model.zero_grad()
loss.backward()
for p in model.parameters():
    p.data -= 0.05 * p.grad
```

Run the XOR demo:

```bash
python train_xor.py
```

Trained output (XOR is not linearly separable, so this requires hidden layers):

```
input [0.0, 0.0] -> predicted +0.000  (target 0.0)
input [0.0, 1.0] -> predicted +1.000  (target 1.0)
input [1.0, 0.0] -> predicted +1.000  (target 1.0)
input [1.0, 1.0] -> predicted +0.001  (target 0.0)
```

## Verifying the gradients

The backward rules are checked against a numerical estimate using the symmetric
difference `(f(x+h) - f(x-h)) / (2h)`. If two independent methods agree, the
gradient is correct.

```bash
python test_engine.py
```

```
[PASS] x=3.0: analytical=6.000000  numerical=6.000000
[PASS] x=0.5: analytical=0.786448  numerical=0.786448
...
All gradient checks passed.
```

## A note on activations

`Neuron`, `Layer`, and `MLP` accept an `activation` argument: `"tanh"` (default),
`"relu"`, or `None` (linear). For tiny networks like the XOR demo, **tanh** is the
reliable choice: ReLU can suffer from "dead units" that output 0 forever once
their input goes negative, which can stall learning on small problems. The output
layer is always linear so predictions aren't squashed into a fixed range.

## Architecture

```
MLP      -> a list of Layers      (the full network)
Layer    -> a list of Neurons     (neurons sharing one input)
Neuron   -> weights + bias        (activation(w . x + b))
Value    -> a scalar + its grad   (the autograd engine underneath)
```

None of the network classes contain gradient logic - they only organize `Value`
objects. The engine does all the differentiation.

## Credits

Built following Andrej Karpathy's micrograd and *Neural Networks: Zero to Hero*.
The structure (Value engine + Module / Neuron / Layer / MLP) mirrors his design.
