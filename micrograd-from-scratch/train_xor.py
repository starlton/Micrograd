"""
train_xor.py - Train the network to solve XOR.

XOR is the classic test: its output classes are not linearly separable, so a
single neuron cannot solve it. A network with hidden layers can, by bending the
decision boundary. Running this script trains an MLP and prints the loss falling.
"""

from nn import MLP

# XOR dataset: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]

# 2 inputs -> hidden 4 -> hidden 4 -> 1 output.
# tanh hidden layers converge reliably on XOR (ReLU can get dead units here).
model = MLP(2, [4, 4, 1], activation="tanh")
print(f"Network has {len(model.parameters())} parameters\n")

learning_rate = 0.05
epochs = 200

for step in range(epochs):
    # 1. Forward pass: predict all four examples.
    ypreds = [model(x) for x in xs]

    # 2. Loss: mean squared error.
    loss = sum((ypred - ygt) ** 2 for ygt, ypred in zip(ys, ypreds))

    # 3. Zero gradients, then backpropagate.
    model.zero_grad()
    loss.backward()

    # 4. Update every parameter downhill (opposite the gradient).
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.data:.6f}")

# Show final predictions
print("\nFinal predictions:")
for x, y in zip(xs, ys):
    pred = model(x)
    print(f"  input {x} -> predicted {pred.data:+.3f}  (target {y})")
