"""
test_engine.py - Gradient checks for the autograd engine.

Each test compares the analytical gradient (from .backward()) against a numerical
estimate using the symmetric difference  (f(x+h) - f(x-h)) / (2h).  If two
independent methods agree, the _backward rule is proven correct.
"""

import math
from engine import Value


def grad_check(micrograd_fn, plain_fn, x_val, h=1e-6, tol=1e-4):
    # Analytical gradient via micrograd
    x = Value(x_val)
    out = micrograd_fn(x)
    out.backward()
    analytical = x.grad

    # Numerical gradient via symmetric difference
    numerical = (plain_fn(x_val + h) - plain_fn(x_val - h)) / (2 * h)

    ok = abs(analytical - numerical) < tol
    print(f"[{'PASS' if ok else 'FAIL'}] x={x_val}: "
          f"analytical={analytical:.6f}  numerical={numerical:.6f}")
    assert ok


if __name__ == "__main__":
    grad_check(lambda x: x ** 2,     lambda t: t ** 2,        3.0)
    grad_check(lambda x: -x,         lambda t: -t,            4.0)
    grad_check(lambda x: x - 2,      lambda t: t - 2,         7.0)
    grad_check(lambda x: x.tanh(),   lambda t: math.tanh(t),  0.5)
    grad_check(lambda x: x.relu(),   lambda t: max(0, t),     3.0)
    grad_check(lambda x: x.relu(),   lambda t: max(0, t),    -2.0)
    print("\nAll gradient checks passed.")
