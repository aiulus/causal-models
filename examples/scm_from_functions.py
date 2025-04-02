from scm.base import SCM
from scm import sampler
import numpy as np

# === Define structural equations ===
functions = {
    "X1": "lambda _: 1",
    "X2": "lambda X1: 2 * X1",
    "X3": "lambda X1, X2: X1 + X2",
    "Y": "lambda X3: 3 * X3"
}

# === Define noise distributions ===
noise = {
    "X1": ("gaussian", [0, 1]),
    "X2": ("gaussian", [0, 1]),
    "X3": ("gaussian", [0, 1]),
    "Y": ("gaussian", [0, 1])
}

# === Build the SCM ===
scm = SCM.from_functions(functions, noise)

# === Sample observational data (L1) ===
data = scm.sample(n_samples=1000, mode='observational')

# === Print summary statistics ===
for var, samples in data.items():
    print(f"{var}: mean={np.mean(samples):.2f}, var={np.var(samples):.2f}")
