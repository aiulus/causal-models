# examples/scm_demo.py

import os
import numpy as np
from scm.base import SCM
from utils import io
from graphs import generator
from scm import sampler

# === Create and save SCM from graph + functions ===
print("== Generating SCM from chain graph ==")
graph_dict = generator.generate_chain_graph(4)
nodes = graph_dict["nodes"]

# Default: linear functions with Gaussian noise
noise_dict = {node: ("gaussian", [0, 1]) for node in nodes}
functions = {
    nodes[0]: "lambda _: 0"
}
for i in range(1, len(nodes)):
    parent = nodes[i - 1]
    functions[nodes[i]] = f"lambda {parent}: 2.0 * {parent}"

scm = SCM.from_functions(functions, noise_dict)
scm.visualize()

# === Sample L1 data ===
print("\n== Sampling observational data (L1) ==")
L1 = scm.sample(n_samples=10, mode='observational')
for k, v in L1.items():
    print(f"{k}: {v[:5]}...")

# === Sample L2 data (atomic intervention) ===
print("\n== Sampling interventional data (L2, do(X1=5)) ==")
L2 = scm.sample(n_samples=10, mode='interventional', interventions={"X3": 5})
for k, v in L2.items():
    print(f"{k}: {v[:5]}...")

# === Extended intervention ===
print("\n== Applying extended intervention to X2 (new function and noise) ==")
scm.intervene({
    "X2": {
        "function": "lambda X1: X1 + 3",
        "noise": ("gaussian", [0, 0.1])
    }
})
L2_ext = scm.sample(n_samples=10, mode='interventional', interventions={"X2": {"function": "lambda X1: X1 + 3"}})
for k, v in L2_ext.items():
    print(f"{k}: {v[:5]}...")

# === Counterfactuals ===
# print("\n== Counterfactual inference ==")
# cf = scm.counterfactual(L1, {"X1": 10}, n_samples=10)
# for k, v in cf.items():
#    print(f"{k}: {v[:5]}...")

# === Save SCM to JSON ===
print("\n== Saving SCM to file ==")
scm_filename = "demo_chain_scm.json"
scm.save_to_json(scm_filename)

# === Load SCM from file ===
print("\n== Loading SCM from saved JSON file ==")
loaded_scm = SCM.from_json(os.path.join(io.config["PATH_SCMs"], scm_filename))
loaded_scm.visualize()

