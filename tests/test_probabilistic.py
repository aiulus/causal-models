import numpy as np
import pytest
from scm.base import SCM
from scm import sampler

def build_chain_scm(n=5, coeff=2.0):
    nodes = [f"X{i+1}" for i in range(n)] + ["Y"]
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

    functions = {}
    for i, node in enumerate(nodes):
        if i == 0:
            functions[node] = "lambda _: 0"  # No parents; noise will be added externally
        else:
            parent = nodes[i-1]
            functions[node] = f"lambda {parent}: {coeff} * {parent}"

    # Format: {node: ("gaussian", [mu, sigma])}
    noise_dict = {node: ("gaussian", [0, 1]) for node in nodes}

    scm_data = {
        "nodes": nodes,
        "edges": edges,
        "functions": functions,
        "noise": noise_dict
    }
    return SCM(scm_data)


def test_chain_variance_propagation():
    scm = build_chain_scm(n=4)
    data = sampler.sample_L1(scm, 10000)

    variances = [np.var(data[node]) for node in scm.G.nodes]
    print("Variances:", variances)

    assert all(x < y for x, y in zip(variances, variances[1:])), \
        "Variance should strictly increase along the causal chain"
