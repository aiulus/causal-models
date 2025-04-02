import numpy as np
import pytest
from scm.base import SCM
from scm import sampler


def build_chain_scm(n=5, coeff=2.0):
    nodes = [f"X{i + 1}" for i in range(n)] + ["Y"]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    functions = {}
    for i, node in enumerate(nodes):
        if i == 0:
            functions[node] = "lambda _: 0"  # No parents; noise will be added externally
        else:
            parent = nodes[i - 1]
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


def build_parallel_scm(n=3):
    nodes = [f"X{i + 1}" for i in range(n)] + ["Y"]
    edges = [(f"X{i + 1}", "Y") for i in range(n)]
    functions = {f"X{i + 1}": "lambda _: 0" for i in range(n)}
    functions["Y"] = "lambda " + ",".join([f"X{i + 1}" for i in range(n)]) + ": " + " + ".join(
        [f"X{i + 1}" for i in range(n)])
    noise = {node: ("gaussian", [0, 1]) for node in nodes}
    return SCM({"nodes": nodes, "edges": edges, "functions": functions, "noise": noise})


def test_parallel_independence():
    scm = build_parallel_scm()
    data = sampler.sample_L1(scm, 10000)

    Xs = [data[f"X{i + 1}"] for i in range(3)]
    corr_matrix = np.corrcoef(Xs)
    upper_triangle = corr_matrix[np.triu_indices(3, k=1)]

    assert np.all(np.abs(upper_triangle) < 0.1), f"Unexpected correlation: {upper_triangle}"


def test_y_high_variance_in_parallel():
    scm = build_parallel_scm()
    data = sampler.sample_L1(scm, 10000)

    y_var = np.var(data["Y"])
    x_vars = [np.var(data[f"X{i + 1}"]) for i in range(3)]

    assert y_var > max(x_vars), "Y should have higher variance due to summation"


def test_intervention_removes_variance():
    scm = build_chain_scm(n=4)
    obs = sampler.sample_L1(scm, 10000)
    inter = sampler.sample_L2(scm, 10000, interventions=["(X1, 5)"])

    # X1 should become constant
    assert np.var(inter["X1"]) < 1e-6

    # X2 onward should have reduced variance compared to observational
    for i in range(1, 5):
        key = f"X{i}"
        assert np.var(inter[key]) < np.var(obs[key])


def test_function_matches_parents():
    scm = build_chain_scm(n=4)
    for node in scm.G.nodes:
        f_str = scm.F[node]
        expected_args = list(scm.G.predecessors(node))
        for arg in expected_args:
            assert arg in f_str, f"{node}'s function is missing expected parent {arg}"


@pytest.mark.parametrize("n_samples", [1, 10, 100])
def test_sampling_shape_consistency(n_samples):
    scm = build_chain_scm(n=3)
    data = sampler.sample_L1(scm, n_samples)

    assert isinstance(data, dict)
    assert all(len(v) == n_samples for v in data.values())


def test_downstream_effect_on_Y():
    scm = build_chain_scm(n=4)
    inter1 = sampler.sample_L2(scm, 1000, interventions=["(X1, 0)"])
    inter2 = sampler.sample_L2(scm, 1000, interventions=["(X1, 10)"])

    y_diff = abs(np.mean(inter2["Y"]) - np.mean(inter1["Y"]))
    assert y_diff > 0.5, f"Expected Y to respond to X1; got ΔY = {y_diff}"





