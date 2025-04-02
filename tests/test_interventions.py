import numpy as np
import pytest
from scm.base import SCM
from scm import sampler


def build_simple_scm():
    """
    Create a simple SCM with a known structure:
    X1 → X2 → Y
    """
    nodes = ["X1", "X2", "Y"]
    edges = [("X1", "X2"), ("X2", "Y")]

    functions = {
        "X1": "lambda _: 0",  # Noise only
        "X2": "lambda X1: 2 * X1",  # X2 = 2*X1 + N_X2
        "Y": "lambda X2: 3 * X2"    # Y = 3*X2 + N_Y
    }

    noise = {
        "X1": ("gaussian", [0, 1]),
        "X2": ("gaussian", [0, 1]),
        "Y": ("gaussian", [0, 1])
    }

    scm_data = {
        "nodes": nodes,
        "edges": edges,
        "functions": functions,
        "noise": noise
    }

    return SCM(scm_data)


def test_single_intervention_constant_value():
    scm = build_simple_scm()
    data = sampler.sample_L2(scm, n_samples=1000, interventions=["(X1, 5)"])

    # X1 should be a constant vector of 5s
    assert np.allclose(data["X1"], 5), "X1 values should all be 5 after intervention"
    assert data["X2"].std() > 0, "X2 should vary due to noise"
    assert data["Y"].std() > 0, "Y should vary due to noise"


def test_multiple_interventions():
    scm = build_simple_scm()
    data = sampler.sample_L2(scm, n_samples=500, interventions=["(X1, 1)", "(X2, 10)"])

    assert np.allclose(data["X1"], 1), "X1 values should all be 1"
    assert np.allclose(data["X2"], 10), "X2 values should all be 10"
    assert data["Y"].std() > 0, "Y should still vary"


def test_intervened_variable_is_constant():
    scm = build_simple_scm()
    interventions = ["(X2, 8)"]
    data = sampler.sample_L2(scm, n_samples=500, interventions=interventions)

    assert np.allclose(data["X2"], 8), "Intervened variable X2 should be constant"
    # Upstream variable X1 should be unaffected
    assert data["X1"].std() > 0, "X1 should remain stochastic"


def test_output_shapes_and_keys():
    scm = build_simple_scm()
    data = sampler.sample_L2(scm, n_samples=100, interventions=["(X1, 3)"])

    assert isinstance(data, dict)
    assert set(data.keys()) == {"X1", "X2", "Y"}
    assert all(len(values) == 100 for values in data.values())


def test_intervene_directly_on_object():
    scm = build_simple_scm()
    scm.intervene({"X2": 7})
    assert "X2" in scm.interventions
    assert scm.F["X2"] == "lambda _: 7"
