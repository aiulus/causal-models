import pytest
import numpy as np
from scm.base import SCM
from scm import sampler, counterfactuals
from scm.functions import generate_functions
from scm.noises import parse_noise
from utils import io
import networkx as nx

# === FIXTURE: Simple SCM ===
@pytest.fixture
def simple_scm():
    # Manually construct a simple SCM dict
    G = nx.DiGraph()
    G.add_edges_from([("X1", "X2"), ("X2", "Y")])
    nodes = ["X1", "X2", "Y"]

    noise_dict = {
        "X1": ("gaussian", [0, 1]),
        "X2": ("gaussian", [0, 1]),
        "Y": ("gaussian", [0, 1]),
    }

    functions = {
        "X1": "lambda _: 0",
        "X2": "lambda X1: X1",
        "Y": "lambda X2: X2",
    }

    scm_dict = {
        "nodes": nodes,
        "edges": list(G.edges),
        "functions": functions,
        "noise": noise_dict,
    }

    return SCM(scm_dict)


# === TEST L1 ===
def test_sample_L1_shape(simple_scm):
    data = sampler.sample_L1(simple_scm, n_samples=100)
    assert all(len(v) == 100 for v in data.values())


# === TEST L2 ===
def test_sample_L2_constant_intervention(simple_scm):
    data = sampler.sample_L2(simple_scm, n_samples=100, interventions=["(X1, 5)"])
    assert np.allclose(data["X1"], 5)


# === TEST L3 ===
def test_sample_L3_output_shape(simple_scm):
    L1 = sampler.sample_L1(simple_scm, 1)
    do = ["(X1, 1)"]
    L3 = counterfactuals.sample_L3(simple_scm, L1_obs={k: v[0] for k, v in L1.items()}, interventions=do, n_samples=50)
    assert all(len(v) == 50 for v in L3.values())


# === TEST FUNCTION STRING PARSE ===
def test_function_parse_eval():
    f_str = "lambda X1, X2: X1 + 2 * X2"
    f = eval(f_str)
    assert f(3, 4) == 11


# === TEST INTERVENTION PARSING ===
def test_parse_interventions():
    do = ["(X1, 0)", "(X2, 1.5)"]
    parsed = io.parse_interventions(do)
    assert parsed == {"X1": "0", "X2": "1.5"}
