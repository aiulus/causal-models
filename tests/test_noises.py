import pytest
from scm import noises as noise_utils
import numpy as np

# === TEST: Parsing valid noise strings ===
@pytest.mark.parametrize("input_str, expected_type, expected_params", [
    ("N(0,1)", "gaussian", [0.0, 1.0]),
    ("Exp(2)", "exponential", [2.0]),
    ("Ber(0.5)", "bernoulli", [0.5]),
])
def test_parse_noise_string(input_str, expected_type, expected_params):
    dist_type, params = noise_utils.parse_noise_string(input_str)
    assert dist_type == expected_type
    assert all(np.isclose(p, e) for p, e in zip(params, expected_params))


# === TEST: Expand to full noise dict ===
def test_parse_noise_expansion():
    nodes = ["X1", "X2", "Y"]
    noise = ["N(0,1)"]
    parsed = noise_utils.parse_noise(noise, nodes)
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == set(nodes)
    assert all(dist[0] == "gaussian" for dist in parsed.values())


# === TEST: Invalid string throws error ===
def test_invalid_noise_string_raises():
    with pytest.raises(ValueError):
        noise_utils.parse_noise_string("WeirdDist(2,3)")


# === TEST: Generate sampler and check shape ===
@pytest.mark.parametrize("noise_tuple", [
    ("gaussian", [0, 1]),
    ("exponential", [1.0]),
    ("bernoulli", [0.8])
])
def test_generate_distribution(noise_tuple):
    gen = noise_utils.generate_distribution(noise_tuple)
    samples = gen(100)
    assert len(samples) == 100
