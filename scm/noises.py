# scm/noise.py

import re
from scipy.stats import norm, bernoulli, expon
from utils import io

# Supported string-to-distribution mappings
DIST_TYPE_MAP = {
    'N': 'gaussian',
    'Exp': 'exponential',
    'Ber': 'bernoulli',
}

# String → ("type", [params...])
def parse_noise_string(noise_str):
    """
    Example: 'N(0,1)' → ('gaussian', [0,1])
    """
    pattern = r'([A-Za-z]+)\(([^)]+)\)'
    match = re.match(pattern, noise_str)
    if not match:
        raise ValueError(f"Invalid distribution format: {noise_str}")

    dist_name, param_str = match.groups()
    params = [float(x.strip()) for x in param_str.split(',')]

    if dist_name not in DIST_TYPE_MAP:
        raise ValueError(f"Unsupported distribution: {dist_name}")

    return DIST_TYPE_MAP[dist_name], params


def parse_noise(noise_list, nodes):
    """
    Expand a list of noise strings to a dictionary keyed by nodes.
    """
    if isinstance(noise_list, str):
        noise_list = [noise_list]

    n = len(nodes)
    if len(noise_list) == 1:
        noise_list *= n
    elif len(noise_list) != n:
        raise ValueError(f"Expected 1 or {n} noise terms, got: {len(noise_list)}")

    return {
        node: parse_noise_string(noise_str)
        for node, noise_str in zip(nodes, noise_list)
    }


def generate_distribution(noise_tuple):
    dist_type, params = noise_tuple
    if dist_type == 'gaussian':
        mu, sigma = params
        return lambda x: norm.rvs(mu, sigma, size=x)
    elif dist_type == 'exponential':
        lam = params[0]
        return lambda x: expon.rvs(scale=1 / lam, size=x)
    elif dist_type == 'bernoulli':
        p = params[0]
        return lambda x: bernoulli.rvs(p, size=x)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
