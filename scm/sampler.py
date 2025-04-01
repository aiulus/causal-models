import networkx as nx
import numpy as np
from scm import noises as noise_utils
from utils import io


def _evaluate_function(f_str, data_dict):
    """
    Safely evaluate a lambda function string on data_dict inputs.
    Assumes noise is added outside of this function.
    """
    import re

    # Extract argument list from lambda string
    match = re.match(r'lambda\s*(.*?):', f_str)
    args = match.group(1).replace('(', '').replace(')', '').strip().split(',')
    args = [arg.strip() for arg in args if arg.strip() != '_']

    f = eval(f_str)  # Preserve original behavior

    input_vals = [data_dict[arg] for arg in args]
    return f(*input_vals) if input_vals else f('_')


def sample_L1(scm, n_samples):
    """
    Observational data: simulate from the full SCM.
    """
    noise_data = {}
    data = {}

    for X_j in scm.G.nodes:
        noise_dist = noise_utils.generate_distribution(scm.N[X_j])
        noise_data[X_j] = noise_dist(n_samples)

    for X_j in nx.topological_sort(scm.G):
        if scm.G.in_degree(X_j) == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j = scm.F[X_j]
            result = _evaluate_function(f_j, data)
            data[X_j] = result + noise_data[X_j]

    return data


def sample_L2(scm, n_samples, interventions):
    """
    Interventional data: simulate from SCM with fixed values for a subset of variables.
    """
    do_dict = io.parse_interventions(interventions)
    scm.intervene(do_dict)

    noise_data = {}
    data = {}

    for X_j in scm.G.nodes:
        if X_j in do_dict:
            continue
        noise_dist = noise_utils.generate_distribution(scm.N[X_j])
        noise_data[X_j] = noise_dist(n_samples)

    for X_j in nx.topological_sort(scm.G):
        if X_j in do_dict:
            data[X_j] = np.repeat(float(do_dict[X_j]), n_samples)
        elif scm.G.in_degree(X_j) == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j = scm.F[X_j]
            result = _evaluate_function(f_j, data)
            data[X_j] = result + noise_data[X_j]

    return data
