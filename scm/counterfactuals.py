import numpy as np
import networkx as nx
from scm import sampler


def abduction(scm, L1_obs):
    """
    Infer latent noise variables N_i from observed values under the observational SCM.
    """
    noise_data = {}

    for X_j in scm.G.nodes:
        f_j = eval(scm.F[X_j])
        parents = list(scm.G.predecessors(X_j))
        parent_vals = [L1_obs[parent] for parent in parents]
        inferred_noise = L1_obs[X_j] - (f_j(*parent_vals) if parent_vals else f_j('_'))
        noise_data[X_j] = inferred_noise

    return noise_data


def predict_with_modified_model(scm, noise_data, n_samples):
    """
    Given the modified SCM and fixed noise, simulate predicted values.
    """
    L3 = {node: np.zeros(n_samples) for node in scm.G.nodes}

    for X_j in nx.topological_sort(scm.G):
        if X_j in scm.interventions:
            # intervention already sets a constant value through lambda _: ...
            L3[X_j] = np.repeat(eval(scm.F[X_j])('_'), n_samples)
        else:
            f_j = eval(scm.F[X_j])
            parents = list(scm.G.predecessors(X_j))
            parent_vals = [L3[parent] for parent in parents]
            L3[X_j] = f_j(*parent_vals) + noise_data[X_j]

    return L3


def sample_L3(scm, L1_obs, interventions, n_samples):
    """
    Counterfactual simulation: L1 -> L2' -> L3 (prediction under altered SCM).
    """
    # Step 1: Abduction
    noise_data = abduction(scm, L1_obs)

    # Step 2: Intervention
    scm.intervene(interventions)

    # Step 3: Prediction
    return predict_with_modified_model(scm, noise_data, n_samples)
