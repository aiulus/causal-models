from typing import Callable, Dict, List
import networkx as nx
import numpy as np


class NoiseDistribution:
    def __init__(self, dist_type: str, params: List[float]):
        self.dist_type = dist_type
        self.params = params

    def sample(self, size: int) -> np.ndarray:
        from scipy.stats import norm, bernoulli, expon

        if self.dist_type in ('gaussian', 'gau'):
            return norm.rvs(*self.params, size=size)
        elif self.dist_type in ('bernoulli', 'ber'):
            return bernoulli.rvs(*self.params, size=size)
        elif self.dist_type in ('exponential', 'exp'):
            return expon.rvs(scale=1 / self.params[0], size=size)
        else:
            raise ValueError(f"Unsupported distribution: {self.dist_type}")


class StructuralEquation:
    def __init__(self, func: Callable):
        self._func = func

    def evaluate(self, *args):
        return self._func(*args)


class SCM:
    def __init__(self, graph: nx.DiGraph,
                 functions: Dict[str, StructuralEquation],
                 noises: Dict[str, NoiseDistribution]):
        self.graph = graph
        self.functions = functions
        self.noises = noises

    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        data = {}
        noise_values = {
            node: self.noises[node].sample(n_samples)
            for node in self.graph.nodes
        }

        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            parent_data = [data[p] for p in parents]
            f_j = self.functions[node]
            result = f_j.evaluate(*parent_data)
            result += noise_values[node]
            data[node] = result

        return data

    def apply_intervention(self, intervention_fn: Callable[["SCM"], None]):
        intervention_fn(self)

    def apply_abduction(self, observations):
        # TODO: must update self.noises such that they comply with 'observations'
        return 0

    def get_structure(self):
        return {
            'nodes': list(self.graph.nodes),
            'edges': list(self.graph.edges),
            'functions': {k: str(f._func) for k, f in self.functions.items()},
            'noise': {k: (v.dist_type, v.params) for k, v in self.noises.items()}
        }
