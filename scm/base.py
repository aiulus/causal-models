import networkx as nx
import numpy as np
from utils import io


class SCM:
    def __init__(self, input):
        self.nodes, self.G, self.F, self.N = io.parse_scm(input)
        self.interventions = {}
        self.rewards = {}

    def intervene(self, interventions):
        if isinstance(interventions, list):  # Accept ["(X1, 5)", "(X2, 0)"]
            interventions = io.parse_interventions(interventions)

        for variable, val in interventions.items():
            lambda_string = f"lambda _: {val}"
            self.interventions[variable] = val
            self.F[variable] = lambda_string

    def abduction(self, L1):
        noise_data = {}
        for X_j in self.G.nodes:
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
            parents_data = [L1[parent] for parent in pa_j]
            inferred_noise = L1[X_j] - f_j(*parents_data)
            noise_data[X_j] = inferred_noise
        return noise_data

    def counterfactual(self, L1, interventions, n_samples):
        noise_data = self.abduction(L1)
        self.intervene(interventions)
        L2 = self.sample(n_samples)

        L3 = {node: np.zeros(n_samples) for node in self.G.nodes}
        for X_j in nx.topological_sort(self.G):
            if X_j in L2:
                L3[X_j] = L2[X_j]
                continue
            N_j = noise_data[X_j]
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
            parents_data = [L3[parent] for parent in pa_j]
            L3[X_j] = f_j(*parents_data) + noise_data[X_j]
        return L3

    def sample(self, n_samples, mode='observational', interventions=None):
        from scm import sampler
        if mode == 'observational':
            data = sampler.sample_L1(self, n_samples)
        elif mode == 'interventional':
            if interventions is None:
                raise ValueError("A set of interventions must be provided for L2-sampling.")
            data = sampler.sample_L2(self, n_samples, interventions)
        return data

    def visualize(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_weight='bold')
        plt.show()

    def save_to_json(self, filename):
        import os
        os.makedirs(io.config['PATH_SCMs'], exist_ok=True)
        scm_data = {
            "nodes": list(self.nodes),
            "edges": list(self.G.edges),
            "functions": {k: str(v) for k, v in self.F.items()},
            "noise": self.N
        }
        file_path = os.path.join(io.config['PATH_SCMs'], filename)
        with open(file_path, 'w') as f:
            import json
            json.dump(scm_data, f, indent=2)
        print(f"SCM saved to {file_path}")

    @staticmethod
    def func_to_str(func):
        return func

    @staticmethod
    def str_to_func(func_str):
        return eval(func_str)


class LatentSCM(SCM):
    def __init__(self, input):
        super().__init__(input)
        self.H = self._extract_hidden_variables()

    def _extract_hidden_variables(self):
        return [node for node in self.G.nodes if node.startswith('H_')]

    def visualize_with_hidden(self):
        import matplotlib.pyplot as plt
        pos = nx.planar_layout(self.G) if nx.is_planar(self.G) else nx.spring_layout(self.G)
        hidden_nodes = self.H
        observable_nodes = [node for node in self.G.nodes if node not in hidden_nodes]
        nx.draw_networkx_nodes(self.G, pos, nodelist=observable_nodes, node_size=1000, node_color='lightblue',
                               font_size=10, font_weight='bold')
        nx.draw_networkx_edges(self.G, pos)
        for hidden in hidden_nodes:
            children = list(self.G.successors(hidden))
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    nx.draw_networkx_edges(self.G, pos,
                                           edgelist=[(children[i], children[j]), (children[i], children[j])],
                                           edge_color='gray', style='dashed')
        nx.draw_networkx_labels(self.G, pos)
        plt.show()
