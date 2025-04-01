import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path
import json
from typing import Dict, List, Union, Callable
from scm import StructuralEquation, NoiseDistribution, SCM


class SCMBuilder:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.functions: Dict[str, StructuralEquation] = {}
        self.noises: Dict[str, NoiseDistribution] = {}

    def add_node(self, node: str,
                 func: Union[str, Callable, None] = None,
                 noise: Union[str, NoiseDistribution, None] = None,
                 parents: List[str] = None) -> 'SCMBuilder':
        self.nodes.append(node)
        if parents:
            for p in parents:
                self.edges.append((p, node))

        if isinstance(func, str):
            self.functions[node] = self._parse_function_string(func)
        elif callable(func):
            self.functions[node] = StructuralEquation(func)

        if isinstance(noise, str):
            self.noises[node] = self._parse_noise_string(noise)
        elif isinstance(noise, NoiseDistribution):
            self.noises[node] = noise

        return self

    def set_function(self, node: str, func: Union[str, Callable]) -> 'SCMBuilder':
        if isinstance(func, str):
            self.functions[node] = self._parse_function_string(func)
        elif callable(func):
            self.functions[node] = StructuralEquation(func)
        return self

    def set_noise(self, node: str, noise: Union[str, NoiseDistribution]) -> 'SCMBuilder':
        if isinstance(noise, str):
            self.noises[node] = self._parse_noise_string(noise)
        elif isinstance(noise, NoiseDistribution):
            self.noises[node] = noise
        return self

    def build(self) -> SCM:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return SCM(graph, self.functions, self.noises)

    @staticmethod
    def from_json(json_path: Union[str, dict]) -> SCM:
        if isinstance(json_path, str):
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = json_path

        builder = SCMBuilder()
        builder.nodes = data['nodes']
        builder.edges = [tuple(edge) for edge in data['edges']]

        for node, func_str in data['functions'].items():
            builder.functions[node] = StructuralEquation(eval(func_str))

        for node, dist_data in data['noise'].items():
            dist_type = dist_data[0]
            params = dist_data[1:]
            builder.noises[node] = NoiseDistribution(dist_type, params)

        return builder.build()

    @staticmethod
    def to_json(scm: SCM, path: str):
        data = {
            "nodes": list(scm.graph.nodes),
            "edges": list(scm.graph.edges),
            "functions": {k: str(f._func) for k, f in scm.functions.items()},
            "noise": {k: [v.dist_type] + v.params for k, v in scm.noises.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _parse_noise_string(noise_str: str) -> NoiseDistribution:
        dist_type_map = {
            'N': 'gaussian',
            'Ber': 'bernoulli',
            'Exp': 'exponential'
        }
        pattern = r'([A-Za-z]+)\(([^)]+)\)'
        match = re.match(pattern, noise_str)
        if not match:
            raise ValueError(f"Invalid noise format: {noise_str}")

        noise_type, params = match.groups()
        param_list = [float(x) for x in params.split(',')]

        if noise_type not in dist_type_map:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        return NoiseDistribution(dist_type_map[noise_type], param_list)

    @staticmethod
    def _parse_function_string(func_str: str) -> StructuralEquation:
        return StructuralEquation(eval(func_str))


def parse_noise_list(noise_list: List[str], nodes: List[str]) -> Dict[str, NoiseDistribution]:
    if len(noise_list) == 1:
        noise_list *= len(nodes)
    if len(noise_list) != len(nodes):
        raise ValueError("Mismatch between number of nodes and noise terms")

    return {node: SCMBuilder._parse_noise_string(noise_str) for node, noise_str in zip(nodes, noise_list)}


def generate_functions(graph: nx.DiGraph,
                       func_type: str = 'linear') -> Dict[str, StructuralEquation]:
    functions = {}
    COEFFS = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    MAX_DEGREE = 3

    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        coeffs = np.random.choice(COEFFS, size=len(parents))

        if func_type == 'linear':
            def linear_func(*args, c=coeffs):
                return sum(c[i] * args[i] for i in range(len(args)))

            functions[node] = StructuralEquation(linear_func)

        elif func_type == 'polynomial':
            degrees = np.random.randint(1, MAX_DEGREE + 1, size=len(parents))

            def poly_func(*args, c=coeffs, d=degrees):
                return sum(c[i] * (args[i] ** d[i]) for i in range(len(args)))

            functions[node] = StructuralEquation(poly_func)

        elif func_type == 'differential':
            def diff_func(*args, c=coeffs):
                return sum(c[i] * np.sin(args[i]) for i in range(len(args)))

            functions[node] = StructuralEquation(diff_func)

        else:
            raise ValueError(f"Unsupported function type: {func_type}")

    return functions


def generate_lazy_graph(topology: str, n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = [f"X{i}" for i in range(1, n + 1)]
    G.add_nodes_from(nodes)
    if topology == 'chain':
        G.add_edges_from([(nodes[i], nodes[i + 1]) for i in range(n - 1)])
    elif topology == 'parallel':
        G.add_node("Y")
        G.add_edges_from([(node, "Y") for node in nodes])
    elif topology == 'erdos':
        p = 0.3
        G = nx.erdos_renyi_graph(n, p, directed=True)
        G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u != v])
        mapping = {i: f"X{i + 1}" for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return G


def plot_samples_dict(samples: Dict[str, np.ndarray], title: str = "SCM Samples"):
    df = pd.DataFrame(samples)
    if len(df.columns) == 1:
        df.hist(bins=30, edgecolor='k', alpha=0.7)
    else:
        pd.plotting.scatter_matrix(df, alpha=0.2)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate SCMs lazily using predefined graph topologies and function types.")
    parser.add_argument("--topology", choices=["chain", "parallel", "erdos"], required=True,
                        help="Type of graph to generate")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes (excluding Y for parallel)")
    parser.add_argument("--func_type", choices=["linear", "polynomial", "differential"], default="linear",
                        help="Function type")
    parser.add_argument("--noise", type=str, default="Ber(0.5)", help="Noise string, e.g., 'N(0,1)', 'Ber(0.5)'")
    parser.add_argument("--mode", choices=["none", "l1", "l2", "l3"], default="none",
                        help="Sampling mode: none, l1, l2, l3")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--do", nargs='*', help="List of do-interventions in the form 'X1=0.5 X2=1.0'")
    parser.add_argument("--plot", action="store_true", help="Show plots of sampled data")

    args = parser.parse_args()

    graph = generate_lazy_graph(args.topology, args.n)
    functions = generate_functions(graph, func_type=args.func_type)
    noises = parse_noise_list([args.noise], list(graph.nodes))
    scm = SCM(graph, functions, noises)

    output_dir = Path.cwd().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"scm_{args.topology}_{args.func_type}_n{args.n}.json"
    output_path = output_dir / filename

    SCMBuilder.to_json(scm, str(output_path))
    print(f"SCM saved to {output_path}")

    # Sampling operations
    if args.mode != "none":
        print(f"Performing {args.mode.upper()} sampling with {args.samples} samples...")

        def parse_do(do_list):
            return {kv.split('=')[0]: float(kv.split('=')[1]) for kv in do_list}

        if args.mode == "l1":
            samples = scm.sample(args.samples)
        elif args.mode == "l2":
            interventions = parse_do(args.do) if args.do else {}
            def apply_do(model):
                for var, val in interventions.items():
                    model.functions[var] = StructuralEquation(lambda *args, v=val: np.full(args[0].shape, v))
            scm.apply_intervention(apply_do)
            samples = scm.sample(args.samples)
        elif args.mode == "l3":
            L1 = scm.sample(args.samples)
            interventions = parse_do(args.do) if args.do else {}
            samples = scm.counterfactual(L1, interventions, args.samples)
        else:
            samples = None

        if samples:
            for k, v in samples.items():
                print(f"{k}: {v[:5]} ...")  # Show first 5 values only
        # Convert to DataFrame and save
        df = pd.DataFrame(samples)
        csv_filename = filename.replace(".json", f"_{args.mode}_samples.csv")
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"Samples saved to {csv_path}")
        if args.plot:
            plot_samples_dict(samples, title=f"{args.mode.upper()} Sampling")


if __name__ == "__main__":
    main()