import argparse
import os
from scm.base import SCM
from scm.functions import generate_functions
from scm.noises import parse_noise
from graphs import generator
from utils import io, plot


def main():
    config = io.config
    parser = argparse.ArgumentParser("Structural Causal Model (SCM) CLI")

    parser.add_argument("--graph_type", required=True, choices=["chain", "parallel", "random"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, help="Edge prob for random DAG")
    parser.add_argument("--pa_n", type=int, default=1, help="Num of parents of Y")
    parser.add_argument("--vstr", type=int, default=-1, help="# of v-structures")
    parser.add_argument("--conf", type=int, default=-1, help="# of confounders")

    parser.add_argument("--noise_types", nargs='+', default=['N(0,1)'])
    parser.add_argument("--funct_type", choices=["linear", "polynomial"], default="linear")

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    path_scm = config['PATH_SCMs']
    path_graphs = config['PATH_GRAPHS']

    # Load or generate graph
    if args.graph_type == "chain":
        graph = generator.generate_chain_graph(args.n)
    elif args.graph_type == "parallel":
        graph = generator.generate_parallel_graph(args.n)
    elif args.graph_type == "random":
        graph = generator.erdos_with_constraints(
            n=args.n, p=args.p,
            pa_y=args.pa_n,
            confs=args.conf,
            vstr=args.vstr
        )

    # Save graph to disk (optional)
    if args.save:
        graph_filename = io.scm_args_to_filename(args, 'json', path_graphs)
        os.makedirs(path_graphs, exist_ok=True)
        with open(graph_filename, 'w') as f:
            import json
            json.dump(graph, f, indent=2)

    # Prepare noise and structural functions
    nodes = graph['nodes']
    noise_dict = parse_noise(args.noise_types, nodes)
    functions = generate_functions(graph=nx_from_dict(graph), noise_vars=noise_dict, funct_type=args.funct_type)

    # Build SCM data
    scm_data = {
        "nodes": nodes,
        "edges": graph['edges'],
        "functions": {k: SCM.func_to_str(v) for k, v in functions.items()},
        "noise": {k: v for k, v in noise_dict.items()}
    }

    scm = SCM(scm_data)
    save_path = io.scm_args_to_filename(args, 'json', path_scm)
    scm.save_to_json(os.path.basename(save_path))

    if args.plot:
        plot.draw_scm(os.path.basename(save_path))
        # scm.visualize()


def nx_from_dict(graph_dict):
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(graph_dict['nodes'])
    G.add_edges_from(graph_dict['edges'])
    return G
