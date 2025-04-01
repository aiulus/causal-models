from pathlib import Path
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import noises, plots, structural_equations, graph_generator, io_mgmt, sampling

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']


def parse_interventions(interventions):
    """
    Parse intervention strings like 'do(X_i=a)' into a dictionary.
    Example: "do(X1=0)" --> {"X1": 0}
    """
    interventions_dict = {}
    for intervention in interventions:
        var, val = intervention.replace('do(', '').replace(')', '').split('=')
        interventions_dict[var.strip()] = float(val.strip())

    return interventions_dict


# TODO: Extend to other than just fully-observed SCM's
class SCM:
    def __init__(self, input):
        self.nodes, self.G, self.F, self.N = io_mgmt.parse_scm(input)
        self.interventions = {}
        self.rewards = {}

    def intervene(self, interventions):
        """Perform interventions on multiple variables.

        Interventions can be perfect (constant value) or soft (stochastic function).
        """
        for variable, func in interventions.items():
            lambda_string = f"lambda _: {func}"
            self.interventions[variable] = func
            self.F[variable] = lambda_string

    def abduction(self, L1):
        """Infer the values of the exogenous variables given observational outputs"""
        noise_data = {}
        for X_j in self.G.nodes:
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
            parents_data = [L1[parent] for parent in pa_j]
            inferred_noise = L1[X_j] - f_j(*parents_data)
            noise_data[X_j] = inferred_noise
        return noise_data

    def counterfactual(self, L1, interventions, n_samples):
        """Compute counterfactual distribution given L1-outputs and an intervention."""
        # Step 1: Abduction - Update the noise distribution given the observations
        noise_data = self.abduction(L1)

        # Step 2: Action - Intervene within the observationally constrained SCM
        self.intervene(interventions)
        L2 = self.sample(n_samples)

        # Step 3: Prediction - Generate samples in the modified model
        L3 = {node: np.zeros(n_samples) for node in self.G.nodes}
        for X_j in nx.topological_sort(self.G):
            if X_j in L2:
                L3[X_j] = L2[X_j]
                continue

            N_j = noise_data[X_j]
            f_j = eval(self.F[X_j])
            pa_j = list(self.G.predecessors(X_j))
            parents_data = [L3[parent] for parent in pa_j]
            L3[X_j] = f_j(*parents_data) + noise_data

        return L3

    def sample(self, n_samples, mode='observational', interventions=None):
        if mode == 'observational':
            data = sampling.sample_L1(self, n_samples)
        elif mode == 'interventional':
            if interventions is None:
                raise ValueError("A set of interventions must be provided for L2-sampling.")
            data = sampling.sample_L2(self, n_samples, interventions)

        return data

    def visualize(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_weight='bold')
        plt.show()

    def save_to_json(self, filename):
        os.makedirs(PATH_SCM, exist_ok=True)
        scm_data = {
            "nodes": [node for node in self.nodes],
            "edges": [edge for edge in self.G.edges],
            "functions": {k: str(v) for k, v in self.F.items()},
            # TODO: Save only the type(s) of distribution for the noises
            "noise": self.N
        }
        file_path = os.path.join(PATH_SCM, filename)
        with open(file_path, 'w') as f:
            json.dump(scm_data, f, indent=2)
        print(f"SCM saved to {file_path}")

    @staticmethod
    def func_to_str(func):
        return func

    @staticmethod
    def str_to_func(func_str):
        return eval(func_str)


class LatentSCM(SCM):
    """
    self.G: Acyclic Directed Mixed Graph (ADMG)
    """
    def __init__(self, input):
        super().__init__(input)
        self.H = self._extract_hidden_variables()

    def _extract_hidden_variables(self):
        """
        TODO
        Assumes a naming convention that hidden nodes are encoded with names starting with 'H_'
        but currently no method to mark certain nodes as hidden (*_) has been implemented.
        (*_) Bidirectional Edges !!
        """
        hidden_variables = [node for node in self.G.nodes if node.startswith('H_')]
        return hidden_variables

    def visualize_with_hidden(self):
        pos = nx.planar_layout(self.G) if nx.is_planar(self.G) else nx.spring_layout(self.G)
        hidden_nodes = self.H
        observable_nodes = [node for node in self.G.nodes if node not in hidden_nodes]
        nx.draw_networkx_nodes(self.G, pos, nodelist=observable_nodes, node_size=1000, node_color='lightblue', font_size=10, font_weight='bold')
        nx.draw_networkx_edges(self.G, pos)
        # Draw bidirectional edges for each pair of the given hidden variable's children
        for hidden in hidden_nodes:
            children = list(self.G.successors(hidden))
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    nx.draw_networkx_edges(self.G, pos, edgelist=[(children[i], children[j]), (children[i], children[j])], edge_color='gray', style='dashed')
        nx.draw_networkx_labels(self.G, pos)
        plt.show()


def main():
    parser = argparse.ArgumentParser("Structural Causal Model (SCM) operations.")
    parser.add_argument("--graph_type", choices=['chain', 'parallel', 'random'],
                        help="Type of graph structure to generate. Currently supported: ['chain', 'parallel', 'random']")
    # TODO: help info
    parser.add_argument('--noise_types', default='N(0,1)', type=str, nargs='+',
                        help='Specify distribution types for noise terms with --noise_types. '
                             'Currently supported: [N(mu, sigma), Exp(lambda), Ber(p)]')

    parser.add_argument('--funct_type', type=str, default='linear', choices=['linear', 'polynomial'],
                        help="Specify the function family "
                             "to be used in structural "
                             "equations. Currently "
                             "supported: ['linear', "
                             "'polynomial']")

    parser.add_argument("--n", type=int, required=True, help="Number of (non-reward) nodes in the graph.")
    # Required for --graph_type random
    parser.add_argument("--p", type=float, help="Denseness of the graph / prob. of including any potential edge.")
    parser.add_argument("--pa_n", type=int, default=-1, help="Cardinality of pa_Y in G.")
    parser.add_argument("--vstr", type=int, default=-1, help="Desired number of v-structures in the causal graph.")
    parser.add_argument("--conf", type=int, default=-1,
                        help="Desired number of confounding variables in the causal graph.")
    parser.add_argument("--intervene", type=str, help="JSON string representing interventions to perform.")
    parser.add_argument("--plot", action='store_true')
    # TODO: Currently no method for re-assigning default source/target paths
    parser.add_argument("--path_graphs", type=str, default=PATH_GRAPHS, help="Path to save/load graph specifications.")
    parser.add_argument("--path_scm", type=str, default=PATH_SCM, help="Path to save/load SCM specifications.")
    parser.add_argument("--path_plots", type=str, default=PATH_PLOTS, help="Path to save the plots.")

    args = parser.parse_args()

    save_path = io_mgmt.scm_args_to_filename(args, 'json', PATH_SCM)

    if args.plot:
        plots.draw_scm(save_path)
        return

    if args.noise_types is not None:
        arg_count = len(args.noise_types)
        if arg_count != 1 and arg_count != args.n + 1:
            raise ValueError(f"Provided: {args.noise_types}. Invalid number of noise terms: {arg_count}\n"
                             f"Specify either exactly one noise distribution or |X| - many !")

    graph_type = f"{args.graph_type}_graph_N{args.n}"
    file_path = f"{PATH_GRAPHS}/{graph_type}.json"
    if args.graph_type == 'random':
        graph_type = f"random_graph_N{args.n}_paY_{args.pa_n}_p_{str(args.p).replace('.', '')}_graph_N{args.n}"
        file_path = f"{PATH_GRAPHS}/{graph_type}.json"
    try:
        graph = io_mgmt.load_graph_from_json(file_path)
    except (FileNotFoundError, UnicodeDecodeError):
        print(f"No such file: {file_path}")
        generate_graph_args = [
            '--graph_type', f"{args.graph_type}",
            '--n', f"{args.n}",
            '--p', f"{args.p}",
            '--pa_n', f"{args.pa_n}",
            '--vstr', f"{args.vstr}",
            '--conf', f"{args.conf}",
            '--save'
        ]
        print("Trying again...")
        sys.argv = ['graph_generator.py'] + generate_graph_args
        print(
            f"Calling the main function of graph_generator.py with the options {generate_graph_args}")  # Debug statement
        graph_generator.main()
        print(f"Graph successfully saved under {file_path}")

        graph = io_mgmt.load_graph_from_json(file_path)

    noise_list = [f'{dist}' for dist in args.noise_types]
    if len(noise_list) == 1:
        noise_list *= len(graph.nodes)

    noises_dict = noises.parse_noise(noise_list, list(graph.nodes))
    functions = structural_equations.generate_functions(graph, noises_dict, args.funct_type)

    scm_data = {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "functions": {k: SCM.func_to_str(v) for k, v in functions.items()},
        "noise": {node: (dist_type, *params) for node, (dist_type, params) in noises_dict.items()}
    }
    scm = SCM(scm_data)

    scm.save_to_json(save_path)


if __name__ == '__main__':
    main()

import argparse
import ast
import json
import csv
from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import noises, plots, structural_equations, graph_generator, SCM, io_mgmt

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
PATH_DATA = config['PATH_DATA']
DISTS = config['DISTS']


def evaluate_structural_equation(function_string, data_dict, noise_dict):
    match_single_arg = re.search(r'lambda\s+(\w+)\s*:', function_string)
    match_multiple_args = re.search(r'lambda\s*\(([^)]*)\)\s*:', function_string)
    match = re.search(r'lambda\s*([^:]+)\s*:', function_string)
    match_constant_function = re.search(r'lambda\s*_\s*:', function_string)
    if match_multiple_args:
        input_vars = match_multiple_args.group(1).split(',')
    elif match_single_arg:
        input_vars = [match_single_arg.group(1)]
    elif match_constant_function:
        input_vars = []
    elif match:
        input_vars = match.group(1).split(',')
    else:
        raise ValueError(f"Invalid lambda function format: {function_string}")
    # Clean up and map the input variables to data_dict entries
    input_vars = [var.strip() for var in input_vars]

    # Replace 'N_Xi" with the corresponding noise data vectors
    noise_vars = re.findall(r'N\w+', function_string)
    noise_vars = [var[2:] for var in noise_vars]

    # TODO: ATTENTION! Currently assuming additive noises. Due to storage preferences,
    #  the noises don't appear in the function strings contained in the .json representation of the SCM.

    function_string = re.sub(r'\s*\+\s*N_\w+', '', function_string)  # Remove terms like '+ N_Xi'

    # Define the lambda function
    SE_lambda = eval(function_string)

    print(f"DATA DICTIONARY: {data_dict}")  # Debug statement

    # Prepare the arguments
    args = [data_dict[var] for var in input_vars if var != '_']

    result = SE_lambda(*args) if args else SE_lambda('_')

    return result


def save_to_csv(dict, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in dict.items():
            value_str = ','.join(map(str, value))
            writer.writerow([key, value_str])


def csv_to_dict(path):
    data = {}

    with open(path, mode='r') as f:
        # reader = csv.DictReader(f)
        reader = csv.reader(f)
        for row in reader:
            node = row[0]
            values = list(map(float, row[1].split(',')))
            data[node] = values

    return data


def sample_L1(scm, n_samples):
    noise_data = {}
    data = {}

    for X_j in scm.G.nodes:
        n_j_str = scm.N[X_j]

        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    for X_j in nx.topological_sort(scm.G):
        if scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            data[X_j] = samples

    return data


def sample_L2(scm, n_samples, interventions):
    interventions_dict = io_mgmt.parse_interventions(interventions)
    scm.intervene(interventions_dict)

    noise_data = {}
    data = {}

    for X_j in scm.G.nodes:
        if X_j in interventions_dict:
            continue

        n_j_str = scm.N[X_j]

        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    for X_j in nx.topological_sort(scm.G):
        if X_j not in interventions_dict and scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            if X_j in interventions_dict:
                samples = np.repeat(samples, n_samples)
                data[X_j] = samples
            else:
                data[X_j] = samples + noise_data[X_j]

    return data


def sample_observational_distribution(scm, n_samples, data_savepath):
    noise_data = {}

    for X_j in scm.G.nodes:
        print(f"Parsing noise for {X_j}")  # Debug statement
        n_j_str = scm.N[X_j]
        print(f"Obtained string representation of N_{X_j}: {n_j_str}")  # Debug statement
        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    data = {}

    for X_j in nx.topological_sort(scm.G):
        if scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            print(f"Now evaluating {f_j_str}...")  # Debug statement
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            data[X_j] = samples

    print(f"Sampled data: {data}")  # Debug statement

    save_to_csv(data, data_savepath)
    print(f"Data saved to {data_savepath}")  # Debug statement

    return data


def sample_interventional_distribution(scm, n_samples, data_savepath, interventions):
    interventions_dict = io_mgmt.parse_interventions(interventions)
    scm.intervene(interventions_dict)
    save_path = data_savepath.strip('.json')
    do_suffix = io_mgmt.make_do_suffix(interventions)
    save_path = save_path + do_suffix + ".json"
    print(f"Attempting to save new SCM to: {save_path}")  # Debug statement
    scm.save_to_json(save_path)
    data_savepath = f"{PATH_DATA}/{save_path}".replace('.json', '.csv')

    noise_data = {}
    print(f"INTERVENTIONS: {interventions_dict}")  # Debug statement

    for X_j in scm.G.nodes:
        if X_j in interventions_dict:
            continue
        print(f"Parsing noise for {X_j}")
        n_j_str = scm.N[X_j]
        samples = noises.generate_distribution(n_j_str)(n_samples)
        noise_data[X_j] = samples

    data = {}

    for X_j in nx.topological_sort(scm.G):
        if X_j not in interventions_dict and scm.G.in_degree[X_j] == 0:
            data[X_j] = noise_data[X_j]
        else:
            f_j_str = scm.F[X_j]
            print(f"Now evaluating {f_j_str}...")
            samples = evaluate_structural_equation(f_j_str, data, noise_data)
            if X_j in interventions_dict:
                samples = np.repeat(samples, n_samples)
                data[X_j] = samples
                print(f"Not using noise for {X_j}")
            else:
                data[X_j] = samples + noise_data[X_j]
                print(f"Additive noise considered for variable {X_j}: {noise_data[X_j]}")

    print(f"Sampled data: {data}")
    save_to_csv(data, data_savepath)
    print(f"Data saved to {data_savepath}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generating L1, L2, L3 data from .json files representing SCM's.")
    parser.add_argument('--file_name', required=True,
                        help="Please specify the name (not the full path!) of the SCM file.")
    parser.add_argument('--mode', required=True, choices=['l1', 'l2', 'l3'],
                        help="Please provide the distribution type: "
                             "'l1' for observational, 'l2' for interventional, 'l3' for counterfactual data.")
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    # Required for the option --mode l2 and --mode l3
    parser.add_argument('--do', type=str, nargs='+',
                        help="Please specify the interventions to be performed. Example: "
                             "'--do (Xi,0)' sets variable X_i to zero. For multiple simultaneous interventions," \
                             "use spaces to seperate the individual do()-operations.")
    # TODO: not a very practical way of passing info
    parser.add_argument('--observations_path', type=str,
                        help="Provide the path to the JSON file that contains the observations"
                             "to be used for constructing the observationally constrained SCM in mode 'l3'.")
    parser.add_argument('--variables', nargs='+', help="Variables to visualize.")
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help="Specify the number of samples to be generated with '--n_samples'")
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    file_path = f"{PATH_SCM}/{args.file_name}"
    data_savepath = f"{PATH_DATA}/{args.file_name}".replace('.json', '.csv')

    if args.plot:
        dictionary = csv_to_dict(data_savepath)
        fig = plots.plot_distributions_from_dict(dictionary)
        if args.save:
            plot_filename = f"{PATH_PLOTS}/{args.file_name}".replace('.json', '.png')
            fig.savefig(plot_filename)
        return

    scm = SCM.SCM(file_path)
    if args.mode == 'l1':
        sample_observational_distribution(scm, args.n_samples, data_savepath)
    if args.mode == 'l2':
        if args.do is None:
            print(f"Please specify at least one intervention using the option --do (Xj, value)")
            return

        sample_interventional_distribution(scm, args.n_samples, args.file_name, args.do)


if __name__ == '__main__':
    main()

import sys
import numpy as np

import io_mgmt

sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']


# TODO: noises should appear as arguments in lambdas

def generate_xor_function(parents, noise):
    # TODO
    return False


def generate_linear_function(parents, noise, coeffs):
    """

    :param parents: immediate predecessors (pa(X_i)) of node X_i in graph G
    :param coeffs: coefficient vector of length |pa(X_i)|
    :return: Linear function f_i(pa(X_i), N_i) = f(pa(X_i)) + N_i as a string.
    """
    terms = [f"{coeffs[i]} * {parent}" for i, parent in enumerate(parents)]
    terms.append(noise)
    function = f"lambda {', '.join(parents)}: " + " + ".join(terms)

    return function


def generate_polynomial(parents, noise, coeffs, degrees):
    """
    Generate f_i(pa(X_i), N_i) = polynomial(pa(X_i), a_j, e_j) + N_i
    :param parents: pa(X_i)
    :param coeffs: a_i
    :param degrees: e_i in f_i(pa(X_i), N_i) = N_i + sum(a_j * X_i^{e_j} for j in pa(X_i))
    :return: Linear function f_i(pa(X_i), N_i) = f(pa(X_i)) + N_i as a string.
    """
    terms = [f"{coeffs[i]}*{parent}**{degrees[i]}" for i, parent in enumerate(parents)]
    terms.append(noise)
    function = f"lambda {', '.join(parents)}: " + " + ".join(terms)
    return function


def generate_functions(graph, noise_vars, funct_type='linear'):
    # TODO
    """

    :param graph:
    :param noise_vars: Noise variables N_i specified as f"lambda ... : ..."
    :param funct_type:
    :return:
    """
    functions = {}
    functions.keys()
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        if funct_type == 'linear':
            # Randomly pick the coefficients
            coeffs = np.random.choice(COEFFS, size=len(parents))
            functions[node] = generate_linear_function(parents, f"N_{node}", coeffs)
        elif funct_type == 'polynomial':
            degrees = np.random.randint(1, MAX_DEGREE + 1, size=len(parents))
            coeffs = np.random.choice(COEFFS, size=len(parents))
            functions[node] = generate_polynomial(parents, f"N_{node}", coeffs, degrees)
        else:
            raise ValueError(f"Unsupported function type: {funct_type}. Use 'linear' or 'polynomial'.")

    return functions


def parse_functions(func_dict):
    """
    Parse strings into lambda functions.
    """
    parsed_functions = {}
    for key, func_str in func_dict.items():
        parsed_functions[key] = eval(func_str)
    return parsed_functions

from scipy.stats.distributions import norm, bernoulli, beta, gamma, expon
import re
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import io_mgmt

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']


def generate_distribution(noise_term):
    dist_type = noise_term[0]
    params = noise_term[1:]

    if dist_type == 'gaussian':
        return lambda x, mu=params[0], sigma=params[1]: norm.rvs(mu, sigma, size=x)
    elif dist_type == 'bernoulli':
        return lambda x, p=params[0]: bernoulli.rvs(p, size=x)
    elif dist_type == 'exponential':
        return lambda x, lam=params[0]: expon.rvs(scale=1 / lam, size=x)
    else:
        raise ValueError(f"Unsupported distribution type:{dist_type}")


def parse_noise_string(noise_str):
    dist_type_map = {
        'N': 'gaussian',
        'Exp': 'exponential',
        'Ber': 'bernoulli'
    }

    pattern = r'([A-Za-z]+)\(([^)]+)\)'
    match = re.match(pattern, noise_str)
    if not match:
        raise ValueError(f"Invalid distribution format: {noise_str}")

    noise_type, params = match.groups()
    params = [float(x) for x in params.split(',')]

    if noise_type not in dist_type_map:
        raise ValueError(f"Unsupported distrbution type: {noise_type}")

    return dist_type_map[noise_type], params


def parse_noise(noise, nodes):
    """
    Parse noise distribution strings into a dictionary format.
    Example: "N(0,1) --> {"X_{node_id}": ("N", 0, 1)}
    """

    if isinstance(noise, str):
        noise = [noise]

    num_nodes = len(nodes)
    counts = {distr_str: noise.count(distr_str) for distr_str in DISTS}
    arg_count = len(noise)

    if arg_count == 1:
        noise = noise * num_nodes
    elif arg_count != num_nodes:
        raise ValueError(f"Expected either 1 or {num_nodes} noise distributions, but got {arg_count}: \n {noise}")

    noise_dict = {node: parse_noise_string(noise_str) for node, noise_str in zip(nodes, noise)}

    return noise_dict

import argparse
import json
import os
import random
import networkx as nx
import io_mgmt

config = io_mgmt.configuration_loader()

PATH_GRAPHS = config['PATH_GRAPHS']

def generate_chain_graph(n, save=False):
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    edges = [[nodes[i], nodes[i + 1]] for i in range(n)]
    G = {"nodes": nodes, "edges": edges}
    if save:
        save_graph(G, 'chain', n)
    return G


def generate_parallel_graph(n, save=False):
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    edges = [[nodes[i], "Y"] for i in range(n)]
    G = {"nodes": nodes, "edges": edges}
    if save:
        save_graph(G, 'parallel', n)
    return G


def generate_random_dag(n, p):
    """
    Generate a random DAG G=(V,Ɛ), using the Erdős–Rényi model.
    :param save:
    :param n: Number of nodes in the graph.
    :param p: Probability of including one of the binomial(n, 2) potential edges in Ɛ.
    :return: Graph G.
    """
    # TODO: Rename nodes
    # G = nx.erdos_renyi_graph(n, p, directed=True)
    G = nx.DiGraph()
    # Construct the graph in topological order to ensure DAG'ness
    nodes = [f"X{n}" for n in list(range(1, n + 1))]
    nodes.append('Y')
    G.add_nodes_from(nodes)  # Add nodes
    edges = [(u, v) for u in nodes for v in nodes if u < v and random.random() < p]  # Include edges with probability p
    G.add_edges_from(edges)  # Add edges

    return G


def erdos_with_properties(n, p, n_pa_Y, confs, vstr, save=False):
    """
    Generate a random DAG G=(V,Ɛ) with certain properties using the Erdős–Rényi model.

    :param save:
    :param n: Number of nodes in the graph.
    :param p: Probability of including one of the binomial(n, 2) potential edges in Ɛ.
    :param n_pa_Y: Cardinality of the parent set of reward node Y.
    :param confs: Number of confounding variables.
    :param vstr: Number of v-structures.
    :return: Graph G.
    """
    G = generate_random_dag(n, p)
    print(f"ERDOS: Graph successfully created. Checking for other specifications...\n") # Debug statement
    # Ensure that the number of confounders in the graph is same as 'confs'
    if confs != -1:
        print("Applying number of confounders-constraint.\n") # Debug statement
        n_confs = count_confounders(G)
        while n_confs < confs:
            add_confounders(G)
    else:
        print("No confounders specified.\n") # Debug statement
    # Ensure the specified number of v-structures
    if vstr != -1:
        print("Applying v-structures constraint.\n") # Debug statement
        n_vs = count_v_structures(G)
        while n_vs < vstr:
            add_v_structures(G)
    else:
        print("No v-structures specified.\n") # Debug statement

    # TODO: Fix the inconsistecy between #nodes when n_pa_Y exercised (n+1) and when not (n)
    # Ensure a specific number of parent nodes for the reward variable
    if n_pa_Y != -1:
        y = n
        pa_Y = random.sample(G.nodes, n_pa_Y)
        for parent in pa_Y:
            G.add_edge(parent, y)

    print("Mapping graph to dict\n") # Debug statement
    G_dict = {"nodes": list(G.nodes), "edges": list(G.edges)}

    if save:
        graph_type = f"random_graph_N{n}_paY_{n_pa_Y}_p_{p}"
        print(f"Saving {graph_type}...\n")
        save_graph(G_dict, graph_type, n)
        print(f"Random graph saved to {PATH_GRAPHS}/{graph_type}")

    return G_dict


def count_confounders(G):
    """
    O(n^3) - complex in the number of nodes. Don't use with denser graphs.
    :return: Confounder count in the provided graph.
    """
    n_confounders = 0
    # For each node u in the graph
    for u in G.nodes:
        ch_u = list(G.successors(u))
        # Check if any pair of u's children are connected by a directed edge
        for i in range(len(ch_u)):
            for j in range(i + 1, len(ch_u)):
                v, w = ch_u[i], ch_u[j]
                if nx.has_path(G, v, w) or nx.has_path(G, w, v):
                    n_confounders += 1
                    # One such structure suffices to mark u as a confounder
                    break
    return n_confounders


def count_v_structures(G):
    """
        O(n^3) - complex in the number of nodes. Don't use with denser graphs.
        :return: v-structure count in the provided graph.
        """
    n_v = 0
    for v in G.nodes:
        pa_v = list(G.predecessors(v))
        for i in range(len(pa_v)):
            for j in range(i + 1, len(pa_v)):
                u, w, = pa_v[i], pa_v[j]
                if not G.has_edge(u, w) and not G.has_edge(w, u):
                    n_v += 1
    return n_v


def add_confounders(G):
    """
    Add confounders to the graph
    :param G: DAG G=(V,Ɛ)
    :param num_confounders: number of confounders (Nodes Z s.t. Z-->X, Z-->Y, X-->Y)
    """
    nodes = list(G.nodes())
    # Make num_confounders - many randomly selected edges bidirectional, effectively inserting a confounder
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v) and not G.has_edge(v, u):
        G.add_edge(u, v)


def add_v_structures(G):
    """
    Add v-structures to the graph.
    :param G: DAG G=(V,Ɛ)
    :param num_v_structures: number of node triples {X,Y,Z} that form a v-structure: X --> Y <-- Z, X -||- Z
    """
    nodes = list(G.nodes())

    u, v, w = random.sample(nodes, 3)
    if not G.has_edge(u, v) and not G.has_edge(v, u) and not G.has_edge(w, v) and not G.has_edge(v, w):
        G.add_edge(u, v)
        G.add_edge(w, v)


def save_graph(graph, graph_type, n):
    os.makedirs(PATH_GRAPHS, exist_ok=True)
    file_path = f"{PATH_GRAPHS}/{graph_type}_graph_N{n}.json"
    with open(file_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"{graph_type.capitalize()} graph with {n} nodes saved to {file_path}.")


def main():
    parser = argparse.ArgumentParser(description="Generate graph structures and save as JSON files.")
    parser.add_argument("--graph_type", choices=['chain', 'parallel', 'random'],
                        help="Type of graph structure to generate. Currently supported: ['chain', 'parallel', 'random']")
    parser.add_argument("--n", type=int, required=True, help="Number of (non-reward) nodes in the graph.")
    # Required for option --graph_type random
    parser.add_argument("--p", type=float, help="Denseness of the graph / prob. of including any potential edge.")
    # Required for option --graph_type random
    parser.add_argument("--pa_n", type=int, default=1, help="Cardinality of pa_Y in G.")
    parser.add_argument("--vstr", type=int, help="Desired number of v-structures in the causal graph.")
    parser.add_argument("--conf", type=int, help="Desired number of confounding variables in the causal graph.")
    parser.add_argument("--save", action='store_true')

    args = parser.parse_args()

    if args.graph_type == 'chain':
        generate_chain_graph(args.n, args.save)
    elif args.graph_type == 'parallel':
        generate_parallel_graph(args.n, args.save)
    elif args.graph_type == 'random':
        if args.p is None:
            print("Please specify the probability of including an edge with --p for random graph generation.")
            return
        if args.pa_n is None:
            print("Please specify the cardinality of the parent set for the reward variable Y.")
        # erdos_with_properties(args.n, args.p, args.pa_n, args.conf, args.vstr, args.save)
        erdos_with_properties(args.n, args.p, args.pa_n, args.conf, args.vstr, args.save)
        # graph_type = f"random_pa{args.pa_n}_conf{args.conf}_vstr{args.vstr}"
    else:
        print("Please specify a type of graph. Currently supported: ['chain', 'parallel', 'random']")
        return


if __name__ == "__main__":
    main()

import json
import matplotlib.pyplot as plt
import networkx as nx
import os

import io_mgmt

config = io_mgmt.configuration_loader()
PATH_GRAPHS = config['PATH_GRAPHS']
PATH_SCM = config['PATH_SCMs']
PATH_PLOTS = config['PATH_PLOTS']
MAX_DEGREE = config['MAX_POLYNOMIAL_DEGREE']
COEFFS = config['COEFFICIENTS']
DISTS = config['DISTS']


def draw_scm(scm_filename):
    scm_path = os.path.join(PATH_SCM, scm_filename)
    try:
        with open(scm_path, 'r') as f:
            scm_data = json.load(f)
    except FileNotFoundError:
        print(f"The file at {scm_path} does not exist.")
        return

    G = nx.DiGraph()
    G.add_nodes_from(scm_data['nodes'])
    G.add_edges_from(scm_data['edges'])

    # TODO: Best layouts for various types of graphs / possibility to specify the layout / custom layout
    # other good options 'nx.spiral_layout(G)', ' nx.arf_layout(G)'

    try:  # Define the layout
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.shell_layout(G)

    # Draw the regular nodes
    nx.draw_networkx_nodes(G, pos, node_color='none', edgecolors='black', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Draw the noise node
    noise = scm_data['noise']
    noise_nodes = []
    noise_labels = {}

    for i, node in enumerate(scm_data['nodes']):
        noise_node = f"N_{i + 1}"
        noise_nodes.append(noise_node)
        G.add_node(noise_node)
        G.add_edge(noise_node, node)
        pos[noise_node] = (pos[node][0], pos[node][1] + 1)
        noise_labels[node] = noise_node

    # TODO : Create labels for noise nodes
    # Draw the noise nodes
    nx.draw_networkx_nodes(G, pos, nodelist=noise_nodes, node_shape='o', node_color='white',
                           edgecolors='black', node_size=1000, alpha=0.5)
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(f"N_{i + 1}", scm_data['nodes'][i]) for i in range(len(scm_data['nodes']))],
                           style='dashed', edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Display the functions next to the graph
    functions = scm_data['functions']
    functions_text = "\n".join([f"{k}: {v}" for k, v in functions.items()])
    # TODO: partially overlaps with the DAG, needs more flexible positioning
    plt.text(1.05, 0.5, functions_text, ha='left', va='center')

    # Save/show the plot
    # TODO: More informative but concise title, better formatted
    plt.title("Structural Causal Model")
    os.makedirs(PATH_SCM, exist_ok=True)
    plot_filename = scm_filename.replace('.json', '.png')
    plot_filename = os.path.join(PATH_PLOTS, plot_filename)
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.show()
    plt.close()


def plot_samples(samples, title, bins=30, xlabel="x", ylabel="f(x)"):
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=bins, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_distributions_from_dict(dict, save=False):
    num_plots = len(dict)
    num_cols = 2  # Select the number of columns in the grid
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create a grid for the plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for idx, (key, values) in enumerate(dict.items()):
        axes[idx].hist(values, bins=30, edgecolor='k', alpha=0.7)
        axes[idx].set_title(key)
        axes[idx].set_xlabel('value')
        axes[idx].set_ylabel('Frequency')
        # axes[idx].set_yscale('log')

    # Hide unused plots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return fig


def plot_outputs(data, alg_type, bandit_type, plot_filename, save=False):
    plot_filename = plot_filename + ".png"
    title = "Rewards Over Time"
    y_label = "Reward"
    if "regret" in plot_filename:
        title = "Regret Over Time"
        y_label = "Regret"
    if "simple" in plot_filename:
        title = "Simple " + title
        y_label = "Simple " + y_label
    elif "cumulative" in plot_filename:
        title = "Cumulative " + title
        y_label = "Cumulative " + y_label
    subtitle = f"Algorithm used: {alg_type}\n Bandit type: {bandit_type}"
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel("Iterations")
    plt.ylabel(y_label)
    plt.title(title)
    plt.suptitle(subtitle)
    plt.grid(True)
    if save:
        plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()

import os
import uuid
import json
import csv
import argparse

import pandas as pd
import networkx as nx


def parse_scm(input):
    if isinstance(input, str):  # JSON input
        try:
            with open(input, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"No such file: {input}")
    elif isinstance(input, dict):  # Dictionary input
        data = input
    else:
        raise ValueError("Input must be a JSON string or a dictionary")

    nodes = data['nodes']
    functions = data['functions']
    noise = data['noise']

    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])

    return nodes, G, functions, noise


def append_counter(filename):
    counter = 1
    filename_stripped, file_type = filename.split('.')
    while os.path.exists(f"{filename}/({counter}).{file_type}"):
        counter += 1
    filename_with_counter = f"{filename}/({counter}).{file_type}"

    return filename_with_counter


# Append universally unique identifier
def append_unique_id(filename):
    id = uuid.uuid4()
    filename_stripped, file_type = filename.split('.')
    filename_with_uid = f"{filename}_{id}.{file_type}"

    return filename_with_uid


def scm_args_to_filename(args, file_type, base_path):
    # Specify name components common to all user inputs
    components = [
        f"SCM_n{args.n}",
        f"{args.graph_type}_graph"
    ]

    # Add non-graph related property specifications
    if args.funct_type is not None:
        components.append(f"{args.funct_type}_functions")
    if args.noise_types is not None:
        # stip away apostrophes to avoid misinterpretation of user input
        noise_str = ''.join(map(lambda s: s.replace("'", ""), args.noise_types))
        components.append(f"{noise_str}")

    # Include additional information in the file name when provided
    if args.graph_type == 'random':
        if args.pa_n >= 0:
            components.append(f"paY_{args.pa_n}")
        if args.vstr >= 0:
            components.append(f"vstr_{args.vstr}")
        if args.conf >= 0:
            components.append(f"conf_{args.conf}")
        components.append(f"p{args.p}")

    filename = '_'.join(components) + f".{file_type}"
    return os.path.join(base_path, filename)


def make_do_suffix(do_list):
    suffix = "_do"
    if isinstance(do_list, dict):  # FOR DEBUGGING ONLY
        for item in do_list:
            variable, value = item.strip('()').strip(' ').split(',')
            suffix += f"{variable}-{value}"
    return suffix


def parse_interventions(do_list):
    do_dict = {}

    if not isinstance(do_list, list):
        do_list = [do_list]
    for intervention in do_list:
        variable, value = intervention.strip('()').strip(' ').split(',')
        do_dict[variable] = value

    return do_dict


def save_rewards_to_csv(rewards, filename):
    df = pd.DataFrame(rewards)
    filename = filename + ".csv"
    df.to_csv(filename, index=False)
    print(f"Rewards saved to {filename}")


def csv_to_dict(path):
    data = {}
    with open(path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            node = row[0]
            values = list(map(float, row[1].split(',')))
            data[node] = values
    return data


def configuration_loader(config_file="global_variables.json"):
    # TODO: This assumes all scripts that have a dependency on configuration_loader are two levels deep
    config_path = f"../../config/{config_file}"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def process_costs_per_arm(costs, n_arms):
    if isinstance(costs, float):
        costs = [costs]
    if len(costs) == 1:
        costs *= n_arms
    return costs


def load_graph_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])
    return G

