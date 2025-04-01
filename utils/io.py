import os
import json
import csv
import uuid
import networkx as nx
import pandas as pd


# Load config from file
def configuration_loader(config_file="global_variables.json"):
    config_path = os.path.join("config", config_file)
    with open(config_path, 'r') as f:
        return json.load(f)


config = configuration_loader()


# === SCM JSON HANDLING ===
def parse_scm(input):
    if isinstance(input, str):
        with open(input, 'r') as f:
            data = json.load(f)
    elif isinstance(input, dict):
        data = input
    else:
        raise ValueError("Input must be a JSON file path or dictionary")

    nodes = data['nodes']
    edges = data['edges']
    functions = data['functions']
    noise = data['noise']

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return nodes, G, functions, noise


# === FILE NAMING HELPERS ===
def scm_args_to_filename(args, file_type, base_path):
    components = [
        f"SCM_n{args.n}",
        f"{args.graph_type}_graph"
    ]

    if args.funct_type:
        components.append(f"{args.funct_type}_functions")
    if args.noise_types:
        if len(set(args.noise_types)) == 1:
            noise_str = args.noise_types[0].replace("'", "")
        else:
            noise_str = '_'.join(s.replace("'", "") for s in args.noise_types)
        components.append(noise_str)

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


def append_unique_id(filename):
    base, ext = filename.rsplit('.', 1)
    return f"{base}_{uuid.uuid4().hex[:6]}.{ext}"


def make_do_suffix(do_list):
    suffix = "_do"
    if isinstance(do_list, dict):  # For internal debug use
        items = do_list.items()
    else:
        items = [x.strip('()').split(',') for x in do_list]

    for variable, value in items:
        suffix += f"{variable.strip()}-{value.strip()}"
    return suffix


# === INTERVENTION PARSING ===
def parse_interventions(do_list):
    do_dict = {}
    if not isinstance(do_list, list):
        do_list = [do_list]

    for intervention in do_list:
        variable, value = intervention.strip('()').split(',')
        do_dict[variable.strip()] = value.strip()
    return do_dict


# === CSV UTILS ===
def csv_to_dict(path):
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = row[0]
            values = list(map(float, row[1].split(',')))
            data[key] = values
    return data


def save_to_csv(data_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, values in data_dict.items():
            row = [key, ','.join(map(str, values))]
            writer.writerow(row)


def save_rewards_to_csv(rewards, filename):
    df = pd.DataFrame(rewards)
    path = filename if filename.endswith(".csv") else filename + ".csv"
    df.to_csv(path, index=False)
    print(f"Rewards saved to {path}")


# === GRAPH HANDLING ===
def load_graph_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])
    return G
