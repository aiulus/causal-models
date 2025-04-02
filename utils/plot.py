import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from utils import io


def draw_scm_with_noise(scm_filename):
    """
    Draw SCM DAG from a saved JSON file and optionally save as .png.
    """
    path = os.path.join(io.config['PATH_SCMs'], scm_filename)
    with open(path, 'r') as f:
        scm_data = json.load(f)

    G = nx.DiGraph()
    G.add_nodes_from(scm_data['nodes'])
    G.add_edges_from(scm_data['edges'])

    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.spring_layout(G)

    # Draw observed nodes
    nx.draw_networkx_nodes(G, pos, node_color='none', edgecolors='black', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Draw noise nodes (visually, not from model graph)
    noise_nodes = []
    for i, node in enumerate(scm_data['nodes']):
        noise_node = f"N_{i+1}"
        noise_nodes.append(noise_node)
        G.add_node(noise_node)
        G.add_edge(noise_node, node)
        pos[noise_node] = (pos[node][0], pos[node][1] + 1)

    nx.draw_networkx_nodes(G, pos, nodelist=noise_nodes, node_shape='o', node_color='white',
                           edgecolors='black', node_size=1000, alpha=0.5)
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(f"N_{i+1}", scm_data['nodes'][i]) for i in range(len(scm_data['nodes']))],
                           style='dashed', edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Add function display
    functions = scm_data['functions']
    functions_text = "\n".join([f"{k}: {v}" for k, v in functions.items()])
    plt.text(1.05, 0.5, functions_text, ha='left', va='center', transform=plt.gca().transAxes)

    # Finalize
    plt.title("Structural Causal Model")
    os.makedirs(io.config['PATH_PLOTS'], exist_ok=True)
    plot_path = os.path.join(io.config['PATH_PLOTS'], scm_filename.replace('.json', '.png'))
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def draw_scm(scm_filename):
    """
    Draw SCM DAG from a saved JSON file and optionally save as .png.
    """
    path = os.path.join(io.config['PATH_SCMs'], scm_filename)
    with open(path, 'r') as f:
        scm_data = json.load(f)

    G = nx.DiGraph()
    G.add_nodes_from(scm_data['nodes'])
    G.add_edges_from(scm_data['edges'])

    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.spring_layout(G)

    # Draw observed nodes
    nx.draw_networkx_nodes(G, pos, node_color='none', edgecolors='black', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True,
                           min_source_margin=14.5, min_target_margin=14.5, arrowsize=15)

    # Add function display
    functions = scm_data['functions']
    functions_text = "\n".join([f"{k}: {v}" for k, v in functions.items()])
    plt.text(1.05, 0.5, functions_text, ha='left', va='center', transform=plt.gca().transAxes)

    # Finalize
    plt.title("Structural Causal Model")
    os.makedirs(io.config['PATH_PLOTS'], exist_ok=True)
    plot_path = os.path.join(io.config['PATH_PLOTS'], scm_filename.replace('.json', '.png'))
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()


def plot_distributions_from_dict(data_dict, save=False):
    """
    Plot histogram for each variable in sampled SCM data.
    """
    num_plots = len(data_dict)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for idx, (key, values) in enumerate(data_dict.items()):
        axes[idx].hist(values, bins=30, edgecolor='k', alpha=0.7)
        axes[idx].set_title(key)
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig


def plot_samples(samples, title, bins=30, xlabel="x", ylabel="f(x)"):
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=bins, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_outputs(data, alg_type, bandit_type, plot_filename, save=False):
    """
    Plot reward or regret curves from bandit experiments.
    """
    if not plot_filename.endswith(".png"):
        plot_filename += ".png"

    title = "Rewards Over Time"
    ylabel = "Reward"
    if "regret" in plot_filename:
        title = "Regret Over Time"
        ylabel = "Regret"
    if "simple" in plot_filename:
        title = "Simple " + title
        ylabel = "Simple " + ylabel
    elif "cumulative" in plot_filename:
        title = "Cumulative " + title
        ylabel = "Cumulative " + ylabel

    subtitle = f"Algorithm used: {alg_type}\nBandit type: {bandit_type}"

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.suptitle(subtitle)
    plt.grid(True)

    if save:
        os.makedirs(io.config['PATH_PLOTS'], exist_ok=True)
        path = os.path.join(io.config['PATH_PLOTS'], plot_filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved plot to {path}")

    plt.show()
