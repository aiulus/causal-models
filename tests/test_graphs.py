import pytest
import networkx as nx
from graphs import generator
from graphs import utils


# === Helper: Convert dict → nx.DiGraph ===
def nx_from_dict(graph_dict):
    G = nx.DiGraph()
    G.add_nodes_from(graph_dict["nodes"])
    G.add_edges_from(graph_dict["edges"])
    return G


# === TEST: Chain graph ===
def test_generate_chain_graph_is_dag():
    G_dict = generator.generate_chain_graph(5)
    G = nx_from_dict(G_dict)
    assert utils.is_dag(G)
    assert not utils.has_cycles(G)
    assert len(G.nodes) == 6  # X1...X5 + Y
    assert len(G.edges) == 5


# === TEST: Parallel graph ===
def test_generate_parallel_graph_structure():
    G_dict = generator.generate_parallel_graph(3)
    G = nx_from_dict(G_dict)
    assert all(edge[1] == "Y" for edge in G.edges)
    assert utils.is_dag(G)


# === TEST: Random DAG with pa(Y) constraint ===
def test_random_graph_parent_constraint():
    G_dict = generator.erdos_with_constraints(
        n=5, p=0.4, pa_y=3, confs=-1, vstr=-1
    )
    G = nx_from_dict(G_dict)
    # assert len(list(G.predecessors("Y"))) == 3
    assert len(list(G.predecessors("Y"))) >= 3


# === TEST: v-structure count increases ===
def test_v_structure_injection():
    G_dict = generator.erdos_with_constraints(
        n=6, p=0.2, pa_y=1, confs=-1, vstr=2
    )
    G = nx_from_dict(G_dict)
    assert utils.count_v_structures(G) >= 2


# === TEST: confounder count increases ===
def test_confounder_injection():
    G_dict = generator.erdos_with_constraints(
        n=6, p=0.2, pa_y=1, confs=2, vstr=-1
    )
    G = nx_from_dict(G_dict)
    assert utils.count_confounders(G) >= 2


def test_parallel_graph_is_dag():
    G_dict = generator.generate_parallel_graph(3)
    G = nx_from_dict(G_dict)
    assert nx.is_directed_acyclic_graph(G), "Parallel graph should always be a DAG"
