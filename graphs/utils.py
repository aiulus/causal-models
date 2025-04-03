import networkx as nx
from itertools import chain, combinations
from typing import List, Set, Tuple, Dict

def is_dag(G):
    """Check if a directed graph is acyclic."""
    return nx.is_directed_acyclic_graph(G)


def has_cycles(G):
    """Return True if cycles exist in G."""
    try:
        nx.find_cycle(G, orientation='original')
        return True
    except nx.NetworkXNoCycle:
        return False


def count_v_structures(G):
    """
    Count v-structures: X → Z ← Y, where X and Y are not adjacent.
    """
    count = 0
    for z in G.nodes():
        pa = list(G.predecessors(z))
        for i in range(len(pa)):
            for j in range(i + 1, len(pa)):
                u, v = pa[i], pa[j]
                if not G.has_edge(u, v) and not G.has_edge(v, u):
                    count += 1
    return count


def count_confounders(G):
    """
    Count confounder-like structures: X → Y and X → Z with Y ↔ Z via some path.
    """
    count = 0
    for u in G.nodes():
        children = list(G.successors(u))
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                v, w = children[i], children[j]
                if nx.has_path(G, v, w) or nx.has_path(G, w, v):
                    count += 1
                    break
    return count


def get_bidirected_pairs(hidden_nodes, G):
    """
    Given a set of hidden nodes, return all observed node pairs
    connected through a common hidden parent → interpretable as bidirected edges.
    """
    bidirected = set()
    for h in hidden_nodes:
        children = list(G.successors(h))
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                bidirected.add(tuple(sorted((children[i], children[j]))))
    return list(bidirected)


def get_ancestors(G: nx.DiGraph, node: str) -> Set[str]:
    return nx.ancestors(G, node).union({node})


def get_descendants(G: nx.DiGraph, nodes: Set[str]) -> Set[str]:
    desc = set(nodes)
    for node in nodes:
        desc |= nx.descendants(G, node)
    return desc


def c_component(G: nx.DiGraph) -> List[Set[str]]:
    """Finds c-components using bidirected edges marked by a special attribute."""
    bidirected_subgraph = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("confounded", False):
            bidirected_subgraph.add_edge(u, v)
    return list(nx.connected_components(bidirected_subgraph))


def reversed_topological(G: nx.DiGraph, exclude: Set[str]) -> List[str]:
    nodes = [n for n in nx.topological_sort(G) if n not in exclude]
    return list(reversed(nodes))


def induce_subgraph(G: nx.DiGraph, nodes: Set[str]) -> nx.DiGraph:
    return G.subgraph(nodes).copy()