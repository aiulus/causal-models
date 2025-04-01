import networkx as nx
import random


def generate_chain_graph(n):
    """
    Returns: Chain: X1 → X2 → ... → Xn → Y
    """
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n)]
    return {"nodes": nodes, "edges": edges}


def generate_parallel_graph(n):
    """
    Returns: Parallel: X1 → Y, X2 → Y, ..., Xn → Y
    """
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    edges = [(f"X{i}", "Y") for i in range(1, n + 1)]
    return {"nodes": nodes, "edges": edges}


def generate_random_dag(n, p):
    """
    Random DAG via topological order + ER edge sampling.
    """
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < p:
                G.add_edge(nodes[i], nodes[j])

    return {"nodes": list(G.nodes), "edges": list(G.edges)}


def add_v_structure(G):
    nodes = list(G.nodes())
    u, v, w = random.sample(nodes, 3)
    if not G.has_edge(u, v) and not G.has_edge(w, v):
        if not G.has_edge(u, w) and not G.has_edge(w, u):
            G.add_edge(u, v)
            G.add_edge(w, v)


def add_confounder(G):
    nodes = list(G.nodes())
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v) and not G.has_edge(v, u):
        G.add_edge(u, v)


def count_v_structures(G):
    count = 0
    for v in G.nodes():
        pa = list(G.predecessors(v))
        for i in range(len(pa)):
            for j in range(i + 1, len(pa)):
                u, w = pa[i], pa[j]
                if not G.has_edge(u, w) and not G.has_edge(w, u):
                    count += 1
    return count


def count_confounders(G):
    count = 0
    for u in G.nodes():
        children = list(G.successors(u))
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                if nx.has_path(G, children[i], children[j]) or nx.has_path(G, children[j], children[i]):
                    count += 1
                    break
    return count


def erdos_with_constraints(n, p, pa_y=1, confs=-1, vstr=-1):
    """
    Erdős-Rényi DAG with optional structural constraints.
    """
    G = nx.DiGraph()
    nodes = [f"X{i}" for i in range(1, n + 1)] + ["Y"]
    G.add_nodes_from(nodes)

    # Topological generation with edge probability
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < p:
                G.add_edge(nodes[i], nodes[j])

    # Force desired number of v-structures
    while vstr != -1 and count_v_structures(G) < vstr:
        add_v_structure(G)

    # Force desired number of confounders
    while confs != -1 and count_confounders(G) < confs:
        add_confounder(G)

    # Ensure Y has the specified number of parents
    if pa_y != -1:
        Y = "Y"
        current_pa = list(G.predecessors(Y))
        required = pa_y - len(current_pa)
        if required > 0:
            available = [x for x in nodes if x != Y and x not in current_pa]
            new_pa = random.sample(available, required)
            for x in new_pa:
                G.add_edge(x, Y)

    return {"nodes": list(G.nodes), "edges": list(G.edges)}
