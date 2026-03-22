from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_connected(G: nx.Graph) -> nx.Graph:
    if nx.is_connected(G):
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def add_unit_weights_if_missing(G: nx.Graph, weight_key: str = "weight") -> None:
    for u, v, d in G.edges(data=True):
        if weight_key not in d:
            d[weight_key] = 1.0


def graph_memory_proxy_bytes(G: nx.Graph) -> int:
    m = G.number_of_edges()
    return int(m * 3 * 8)


def sample_node_pairs(G: nx.Graph, k: int, seed: int = 0) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    pairs = set()
    while len(pairs) < min(k, n * (n - 1) // 2):
        a, b = rng.sample(nodes, 2)
        if a != b:
            pairs.add((a, b) if a < b else (b, a))
    return list(pairs)


def shortest_path_length_safe(G: nx.Graph, s: int, t: int, weight: str = "weight") -> float:
    try:
        return nx.shortest_path_length(G, source=s, target=t, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf


# -----------------------------
# Graph generators
# -----------------------------

def generate_er(n: int, p: float, seed: int = 0) -> nx.Graph:
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    G = ensure_connected(G)
    add_unit_weights_if_missing(G)
    return G


def generate_ba(n: int, m_attach: int, seed: int = 0) -> nx.Graph:
    G = nx.barabasi_albert_graph(n=n, m=m_attach, seed=seed)
    G = ensure_connected(G)
    add_unit_weights_if_missing(G)
    return G


def generate_sbm(sizes: List[int], p_in: float, p_out: float, seed: int = 0) -> nx.Graph:
    k = len(sizes)
    p = [[p_out for _ in range(k)] for __ in range(k)]
    for i in range(k):
        p[i][i] = p_in

    G = nx.stochastic_block_model(sizes=sizes, p=p, seed=seed)
    G = ensure_connected(G)
    add_unit_weights_if_missing(G)
    return G


# -----------------------------
# Method 1: Greedy Spanner
# -----------------------------

def greedy_spanner(G: nx.Graph, stretch: float = 2.0, weight: str = "weight") -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    edges = [(u, v, d.get(weight, 1.0)) for u, v, d in G.edges(data=True)]
    edges.sort(key=lambda x: x[2])

    for u, v, w in edges:
        if u == v:
            continue

        try:
            dist_uv = nx.shortest_path_length(H, u, v, weight=weight)
        except nx.NetworkXNoPath:
            dist_uv = math.inf

        if dist_uv > stretch * w:
            H.add_edge(u, v, **{weight: w})

    return H


# -----------------------------
# Method 2: Partition-Based
# -----------------------------

def partition_greedy_communities(G: nx.Graph) -> List[set]:
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    if len(comms) == 0:
        return [set(G.nodes())]
    return comms


def sparsify_partition_based(G: nx.Graph, stretch: float = 2.0, weight: str = "weight") -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    comms = partition_greedy_communities(G)
    node_to_comm = {}
    for i, c in enumerate(comms):
        for v in c:
            node_to_comm[v] = i

    for c in comms:
        sub = G.subgraph(c).copy()
        if sub.number_of_edges() == 0:
            continue
        sub_sp = greedy_spanner(sub, stretch=stretch, weight=weight)
        H.add_edges_from(sub_sp.edges(data=True))

    best_edge: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    for u, v, d in G.edges(data=True):
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu == cv:
            continue
        w = d.get(weight, 1.0)
        key = (cu, cv) if cu < cv else (cv, cu)
        if key not in best_edge or w < best_edge[key][2]:
            best_edge[key] = (u, v, w)

    for (u, v, w) in best_edge.values():
        H.add_edge(u, v, **{weight: w})

    return H


# -----------------------------
# Method 3: Multi-Resolution
# -----------------------------

def contract_communities_to_supergraph(
    G: nx.Graph,
    communities: List[set],
    weight: str = "weight"
):
    node_to_comm = {}
    for i, c in enumerate(communities):
        for v in c:
            node_to_comm[v] = i

    coarse = nx.Graph()
    coarse.add_nodes_from(range(len(communities)))

    best = {}
    for u, v, d in G.edges(data=True):
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu == cv:
            continue
        w = d.get(weight, 1.0)
        key = (cu, cv) if cu < cv else (cv, cu)
        if key not in best or w < best[key]:
            best[key] = w

    for (cu, cv), w in best.items():
        coarse.add_edge(cu, cv, **{weight: w})

    return coarse, node_to_comm


def multi_resolution_sparsify(
    G: nx.Graph,
    stretch_fine: float = 2.0,
    stretch_coarse: float = 2.0,
    weight: str = "weight"
) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    comms = partition_greedy_communities(G)

    node_to_comm = {}
    for i, c in enumerate(comms):
        for v in c:
            node_to_comm[v] = i

    for c in comms:
        sub = G.subgraph(c).copy()
        if sub.number_of_edges() == 0:
            continue
        sub_sp = greedy_spanner(sub, stretch=stretch_fine, weight=weight)
        H.add_edges_from(sub_sp.edges(data=True))

    coarse, node_to_comm = contract_communities_to_supergraph(G, comms, weight=weight)
    coarse_sp = greedy_spanner(coarse, stretch=stretch_coarse, weight=weight)

    best_cross = {}
    for u, v, d in G.edges(data=True):
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu == cv:
            continue
        w = d.get(weight, 1.0)
        key = (cu, cv) if cu < cv else (cv, cu)
        if key not in best_cross or w < best_cross[key][2]:
            best_cross[key] = (u, v, w)

    for ci, cj, _ in coarse_sp.edges(data=True):
        key = (ci, cj) if ci < cj else (cj, ci)
        if key in best_cross:
            u, v, w = best_cross[key]
            H.add_edge(u, v, **{weight: w})

    return H


# -----------------------------
# Evaluation
# -----------------------------

@dataclass
class EvalResult:
    name: str
    n: int
    m: int
    m_sp: int
    edge_ratio: float
    time_sec: float


def run_one_graph_quick(G: nx.Graph, stretch: float = 2.0) -> List[EvalResult]:
    """
    Quick runner for scalability:
    only collects runtime + edge ratio (fast & simple).
    """
    add_unit_weights_if_missing(G)

    results = []

    t0 = time.perf_counter()
    H1 = greedy_spanner(G, stretch=stretch)
    t1 = time.perf_counter()
    results.append(EvalResult("GreedySpanner", G.number_of_nodes(), G.number_of_edges(),
                              H1.number_of_edges(), H1.number_of_edges() / G.number_of_edges(), t1 - t0))

    t0 = time.perf_counter()
    H2 = sparsify_partition_based(G, stretch=stretch)
    t1 = time.perf_counter()
    results.append(EvalResult("PartitionBased", G.number_of_nodes(), G.number_of_edges(),
                              H2.number_of_edges(), H2.number_of_edges() / G.number_of_edges(), t1 - t0))

    t0 = time.perf_counter()
    H3 = multi_resolution_sparsify(G, stretch_fine=stretch, stretch_coarse=stretch)
    t1 = time.perf_counter()
    results.append(EvalResult("MultiResolution", G.number_of_nodes(), G.number_of_edges(),
                              H3.number_of_edges(), H3.number_of_edges() / G.number_of_edges(), t1 - t0))

    return results
