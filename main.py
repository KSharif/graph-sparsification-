"""
Graph Sparsification Methods for Efficient Network Analysis
End-to-end implementation + SAVE-TO-FILE visualizations (no interactive windows)

Creates plots in ./plots as PNGs (300 DPI):
  - edge_ratio.png
  - runtime.png
  - mse_log.png
  - relative_error.png
  - stretch_p90.png
  - tradeoff.png
  - radar_ER.png
  - radar_BA.png
  - radar_SBM.png
"""

from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx

# --- Matplotlib: save-to-file backend (no windows) ---
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: saves figures without needing a GUI
import matplotlib.pyplot as plt

# Folder to store plots
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_connected(G: nx.Graph) -> nx.Graph:
    """Keep largest connected component so shortest-path distances exist for most pairs."""
    if nx.is_connected(G):
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def add_unit_weights_if_missing(G: nx.Graph, weight_key: str = "weight") -> None:
    """Ensure every edge has a weight."""
    for u, v, d in G.edges(data=True):
        if weight_key not in d:
            d[weight_key] = 1.0


def graph_memory_proxy_bytes(G: nx.Graph) -> int:
    """
    Simple memory proxy for comparison (not exact RAM):
    store edges as (u,v,w) triples of 3*8 bytes (ignore overhead).
    """
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
# Graph generators (ER / BA / SBM)
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
    """
    Practical greedy spanner:
      sort edges by weight,
      add (u,v,w) if dist_H(u,v) > stretch*w
    """
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
# Method 2: Partition-Based Sparsification
# -----------------------------

def partition_greedy_communities(G: nx.Graph) -> List[set]:
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    if len(comms) == 0:
        return [set(G.nodes())]
    return comms


def sparsify_partition_based(G: nx.Graph, stretch: float = 2.0, weight: str = "weight") -> nx.Graph:
    """
    1) Detect communities
    2) Build local greedy spanner within each community
    3) Keep best (min-weight) inter-community edge per community pair
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    comms = partition_greedy_communities(G)
    node_to_comm = {}
    for i, c in enumerate(comms):
        for v in c:
            node_to_comm[v] = i

    # Local spanners
    for c in comms:
        sub = G.subgraph(c).copy()
        if sub.number_of_edges() == 0:
            continue
        sub_sp = greedy_spanner(sub, stretch=stretch, weight=weight)
        H.add_edges_from(sub_sp.edges(data=True))

    # Best inter-community edges
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
# Method 3: Multi-Resolution Sparsification
# -----------------------------

def contract_communities_to_supergraph(
    G: nx.Graph,
    communities: List[set],
    weight: str = "weight"
) -> Tuple[nx.Graph, Dict[int, int], Dict[int, set]]:
    node_to_comm = {}
    for i, c in enumerate(communities):
        for v in c:
            node_to_comm[v] = i

    coarse = nx.Graph()
    coarse.add_nodes_from(range(len(communities)))
    comm_to_nodes = {i: set(c) for i, c in enumerate(communities)}

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

    return coarse, node_to_comm, comm_to_nodes


def multi_resolution_sparsify(
    G: nx.Graph,
    levels: int = 2,
    stretch_fine: float = 2.0,
    stretch_coarse: float = 2.0,
    weight: str = "weight"
) -> nx.Graph:
    """
    Fine: local spanners inside communities
    Coarse: spanner on contracted graph, then lift coarse edges back
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    comms = partition_greedy_communities(G)

    node_to_comm = {}
    for i, c in enumerate(comms):
        for v in c:
            node_to_comm[v] = i

    # Local spanners (fine)
    for c in comms:
        sub = G.subgraph(c).copy()
        if sub.number_of_edges() == 0:
            continue
        sub_sp = greedy_spanner(sub, stretch=stretch_fine, weight=weight)
        H.add_edges_from(sub_sp.edges(data=True))

    # Coarse graph + coarse spanner
    coarse, node_to_comm, _ = contract_communities_to_supergraph(G, comms, weight=weight)
    coarse_sp = greedy_spanner(coarse, stretch=stretch_coarse, weight=weight)

    # Lift: add best original cross-edge for each coarse edge
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
    mem_proxy_bytes: int
    connected: bool
    mse: float
    mean_rel_err: float
    stretch_max: float
    stretch_p50: float
    stretch_p90: float


def evaluate_sparsifier(
    G: nx.Graph,
    H: nx.Graph,
    name: str,
    build_time_sec: float,
    pairs_k: int = 2000,
    weight: str = "weight",
    seed: int = 0
) -> EvalResult:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    m2 = H.number_of_edges()

    edge_ratio = (m2 / m) if m > 0 else 0.0
    mem_proxy = graph_memory_proxy_bytes(H)
    connected = nx.is_connected(H) if H.number_of_nodes() > 0 else False

    pairs = sample_node_pairs(G, k=pairs_k, seed=seed)

    sq_errs = []
    rel_errs = []
    stretches = []

    for s, t in pairs:
        dG = shortest_path_length_safe(G, s, t, weight=weight)
        dH = shortest_path_length_safe(H, s, t, weight=weight)

        if math.isinf(dG) or dG == 0:
            continue
        if math.isinf(dH):
            continue

        diff = dH - dG
        sq_errs.append(diff * diff)
        rel_errs.append(abs(diff) / dG)
        stretches.append(dH / dG)

    if len(sq_errs) == 0:
        mse = mean_rel = smax = sp50 = sp90 = math.inf
    else:
        mse = float(np.mean(sq_errs))
        mean_rel = float(np.mean(rel_errs))
        smax = float(np.max(stretches))
        sp50 = float(np.percentile(stretches, 50))
        sp90 = float(np.percentile(stretches, 90))

    return EvalResult(
        name=name,
        n=n, m=m,
        m_sp=m2,
        edge_ratio=edge_ratio,
        time_sec=build_time_sec,
        mem_proxy_bytes=mem_proxy,
        connected=connected,
        mse=mse,
        mean_rel_err=mean_rel,
        stretch_max=smax,
        stretch_p50=sp50,
        stretch_p90=sp90,
    )


# -----------------------------
# Runner
# -----------------------------

def run_one_graph(G: nx.Graph, stretch: float = 2.0, seed: int = 0) -> List[EvalResult]:
    add_unit_weights_if_missing(G)

    results = []

    t0 = time.perf_counter()
    H1 = greedy_spanner(G, stretch=stretch)
    t1 = time.perf_counter()
    results.append(evaluate_sparsifier(G, H1, "GreedySpanner", t1 - t0, seed=seed))

    t0 = time.perf_counter()
    H2 = sparsify_partition_based(G, stretch=stretch)
    t1 = time.perf_counter()
    results.append(evaluate_sparsifier(G, H2, "PartitionBased", t1 - t0, seed=seed))

    t0 = time.perf_counter()
    H3 = multi_resolution_sparsify(G, levels=2, stretch_fine=stretch, stretch_coarse=stretch)
    t1 = time.perf_counter()
    results.append(evaluate_sparsifier(G, H3, "MultiResolution", t1 - t0, seed=seed))

    return results


def print_results_table(results: List[EvalResult]) -> None:
    headers = [
        "Method", "n", "m→m'", "m'/m", "time(s)", "mem_proxy(B)", "conn",
        "MSE", "rel_err", "stretch(max/p50/p90)"
    ]
    print("\n" + "-" * 120)
    print("{:<15} {:>6} {:>14} {:>7} {:>8} {:>13} {:>6} {:>10} {:>10} {:>22}".format(*headers))
    print("-" * 120)
    for r in results:
        mm = f"{r.m}->{r.m_sp}"
        st = f"{r.stretch_max:.2f}/{r.stretch_p50:.2f}/{r.stretch_p90:.2f}" if math.isfinite(r.stretch_max) else "inf"
        print("{:<15} {:>6} {:>14} {:>7.3f} {:>8.3f} {:>13} {:>6} {:>10.4f} {:>10.4f} {:>22}".format(
            r.name, r.n, mm, r.edge_ratio, r.time_sec, r.mem_proxy_bytes, str(r.connected),
            r.mse if math.isfinite(r.mse) else float("inf"),
            r.mean_rel_err if math.isfinite(r.mean_rel_err) else float("inf"),
            st
        ))
    print("-" * 120)


# -----------------------------
# Plotting (SAVE to plots/)
# -----------------------------

def results_to_rows(all_results: Dict[str, List[EvalResult]]):
    rows = []
    for gname, res_list in all_results.items():
        for r in res_list:
            rows.append({
                "graph": gname,
                "method": r.name,
                "edge_ratio": r.edge_ratio,
                "time_sec": r.time_sec,
                "mse": r.mse,
                "mean_rel_err": r.mean_rel_err,
                "stretch_max": r.stretch_max,
                "stretch_p50": r.stretch_p50,
                "stretch_p90": r.stretch_p90,
                "connected": r.connected,
                "m": r.m,
                "m_sp": r.m_sp,
            })
    return rows


def _grouped_bar(rows, metric: str, title: str, ylabel: str):
    graphs = sorted(list({r["graph"] for r in rows}))
    methods = sorted(list({r["method"] for r in rows}))

    mat = np.zeros((len(graphs), len(methods)))
    for i, g in enumerate(graphs):
        for j, m in enumerate(methods):
            mat[i, j] = next(rr[metric] for rr in rows if rr["graph"] == g and rr["method"] == m)

    x = np.arange(len(graphs))
    width = 0.22 if len(methods) >= 3 else 0.3

    plt.figure()
    for j, m in enumerate(methods):
        plt.bar(x + (j - (len(methods) - 1) / 2) * width, mat[:, j], width, label=m)

    plt.xticks(x, graphs)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Graph type")
    plt.legend()
    plt.tight_layout()


def plot_edge_ratio(rows):
    _grouped_bar(rows, "edge_ratio", "Edge Retention Ratio (m'/m) by Method", "m'/m (lower is more sparse)")
    plt.savefig(os.path.join(PLOT_DIR, "edge_ratio.png"), dpi=300)
    plt.close()


def plot_runtime(rows):
    _grouped_bar(rows, "time_sec", "Runtime (seconds) by Method", "Seconds (lower is faster)")
    plt.savefig(os.path.join(PLOT_DIR, "runtime.png"), dpi=300)
    plt.close()


def plot_mse_log(rows):
    _grouped_bar(rows, "mse", "Distance Distortion (MSE, log scale) by Method", "MSE (lower is better)")
    plt.yscale("log")
    plt.savefig(os.path.join(PLOT_DIR, "mse_log.png"), dpi=300)
    plt.close()


def plot_rel_error(rows):
    _grouped_bar(rows, "mean_rel_err", "Mean Relative Error by Method", "Mean relative error (lower is better)")
    plt.savefig(os.path.join(PLOT_DIR, "relative_error.png"), dpi=300)
    plt.close()


def plot_stretch_p90(rows):
    _grouped_bar(rows, "stretch_p90", "Empirical Stretch (90th percentile) by Method", "Stretch p90 (lower is better)")
    plt.savefig(os.path.join(PLOT_DIR, "stretch_p90.png"), dpi=300)
    plt.close()


def plot_tradeoff_scatter(rows):
    plt.figure()
    for r in rows:
        plt.scatter(r["edge_ratio"], r["mse"])
        plt.text(r["edge_ratio"], r["mse"], f'{r["graph"]}-{r["method"]}', fontsize=7)

    plt.title("Trade-off: Sparsity vs Distortion (edge_ratio vs MSE)")
    plt.xlabel("Edge ratio m'/m (lower = sparser)")
    plt.ylabel("MSE (lower = better)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "tradeoff.png"), dpi=300)
    plt.close()


def plot_radar_for_graph(rows, graph_name: str):
    subset = [r for r in rows if r["graph"] == graph_name]
    methods = sorted(list({r["method"] for r in subset}))

    metrics = ["edge_ratio", "time_sec", "mse", "mean_rel_err", "stretch_p90"]
    labels = ["EdgeRatio", "Time", "MSE", "RelErr", "StretchP90"]

    max_vals = {met: max(1e-12, max(r[met] for r in subset)) for met in metrics}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure()
    ax = plt.subplot(111, polar=True)
    plt.title(f"Radar Comparison (lower is better) — {graph_name}")

    for m in methods:
        r = next(rr for rr in subset if rr["method"] == m)
        vals = [(r[met] / max_vals[met]) for met in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, label=m)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"radar_{graph_name}.png"), dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    set_seed(42)

    # Use moderate n to keep greedy spanner fast
    n = 250

    # ER
    G_er = generate_er(n=n, p=0.04, seed=1)
    res_er = run_one_graph(G_er, stretch=2.0, seed=1)
    print("\n=== ER Graph ===")
    print_results_table(res_er)

    # BA
    G_ba = generate_ba(n=n, m_attach=3, seed=2)
    res_ba = run_one_graph(G_ba, stretch=2.0, seed=2)
    print("\n=== BA Graph ===")
    print_results_table(res_ba)

    # SBM
    G_sbm = generate_sbm(sizes=[80, 80, 90], p_in=0.08, p_out=0.01, seed=3)
    res_sbm = run_one_graph(G_sbm, stretch=2.0, seed=3)
    print("\n=== SBM Graph ===")
    print_results_table(res_sbm)

    # --- Plot + Save ---
    all_results = {"ER": res_er, "BA": res_ba, "SBM": res_sbm}
    rows = results_to_rows(all_results)

    plot_edge_ratio(rows)
    plot_runtime(rows)
    plot_mse_log(rows)
    plot_rel_error(rows)
    plot_stretch_p90(rows)
    plot_tradeoff_scatter(rows)

    plot_radar_for_graph(rows, "ER")
    plot_radar_for_graph(rows, "BA")
    plot_radar_for_graph(rows, "SBM")

    print(f"\nSaved plots to: ./{PLOT_DIR}/")
    print("Check with: ls plots")


if __name__ == "__main__":
    main()


#_________________________________________________________________--
# """
# Graph Sparsification Methods for Efficient Network Analysis
# End-to-end implementation for:
#   (1) Greedy Spanner (distance-preserving)
#   (2) Partition-Based Sparsification (divide & conquer)
#   (3) Multi-Resolution Sparsification (hierarchical)

# Matches project plan: synthetic graphs (ER/BA/SBM), evaluation metrics:
# edge reduction, runtime, memory proxy, MSE/relative error, stretch, connectivity.

# Author: (you)
# """

# from __future__ import annotations

# import time
# import math
# import matplotlib
# matplotlib.use("MacOSX")   # important for mac
# import matplotlib.pyplot as plt

# import random
# from dataclasses import dataclass
# from typing import Dict, Tuple, List, Iterable, Optional

# import numpy as np
# import networkx as nx


# # -----------------------------
# # Utilities
# # -----------------------------

# def set_seed(seed: int = 42) -> None:
#     random.seed(seed)
#     np.random.seed(seed)


# def ensure_connected(G: nx.Graph) -> nx.Graph:
#     """Keep largest connected component so shortest-path distances exist for most pairs."""
#     if nx.is_connected(G):
#         return G
#     comp = max(nx.connected_components(G), key=len)
#     return G.subgraph(comp).copy()


# def add_unit_weights_if_missing(G: nx.Graph, weight_key: str = "weight") -> None:
#     """Ensure every edge has a weight."""
#     for u, v, d in G.edges(data=True):
#         if weight_key not in d:
#             d[weight_key] = 1.0


# def graph_memory_proxy_bytes(G: nx.Graph) -> int:
#     """
#     A simple, consistent memory proxy (NOT exact RAM):
#     store edges as (u,v,w) triples of 3*8 bytes + overhead ignored.
#     Useful for relative comparison across sparsifiers.
#     """
#     m = G.number_of_edges()
#     return int(m * 3 * 8)


# def sample_node_pairs(G: nx.Graph, k: int, seed: int = 0) -> List[Tuple[int, int]]:
#     rng = random.Random(seed)
#     nodes = list(G.nodes())
#     n = len(nodes)
#     pairs = set()
#     while len(pairs) < min(k, n * (n - 1) // 2):
#         a, b = rng.sample(nodes, 2)
#         if a != b:
#             pairs.add((a, b) if a < b else (b, a))
#     return list(pairs)


# def shortest_path_length_safe(
#     G: nx.Graph, s: int, t: int, weight: str = "weight"
# ) -> float:
#     try:
#         return nx.shortest_path_length(G, source=s, target=t, weight=weight)
#     except (nx.NetworkXNoPath, nx.NodeNotFound):
#         return math.inf


# # -----------------------------
# # Graph generators (ER / BA / SBM)
# # -----------------------------

# def generate_er(n: int, p: float, seed: int = 0) -> nx.Graph:
#     G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
#     G = ensure_connected(G)
#     add_unit_weights_if_missing(G)
#     return G


# def generate_ba(n: int, m_attach: int, seed: int = 0) -> nx.Graph:
#     G = nx.barabasi_albert_graph(n=n, m=m_attach, seed=seed)
#     G = ensure_connected(G)
#     add_unit_weights_if_missing(G)
#     return G


# def generate_sbm(
#     sizes: List[int],
#     p_in: float,
#     p_out: float,
#     seed: int = 0
# ) -> nx.Graph:
#     # Block probability matrix
#     k = len(sizes)
#     p = [[p_out for _ in range(k)] for __ in range(k)]
#     for i in range(k):
#         p[i][i] = p_in

#     G = nx.stochastic_block_model(sizes=sizes, p=p, seed=seed)
#     G = ensure_connected(G)
#     add_unit_weights_if_missing(G)
#     return G


# # -----------------------------
# # Method 1: Greedy Spanner (Transform & Conquer)
# # -----------------------------
# # Proposal: add edge only if removing it increases endpoint distance beyond allowed limit. :contentReference[oaicite:2]{index=2}

# def greedy_spanner(
#     G: nx.Graph,
#     stretch: float = 2.0,
#     weight: str = "weight"
# ) -> nx.Graph:
#     """
#     A practical greedy spanner:
#       - Sort edges by weight (classic spanner approach for weighted graphs)
#       - Maintain H initially empty with same nodes
#       - For each edge (u,v,w):
#            if dist_H(u,v) > stretch * w: add edge
#     Complexity can be high; use for moderate graph sizes.
#     """
#     H = nx.Graph()
#     H.add_nodes_from(G.nodes())

#     edges = [(u, v, d.get(weight, 1.0)) for u, v, d in G.edges(data=True)]
#     edges.sort(key=lambda x: x[2])

#     for u, v, w in edges:
#         if u == v:
#             continue

#         # Distance in current sparsifier
#         try:
#             dist_uv = nx.shortest_path_length(H, u, v, weight=weight)
#         except nx.NetworkXNoPath:
#             dist_uv = math.inf

#         if dist_uv > stretch * w:
#             H.add_edge(u, v, **{weight: w})

#     return H


# # -----------------------------
# # Method 2: Partition-Based Sparsification (Divide & Conquer)
# # -----------------------------
# # Proposal: partition graph, sparsify each part locally, keep bridge edges. :contentReference[oaicite:3]{index=3}

# def partition_greedy_communities(G: nx.Graph) -> List[set]:
#     """
#     Uses NetworkX greedy modularity community detection (no extra deps).
#     Returns list of node-sets.
#     """
#     # If graph is small, this is fine; for large graphs, consider faster heuristics.
#     comms = list(nx.algorithms.community.greedy_modularity_communities(G))
#     if len(comms) == 0:
#         return [set(G.nodes())]
#     return comms


# def sparsify_partition_based(
#     G: nx.Graph,
#     stretch: float = 2.0,
#     weight: str = "weight"
# ) -> nx.Graph:
#     """
#     1) Detect communities/partitions
#     2) Build a local greedy spanner inside each partition (subgraph)
#     3) Add a limited set of inter-partition edges to keep connectivity:
#         - For each pair of communities that has cross edges,
#           keep the minimum-weight cross edge (cheap + reasonable).
#     """
#     H = nx.Graph()
#     H.add_nodes_from(G.nodes())

#     comms = partition_greedy_communities(G)
#     node_to_comm = {}
#     for i, c in enumerate(comms):
#         for v in c:
#             node_to_comm[v] = i

#     # Local spanners
#     for c in comms:
#         sub = G.subgraph(c).copy()
#         if sub.number_of_edges() == 0:
#             continue
#         sub_sp = greedy_spanner(sub, stretch=stretch, weight=weight)
#         H.add_edges_from(sub_sp.edges(data=True))

#     # Inter-community edges: keep one best edge per (comm_i, comm_j)
#     best_edge: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
#     for u, v, d in G.edges(data=True):
#         cu, cv = node_to_comm[u], node_to_comm[v]
#         if cu == cv:
#             continue
#         w = d.get(weight, 1.0)
#         key = (cu, cv) if cu < cv else (cv, cu)
#         if key not in best_edge or w < best_edge[key][2]:
#             best_edge[key] = (u, v, w)

#     for (u, v, w) in best_edge.values():
#         H.add_edge(u, v, **{weight: w})

#     return H


# # -----------------------------
# # Method 3: Multi-Resolution Sparsification
# # -----------------------------
# # Proposal: build hierarchy; preserve global structure at coarse levels + local at fine levels. :contentReference[oaicite:4]{index=4}

# def contract_communities_to_supergraph(
#     G: nx.Graph,
#     communities: List[set],
#     weight: str = "weight"
# ) -> Tuple[nx.Graph, Dict[int, int], Dict[int, set]]:
#     """
#     Create a coarse graph where each community becomes a super-node.
#     Edge weight between super-nodes = minimum cross-edge weight (simple).
#     """
#     node_to_comm = {}
#     for i, c in enumerate(communities):
#         for v in c:
#             node_to_comm[v] = i

#     coarse = nx.Graph()
#     coarse.add_nodes_from(range(len(communities)))

#     # map comm -> nodes
#     comm_to_nodes = {i: set(c) for i, c in enumerate(communities)}

#     # Add coarse edges
#     best = {}
#     for u, v, d in G.edges(data=True):
#         cu, cv = node_to_comm[u], node_to_comm[v]
#         if cu == cv:
#             continue
#         w = d.get(weight, 1.0)
#         key = (cu, cv) if cu < cv else (cv, cu)
#         if key not in best or w < best[key]:
#             best[key] = w
#     for (cu, cv), w in best.items():
#         coarse.add_edge(cu, cv, **{weight: w})

#     return coarse, node_to_comm, comm_to_nodes


# def multi_resolution_sparsify(
#     G: nx.Graph,
#     levels: int = 2,
#     stretch_fine: float = 2.0,
#     stretch_coarse: float = 2.0,
#     weight: str = "weight"
# ) -> nx.Graph:
#     """
#     A practical multi-resolution scheme:
#       Level 0 (fine): local spanners inside communities
#       Level 1..L (coarser): spanner on contracted graph to preserve global structure
#     Then lift coarse edges back to original graph by selecting best representative edges.
#     """
#     H = nx.Graph()
#     H.add_nodes_from(G.nodes())

#     current_G = G

#     # 1) Fine level: local partition + local spanners
#     comms = partition_greedy_communities(current_G)
#     node_to_comm = {}
#     for i, c in enumerate(comms):
#         for v in c:
#             node_to_comm[v] = i

#     # Local spanners
#     for c in comms:
#         sub = current_G.subgraph(c).copy()
#         if sub.number_of_edges() == 0:
#             continue
#         sub_sp = greedy_spanner(sub, stretch=stretch_fine, weight=weight)
#         H.add_edges_from(sub_sp.edges(data=True))

#     # 2) Build coarse graph and sparsify it (repeat if levels > 2)
#     coarse, node_to_comm, comm_to_nodes = contract_communities_to_supergraph(
#         current_G, comms, weight=weight
#     )

#     # Spanner at coarse level (captures long-range structure)
#     coarse_sp = greedy_spanner(coarse, stretch=stretch_coarse, weight=weight)

#     # 3) Lift coarse edges back: for each coarse edge (ci,cj), add best original cross-edge
#     # Find best cross-edge between these communities in original graph
#     best_cross = {}
#     for u, v, d in G.edges(data=True):
#         cu, cv = node_to_comm[u], node_to_comm[v]
#         if cu == cv:
#             continue
#         w = d.get(weight, 1.0)
#         key = (cu, cv) if cu < cv else (cv, cu)
#         if key not in best_cross or w < best_cross[key][2]:
#             best_cross[key] = (u, v, w)

#     for ci, cj, d in coarse_sp.edges(data=True):
#         key = (ci, cj) if ci < cj else (cj, ci)
#         if key in best_cross:
#             u, v, w = best_cross[key]
#             H.add_edge(u, v, **{weight: w})

#     return H


# # -----------------------------
# # Evaluation
# # -----------------------------
# @dataclass
# class EvalResult:
#     name: str
#     n: int
#     m: int
#     m_sp: int
#     edge_ratio: float
#     time_sec: float
#     mem_proxy_bytes: int
#     connected: bool
#     mse: float
#     mean_rel_err: float
#     stretch_max: float
#     stretch_p50: float
#     stretch_p90: float


# def evaluate_sparsifier(
#     G: nx.Graph,
#     H: nx.Graph,
#     name: str,
#     build_time_sec: float,
#     pairs_k: int = 2000,
#     weight: str = "weight",
#     seed: int = 0
# ) -> EvalResult:
#     """
#     Metrics from proposal: edge ratio, time, memory usage, MSE/relative error,
#     empirical stretch (max, p50, p90), connectivity. :contentReference[oaicite:5]{index=5}
#     """
#     n = G.number_of_nodes()
#     m = G.number_of_edges()
#     m2 = H.number_of_edges()

#     edge_ratio = (m2 / m) if m > 0 else 0.0
#     mem_proxy = graph_memory_proxy_bytes(H)
#     connected = nx.is_connected(H) if H.number_of_nodes() > 0 else False

#     pairs = sample_node_pairs(G, k=pairs_k, seed=seed)

#     sq_errs = []
#     rel_errs = []
#     stretches = []

#     for s, t in pairs:
#         dG = shortest_path_length_safe(G, s, t, weight=weight)
#         dH = shortest_path_length_safe(H, s, t, weight=weight)

#         if math.isinf(dG) or dG == 0:
#             continue
#         if math.isinf(dH):
#             # No path in sparsified graph: treat as huge distortion
#             # (You can also record separately as "disconnected pairs")
#             continue

#         diff = dH - dG
#         sq_errs.append(diff * diff)
#         rel_errs.append(abs(diff) / dG)
#         stretches.append(dH / dG)

#     if len(sq_errs) == 0:
#         mse = math.inf
#         mean_rel = math.inf
#         smax = sp50 = sp90 = math.inf
#     else:
#         mse = float(np.mean(sq_errs))
#         mean_rel = float(np.mean(rel_errs))
#         smax = float(np.max(stretches))
#         sp50 = float(np.percentile(stretches, 50))
#         sp90 = float(np.percentile(stretches, 90))

#     return EvalResult(
#         name=name,
#         n=n, m=m,
#         m_sp=m2,
#         edge_ratio=edge_ratio,
#         time_sec=build_time_sec,
#         mem_proxy_bytes=mem_proxy,
#         connected=connected,
#         mse=mse,
#         mean_rel_err=mean_rel,
#         stretch_max=smax,
#         stretch_p50=sp50,
#         stretch_p90=sp90,
#     )

# def results_to_rows(all_results: Dict[str, List[EvalResult]]):
#     """
#     all_results = {
#       "ER": [EvalResult, EvalResult, ...],
#       "BA": [...],
#       "SBM": [...]
#     }
#     Returns list of dict rows for easy plotting.
#     """
#     rows = []
#     for gname, res_list in all_results.items():
#         for r in res_list:
#             rows.append({
#                 "graph": gname,
#                 "method": r.name,
#                 "edge_ratio": r.edge_ratio,
#                 "time_sec": r.time_sec,
#                 "mse": r.mse,
#                 "mean_rel_err": r.mean_rel_err,
#                 "stretch_max": r.stretch_max,
#                 "stretch_p50": r.stretch_p50,
#                 "stretch_p90": r.stretch_p90,
#                 "connected": r.connected,
#                 "m": r.m,
#                 "m_sp": r.m_sp,
#             })
#     return rows


# def _grouped_bar(rows, metric: str, title: str, ylabel: str):
#     """
#     Grouped bar: x-axis = graph types (ER/BA/SBM),
#     bars = methods.
#     """
#     graphs = sorted(list({r["graph"] for r in rows}))
#     methods = sorted(list({r["method"] for r in rows}))

#     # Build matrix: graph x method
#     mat = np.zeros((len(graphs), len(methods)))
#     for i, g in enumerate(graphs):
#         for j, m in enumerate(methods):
#             val = next(rr[metric] for rr in rows if rr["graph"] == g and rr["method"] == m)
#             mat[i, j] = val

#     x = np.arange(len(graphs))
#     width = 0.22 if len(methods) >= 3 else 0.3

#     plt.figure()
#     for j, m in enumerate(methods):
#         plt.bar(x + (j - (len(methods)-1)/2)*width, mat[:, j], width, label=m)

#     plt.xticks(x, graphs)
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.xlabel("Graph type")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def plot_edge_ratio(rows):
#     _grouped_bar(rows, "edge_ratio", "Edge Retention Ratio (m'/m) by Method", "m'/m (lower is more sparse)")


# def plot_runtime(rows):
#     _grouped_bar(rows, "time_sec", "Runtime (seconds) by Method", "Seconds (lower is faster)")


# def plot_mse(rows, log_scale: bool = True):
#     # MSE can vary a lot; log scale helps
#     _grouped_bar(rows, "mse", "Distance Distortion (MSE) by Method", "MSE (lower is better)")
#     if log_scale:
#         plt.figure()
#         graphs = sorted(list({r["graph"] for r in rows}))
#         methods = sorted(list({r["method"] for r in rows}))
#         x = np.arange(len(graphs))
#         width = 0.22
#         mat = np.zeros((len(graphs), len(methods)))
#         for i, g in enumerate(graphs):
#             for j, m in enumerate(methods):
#                 mat[i, j] = next(rr["mse"] for rr in rows if rr["graph"] == g and rr["method"] == m)

#         for j, m in enumerate(methods):
#             plt.bar(x + (j - (len(methods)-1)/2)*width, mat[:, j], width, label=m)

#         plt.yscale("log")
#         plt.xticks(x, graphs)
#         plt.title("Distance Distortion (MSE, log scale) by Method")
#         plt.ylabel("MSE (log scale)")
#         plt.xlabel("Graph type")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


# def plot_rel_error(rows):
#     _grouped_bar(rows, "mean_rel_err", "Mean Relative Error by Method", "Mean relative error (lower is better)")


# def plot_stretch_p90(rows):
#     _grouped_bar(rows, "stretch_p90", "Empirical Stretch (90th percentile) by Method", "Stretch p90 (lower is better)")


# def plot_tradeoff_scatter(rows):
#     """
#     Scatter: edge_ratio vs MSE
#     Best region: bottom-left (low edge_ratio, low MSE)
#     """
#     plt.figure()
#     for r in rows:
#         plt.scatter(r["edge_ratio"], r["mse"])
#         plt.text(r["edge_ratio"], r["mse"], f'{r["graph"]}-{r["method"]}', fontsize=8)

#     plt.title("Trade-off: Sparsity vs Distortion (edge_ratio vs MSE)")
#     plt.xlabel("Edge ratio m'/m (lower = sparser)")
#     plt.ylabel("MSE (lower = better)")
#     plt.tight_layout()
#     plt.show()


# def plot_radar_for_graph(rows, graph_name: str):
#     """
#     Radar chart for ONE graph type comparing methods across normalized metrics.
#     Metrics used: edge_ratio, time_sec, mse, mean_rel_err, stretch_p90
#     Lower is better for all; we normalize by max in that graph.
#     """
#     subset = [r for r in rows if r["graph"] == graph_name]
#     methods = sorted(list({r["method"] for r in subset}))

#     metrics = ["edge_ratio", "time_sec", "mse", "mean_rel_err", "stretch_p90"]
#     labels = ["EdgeRatio", "Time", "MSE", "RelErr", "StretchP90"]

#     # compute max per metric within this graph (avoid divide by 0)
#     max_vals = {}
#     for met in metrics:
#         max_vals[met] = max(1e-12, max(r[met] for r in subset))

#     # Radar angles
#     angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
#     angles += angles[:1]  # close loop

#     plt.figure()
#     ax = plt.subplot(111, polar=True)
#     plt.title(f"Radar Comparison (lower is better) — {graph_name}")

#     for m in methods:
#         r = next(rr for rr in subset if rr["method"] == m)
#         # normalize to [0,1] where 1 = worst (max), 0 = best (min-ish)
#         vals = [(r[met] / max_vals[met]) for met in metrics]
#         vals += vals[:1]
#         ax.plot(angles, vals, label=m)
#         ax.fill(angles, vals, alpha=0.1)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels([])

#     plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
#     plt.tight_layout()
#     plt.show()


# # -----------------------------
# # Experiment runner (ER / BA / SBM)
# # -----------------------------
# def run_one_graph(
#     G: nx.Graph,
#     stretch: float = 2.0,
#     seed: int = 0
# ) -> List[EvalResult]:
#     add_unit_weights_if_missing(G)

#     results = []

#     # 1) Greedy Spanner
#     t0 = time.perf_counter()
#     H1 = greedy_spanner(G, stretch=stretch)
#     t1 = time.perf_counter()
#     results.append(evaluate_sparsifier(G, H1, "GreedySpanner", t1 - t0, seed=seed))

#     # 2) Partition-Based
#     t0 = time.perf_counter()
#     H2 = sparsify_partition_based(G, stretch=stretch)
#     t1 = time.perf_counter()
#     results.append(evaluate_sparsifier(G, H2, "PartitionBased", t1 - t0, seed=seed))

#     # 3) Multi-Resolution
#     t0 = time.perf_counter()
#     H3 = multi_resolution_sparsify(G, levels=2, stretch_fine=stretch, stretch_coarse=stretch)
#     t1 = time.perf_counter()
#     results.append(evaluate_sparsifier(G, H3, "MultiResolution", t1 - t0, seed=seed))

#     return results


# def print_results_table(results: List[EvalResult]) -> None:
#     headers = [
#         "Method", "n", "m→m'", "m'/m", "time(s)", "mem_proxy(B)", "conn",
#         "MSE", "rel_err", "stretch(max/p50/p90)"
#     ]
#     print("\n" + "-" * 120)
#     print("{:<15} {:>6} {:>14} {:>7} {:>8} {:>13} {:>6} {:>10} {:>10} {:>22}".format(*headers))
#     print("-" * 120)
#     for r in results:
#         mm = f"{r.m}->{r.m_sp}"
#         st = f"{r.stretch_max:.2f}/{r.stretch_p50:.2f}/{r.stretch_p90:.2f}" if math.isfinite(r.stretch_max) else "inf"
#         print("{:<15} {:>6} {:>14} {:>7.3f} {:>8.3f} {:>13} {:>6} {:>10.4f} {:>10.4f} {:>22}".format(
#             r.name, r.n, mm, r.edge_ratio, r.time_sec, r.mem_proxy_bytes, str(r.connected),
#             r.mse if math.isfinite(r.mse) else float("inf"),
#             r.mean_rel_err if math.isfinite(r.mean_rel_err) else float("inf"),
#             st
#         ))
#     print("-" * 120)


# def main() -> None:
#     set_seed(42)

#     # Keep sizes moderate for greedy spanner (it can be slow on huge dense graphs)
#     n = 250

#     # ER
#     G_er = generate_er(n=n, p=0.04, seed=1)
#     res_er = run_one_graph(G_er, stretch=2.0, seed=1)
#     print("\n=== ER Graph ===")
#     print_results_table(res_er)

#     # BA
#     G_ba = generate_ba(n=n, m_attach=3, seed=2)
#     res_ba = run_one_graph(G_ba, stretch=2.0, seed=2)
#     print("\n=== BA Graph ===")
#     print_results_table(res_ba)

#     # SBM (community structure)
#     G_sbm = generate_sbm(sizes=[80, 80, 90], p_in=0.08, p_out=0.01, seed=3)
#     res_sbm = run_one_graph(G_sbm, stretch=2.0, seed=3)
#     print("\n=== SBM Graph ===")
#     print_results_table(res_sbm)


# if __name__ == "__main__":
#     main()
