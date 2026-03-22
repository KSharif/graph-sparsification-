"""
advanced_studies.py
-------------------
Adds 3 research-style studies + plots (all saved as PNGs in ./plots):

1) Error vs Graph Density study (ER graphs): how distortion changes as p increases
2) Edge Reduction vs Stretch curve: trade-off by varying stretch parameter t
3) Pareto Frontier plot: show nondominated points (best trade-offs) for (edge_ratio, stretch_p90)

How to run:
  (venv) python advanced_studies.py

It depends on your existing `sparsify_core.py`.
This file does NOT open any plot windows; it only saves images.
"""

from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from your core file
from sparsify_core import (
    set_seed,
    generate_er,
    greedy_spanner,
    sparsify_partition_based,
    multi_resolution_sparsify,
)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================
# Metric computation helpers
# =========================

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


@dataclass
class MetricRow:
    graph: str
    method: str
    n: int
    p: float
    stretch_param: float
    m: int
    m_sp: int
    edge_ratio: float
    edge_reduction: float
    time_sec: float
    mse: float
    mean_rel_err: float
    stretch_p90: float


def add_unit_weights_if_missing(G: nx.Graph, weight_key: str = "weight") -> None:
    for u, v, d in G.edges(data=True):
        if weight_key not in d:
            d[weight_key] = 1.0


def compute_metrics(
    G: nx.Graph,
    H: nx.Graph,
    graph_name: str,
    method_name: str,
    n: int,
    p: float,
    stretch_param: float,
    build_time_sec: float,
    pairs_k: int = 1200,
    seed: int = 0,
    weight: str = "weight",
) -> MetricRow:
    m = G.number_of_edges()
    m_sp = H.number_of_edges()
    edge_ratio = (m_sp / m) if m > 0 else 0.0
    edge_reduction = 1.0 - edge_ratio

    pairs = sample_node_pairs(G, k=pairs_k, seed=seed)

    sq_errs = []
    rel_errs = []
    stretches = []

    for s, t in pairs:
        dG = shortest_path_length_safe(G, s, t, weight=weight)
        dH = shortest_path_length_safe(H, s, t, weight=weight)

        if math.isinf(dG) or dG <= 0:
            continue
        if math.isinf(dH):
            # If disconnected pair, skip for now (graph should be connected)
            continue

        diff = dH - dG
        sq_errs.append(diff * diff)
        rel_errs.append(abs(diff) / dG)
        stretches.append(dH / dG)

    if len(sq_errs) == 0:
        mse = mean_rel = stretch_p90 = math.inf
    else:
        mse = float(np.mean(sq_errs))
        mean_rel = float(np.mean(rel_errs))
        stretch_p90 = float(np.percentile(stretches, 90))

    return MetricRow(
        graph=graph_name,
        method=method_name,
        n=n,
        p=p,
        stretch_param=stretch_param,
        m=m,
        m_sp=m_sp,
        edge_ratio=edge_ratio,
        edge_reduction=edge_reduction,
        time_sec=build_time_sec,
        mse=mse,
        mean_rel_err=mean_rel,
        stretch_p90=stretch_p90,
    )


def run_three_methods(G: nx.Graph, graph_name: str, p: float, stretch_param: float, seed: int) -> List[MetricRow]:
    add_unit_weights_if_missing(G)

    rows: List[MetricRow] = []
    n = G.number_of_nodes()

    # 1) GreedySpanner
    t0 = time.perf_counter()
    H1 = greedy_spanner(G, stretch=stretch_param)
    t1 = time.perf_counter()
    rows.append(compute_metrics(G, H1, graph_name, "GreedySpanner", n, p, stretch_param, t1 - t0, seed=seed))

    # 2) PartitionBased
    t0 = time.perf_counter()
    H2 = sparsify_partition_based(G, stretch=stretch_param)
    t1 = time.perf_counter()
    rows.append(compute_metrics(G, H2, graph_name, "PartitionBased", n, p, stretch_param, t1 - t0, seed=seed))

    # 3) MultiResolution
    t0 = time.perf_counter()
    H3 = multi_resolution_sparsify(G, stretch_fine=stretch_param, stretch_coarse=stretch_param)
    t1 = time.perf_counter()
    rows.append(compute_metrics(G, H3, graph_name, "MultiResolution", n, p, stretch_param, t1 - t0, seed=seed))

    return rows


# =========================
# 1) Error vs Graph Density
# =========================

def study_error_vs_density(n: int = 400, ps: List[float] = None, stretch_param: float = 2.0, seed: int = 42):
    """
    Vary ER density p, measure error metrics for each method.
    Saves:
      - density_vs_relerr.png
      - density_vs_stretchp90.png
      - density_vs_mse_log.png
    """
    if ps is None:
        ps = [0.01, 0.02, 0.04, 0.06, 0.08]

    all_rows: List[MetricRow] = []

    for p in ps:
        print(f"[Density Study] n={n}, p={p}")
        G = generate_er(n=n, p=p, seed=seed)
        rows = run_three_methods(G, graph_name="ER", p=p, stretch_param=stretch_param, seed=seed)
        all_rows.extend(rows)

        for r in rows:
            print(f"  {r.method:15s} edge_ratio={r.edge_ratio:.3f} rel_err={r.mean_rel_err:.3f} stretch_p90={r.stretch_p90:.3f} time={r.time_sec:.3f}s")

    methods = sorted({r.method for r in all_rows})

    # Plot A: Mean Relative Error vs p
    plt.figure()
    for m in methods:
        xs = [r.p for r in all_rows if r.method == m]
        ys = [r.mean_rel_err for r in all_rows if r.method == m]
        plt.plot(xs, ys, marker="o", label=m)

    plt.xlabel("Graph density p (ER)")
    plt.ylabel("Mean relative error (lower is better)")
    plt.title(f"Error vs Density (n={n}, stretch={stretch_param})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "density_vs_relerr.png"), dpi=300)
    plt.close()

    # Plot B: Stretch p90 vs p
    plt.figure()
    for m in methods:
        xs = [r.p for r in all_rows if r.method == m]
        ys = [r.stretch_p90 for r in all_rows if r.method == m]
        plt.plot(xs, ys, marker="o", label=m)

    plt.xlabel("Graph density p (ER)")
    plt.ylabel("Stretch p90 (lower is better)")
    plt.title(f"Stretch vs Density (n={n}, stretch={stretch_param})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "density_vs_stretchp90.png"), dpi=300)
    plt.close()

    # Plot C: MSE vs p (log)
    plt.figure()
    for m in methods:
        xs = [r.p for r in all_rows if r.method == m]
        ys = [r.mse for r in all_rows if r.method == m]
        plt.plot(xs, ys, marker="o", label=m)

    plt.yscale("log")
    plt.xlabel("Graph density p (ER)")
    plt.ylabel("MSE (log scale, lower is better)")
    plt.title(f"MSE vs Density (n={n}, stretch={stretch_param})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "density_vs_mse_log.png"), dpi=300)
    plt.close()

    print("Saved density plots: density_vs_relerr.png, density_vs_stretchp90.png, density_vs_mse_log.png")


# ==================================
# 2) Edge Reduction vs Stretch Curve
# ==================================

def study_edge_reduction_vs_stretch(n: int = 350, p: float = 0.04, stretches: List[float] = None, seed: int = 42):
    """
    Vary stretch parameter t and plot trade-off:
      x = edge reduction (1 - m'/m)
      y = stretch_p90
    Saves:
      - edge_reduction_vs_stretchp90.png
    """
    if stretches is None:
        stretches = [1.5, 2.0, 3.0, 4.0]

    all_rows: List[MetricRow] = []

    for t in stretches:
        print(f"[EdgeReduction-Stretch] n={n}, p={p}, stretch={t}")
        G = generate_er(n=n, p=p, seed=seed)
        rows = run_three_methods(G, graph_name="ER", p=p, stretch_param=t, seed=seed)
        all_rows.extend(rows)

        for r in rows:
            print(f"  {r.method:15s} edge_reduction={r.edge_reduction:.3f} stretch_p90={r.stretch_p90:.3f} time={r.time_sec:.3f}s")

    methods = sorted({r.method for r in all_rows})

    plt.figure()
    for m in methods:
        xs = [r.edge_reduction for r in all_rows if r.method == m]
        ys = [r.stretch_p90 for r in all_rows if r.method == m]
        plt.plot(xs, ys, marker="o", label=m)

    plt.xlabel("Edge reduction (1 - m'/m)  (higher = fewer edges)")
    plt.ylabel("Stretch p90 (lower = better)")
    plt.title(f"Edge Reduction vs Stretch (ER, n={n}, p={p})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "edge_reduction_vs_stretchp90.png"), dpi=300)
    plt.close()

    print("Saved: edge_reduction_vs_stretchp90.png")


# =========================
# 3) Pareto Frontier Plot
# =========================

def pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """
    Compute Pareto frontier indices for 2 objectives we want to MINIMIZE:
      - x = edge_ratio (lower is better)
      - y = stretch_p90 (lower is better)

    A point i is dominated if there exists j with:
      xj <= xi and yj <= yi and at least one strict.
    """
    idxs = list(range(len(points)))
    frontier = []

    for i in idxs:
        xi, yi = points[i]
        dominated = False
        for j in idxs:
            if i == j:
                continue
            xj, yj = points[j]
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def study_pareto_frontier(
    n: int = 350,
    ps: List[float] = None,
    stretches: List[float] = None,
    seed: int = 42
):
    """
    Build many points by varying density p and stretch t, then plot:
      x = edge_ratio
      y = stretch_p90
    Highlight Pareto frontier (best trade-offs).

    Saves:
      - pareto_edgeRatio_vs_stretchP90.png
    """
    if ps is None:
        ps = [0.02, 0.04, 0.06]
    if stretches is None:
        stretches = [1.5, 2.0, 3.0, 4.0]

    all_rows: List[MetricRow] = []

    for p in ps:
        for t in stretches:
            print(f"[Pareto] n={n}, p={p}, stretch={t}")
            G = generate_er(n=n, p=p, seed=seed)
            rows = run_three_methods(G, graph_name="ER", p=p, stretch_param=t, seed=seed)
            all_rows.extend(rows)

    # Build points
    points = [(r.edge_ratio, r.stretch_p90) for r in all_rows]
    fidx = set(pareto_frontier(points))

    # Plot all points
    plt.figure()
    for i, r in enumerate(all_rows):
        plt.scatter(r.edge_ratio, r.stretch_p90, alpha=0.8)
        # label compactly
        label = f"{r.method},p={r.p},t={r.stretch_param}"
        plt.text(r.edge_ratio, r.stretch_p90, label, fontsize=6)

    # Highlight frontier points
    fx = [all_rows[i].edge_ratio for i in fidx]
    fy = [all_rows[i].stretch_p90 for i in fidx]
    plt.scatter(fx, fy, marker="x", s=120)

    plt.xlabel("Edge ratio m'/m (lower = sparser)")
    plt.ylabel("Stretch p90 (lower = better)")
    plt.title(f"Pareto Frontier (ER, n={n}) — minimize (edge_ratio, stretch_p90)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pareto_edgeRatio_vs_stretchP90.png"), dpi=300)
    plt.close()

    print("Saved: pareto_edgeRatio_vs_stretchP90.png")


# =========================
# Main entry
# =========================

def main():
    set_seed(42)

    # 1) Error vs density (keep n moderate so GreedySpanner doesn’t explode)
    study_error_vs_density(n=400, ps=[0.01, 0.02, 0.04, 0.06, 0.08], stretch_param=2.0, seed=42)

    # 2) Edge reduction vs stretch curve
    study_edge_reduction_vs_stretch(n=350, p=0.04, stretches=[1.5, 2.0, 3.0, 4.0], seed=42)

    # 3) Pareto frontier (a grid of p and t)
    study_pareto_frontier(n=350, ps=[0.02, 0.04, 0.06], stretches=[1.5, 2.0, 3.0, 4.0], seed=42)

    print("\nAll advanced-study plots saved to ./plots/")
    print("Check with: ls plots | grep -E 'density|edge_reduction|pareto'")


if __name__ == "__main__":
    main()
