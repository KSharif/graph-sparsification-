import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsify_core import set_seed, generate_er, run_one_graph_quick

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def run_scalability():
    set_seed(42)

    sizes = [150, 250, 400, 600, 800]  # increase slowly
    p = 0.04                           # keep density constant

    runtime = {"GreedySpanner": [], "PartitionBased": [], "MultiResolution": []}
    edge_ratio = {"GreedySpanner": [], "PartitionBased": [], "MultiResolution": []}

    for n in sizes:
        print(f"\nRunning scalability for n={n}")
        G = generate_er(n=n, p=p, seed=42)

        res_list = run_one_graph_quick(G, stretch=2.0)

        for r in res_list:
            runtime[r.name].append(r.time_sec)
            edge_ratio[r.name].append(r.edge_ratio)

        # Print quick summary for this n
        for r in res_list:
            print(f"{r.name:15s}  time={r.time_sec:.3f}s   edge_ratio={r.edge_ratio:.3f}")

    # ---- Plot 1: Runtime vs n ----
    plt.figure()
    for method in runtime:
        plt.plot(sizes, runtime[method], marker="o", label=method)

    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Scalability: Runtime vs Graph Size (ER)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scalability_runtime.png"), dpi=300)
    plt.close()

    # ---- Plot 2: Edge Ratio vs n ----
    plt.figure()
    for method in edge_ratio:
        plt.plot(sizes, edge_ratio[method], marker="o", label=method)

    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Edge Ratio (m'/m)")
    plt.title("Scalability: Edge Ratio vs Graph Size (ER)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scalability_edge_ratio.png"), dpi=300)
    plt.close()

    print("\nSaved scalability plots in ./plots/")
    print("Files: scalability_runtime.png, scalability_edge_ratio.png")


if __name__ == "__main__":
    run_scalability()
