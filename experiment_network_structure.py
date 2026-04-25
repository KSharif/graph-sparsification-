import os
import csv
import matplotlib.pyplot as plt

from main import (
    set_seed,
    generate_er,
    generate_ba,
    generate_sbm,
    run_one_graph,
)

# -----------------------------
# Experiment 4: Effect of network structure
# -----------------------------

RESULT_DIR = "experiment_structure_results"
os.makedirs(RESULT_DIR, exist_ok=True)


def save_csv(rows, filename):
    filepath = os.path.join(RESULT_DIR, filename)

    fieldnames = [
        "graph_type",
        "method",
        "n",
        "original_edges",
        "sparsified_edges",
        "edge_ratio",
        "time_sec",
        "mem_proxy_bytes",
        "connected",
        "mse",
        "mean_rel_err",
        "stretch_max",
        "stretch_p50",
        "stretch_p90",
    ]

    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved at: {filepath}")


def plot_metric(rows, metric_name, ylabel, filename):
    methods = sorted(list(set(row["method"] for row in rows)))
    graph_types = ["ER", "BA", "SBM"]

    x_positions = list(range(len(graph_types)))
    bar_width = 0.25

    plt.figure(figsize=(9, 5))

    for i, method in enumerate(methods):
        y_values = []

        for graph_type in graph_types:
            matched_rows = [
                row for row in rows
                if row["method"] == method and row["graph_type"] == graph_type
            ]

            if matched_rows:
                y_values.append(matched_rows[0][metric_name])
            else:
                y_values.append(0)

        shifted_positions = [
            x + (i - 1) * bar_width for x in x_positions
        ]

        plt.bar(
            shifted_positions,
            y_values,
            width=bar_width,
            label=method
        )

    plt.xlabel("Network Structure")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by Network Structure")
    plt.xticks(x_positions, graph_types)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    filepath = os.path.join(RESULT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Plot saved at: {filepath}")


def run_structure_experiment():
    set_seed(42)

    # You are changing only this condition
    graph_types = ["ER", "BA", "SBM"]

    # Keep these fixed for this experiment
    n = 500
    stretch = 2.0

    # Graph-specific fixed settings
    er_p = 0.04
    ba_m_attach = 4

    # SBM sizes should sum to n
    sbm_sizes = [150, 150, 200]
    sbm_p_in = 0.06
    sbm_p_out = 0.005

    all_rows = []

    for graph_type in graph_types:
        print("\n" + "=" * 80)
        print(f"Running experiment for network structure = {graph_type}")
        print("=" * 80)

        if graph_type == "ER":
            G = generate_er(
                n=n,
                p=er_p,
                seed=42
            )

        elif graph_type == "BA":
            G = generate_ba(
                n=n,
                m_attach=ba_m_attach,
                seed=42
            )

        elif graph_type == "SBM":
            G = generate_sbm(
                sizes=sbm_sizes,
                p_in=sbm_p_in,
                p_out=sbm_p_out,
                seed=42
            )

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        results = run_one_graph(
            G,
            stretch=stretch,
            seed=42
        )

        for result in results:
            row = {
                "graph_type": graph_type,
                "method": result.name,
                "n": result.n,
                "original_edges": result.m,
                "sparsified_edges": result.m_sp,
                "edge_ratio": result.edge_ratio,
                "time_sec": result.time_sec,
                "mem_proxy_bytes": result.mem_proxy_bytes,
                "connected": result.connected,
                "mse": result.mse,
                "mean_rel_err": result.mean_rel_err,
                "stretch_max": result.stretch_max,
                "stretch_p50": result.stretch_p50,
                "stretch_p90": result.stretch_p90,
            }

            all_rows.append(row)

            print(
                f"{result.name}: "
                f"graph={graph_type}, "
                f"n={result.n}, "
                f"edges={result.m}->{result.m_sp}, "
                f"edge_ratio={result.edge_ratio:.3f}, "
                f"time={result.time_sec:.3f}s, "
                f"MSE={result.mse:.4f}, "
                f"rel_err={result.mean_rel_err:.4f}, "
                f"stretch_p90={result.stretch_p90:.2f}"
            )

    save_csv(all_rows, "network_structure_experiment.csv")

    # -----------------------------
    # Visualization comparison
    # -----------------------------

    plot_metric(
        rows=all_rows,
        metric_name="original_edges",
        ylabel="Original Number of Edges",
        filename="original_edges_by_structure.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="edge_ratio",
        ylabel="Edge Ratio",
        filename="edge_ratio_by_structure.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="time_sec",
        ylabel="Runtime in Seconds",
        filename="runtime_by_structure.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="mse",
        ylabel="Mean Squared Error",
        filename="mse_by_structure.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="mean_rel_err",
        ylabel="Mean Relative Error",
        filename="relative_error_by_structure.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="stretch_p90",
        ylabel="90th Percentile Stretch",
        filename="stretch_p90_by_structure.png"
    )

    print("\nExperiment completed successfully.")
    print(f"All results are saved inside: {RESULT_DIR}")


if __name__ == "__main__":
    run_structure_experiment()
