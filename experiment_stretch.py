import os
import csv
import matplotlib.pyplot as plt

from main import (
    set_seed,
    generate_er,
    run_one_graph,
)


# Experiment 2: Effect of sparsification stretch


RESULT_DIR = "experiment_stretch_results"
os.makedirs(RESULT_DIR, exist_ok=True)


def save_csv(rows, filename):
    filepath = os.path.join(RESULT_DIR, filename)

    fieldnames = [
        "stretch",
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
    stretch_values = sorted(list(set(row["stretch"] for row in rows)))

    plt.figure(figsize=(8, 5))

    for method in methods:
        x_values = []
        y_values = []

        for stretch in stretch_values:
            matched_rows = [
                row for row in rows
                if row["method"] == method and row["stretch"] == stretch
            ]

            if matched_rows:
                x_values.append(stretch)
                y_values.append(matched_rows[0][metric_name])

        plt.plot(x_values, y_values, marker="o", label=method)

    plt.xlabel("Sparsification Stretch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Sparsification Stretch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filepath = os.path.join(RESULT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Plot saved at: {filepath}")


def run_stretch_experiment():
    set_seed(42)

    # You are changing only this parameter
    stretch_values = [1.5, 2.0, 3.0, 4.0]

    # Keep these fixed for this experiment
    graph_type = "ER"
    n = 500
    edge_probability = 0.04

    all_rows = []

    for stretch in stretch_values:
        print("\n" + "=" * 80)
        print(f"Running experiment for stretch = {stretch}")
        print("=" * 80)

        # Same graph setting for every stretch value
        G = generate_er(
            n=n,
            p=edge_probability,
            seed=42
        )

        results = run_one_graph(
            G,
            stretch=stretch,
            seed=42
        )

        for result in results:
            row = {
                "stretch": stretch,
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
                f"stretch={stretch}, "
                f"n={result.n}, "
                f"edges={result.m}->{result.m_sp}, "
                f"edge_ratio={result.edge_ratio:.3f}, "
                f"time={result.time_sec:.3f}s, "
                f"MSE={result.mse:.4f}, "
                f"rel_err={result.mean_rel_err:.4f}, "
                f"stretch_p90={result.stretch_p90:.2f}"
            )

    save_csv(all_rows, "stretch_experiment.csv")


    # Visualization comparison


    plot_metric(
        rows=all_rows,
        metric_name="edge_ratio",
        ylabel="Edge Ratio",
        filename="edge_ratio_vs_stretch.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="time_sec",
        ylabel="Runtime in Seconds",
        filename="runtime_vs_stretch.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="mse",
        ylabel="Mean Squared Error",
        filename="mse_vs_stretch.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="mean_rel_err",
        ylabel="Mean Relative Error",
        filename="relative_error_vs_stretch.png"
    )

    plot_metric(
        rows=all_rows,
        metric_name="stretch_p90",
        ylabel="90th Percentile Stretch",
        filename="stretch_p90_vs_stretch.png"
    )

    print("\nExperiment completed successfully.")
    print(f"All results are saved inside: {RESULT_DIR}")


if __name__ == "__main__":
    run_stretch_experiment()
