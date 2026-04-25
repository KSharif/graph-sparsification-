# Graph Sparsification: Experimental Comparison of GreedySpanner, PartitionBased, and MultiResolution Methods

This project implements and evaluates multiple graph sparsification methods on synthetic graph datasets. The main goal is to reduce the number of edges in a graph while preserving important structural properties such as connectivity and shortest-path distances.

The project compares three graph sparsification methods:

1. **GreedySpanner**
2. **PartitionBased**
3. **MultiResolution**

The experiments evaluate how these methods behave under different graph sizes, graph densities, stretch values, and network structures.

---

## 1. Project Motivation

Large graphs appear in many real-world applications, including social networks, communication systems, biological networks, transportation systems, and recommendation systems. As graphs become larger and denser, storing and processing all edges becomes computationally expensive.

Graph sparsification aims to produce a smaller graph that keeps only the most important edges while preserving the original graph’s structure as much as possible.

This project studies the trade-off between:

- **Compression:** how many edges are removed
- **Accuracy:** how well shortest-path distances are preserved
- **Runtime:** how efficiently the sparsification method runs
- **Connectivity:** whether the sparsified graph remains connected

---

## 2. Project Objective

The main objectives of this project are:

- To implement and compare different graph sparsification methods.
- To evaluate edge reduction and structural preservation.
- To study how graph size affects sparsification performance.
- To analyze the effect of stretch values on sparsification quality.
- To examine how graph density influences edge reduction and error.
- To compare method behavior across different network structures.
- To identify which method provides the best trade-off between efficiency and accuracy.

---

## 3. Sparsification Methods

### 3.1 GreedySpanner

The GreedySpanner method keeps edges that are important for preserving shortest-path distances. It provides strong distance preservation but usually keeps more edges than the other methods.

**Strengths:**

- Very low distance error
- Strong shortest-path preservation
- Stable across different graph types

**Weaknesses:**

- Higher runtime for larger graphs
- Lower edge reduction compared to other methods

---

### 3.2 PartitionBased

The PartitionBased method divides the graph into partitions and performs sparsification in a more structured way. It usually removes more edges than GreedySpanner and runs faster.

**Strengths:**

- Strong edge reduction
- Faster runtime
- Good trade-off between compression and accuracy

**Weaknesses:**

- Higher error than GreedySpanner
- Some shortest-path distortion is introduced

---

### 3.3 MultiResolution

The MultiResolution method performs sparsification using a multi-level or multi-scale strategy. It is also aggressive in reducing edges.

**Strengths:**

- Strong compression
- Fast runtime
- Similar edge reduction to PartitionBased

**Weaknesses:**

- Higher MSE and relative error in several experiments
- Less stable on some network structures, especially BA graphs

---

## 4. Graph Models Used

This project evaluates the sparsification methods on three synthetic graph models.

### 4.1 ER Graph

**Erdős–Rényi graph**

An ER graph is a random graph where each possible edge is created with probability `p`.

Used to represent random network connectivity.

---

### 4.2 BA Graph

**Barabási–Albert graph**

A BA graph is a scale-free graph generated using preferential attachment. Some nodes become high-degree hubs.

Used to represent networks with hub-based structures, such as social networks or web networks.

---

### 4.3 SBM Graph

**Stochastic Block Model graph**

An SBM graph contains community structure. Nodes are divided into groups, and edge probabilities are different within and across communities.

Used to represent community-based networks.

---

## 5. Evaluation Metrics

The following metrics are used to compare sparsification methods.

### 5.1 Edge Ratio

```text
edge_ratio = number of sparsified edges / number of original edges

A lower edge ratio means stronger sparsification.

5.2 Edge Reduction Percentage
edge_reduction_percent = (1 - edge_ratio) × 100

This shows how many edges were removed.

5.3 Runtime

Runtime measures how long each method takes to sparsify the graph.

Lower runtime means better computational efficiency.

5.4 Memory Proxy

This is an approximate memory indicator based on the number of edges retained in the sparsified graph.

Lower memory proxy means the sparsified graph is smaller.

5.5 MSE

Mean Squared Error measures the difference between shortest-path distances in the original graph and the sparsified graph.

Lower MSE means better preservation of graph distances.

5.6 Mean Relative Error

Mean relative error measures the relative shortest-path distortion.

Lower relative error means the sparsified graph preserves original distances better.

5.7 Stretch

Stretch measures how much longer paths become after sparsification.

This project reports:

stretch_max
stretch_p50
stretch_p90

The stretch_p90 value is especially useful because it shows the 90th percentile path distortion.

5.8 Connectivity

Connectivity checks whether the sparsified graph remains connected.

All methods aim to preserve connectivity after sparsification.

6. Repository Structure
graph-sparsification-
│
├── main.py
├── sparsify_core.py
├── advanced_studies.py
├── scalability_test.py
│
├── experiment_n_nodes.py
├── experiment_stretch.py
├── experiment_graph_density.py
├── experiment_network_structure.py
├── experiment_method_comparison.py
│
├── plots/
│   ├── edge_ratio.png
│   ├── runtime.png
│   ├── mse_log.png
│   ├── relative_error.png
│   ├── stretch_p90.png
│   ├── tradeoff.png
│   ├── radar_ER.png
│   ├── radar_BA.png
│   └── radar_SBM.png
│
├── experiment_n_results/
│   ├── n_nodes_experiment.csv
│   ├── edge_ratio_vs_n.png
│   ├── runtime_vs_n.png
│   ├── mse_vs_n.png
│   ├── relative_error_vs_n.png
│   └── stretch_p90_vs_n.png
│
├── experiment_stretch_results/
│   ├── stretch_experiment.csv
│   ├── edge_ratio_vs_stretch.png
│   ├── runtime_vs_stretch.png
│   ├── mse_vs_stretch.png
│   ├── relative_error_vs_stretch.png
│   └── stretch_p90_vs_stretch.png
│
├── experiment_density_results/
│   ├── density_experiment.csv
│   ├── original_edges_vs_density.png
│   ├── edge_ratio_vs_density.png
│   ├── runtime_vs_density.png
│   ├── mse_vs_density.png
│   ├── relative_error_vs_density.png
│   └── stretch_p90_vs_density.png
│
├── experiment_structure_results/
│   ├── network_structure_experiment.csv
│   ├── original_edges_by_structure.png
│   ├── edge_ratio_by_structure.png
│   ├── runtime_by_structure.png
│   ├── mse_by_structure.png
│   ├── relative_error_by_structure.png
│   └── stretch_p90_by_structure.png
│
├── experiment_method_results/
│   ├── method_comparison_experiment.csv
│   ├── edge_ratio_by_method.png
│   ├── edge_reduction_by_method.png
│   ├── runtime_by_method.png
│   ├── mse_by_method.png
│   ├── relative_error_by_method.png
│   ├── stretch_p90_by_method.png
│   ├── average_edge_ratio_by_method.png
│   ├── average_edge_reduction_by_method.png
│   ├── average_runtime_by_method.png
│   └── average_mse_by_method.png
│
├── README.md
└── .gitignore
7. Installation and Setup
7.1 Clone the Repository
git clone https://github.com/KSharif/graph-sparsification-.git
cd graph-sparsification-
7.2 Create a Virtual Environment

For macOS or Linux:

python3 -m venv venv
source venv/bin/activate

For Windows:

python -m venv venv
venv\Scripts\activate
7.3 Install Required Packages
pip install numpy networkx matplotlib
8. Running the Baseline Experiment

The baseline experiment runs all three methods on ER, BA, and SBM graphs.

python main.py

This generates the baseline plots inside the plots/ folder.

Baseline Output Plots
plots/edge_ratio.png
plots/runtime.png
plots/mse_log.png
plots/relative_error.png
plots/stretch_p90.png
plots/tradeoff.png
plots/radar_ER.png
plots/radar_BA.png
plots/radar_SBM.png
9. Baseline Experiment Summary

The baseline experiment compares the three methods on ER, BA, and SBM graphs.

Main Observation

GreedySpanner achieves the best distance preservation but keeps more edges. PartitionBased and MultiResolution remove more edges but introduce higher distance distortion.

Baseline Result Summary
Method	Main Strength	Main Weakness
GreedySpanner	Lowest MSE and relative error	Keeps more edges and slower
PartitionBased	Best balance of compression and runtime	Higher error than GreedySpanner
MultiResolution	Strong compression	Highest error in many cases
10. Experiment 1: Effect of Number of Nodes
Purpose

This experiment studies how sparsification methods behave as the graph size increases.

Parameter Changed
n = 100, 250, 500, 1000
Fixed Settings
Graph type = ER
Edge probability p = 0.04
Stretch = 2.0
Run Command
python experiment_n_nodes.py
Output Folder
experiment_n_results/
Generated Files
n_nodes_experiment.csv
edge_ratio_vs_n.png
runtime_vs_n.png
mse_vs_n.png
relative_error_vs_n.png
stretch_p90_vs_n.png
Key Findings

As the number of nodes increases, runtime increases for all methods. GreedySpanner remains the most accurate method but becomes much slower for larger graphs.

For example, at n = 1000:

GreedySpanner runtime: 22.724s
PartitionBased runtime: 3.817s
MultiResolution runtime: 3.774s

GreedySpanner preserves distances best, while PartitionBased and MultiResolution scale better in runtime.

Recommended Plots for Presentation

Use:

runtime_vs_n.png
edge_ratio_vs_n.png
mse_vs_n.png
11. Experiment 2: Effect of Sparsification Stretch
Purpose

This experiment studies how the allowed stretch value affects edge reduction and distance error.

Parameter Changed
stretch = 1.5, 2.0, 3.0, 4.0
Fixed Settings
Graph type = ER
n = 500
p = 0.04
Run Command
python experiment_stretch.py
Output Folder
experiment_stretch_results/
Generated Files
stretch_experiment.csv
edge_ratio_vs_stretch.png
runtime_vs_stretch.png
mse_vs_stretch.png
relative_error_vs_stretch.png
stretch_p90_vs_stretch.png
Key Findings

Increasing the stretch value allows methods to remove more edges. However, this also increases distortion.

For GreedySpanner:

At stretch 1.5, edge ratio = 1.000, MSE = 0.0000
At stretch 4.0, edge ratio = 0.303, MSE = 1.8090

This shows a clear trade-off between compression and accuracy.

Recommended Plots for Presentation

Use:

edge_ratio_vs_stretch.png
mse_vs_stretch.png
relative_error_vs_stretch.png
12. Experiment 3: Effect of Graph Density
Purpose

This experiment studies how sparsification methods behave when the original ER graph becomes denser.

Parameter Changed
p = 0.01, 0.02, 0.04, 0.08
Fixed Settings
Graph type = ER
n = 500
Stretch = 2.0
Run Command
python experiment_graph_density.py
Output Folder
experiment_density_results/
Generated Files
density_experiment.csv
original_edges_vs_density.png
edge_ratio_vs_density.png
runtime_vs_density.png
mse_vs_density.png
relative_error_vs_density.png
stretch_p90_vs_density.png
Key Findings

As graph density increases, the number of original edges increases. Denser graphs contain more redundant edges, so sparsification can remove more edges.

For example:

GreedySpanner edge ratio decreases from 0.987 at p = 0.01 to 0.616 at p = 0.08
PartitionBased edge ratio decreases from 0.581 to 0.312
MultiResolution edge ratio decreases from 0.529 to 0.311

Runtime also increases as the graph becomes denser.

Recommended Plots for Presentation

Use:

original_edges_vs_density.png
edge_ratio_vs_density.png
runtime_vs_density.png
mse_vs_density.png
13. Experiment 4: Effect of Network Structure
Purpose

This experiment studies whether the performance of each method depends on the graph topology.

Network Structures Compared
ER = Random graph
BA = Scale-free graph
SBM = Community graph
Fixed Settings
n = 500
Stretch = 2.0
Run Command
python experiment_network_structure.py
Output Folder
experiment_structure_results/
Generated Files
network_structure_experiment.csv
original_edges_by_structure.png
edge_ratio_by_structure.png
runtime_by_structure.png
mse_by_structure.png
relative_error_by_structure.png
stretch_p90_by_structure.png
Key Findings

The results show that graph structure affects sparsification performance.

GreedySpanner is stable across all graph types and has the lowest error.

PartitionBased provides a good balance between compression and accuracy.

MultiResolution performs well in compression but has high error on BA graphs. In the BA graph, MultiResolution reaches an MSE of 40.5705, which is much higher than the other methods.

Recommended Plots for Presentation

Use:

edge_ratio_by_structure.png
mse_by_structure.png
relative_error_by_structure.png
stretch_p90_by_structure.png
14. Experiment 5: Method Comparison
Purpose

This experiment directly compares the three sparsification methods across ER, BA, and SBM graphs.

Methods Compared
GreedySpanner
PartitionBased
MultiResolution
Graph Types Compared
ER
BA
SBM
Fixed Settings
n = 500
Stretch = 2.0
Run Command
python experiment_method_comparison.py
Output Folder
experiment_method_results/
Generated Files
method_comparison_experiment.csv
edge_ratio_by_method.png
edge_reduction_by_method.png
runtime_by_method.png
mse_by_method.png
relative_error_by_method.png
stretch_p90_by_method.png
average_edge_ratio_by_method.png
average_edge_reduction_by_method.png
average_runtime_by_method.png
average_mse_by_method.png
Key Findings

GreedySpanner provides the best accuracy but removes fewer edges.

PartitionBased provides the best overall balance between compression, runtime, and error.

MultiResolution removes many edges but often produces higher error.

Method-Level Summary
Method	Accuracy	Compression	Runtime	Overall Behavior
GreedySpanner	Best	Lowest	Slowest	Best for distance preservation
PartitionBased	Moderate	Strong	Fast	Best overall trade-off
MultiResolution	Lower	Strong	Fast	Aggressive but less stable
Recommended Plots for Presentation

Use:

average_edge_reduction_by_method.png
average_runtime_by_method.png
average_mse_by_method.png
edge_reduction_by_method.png
15. Overall Results Summary

Across all experiments, the results show a clear trade-off between edge reduction and distance preservation.

GreedySpanner

GreedySpanner is the most accurate method. It consistently produces the lowest MSE and relative error. However, it keeps more edges and becomes slower as graph size increases.

PartitionBased

PartitionBased gives the best overall balance. It removes a large percentage of edges, runs faster than GreedySpanner, and keeps error lower than MultiResolution in most cases.

MultiResolution

MultiResolution is aggressive in edge reduction and has fast runtime. However, it often produces higher MSE and relative error, especially for BA graphs.

16. Final Conclusion

This project demonstrates that no single sparsification method is best for all goals.

If the goal is maximum distance preservation, GreedySpanner is the best method.

If the goal is a practical balance between edge reduction, runtime, and accuracy, PartitionBased is the best choice.

If the goal is aggressive compression and runtime efficiency, MultiResolution can be useful, but it may introduce higher structural distortion.

Overall, PartitionBased provides the strongest trade-off across the conducted experiments.

17. Recommended Plots for Final Presentation

The most useful plots for presentation are:

Baseline
plots/edge_ratio.png
plots/runtime.png
plots/mse_log.png
plots/relative_error.png
Experiment 1
experiment_n_results/runtime_vs_n.png
experiment_n_results/edge_ratio_vs_n.png
experiment_n_results/mse_vs_n.png
Experiment 2
experiment_stretch_results/edge_ratio_vs_stretch.png
experiment_stretch_results/mse_vs_stretch.png
experiment_stretch_results/relative_error_vs_stretch.png
Experiment 3
experiment_density_results/original_edges_vs_density.png
experiment_density_results/edge_ratio_vs_density.png
experiment_density_results/runtime_vs_density.png
experiment_density_results/mse_vs_density.png
Experiment 4
experiment_structure_results/edge_ratio_by_structure.png
experiment_structure_results/mse_by_structure.png
experiment_structure_results/relative_error_by_structure.png
Experiment 5
experiment_method_results/average_edge_reduction_by_method.png
experiment_method_results/average_runtime_by_method.png
experiment_method_results/average_mse_by_method.png
experiment_method_results/edge_reduction_by_method.png
18. How to Reproduce All Experiments

Run the following commands in order:

python main.py
python experiment_n_nodes.py
python experiment_stretch.py
python experiment_graph_density.py
python experiment_network_structure.py
python experiment_method_comparison.py

After running these commands, all CSV files and visualization plots will be generated inside their corresponding result folders.

19. Notes
If an ER graph is disconnected, the implementation may evaluate the largest connected component.
This behavior explains why some experiments show a slightly smaller number of nodes than the requested value.
For example, when n = 100, the evaluated graph may show n = 97 because the largest connected component contains 97 nodes.
All experiments preserve connectivity in the sparsified graph.
20. Future Work

Future extensions of this project may include:

Testing on real-world graph datasets.
Adding more sparsification algorithms.
Evaluating spectral similarity.
Measuring centrality preservation.
Studying community preservation after sparsification.
Adding larger-scale experiments with thousands or millions of nodes.
Optimizing the GreedySpanner method for better scalability.



21. Author

Kazi Shaharair Sharif, Tasnia Tahsin Khan, Sadri Islam

Graph sparsification experiment project using Python, NetworkX, NumPy, and Matplotlib.