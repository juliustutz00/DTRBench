# DTRBench

Comparing decision trees is inherently non-trivial due to their structure. While there exist potential tree distance measures, these focus on structure and do not capture the functionality of the underlying decision trees.

Representations of said decision trees enable a structural and functional comparison by abstracting some information and thereby gaining the ability to quantify similarity.

This benchmark therefore explores the usefulness of different decision tree representations by  
(i) assessing the representations in an isolated setting by using controlled perturbations and measuring correlations between representation distances, performance differences, and feature importance shift,  
(ii) estimating the representation’s effectiveness on downstream tasks by using their distances for a diverse subforest selection which is then compared against a single decision tree and subforests chosen at random or solely based on out-of-bag (OOB) accuracy/MCC, and  
(iii) measuring the runtime and memory requirements of each representation.

<details open>
<summary><h2>Quickstart</h2></summary>

<h3>Installing</h3>

```sh
git clone https://github.com/juliustutz00/DTRBench.git
cd DTRBench

# DTRBench requires Python 3.10
# Recommended: create and activate a virtual environment
# Linux/macOS
python3.10 -m venv .venv
source .venv/bin/activate
# Windows (PowerShell)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1  

python -m pip install -U pip
# for reproducing the exact dependency versions used in our experiments, install the pinned dependencies first (uncomment line below)
# python -m pip install -r requirements.txt
python -m pip install -e .
```

<h3>Usage</h3>

After installation, DTRBench can be used via the `dtrbench` command.

The typical workflow consists of running benchmarks and generating reports from the results:

```sh
# Run benchmarks using benchmark_config.yaml
dtrbench run

# Generate reports using report_config.yaml
dtrbench report
```

By default, DTRBench searches for `benchmark_config.yaml` and `report_config.yaml` in the DTRBench folder. Custom configuration files can be provided using the `--config` option:

```sh
dtrbench run --config path/to/benchmark_config.yaml
dtrbench report --config path/to/report_config.yaml
```

For quick execution of a single benchmark, a dataset and benchmark mode can be specified directly:

```sh
dtrbench run --dataset "dataset_name" --mode perturbation
```

Available benchmark modes are:
- perturbation
- subforest
- resource

The available dataset names are listed in Section [Configs](#configs).

Benchmark results are stored in the results folder. Each benchmark run produces a .csv file with the results and a .json file containing benchmark metadata. Reports generate plots and/or tables according to the selected options in the report configuration.


</details>

<a name="benchmarks"></a>
<details>
<summary><h2>Benchmarks</h2></summary>

<h3>Perturbation Benchmark</h3>

This benchmark evaluates how well different tree representations capture meaningful differences between decision trees. For each base tree, we compute a representation embedding and compare it to embeddings of systematically perturbed versions of the same tree. The resulting representation distances are then related to (a) changes in predictive performance and (b) a shift in feature importance, to assess whether a representation is sensitive to relevant model changes.

<h4>Perturbations</h4>

Perturbations are controlled modifications applied directly to sklearn DecisionTreeClassifiers to generate tree variants with different degrees of change. Implemented perturbations include threshold changes, feature changes, node swaps, node removals, and node additions. After each perturbation, affected subtree statistics are updated so the modified tree remains executable, enabling consistent measurement of both representation distances and structural differences.

<h3>Subforest Benchmark</h3>

This benchmark studies how to choose a small, diverse subset of trees from a larger random forest while retaining predictive quality. The code builds pairwise distance matrices from the selected representations and applies some selection strategy to pick representative trees. Selected subforests are evaluated on held-out test data and compared against baselines such as a single decision tree, random selection, and top-OOB trees.

<h4>Selection Strategies</h4>

Selection strategies are used to select a subset of trees from a random forest, based on the pairwise distances between the trees and their OOB-performance. Implemented selection strategies include normal clustering (k-medoid, agglomerative; choosing the central tree in each cluster), performance clustering (k-medoid-performance, agglomerative-performance; choosing the tree with the best OOB-performance in each cluster), density-based selection (density; iteratively sampling trees using Gaussian density kernels adjust dynamically by pairwise distances), and multi-objective based selection (greedy, simulated_annealing, genetic; selects trees by optimizing for both diversity (using the distance matrix) and performance (using the OOB-performace) at once).

<h3>Resource Benchmark</h3>

The Resource Benchmark measures the computational efficiency in terms of runtime and memory requirements of decision tree representations across different random forest sizes. The framework explicitly decouples resource consumption into representation-building and representation-comparison.

</details>

<details>
<summary><h2>Reporting</h2></summary>

The reporting module analyzes benchmark results and generates plots and summary statistics based on the selected options in the [report config file](report_config.yaml).

Reports can be generated from results of one or multiple datasets. When multiple datasets are provided, results are aggregated across datasets before creating plots and statistics.

The generated reports are organized according to the three benchmark types:

<h3>Perturbation Benchmark Reports</h3>

These reports analyze whether representation distances reflect meaningful changes in decision trees. Generated analyses include:
- Representation similarity versus predictive performance and feature importance shift.
- Representation similarity as a function of perturbation intensity for each perturbation type.

<h3>Subforest Benchmark Reports</h3>

These reports evaluate the effectiveness of representation-based tree selection for random forest compression. Generated analyses include:
- Random forest compression performance.
- MCC comparisons between representations and selection strategies.
- Standard deviation comparisons across configurations.
- Agreement between configurations using Kendall's W.
- Correlation between representation distances and subforest size using Spearman correlation.
- Summary tables comparing representations and configurations across different subforest sizes.

<h3>Resource Benchmark Reports</h3>

These reports evaluate the computational requirements of decision tree representations. Generated analyses include:
- Runtime comparisons for representation generation and similarity computation.
- Memory usage comparisons for representation generation and similarity computation.

All generated plots and tables are stored in the configured output directory. Individual reports can be enabled or disabled through the corresponding options in the [report config file](report_config.yaml).

</details>

<details>
<summary><h2>Representations</h2></summary>

| **Representation** | **Reference** | **Type** | **Distance** |
| --- | --- | --- | --- |
| Tree Descriptor | (novel) | Metric Vector | Cosine Distance |
| Leaf Profile | (novel) | Distribution Vector | Earth Mover's Distance |
| Feature Graph | [Sirocchi et al.](https://doi.org/10.1186/s13040-025-00430-3) | Graph | Correlation-adjusted Frobenius Distance |
| Topological Forest | [Bayir et al.](https://doi.org/10.1109/ACCESS.2022.3229008) | Metric Vector | Mapper Graph Shortest-Path |
| INDTree | [Spinnato et al.](https://www.esann.org/sites/default/files/proceedings/2025/ES2025-85.pdf) | Network Weights | Embedding Space Euclidean Distance |

The following figure shows how a simple decision tree is converted into each respective representation.
<img width="1002" height="1178" alt="image" src="https://github.com/user-attachments/assets/cab588d5-2b79-4e3f-a66d-8220eac6c427" />

</details>

<details>
<summary><h2>Data expectations</h2></summary>

The framework includes 25 UCI datasets out of the box. These were chosen as they have large performance differences between a single decision tree and a random forest, ensuring that representations are evaluated on classification problems where ensembles have a substantial predictive advantage over individual decision trees. 

Expected dataset layout:
```
dataset_name/
  X.npy (numpy array)
  y.npy (numpy array)
  features.csv (csv)
```

Notes: 
* Categorical/binary features are **dropped** as sklearn DecisionTreeClassifier cannot use them (only continuous/integer-like types are kept).
* Splits are created via stratified cross-validation.
* Custom datasets can easily be added to the framework. For further information, refer to Section [Benchmarks](#benchmarks).

</details>

<details>
<summary><h2>Adding custom content</h2></summary>

The framework is designed to be modular and easily extensible, allowing users to evaluate their own decision tree representations without modifying the core benchmark implementation. Besides custom representations, users can also add their own datasets, perturbations, and subforest selection strategies.

All user-defined extensions are placed in the [`user_extensions/`](user_extensions) directory. This directory contains template implementations for each extension type together with inline documentation explaining the required interface and expected behavior.

Once implemented, a custom extension only needs to be registered using the provided registration decorator/function. Registered extensions are discovered automatically by the framework and become available throughout the codebase. In particular, they can be referenced directly from the benchmark and report configuration files in exactly the same way as the built-in implementations.

The framework supports the following extension types:

* Representations – Implement custom decision tree representations together with their corresponding similarity measure.
* Datasets – Add custom datasets to the benchmark by providing the expected dataset structure.
* Perturbations – Implement additional perturbation operators for the perturbation benchmark.
* Selection Strategies – Add new algorithms for selecting representative subforests in the subforest benchmark.

This modular design keeps the benchmark implementation independent of individual methods while making it straightforward to compare new approaches against the built-in baselines.

</details>

<a name="configs"></a>
<details>
<summary><h2>Configs</h2></summary>

<h3>Benchmark Configuration (benchmark_config.yaml)</h3>

The benchmark can be customized through the following parameters. Unless specified otherwise, the default configuration runs all available benchmarks on the `iris` dataset using all default representations, perturbations, and selection strategies.

<h4>General Settings</h4>

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `seed` | Random seed used to ensure reproducible benchmark results. | `int` | Any integer | `42` |
| `print_progress` | Whether to display progress information while running the benchmark. | `bool` | `True`, `False` | `True` |
| `dataset` | Dataset used for evaluation. Custom datasets can be added through the dataset register. | `str` | Registered dataset names | `iris` |
| `n_splits` | Number of cross-validation folds used during evaluation. | `int` | `2`–`10` | `3` |
| `n_samples` | Maximum number of samples taken from the dataset. Leave empty to use the complete dataset. | `int` or `None` | Any integer ≥ `n_splits`, or empty | Empty |

Available built-in datasets:

- `balance_scale`
- `cervical_cancer`
- `cirrhosis`
- `connectionist`
- `credit_approval`
- `cylinder_bands`
- `DARWIN`
- `diabetic_retinopathy`
- `eeg_eye`
- `fertility`
- `heart_disease`
- `heart_failure`
- `iris`
- `isolet`
- `japanese_credit`
- `letter_recognition`
- `monk_problem`
- `musk_1`
- `statlog_australian`
- `statlog_german`
- `statlog_vehicle`
- `support2`
- `vertebral_column`
- `waveform`
- `wine`
<br>

<h4>Benchmark Selection</h4>

These options control which benchmark modules are executed.

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `perturbation_benchmark` | Enable or disable the perturbation benchmark. | `bool` | `True`, `False` | `True` |
| `subforest_benchmark` | Enable or disable the subforest benchmark. | `bool` | `True`, `False` | `True` |
| `resource_benchmark` | Enable or disable the resource benchmark. | `bool` | `True`, `False` | `True` |
<br>

<h4>Perturbation Benchmark Configuration</h4>

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `perturbations` | Perturbation methods applied during the perturbation benchmark. Custom perturbations can be added through the perturbation register. | `list[str]` | Registered perturbation names | `["change_threshold", "change_feature", "swap_nodes", "remove_nodes", "add_nodes"]` |
| `intensities` | Strength levels used when applying perturbations. | `list[float]` | Values between `0` and `1` | `[0.2, 0.4, 0.6, 0.8, 1]` |

Available built-in perturbations:

- `change_threshold`
- `change_feature`
- `swap_nodes`
- `remove_nodes`
- `add_nodes`
<br>

<h4>Subforest Benchmark Configuration</h4>

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `random_forest_size` | Number of trees in the random forest. For perturbation benchmarks, this defines the number of trees being perturbed. For subforest benchmarks, it defines the pool from which trees are selected. | `int` | Any integer ≥ 1 | `300` |
| `subforest_size` | Number of trees selected for each subforest evaluation. | `list[int]` | Values between `1` and `random_forest_size` | `[5, 10, 15, 20, 25, 30]` |
| `selection_strategies` | Strategies used to select subforests. Custom strategies can be added through the selection strategy register. | `list[str]` | Registered selection strategy names | Built-in strategies |

Available built-in selection strategies:

- `k-medoid`
- `k-medoid-performance`
- `agglomerative`
- `agglomerative-performance`
- `density`
- `combination-greedy`
- `combination-simulated_annealing`
- `combination-genetic`
<br>

<h4>Resource Benchmark Configuration</h4>

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `resource_benchmark_sizes` | Random forest sizes evaluated during the resource benchmark. This parameter only affects the resource benchmark. | `list[int]` | Any integer ≥ 1 | `[5, 10, 15, 20, 25, 30]` |
<br>

<h4>Representation Configuration</h4>

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `representations` | Representations evaluated during the benchmark. Custom representations can be added through the representation register. | `list[str]` | Registered representation names | `["Tree Descriptor", "Leaf Profile", "Feature Graph", "Topological Forest", "INDTree"]` |

Available built-in representations:

- `Tree Descriptor`
- `Leaf Profile`
- `Feature Graph`
- `Topological Forest`
- `INDTree`
<br>

<h4>Result File Handling</h4>

Existing result files can be provided to append new benchmark results instead of creating new files.

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `existing_perturbation_results_path` | Path to an existing CSV file where perturbation benchmark results should be appended. | `str` or `None` | Valid file path or empty | Empty |
| `existing_subforest_results_path` | Path to an existing CSV file where subforest benchmark results should be appended. | `str` or `None` | Valid file path or empty | Empty |
| `existing_resource_represent_results_path` | Path to an existing CSV file where resource benchmark results for representation operations should be appended. | `str` or `None` | Valid file path or empty | Empty |
| `existing_resource_similarity_results_path` | Path to an existing CSV file where resource benchmark results for similarity operations should be appended. | `str` or `None` | Valid file path or empty | Empty |
<br>
<br>


<h3>Report Configuration (report_config.yaml)</h3>

The report generation can be customized through the following parameters. Unless specified otherwise, the default configuration creates all available plots and tables.

<h4>Content Configuration</h4>

These options control which representations, perturbations, subforest sizes, and selection strategies are included in generated plots and tables.

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `representations` | Representations to include in plots and tables. The order of the list determines the order in which representations are displayed. Custom representations can be added through the representation register. | `list[str]` | Registered representation names | `["Tree Descriptor", "Leaf Profile", "Feature Graph", "Topological Forest", "INDTree"]` |
| `perturbations` | Perturbation methods to include in plots and tables. The order of the list determines the order in which perturbations are displayed. Custom perturbations can be added through the perturbation register. | `list[str]` | Registered perturbation names | `["change_threshold", "change_feature", "swap_nodes", "remove_nodes", "add_nodes"]` |
| `subforest_sizes` | Subforest sizes considered for plots and tables. Values must not exceed the configured `random_forest_size`. | `list[int]` | Values between `1` and `random_forest_size` | `[5, 10, 15, 20, 25, 30]` |
| `selection_strategies` | Selection strategies to include in plots and tables. The order of the list determines the order in which strategies are displayed. Custom strategies can be added through the selection strategy register. | `list[str]` | Registered selection strategy names | `["k-medoid", "k-medoid-performance", "agglomerative", "agglomerative-performance", "density", "combination-greedy", "combination-simulated_annealing", "combination-genetic"]` |

Available built-in representations:

- `Tree Descriptor`
- `Leaf Profile`
- `Feature Graph`
- `Topological Forest`
- `INDTree`

Available built-in perturbations:

- `change_threshold`
- `change_feature`
- `swap_nodes`
- `remove_nodes`
- `add_nodes`

Available built-in selection strategies:

- `k-medoid`
- `k-medoid-performance`
- `agglomerative`
- `agglomerative-performance`
- `density`
- `combination-greedy`
- `combination-simulated_annealing`
- `combination-genetic`
<br>

<h4>Output File Configuration</h4>

These options control where benchmark results and generated reports are stored.

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `output_dir` | Directory where benchmark results and generated plots/tables are stored. | `str` | Valid directory path | Empty |
| `perturbation_benchmark_results_path` | Path to the CSV file containing perturbation benchmark results. | `str` or `None` | Valid file path or empty | Empty |
| `subforest_selection_results_path` | Path to the CSV file containing subforest selection benchmark results. | `str` or `None` | Valid file path or empty | Empty |
| `resource_benchmark_represent_results_path` | Path to the CSV file containing resource benchmark results for representat-operations. | `str` or `None` | Valid file path or empty | Empty |
| `resource_benchmark_similarity_results_path` | Path to the CSV file containing resource benchmark results for similarity-operations. | `str` or `None` | Valid file path or empty | Empty |
<br>

<h4>Plot and Table Generation Configuration</h4>

These options control which individual plots and tables are generated.

| Parameter | Description | Type | Allowed values | Default |
|-----------|-------------|------|----------------|---------|
| `rep_similarity_vs_performance_feature_importance` | Generate the representation similarity versus performance feature importance plots (perturbation benchmark). | `bool` | `True`, `False` | `True` |
| `similarity_vs_intensity_per_perturbation` | Generate similarity versus perturbation intensity plot (perturbation benchmark). | `bool` | `True`, `False` | `True` |
| `rf_compression` | Generate random forest compression plots (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `mcc_boxplots` | Generate MCC boxplots for benchmark comparisons (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `mcc_representation_selection_strategy` | Generate MCC comparison plot across representations and selection strategies (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `std_representation_selection_strategy` | Generate standard deviation comparison plots across representations and selection strategies (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `kendalls_w_vs_config` | Generate Kendall's W agreement plot across configurations (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `spearman_vs_subforest_size` | Generate Spearman correlation plot with respect to subforest size (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `representation_vs_subforest_size` | Generate representation comparison table with respect to subforest size (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `config_vs_subforest_size` | Generate configuration comparison table with respect to subforest size  (subforest benchmark). | `bool` | `True`, `False` | `True` |
| `resource_benchmark_represent` | Generate runtime and memory requirement plots for represent-operations  (resource benchmark). | `bool` | `True`, `False` | `True` |
| `resource_benchmark_similarity` | Generate runtime and memory requirement plots for similarity-operations (resource benchmark). | `bool` | `True`, `False` | `True` |
<br>

</details>

<details>
<summary><h2>Limitations</h2></summary>

* Only works for classification tasks; regression tasks are not supported
* Only works with numerical features; categorical/binary features are not supported (except for the perturbations, they support categorical data)
* Only implements 25 UCI dataset; other types of datasets (e.g., HDLSS) are not included by default

</details>
