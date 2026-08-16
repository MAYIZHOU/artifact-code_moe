# Cost-Aware Mixture-of-Experts Coordination for Model Markets

This repository contains the reproducible experimental pipeline for **Cost-Aware
Mixture-of-Experts Coordination for Model Markets**. A single configuration-driven
entry point supports every dataset reported in the paper and can regenerate the
main table data and Figures.

## Reproduction protocol

All public experiment profiles use the same controlled protocol:

- five random seeds: `1, 2, 3, 4, 5`;
- four heterogeneous experts: an MLP, a Transformer, XGBoost, and CatBoost;
- every expert is fitted independently on the seed-specific training split and
  frozen before the gating network is trained;
- Standard MoE and MoE Market use the same fitted expert bank and data split;
- Standard MoE minimizes predictive loss, whereas MoE Market minimizes the
  market-aware objective with weighted expert execution cost;
- expert unit costs are measured from inference latency and normalized by their
  mean for the dataset and hardware environment;
- the main experiment uses `beta = 0.3` and the allocation experiment uses
  `lambda = 0.6`;
- every summary reports the mean and sample standard deviation across seeds.

The tabular expert pool uses a two-hidden-layer MLP, a two-layer Transformer,
XGBoost, and CatBoost. The image pool uses an image MLP, a patch Transformer,
XGBoost, and CatBoost. Images are resized to 32 by 32 pixels and exposed to the
tree experts as flattened features.

The comparison methods are Welfare-aware Static Selection (WSS), Uniform Average
(UA), Standard MoE (SM), and MoE Market (MM). Revenue-allocation figures compare
participation-only allocation, exact Shapley allocation, and the proposed
cost-adjusted allocation.

## Supported datasets

Tabular classification:

```text
breast_cancer, page_blocks, yeast, adult, ionosphere
```

Tabular regression:

```text
abalone, concrete, forest_fires, bike_sharing, diabetes, wine_quality
```

Image classification:

```text
mnist, fashion_mnist, kmnist, svhn
```

List the datasets recognized by the installed code:

```bash
python experiments/run_dataset.py --list-datasets
```

## Environment

The reference environment uses Python 3.8.20, PyTorch 2.4.1, torchvision 0.19.1,
and CUDA 12.1. The recommended Conda installation is:

```bash
conda env create -f environment.yml
conda activate moe_edbt
python -m pip install -e .
```

Alternatively, create an environment manually. Install the PyTorch build matching
the local CUDA driver first, then install the remaining pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

CPU execution is supported by replacing `--device cuda` with `--device cpu`, but
the image and full sensitivity experiments can be substantially slower.

## Data acquisition

No dataset needs to be committed to the repository.

- Registered tabular datasets are downloaded through `ucimlrepo`; Diabetes is
  loaded from scikit-learn.
- Image datasets are downloaded by torchvision into `data/` on first use.
- `configs/dataset_sources.yml` lists optional local mirrors. If a listed file exists,
  it is used automatically; otherwise the runner falls back to the public source.

The local mirrors and downloaded data are intentionally excluded by `.gitignore`.
To use a different local path, edit only the matching entry in
`configs/dataset_sources.yml`.

## One-dataset reproduction

First validate the resolved protocol without loading data or training models:

```bash
python experiments/run_dataset.py --dataset adult --stage all --dry-run
```

Run only the five-seed main table experiment at `beta = 0.3`:

```bash
python experiments/run_dataset.py --dataset adult --stage table --device cuda
```

Run the main table and the complete Figure workflow:

```bash
python experiments/run_dataset.py --dataset adult --stage all --device cuda
```

`--stage figures` is accepted as an alias for the full figure workflow. The main
table is still emitted because the `beta = 0.3` model used by the figures already
contains the required table result.

Before a full run, use the reduced one-seed smoke protocol:

```bash
python experiments/run_dataset.py --dataset adult --stage all --device cuda --quick
```

The quick protocol uses fewer samples, one expert-training epoch, one gate-training
epoch, and the beta grid `[0.0, 0.3, 1.0]`. It checks the pipeline only.

Seeds can be overridden explicitly:

```bash
python experiments/run_dataset.py --dataset adult --stage table --device cuda --seeds 1 2 3 4 5
```

## Generated artifacts

For `--dataset adult`, outputs have the following structure:

```text
results/adult/
|-- raw/
|   |-- resolved_config.yaml
|   |-- run_metadata.json
|   |-- cost_profile.json
|   |-- seed_progress.json
|   |-- main_runs.csv
|   |-- main_summary.csv
|   |-- beta_sensitivity_runs.csv
|   |-- beta_sensitivity_summary.csv
|   |-- allocation_behavior_runs.csv
|   |-- allocation_behavior_summary.csv
|   |-- allocation_comparison_runs.csv
|   |-- allocation_comparison_summary.csv
|   |-- lambda_sensitivity_runs.csv
|   |-- lambda_sensitivity_summary.csv
|   |-- allocation_runtime_runs.csv
|   `-- allocation_runtime_summary.csv
|-- tables/
|   |-- main_table.csv
|   `-- main_table.tex
`-- figures/
    |-- figure3_beta_sensitivity.png
    |-- final_ablation_*.pdf
    |-- figure4_allocation_behavior.pdf
    |-- figure5_allocation_comparison.pdf
    |-- figure6_lambda_sensitivity.pdf
    `-- figure7_allocation_runtime.pdf
```

CSV checkpoints are rewritten after every completed seed. An interrupted run
therefore retains all completed seed results. PNG and PDF figures are generated
from the saved summaries, not from hidden in-memory state.

Figure meanings:

These figure numbers are internal identifiers used by the experimental pipeline
and do not necessarily correspond to the figure numbers in the paper.

- **Figure 3:** sensitivity to the market cost coefficient `beta`. The first panel's
  legend applies to every panel; shaded regions show one standard deviation.
- **Figure 4:** expert participation, measured normalized cost, and revenue share.
- **Figure 5:** participation-only, exact Shapley, and cost-adjusted allocations.
- **Figure 6:** sensitivity of expert revenue shares to `lambda`.
- **Figure 7:** exact Shapley versus cost-adjusted allocation runtime, including
  per-seed speedup.

## Reproduce multiple datasets

Run the main table experiment for all registered paper datasets:

```bash
python scripts/reproduce_all.py --stage table --device cuda
```

Select a subset:

```bash
python scripts/reproduce_all.py --stage all --device cuda --datasets adult abalone mnist
```

Running `--stage all` for every dataset performs many five-seed beta sweeps and can
take a long time. Start with `--quick`, then schedule full datasets individually so
that failures and resource use remain easy to inspect.

## Interpreting reproduced values

Exact numerical identity across machines is not expected. Inference latency is
hardware-dependent, and GPU kernels, library versions, thread scheduling,
floating-point order, and dataset mirror versions can introduce small numerical
differences. Since costs are measured locally, a different CPU or GPU can also
change the relative normalized expert costs and therefore the learned routing.

A valid reproduction should instead preserve the statistical behavior reported in
the paper across the configured seeds: MoE Market should learn cost-sensitive
routing, substantially reduce unnecessary expert cost in appropriate settings,
remain predictively competitive with the learned baselines, and improve or preserve
deployment welfare over the relevant beta range. The raw per-seed CSV files, means,
standard deviations, resolved configuration, and measured latency profile are saved
so that deviations can be diagnosed rather than hidden.

This statement is not a guarantee that every metric on every seed must favor MoE
Market. The paper's claims concern aggregate quality-cost behavior and allocation
properties, not bit-for-bit equality or universal per-dataset dominance.

## License

The original code in this repository is released under the Apache License 2.0.
See `LICENSE` for the full terms. Datasets, pretrained resources, and third-party
dependencies are not redistributed by this repository and remain subject to their
respective licenses and terms of use.

## Verification

Run the offline tests and syntax checks before launching long experiments:

```bash
python -m pytest
python -m compileall -q src experiments scripts
python experiments/run_dataset.py --dataset adult --stage all --dry-run
```

The canonical public entry points are `experiments/run_dataset.py` and
`scripts/reproduce_all.py`. The implementation is organized as follows:

```text
configs/                       experiment and optional local-data configuration
experiments/run_dataset.py     one-dataset table and Figure 3-7 entry point
experiments/workflow_common.py shared training and allocation workflow
scripts/make_paper_figures.py  publication-style Figure 3-7 generation
src/moe_market/data/           loading and train-only-fitted preprocessing
src/moe_market/models/         experts, gating network, and MoE composition
src/moe_market/                costs, objectives, training, evaluation, allocation
tests/                         offline unit and smoke tests
```
