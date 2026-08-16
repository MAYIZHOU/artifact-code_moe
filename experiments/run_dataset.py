from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from common import PROJECT_ROOT, resolve_device
from workflow_common import (
    METHOD_ORDER,
    allocation_outputs,
    fit_expert_bank,
    train_model,
    wide_summary,
    write_outputs,
)

from moe_market.config import load_config, project_path, with_overrides
from moe_market.costs import measure_latency_profile
from moe_market.data.cleaning import clean_tabular_data, prepare_image_data
from moe_market.data.loaders import RawTabularData, load_raw_tabular_data
from moe_market.data.registry import IMAGE_DATASETS, TABULAR_DATASETS
from moe_market.evaluation import evaluate_all_methods
from moe_market.io import environment_metadata
from moe_market.reproducibility import set_seed


PROFILE_CONFIGS = {
    "tabular_classification": "configs/paper_tabular_classification.yaml",
    "tabular_regression": "configs/paper_tabular_regression.yaml",
    "image_classification": "configs/paper_image_classification.yaml",
}
METHOD_LABELS = {
    "Welfare-aware Static Selection": "WSS",
    "Uniform Average": "UA",
    "Standard MoE": "SM",
    "MoE Market": "MM",
}
METRIC_LABELS = {
    "auc": "AUC",
    "f1": "F1-Score",
    "accuracy": "Accuracy",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "cost": "Cost",
    "welfare": "Welfare",
}


def _profile_for(dataset_name: str) -> str:
    if dataset_name in IMAGE_DATASETS:
        return "image_classification"
    task = TABULAR_DATASETS[dataset_name]["task"]
    return f"tabular_{task}"


def _available_datasets() -> List[str]:
    return list(TABULAR_DATASETS) + list(IMAGE_DATASETS)


def _path_exists(spec: Dict[str, Any], project_root: Path) -> bool:
    if spec.get("source") == "adult_files":
        directory = project_root / spec["directory"]
        return all(
            (directory / spec.get(key, default)).exists()
            for key, default in (("train_file", "adult.data"), ("test_file", "adult.test"))
        )
    path_value = spec.get("path")
    if not path_value:
        return False
    path = Path(path_value)
    return (path if path.is_absolute() else project_root / path).exists()


def _optional_local_override(
    dataset_name: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    mapping_path = PROJECT_ROOT / "configs" / "dataset_sources.yml"
    with mapping_path.open("r", encoding="utf-8") as stream:
        mappings = yaml.safe_load(stream) or {}
    spec = copy.deepcopy(mappings.get(dataset_name))
    if not spec:
        return None
    data_options = spec.pop("data_options", {})
    config["data"].update(data_options)
    if not _path_exists(spec, PROJECT_ROOT):
        return None
    return spec


def _configure(arguments: argparse.Namespace) -> Dict[str, Any]:
    dataset_name = arguments.dataset
    profile = _profile_for(dataset_name)
    config_path = arguments.config or str(PROJECT_ROOT / PROFILE_CONFIGS[profile])
    config = with_overrides(load_config(config_path), dataset_name, arguments.seeds)
    config["experiment"] = f"{dataset_name}_paper_reproduction"
    output_root = Path(arguments.output_root)
    dataset_root = output_root / dataset_name
    config["output_dir"] = str(dataset_root / "raw")
    config["figure_dir"] = str(dataset_root / "figures")
    config["table_dir"] = str(dataset_root / "tables")
    config["workflow"] = {
        "beta_sensitivity": arguments.stage in {"figures", "all"},
        "revenue_allocation": arguments.stage in {"figures", "all"},
        "figures": arguments.stage in {"figures", "all"} and not arguments.no_plots,
    }
    config["_quick_run"] = bool(arguments.quick)
    local_override = _optional_local_override(dataset_name, config)
    config["dataset_overrides"] = {dataset_name: local_override} if local_override else {}

    if arguments.quick:
        quick = config["quick"]
        config["seeds"] = [int(config["seeds"][0])]
        config["training"]["epochs"] = int(quick["epochs"])
        config["training"]["expert_fit_epochs"] = int(quick["expert_fit_epochs"])
        config["training"]["patience"] = int(quick["epochs"])
        config["cost"]["warmup"] = int(quick["latency_warmup"])
        config["cost"]["repeats"] = int(quick["latency_repeats"])
        config["sensitivity"]["beta_grid"] = list(quick["beta_grid"])
        config["allocation"]["shapley_runtime_repeats"] = int(
            quick["shapley_runtime_repeats"]
        )
        config["allocation"]["proposed_runtime_repeats"] = int(
            quick["proposed_runtime_repeats"]
        )
        if profile == "image_classification":
            config["data"]["max_train_samples"] = int(quick["image_train_samples"])
            config["data"]["max_validation_samples"] = int(
                quick["image_validation_samples"]
            )
            config["data"]["max_test_samples"] = int(quick["image_test_samples"])
    return config


def _limit_raw_data(raw: RawTabularData, samples: int, seed: int) -> RawTabularData:
    if samples <= 0 or samples >= len(raw.target):
        return raw
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(raw.target), size=samples, replace=False))
    return RawTabularData(
        name=raw.name,
        task=raw.task,
        features=raw.features.iloc[indices].reset_index(drop=True),
        target=raw.target.iloc[indices].reset_index(drop=True),
    )


def _measure_expert_costs(
    config: Dict[str, Any],
    expert_bank: Any,
    train_loader: Iterable,
    device: torch.device,
) -> Dict[str, Any]:
    if expert_bank is None:
        raise RuntimeError("The paper protocol requires an independently trained expert bank.")
    inputs, _ = next(iter(train_loader))
    batch_size = min(int(config["cost"].get("measurement_batch_size", 128)), len(inputs))
    inputs = inputs[:batch_size].to(device)
    expert_bank.to(device)
    profile = measure_latency_profile(
        expert_bank.experts,
        inputs,
        warmup=int(config["cost"].get("warmup", 10)),
        repeats=int(config["cost"].get("repeats", 50)),
        normalization=str(config["cost"].get("normalization", "mean")),
    )
    return {
        "expert": [f"E{index + 1}" for index in range(len(expert_bank.experts))],
        "expert_type": [type(expert).__name__ for expert in expert_bank.experts],
        "measurement_batch_size": batch_size,
        "device": str(device),
        "latency_ms_per_instance": profile["latency_ms_per_instance"],
        "normalized_unit_cost": profile["normalized_unit_cost"],
        "normalization": config["cost"].get("normalization", "mean"),
    }


def _method_rows(
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    seed: int,
    beta: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        metrics = results[method]
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "seed": seed,
            "method": method,
            "training_beta": float(beta),
        }
        row.update(metrics)
        row["training_welfare"] = float(metrics["utility"] - beta * metrics["cost"])
        row["deployment_welfare"] = float(metrics["welfare"])
        rows.append(row)
    return rows


def _metrics(task: str) -> Tuple[List[str], List[str]]:
    predictive = ["auc", "f1", "accuracy"] if task == "classification" else ["mse", "rmse", "mae"]
    return predictive + ["cost", "welfare"], [
        "training_welfare",
        "deployment_welfare",
        "utility",
        "cost",
        *( ["ce", "accuracy", "f1", "auc"] if task == "classification" else predictive ),
    ]


def _write_table(summary: pd.DataFrame, task: str, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    main_metrics, _ = _metrics(task)
    indexed = summary.set_index("method")
    display_rows: List[Dict[str, str]] = []
    maximize = {"auc", "f1", "accuracy", "welfare"}
    for metric in main_metrics:
        means = indexed[f"{metric}_mean"].reindex(METHOD_ORDER)
        best = means.max() if metric in maximize else means.min()
        row = {"Metric": METRIC_LABELS[metric]}
        for method in METHOD_ORDER:
            mean = float(indexed.loc[method, f"{metric}_mean"])
            std = indexed.loc[method, f"{metric}_std"]
            value = f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} $\\pm$ {float(std):.3f}"
            if np.isclose(mean, best, rtol=1e-9, atol=1e-12):
                value = f"\\textbf{{{value}}}"
            row[METHOD_LABELS[method]] = value
        display_rows.append(row)
    display = pd.DataFrame(display_rows)
    display.to_csv(table_dir / "main_table.csv", index=False)
    latex = display.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        caption="Reproduced main results (mean $\\pm$ standard deviation).",
        label="tab:reproduced_main_results",
    )
    (table_dir / "main_table.tex").write_text(latex, encoding="utf-8")


def _write_resolved_metadata(
    config: Dict[str, Any],
    output_dir: Path,
    dataset_source: str,
) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable, stream, sort_keys=False)
    metadata = {
        "environment": environment_metadata(),
        "dataset_source": dataset_source,
        "expert_protocol": ["MLP", "Transformer", "XGBoost", "CatBoost"],
        "expert_training": "independent_frozen",
        "notes": (
            "Latency-derived costs and floating-point training can vary across hardware and "
            "software environments; evaluate replicated trends using the seeded summaries."
        ),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=True)


def _checkpoint(
    config: Dict[str, Any],
    output_dir: Path,
    table_dir: Path,
    main_rows: List[Dict[str, Any]],
    beta_rows: List[Dict[str, Any]],
    behavior_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    lambda_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
) -> None:
    main_metrics, beta_metrics = _metrics(config["task"])
    main_frame = write_outputs(output_dir, "main", main_rows)
    main_summary = wide_summary(main_frame, ["method"], main_metrics)
    main_summary.to_csv(output_dir / "main_summary.csv", index=False)
    _write_table(main_summary, config["task"], table_dir)
    if beta_rows:
        beta_frame = write_outputs(output_dir, "beta_sensitivity", beta_rows)
        wide_summary(beta_frame, ["training_beta", "method"], beta_metrics).to_csv(
            output_dir / "beta_sensitivity_summary.csv", index=False
        )
    if behavior_rows:
        behavior_frame = write_outputs(output_dir, "allocation_behavior", behavior_rows)
        comparison_frame = write_outputs(output_dir, "allocation_comparison", comparison_rows)
        lambda_frame = write_outputs(output_dir, "lambda_sensitivity", lambda_rows)
        runtime_frame = write_outputs(output_dir, "allocation_runtime", runtime_rows)
        wide_summary(
            behavior_frame,
            ["expert"],
            ["participation", "unit_cost", "cost_adjusted_score", "revenue_share"],
        ).to_csv(output_dir / "allocation_behavior_summary.csv", index=False)
        wide_summary(comparison_frame, ["method", "expert"], ["share", "payment"]).to_csv(
            output_dir / "allocation_comparison_summary.csv", index=False
        )
        wide_summary(lambda_frame, ["lambda", "expert"], ["share", "payment"]).to_csv(
            output_dir / "lambda_sensitivity_summary.csv", index=False
        )
        wide_summary(runtime_frame, ["method", "n_experts"], ["runtime_ms"]).to_csv(
            output_dir / "allocation_runtime_summary.csv", index=False
        )


def _print_plan(config: Dict[str, Any], stage: str) -> None:
    dataset_name = config["datasets"][0]
    plan = {
        "dataset": dataset_name,
        "task": config["task"],
        "stage": stage,
        "seeds": config["seeds"],
        "expert_pool": config["model"]["expert_pool"],
        "expert_training": config["model"]["training_mode"],
        "beta": config["objective"]["beta"],
        "beta_grid": config["sensitivity"]["beta_grid"] if stage != "table" else [config["objective"]["beta"]],
        "lambda": config["allocation"]["lambda"],
        "output_dir": config["output_dir"],
        "figure_dir": config["figure_dir"],
        "table_dir": config["table_dir"],
        "local_data": bool(config.get("dataset_overrides")),
    }
    print(json.dumps(plan, indent=2))


def run(config: Dict[str, Any], device_name: str, stage: str, make_plots: bool) -> None:
    dataset_name = config["datasets"][0]
    is_image = dataset_name in IMAGE_DATASETS
    device = resolve_device(device_name)
    output_dir = project_path(config, config["output_dir"])
    figure_dir = project_path(config, config["figure_dir"])
    table_dir = project_path(config, config["table_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    local_override = config.get("dataset_overrides", {}).get(dataset_name)
    dataset_source = local_override.get("source", "local") if local_override else (
        "torchvision" if is_image else TABULAR_DATASETS[dataset_name]["source"]
    )
    _write_resolved_metadata(config, output_dir, dataset_source)

    raw_full: Optional[RawTabularData] = None
    if not is_image:
        raw_full = load_raw_tabular_data(
            dataset_name,
            config["task"],
            project_path(config, "."),
            local_override,
        )

    def prepare(seed: int) -> Any:
        if is_image:
            return prepare_image_data(dataset_name, config["data"], project_path(config, "."), seed)
        raw = raw_full
        if raw is None:
            raise RuntimeError("Tabular data were not loaded.")
        if config.get("_quick_run", False):
            raw = _limit_raw_data(raw, int(config["quick"]["samples"]), seed)
        return clean_tabular_data(raw, config["data"], seed)

    run_figures = stage in {"figures", "all"}
    main_beta = float(config["objective"]["beta"])
    beta_grid = (
        sorted({float(value) for value in config["sensitivity"]["beta_grid"]})
        if run_figures
        else [main_beta]
    )
    if main_beta not in beta_grid:
        beta_grid.append(main_beta)
        beta_grid.sort()

    main_rows: List[Dict[str, Any]] = []
    beta_rows: List[Dict[str, Any]] = []
    behavior_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    lambda_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    progress: List[Dict[str, Any]] = []

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        print(f"[{dataset_name}] seed={seed} device={device} stage={stage}")
        data = prepare(seed)
        model_input = data.input_shape if is_image else data.input_dim
        expert_bank = fit_expert_bank(config, data, seed, device)
        if "values" not in config["cost"]:
            cost_profile = _measure_expert_costs(config, expert_bank, data.train_loader, device)
            config["cost"]["values"] = list(cost_profile["normalized_unit_cost"])
            with (output_dir / "cost_profile.json").open("w", encoding="utf-8") as stream:
                json.dump(cost_profile, stream, indent=2)
            _write_resolved_metadata(config, output_dir, dataset_source)
            print("Measured normalized expert unit costs:", config["cost"]["values"])

        standard = train_model(
            config,
            model_input,
            data.num_outputs,
            data.train_loader,
            data.validation_loader,
            seed,
            device,
            "predictive",
            main_beta,
            expert_bank,
        )
        main_market = None
        for beta in beta_grid:
            print(f"  training MoE Market with beta={beta:.2f}")
            market = train_model(
                config,
                model_input,
                data.num_outputs,
                data.train_loader,
                data.validation_loader,
                seed,
                device,
                "market",
                beta,
                expert_bank,
            )
            objective = copy.deepcopy(config["objective"])
            objective["beta"] = beta
            results = evaluate_all_methods(
                standard,
                market,
                data.validation_loader,
                data.test_loader,
                config["task"],
                device,
                objective,
                config["cost"],
                best_expert_split=config["evaluation"]["best_expert_split"],
                include_cost_aware_selection=bool(
                    config["evaluation"]["include_cost_aware_selection"]
                ),
            )
            rows = _method_rows(dataset_name, results, seed, beta)
            if run_figures:
                beta_rows.extend(rows)
            if abs(beta - main_beta) < 1e-12:
                main_market = market
                main_rows.extend(rows)

        if main_market is None:
            raise RuntimeError("The main-beta MoE Market model was not trained.")
        if run_figures:
            behavior, comparison, sensitivity, runtime = allocation_outputs(
                config,
                main_market,
                data.test_loader,
                seed,
                device,
                dataset_name=dataset_name,
            )
            behavior_rows.extend(behavior)
            comparison_rows.extend(comparison)
            lambda_rows.extend(sensitivity)
            runtime_rows.extend(runtime)

        progress.append(
            {
                "seed": seed,
                "train": len(data.train_dataset),
                "validation": len(data.validation_dataset),
                "test": len(data.test_dataset),
                "completed": True,
            }
        )
        with (output_dir / "seed_progress.json").open("w", encoding="utf-8") as stream:
            json.dump(progress, stream, indent=2)
        _checkpoint(
            config,
            output_dir,
            table_dir,
            main_rows,
            beta_rows,
            behavior_rows,
            comparison_rows,
            lambda_rows,
            runtime_rows,
        )
        print(f"  completed seed={seed}; checkpoint saved")

    if run_figures and make_plots:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "make_paper_figures.py"),
                "--input-dir",
                str(output_dir),
                "--output-dir",
                str(figure_dir),
                "--task",
                config["task"],
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print(f"Completed {dataset_name}. Raw results: {output_dir}")
    print(f"Table artifacts: {table_dir}")
    if run_figures and make_plots:
        print(f"Figures 3-7: {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the table and Figures 3-7 for any paper dataset."
    )
    parser.add_argument("--dataset", choices=_available_datasets())
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stage", default="all", choices=["table", "figures", "all"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--config", help="Optional protocol config override.")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--quick", action="store_true", help="Run a one-seed smoke experiment.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV summaries without plots.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only.")
    parser.add_argument("--list-datasets", action="store_true")
    arguments = parser.parse_args()
    if arguments.list_datasets:
        for name in _available_datasets():
            print(f"{name}: {_profile_for(name)}")
        return
    if not arguments.dataset:
        parser.error("--dataset is required unless --list-datasets is used.")
    config = _configure(arguments)
    _print_plan(config, arguments.stage)
    if arguments.dry_run:
        return
    set_seed(int(config["seeds"][0]))
    run(config, arguments.device, arguments.stage, not arguments.no_plots)


if __name__ == "__main__":
    main()
