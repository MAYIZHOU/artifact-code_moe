from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = [
    ".gitignore",
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
]
CONFIG_FILES = [
    "dataset_sources.yml",
    "paper_tabular_classification.yaml",
    "paper_tabular_regression.yaml",
    "paper_image_classification.yaml",
]
EXPERIMENT_FILES = ["__init__.py", "common.py", "workflow_common.py", "run_dataset.py"]
SCRIPT_FILES = [
    "build_release.py",
    "make_paper_figures.py",
    "make_paper_style_revenue_figures.py",
    "plot_beta_sensitivity.py",
    "reproduce_all.py",
]


def _copy_files(source_dir: Path, destination_dir: Path, names: Sequence[str]) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Required release file is missing: {source}")
        shutil.copy2(source, destination_dir / name)


def _ignore_generated(_: str, names: Iterable[str]):
    return [
        name
        for name in names
        if name == "__pycache__"
        or name == ".pytest_cache"
        or name.endswith((".pyc", ".pyo"))
    ]


def build_release(destination: Path) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Release destination must be absent or empty; refusing to overwrite {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    _copy_files(PROJECT_ROOT, destination, ROOT_FILES)
    license_path = PROJECT_ROOT / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, destination / "LICENSE")
    _copy_files(PROJECT_ROOT / "configs", destination / "configs", CONFIG_FILES)
    _copy_files(PROJECT_ROOT / "experiments", destination / "experiments", EXPERIMENT_FILES)
    _copy_files(PROJECT_ROOT / "scripts", destination / "scripts", SCRIPT_FILES)
    shutil.copytree(
        PROJECT_ROOT / "src",
        destination / "src",
        ignore=_ignore_generated,
        dirs_exist_ok=True,
    )
    shutil.copytree(
        PROJECT_ROOT / "tests",
        destination / "tests",
        ignore=_ignore_generated,
        dirs_exist_ok=True,
    )
    (destination / "results").mkdir(exist_ok=True)
    (destination / "results" / ".gitkeep").touch()
    print(f"Built upload-ready release at {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean, upload-ready artifact directory.")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build_release(arguments.output)


if __name__ == "__main__":
    main()
