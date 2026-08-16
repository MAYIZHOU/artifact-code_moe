import numpy as np
import pandas as pd
from pathlib import Path

from moe_market.data.cleaning import clean_tabular_data
from moe_market.data.loaders import RawTabularData
from moe_market.data.loaders import load_raw_tabular_data


def test_cleaning_handles_mixed_types_and_missing_values():
    size = 100
    raw = RawTabularData(
        name="synthetic",
        task="classification",
        features=pd.DataFrame(
            {
                "numeric": [float(index) if index % 11 else np.nan for index in range(size)],
                "category": ["a" if index % 2 else "b" for index in range(size)],
            }
        ),
        target=pd.Series([index % 2 for index in range(size)]),
    )
    config = {"validation_ratio": 0.1, "test_ratio": 0.1, "batch_size": 16, "num_workers": 0}
    first = clean_tabular_data(raw, config, seed=7)
    second = clean_tabular_data(raw, config, seed=7)

    assert len(first.train_dataset) == 80
    assert len(first.validation_dataset) == 10
    assert len(first.test_dataset) == 10
    assert first.input_dim == 3
    assert first.num_outputs == 2
    assert np.isfinite(first.train_dataset.features.numpy()).all()
    assert np.array_equal(first.test_dataset.features.numpy(), second.test_dataset.features.numpy())


def test_local_adult_files_are_combined_and_target_suffix_is_removed(tmp_path: Path):
    directory = tmp_path / "adult"
    directory.mkdir()
    row = "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"
    (directory / "adult.data").write_text(row, encoding="utf-8")
    (directory / "adult.test").write_text("|1x3 Cross validator\n" + row.replace("<=50K", ">50K."), encoding="utf-8")
    raw = load_raw_tabular_data(
        "adult",
        "classification",
        tmp_path,
        {"source": "adult_files", "directory": "adult"},
    )
    assert len(raw.target) == 2
    assert raw.target.tolist() == ["<=50K", ">50K"]
