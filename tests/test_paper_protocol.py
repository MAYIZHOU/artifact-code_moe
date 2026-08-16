from pathlib import Path

from moe_market.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    "paper_tabular_classification.yaml",
    "paper_tabular_regression.yaml",
    "paper_image_classification.yaml",
]


def test_public_profiles_share_the_five_seed_strong_expert_protocol():
    expected_pools = {
        "classification": {
            "tabular_classification_hybrid",
            "image_classification_hybrid",
        },
        "regression": {"tabular_regression_hybrid"},
    }
    for filename in PROFILES:
        config = load_config(PROJECT_ROOT / "configs" / filename)
        assert config["seeds"] == [1, 2, 3, 4, 5]
        assert config["model"]["expert_pool"] in expected_pools[config["task"]]
        assert config["model"]["training_mode"] == "independent_frozen"
        assert config["model"]["provider_split"] == "full"
        assert config["training"]["expert_fit_epochs"] == 10
        assert config["objective"]["beta"] == 0.3
        assert config["cost"]["source"] == "latency"
        assert config["cost"]["training_mode"] == "weighted"
        assert config["cost"]["evaluation_mode"] == "weighted"
        assert config["allocation"]["lambda"] == 0.6
        assert config["sensitivity"]["beta_grid"] == [value / 10 for value in range(11)]
