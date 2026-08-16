import torch
from torch.utils.data import DataLoader

from moe_market.data.cleaning import ArrayDataset
from moe_market.evaluation import evaluate_all_methods
from moe_market.models.experts import build_expert_pool
from moe_market.models.gating import build_gate
from moe_market.models.moe import MoE
from moe_market.reproducibility import set_seed
from moe_market.training import train_moe


def _model():
    experts = build_expert_pool("tabular_regression", 4, 1)
    return MoE(experts, build_gate(4, 4, hidden=8), "regression")


def test_training_and_evaluation_pipeline_runs_end_to_end():
    set_seed(3)
    features = torch.randn(32, 4).numpy()
    target = (features[:, 0] - features[:, 1]).astype("float32")
    dataset = ArrayDataset(features, target, "regression")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    training = {"epochs": 1, "learning_rate": 1e-3, "patience": 1}
    objective = {"alpha": 1.0, "beta": 0.1, "evaluation_beta": 0.1, "loss_reference": 1.0}
    cost = {"values": [1.2, 0.7, 0.8, 1.5], "mode": "weighted", "input_scale": 0.0}
    standard = train_moe(_model(), loader, loader, "regression", "predictive", torch.device("cpu"), training, objective, cost)
    market = train_moe(_model(), loader, loader, "regression", "market", torch.device("cpu"), training, objective, cost)
    results = evaluate_all_methods(
        standard, market, loader, loader, "regression", torch.device("cpu"), objective, cost
    )
    assert set(results) == {
        "Welfare-aware Static Selection",
        "Uniform Average",
        "Standard MoE",
        "MoE Market",
    }
    assert all("welfare" in metrics for metrics in results.values())
