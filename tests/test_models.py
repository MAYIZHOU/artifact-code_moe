import torch
from torch.utils.data import DataLoader, TensorDataset

from moe_market.models.experts import build_expert_pool, fit_native_experts
from moe_market.models.gating import build_gate
from moe_market.models.moe import MoE


def test_tabular_classification_forward_and_top_k_routing():
    experts = build_expert_pool("tabular_classification", 6, 3)
    model = MoE(experts, build_gate(6, len(experts)), "classification", routing="top_k", top_k=2)
    output = model(torch.randn(5, 6))

    assert output.prediction.shape == (5, 3)
    assert output.weights.shape == (5, 4)
    assert torch.allclose(output.prediction.sum(dim=1), torch.ones(5), atol=1e-5)
    assert torch.equal(output.active_mask.sum(dim=1), torch.full((5,), 2))


def test_hybrid_tabular_pool_fits_tree_experts_and_backpropagates():
    features = torch.randn(32, 6)
    target = (features[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(features, target), batch_size=16, shuffle=False)
    options = {"tree": {"iterations": 2, "depth": 2, "threads": 1}}
    experts = build_expert_pool("tabular_classification_hybrid", 6, 2, options)
    model = MoE(experts, build_gate(6, len(experts)), "classification")

    fit_native_experts(model, loader, seed=1)
    output = model(features[:5])
    loss = -output.prediction[:, 1].log().mean()
    loss.backward()

    assert output.prediction.shape == (5, 2)
    assert torch.allclose(output.prediction.sum(dim=1), torch.ones(5), atol=1e-5)
    assert any(parameter.grad is not None for parameter in model.gate.parameters())


def test_hybrid_regression_pool_fits_tree_experts_and_backpropagates():
    features = torch.randn(32, 6)
    target = (2.0 * features[:, 0] - features[:, 1]).reshape(-1, 1)
    loader = DataLoader(TensorDataset(features, target), batch_size=16, shuffle=False)
    options = {"tree": {"iterations": 2, "depth": 2, "threads": 1}}
    experts = build_expert_pool("tabular_regression_hybrid", 6, 1, options)
    model = MoE(experts, build_gate(6, len(experts)), "regression")

    fit_native_experts(model, loader, seed=1)
    output = model(features[:5])
    loss = output.prediction.square().mean()
    loss.backward()

    assert output.prediction.shape == (5, 1)
    assert output.expert_predictions.shape == (5, 4, 1)
    assert any(parameter.grad is not None for parameter in model.gate.parameters())
