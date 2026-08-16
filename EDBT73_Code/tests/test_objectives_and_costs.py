import torch

from moe_market.costs import expert_cost_matrix, measure_latency_profile, service_cost
from moe_market.objectives import market_objective


def test_weighted_and_discrete_costs_are_distinct():
    inputs = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    costs = expert_cost_matrix(inputs, [1.0, 2.0], input_scale=0.5)
    weights = torch.tensor([[0.75, 0.25], [0.0, 1.0]])
    active = weights > 0

    weighted = service_cost(weights, costs, active, "weighted")
    discrete = service_cost(weights, costs, active, "discrete")
    assert torch.all(discrete >= weighted)
    assert not torch.allclose(discrete, weighted)


def test_market_objective_rewards_utility_and_penalizes_cost():
    prediction = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    target = torch.tensor([0, 1])
    low, utility, _ = market_objective(
        "classification", prediction, target, torch.tensor([0.5, 0.5]), 1.0, 0.3, 1.0
    )
    high, _, _ = market_objective(
        "classification", prediction, target, torch.tensor([2.0, 2.0]), 1.0, 0.3, 1.0
    )
    assert utility > 0
    assert high > low


def test_latency_profile_is_normalized_to_mean_cost_one():
    experts = [torch.nn.Linear(2, 2), torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))]
    profile = measure_latency_profile(experts, torch.randn(8, 2), warmup=0, repeats=2)
    assert len(profile["latency_ms_per_instance"]) == 2
    assert len(profile["normalized_unit_cost"]) == 2
    assert abs(sum(profile["normalized_unit_cost"]) / 2.0 - 1.0) < 1e-8
