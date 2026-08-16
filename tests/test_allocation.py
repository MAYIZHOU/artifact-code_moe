import numpy as np
import torch

from moe_market.allocation import (
    allocation_shares,
    coalition_values,
    exact_shapley,
    payment_shares_from_shapley,
)


def test_allocation_rules_are_budget_balanced():
    weights = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    shares = allocation_shares(weights, [1.0, 2.0, 3.0], discount=0.5)
    for values in shares.values():
        assert np.isclose(values.sum(), 1.0)
        assert np.all(values >= 0)


def test_exact_shapley_satisfies_efficiency():
    expert_predictions = torch.tensor(
        [
            [[0.9, 0.1], [0.6, 0.4]],
            [[0.2, 0.8], [0.4, 0.6]],
        ]
    )
    weights = torch.full((2, 2), 0.5)
    target = torch.tensor([0, 1])
    values = coalition_values("classification", expert_predictions, weights, target, 1.0, 1.0)
    shapley = exact_shapley(values, 2)
    assert np.isclose(shapley.sum(), values[frozenset({0, 1})], atol=1e-7)
    assert np.isclose(payment_shares_from_shapley(shapley).sum(), 1.0)
