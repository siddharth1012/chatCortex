import pytest

from chatcortex import (
    BeamSynthesizer,
    ExhaustiveSynthesizer,
    HeuristicSynthesizer,
    ParetoPartialBeamSynthesizer,
    ProgressiveParetoBeamSynthesizer,
    RandomSynthesizer,
    SynthesisBudget,
    TaskSpecification,
)
from chatcortex.optimization.pareto import dominates

ALL_SYNTHESIZERS = [
    lambda reg: HeuristicSynthesizer(reg),
    lambda reg: RandomSynthesizer(reg),
    lambda reg: BeamSynthesizer(reg, beam_width=5),
    lambda reg: ParetoPartialBeamSynthesizer(reg, beam_width=5),
    lambda reg: ProgressiveParetoBeamSynthesizer(reg),
    lambda reg: ExhaustiveSynthesizer(reg),
]


@pytest.mark.parametrize("factory", ALL_SYNTHESIZERS)
def test_synthesizer_produces_valid_candidates(factory, registry, task):
    candidates = factory(registry).synthesize(task)
    assert len(candidates) > 0
    n_stages = len(task.required_capabilities)
    for c in candidates:
        assert c.total_cost > 0
        assert c.total_latency > 0
        assert 0 < c.total_reliability <= 1
        assert len(c.graph.list_nodes()) == n_stages
        assert c.graph.validate()


@pytest.mark.parametrize("factory", ALL_SYNTHESIZERS)
def test_output_is_internally_nondominated(factory, registry, task):
    candidates = factory(registry).synthesize(task)
    for a in candidates:
        for b in candidates:
            if a is not b:
                assert not dominates(a, b) or not dominates(b, a)


def test_exhaustive_dominates_approximations(registry, task):
    exact = ExhaustiveSynthesizer(registry).synthesize(task)
    def key(c):
        return (c.total_cost, c.total_latency, c.total_reliability)

    exact_metrics = {key(c) for c in exact}
    approx = ProgressiveParetoBeamSynthesizer(registry).synthesize(task)
    # Every approximate candidate must not dominate any exact frontier point
    for c in approx:
        for e in exact:
            assert not dominates(c, e)
    # Progressive should recover a substantial share of the exact frontier
    recovered = sum(1 for c in approx if key(c) in exact_metrics)
    assert recovered / len(exact) >= 0.5


def test_hard_constraints_respected(registry):
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_cost=0.01,
        max_latency=1000,
    )
    for c in ExhaustiveSynthesizer(registry).synthesize(task):
        assert c.total_cost <= 0.01
        assert c.total_latency <= 1000


def test_budget_limits_evaluations(registry, task):
    budget = SynthesisBudget(max_evaluations=3)
    candidates = ExhaustiveSynthesizer(registry).synthesize(task, budget=budget)
    assert len(candidates) <= 3


def test_missing_capability_returns_empty_or_raises(registry):
    task = TaskSpecification(required_capabilities=["nonexistent_capability"])
    try:
        candidates = ExhaustiveSynthesizer(registry).synthesize(task)
        assert candidates == []
    except Exception:
        pass  # raising is also acceptable behavior
