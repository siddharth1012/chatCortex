"""Per-synthesizer sanity checks that no regression gate duplicates.

Invariants owned by ``tests/test_regression_gates.py`` — exact-frontier
coverage floors, non-domination against the exact reference, hard-constraint
feasibility, budget behavior, and the infeasible/missing-capability
divergence — live only there. This module pins the two checks the gates
do not make: the objective tuple stays positive/in-range for every
returned candidate, and no returned set contains a dominated member.
"""

import pytest

from chatcortex import (
    BeamSynthesizer,
    ExhaustiveSynthesizer,
    HeuristicSynthesizer,
    ParetoPartialBeamSynthesizer,
    ProgressiveParetoBeamSynthesizer,
    RandomSynthesizer,
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
