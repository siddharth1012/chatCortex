"""Seeded regression gates for the synthesis layer.

The beam-family synthesizers share almost all of their code and are about to
be consolidated into one search engine. These tests pin their current
observable behavior FIRST, so a future refactor that degrades Pareto
recovery, breaks determinism, exceeds an evaluation budget, returns
constraint-violating architectures, or changes frontier content trips CI
instead of silently reshaping the project's headline claims.

Baselines were recorded on the demo 3-stage registry
(``build_demo_registry(("retrieval", "generation", "verification"))``):

* ``ExhaustiveSynthesizer`` is the exact reference. Under the constrained
  task (max_cost=0.01, max_latency=900) its frontier has 6 unique metric
  points; under the unconstrained task it has 13. Both frontier metric
  sets are pinned verbatim below.
* ``ParetoSet`` deduplicates metric-identical entries and iterates in
  sorted objective order, so every recorded value is an exact measurement
  of the deduplicated frontier; coverage floors sit exactly at the
  measured values (all synthesizers here are deterministic, so equality
  is stable).
* Unconstrained coverage floors (measured against the 13-point exact
  frontier): beam 6/13, pareto_partial 12/13, progressive and seeded
  random 13/13. The floor sits exactly at the measured value, so any
  degradation on the unconstrained task trips the gate — including
  regressions invisible to the constrained frontiers above (the beam
  floor there is 4/6 ≈ 0.667, here only 6/13 ≈ 0.462).
* Budget accounting: ``SynthesisContext`` tracks evaluations but is
  created inside ``synthesize()`` and never returned, so no public
  evaluations-consumed counter exists. The cap is instead pinned
  observably: a cap smaller than the search's full work returns strictly
  fewer candidates than the completed search (provable early stop), a
  cap at exactly that work returns the identical set, and no cap is ever
  exceeded (``test_evaluation_budget_caps_returned_candidates``).

All synthesizers except ``RandomSynthesizer`` contain no RNG, and the seeded
``RandomSynthesizer`` uses ``random.Random(42)`` — so every gate is
deterministic across repeated runs on any Python the CI matrix covers.

The ``SYNTHESIZERS`` descriptor map below is the single owner of "who gets
which budget": adding a synthesizer means one entry here plus its measured
floors. The invariants this file owns are the ones worth a regression gate;
plain sanity checks that no gate duplicates live in
``tests/test_synthesizers.py``.

Run just this module with::

    pytest tests/test_regression_gates.py -q
"""

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
from chatcortex.evaluation.approximation_metrics import average_regret
from chatcortex.optimization.pareto import dominates, frontier_coverage
from chatcortex.synthesis.heuristic_synthesizer import SynthesisError

# --- Recorded reference frontier -----------------------------------------------
# metrics are (total_cost, total_latency, total_reliability), rounded to 6dp.

EXACT_CONSTRAINED_METRICS = {
    (0.0056, 900.0, 0.524472),
    (0.0068, 780.0, 0.694048),
    (0.0078, 660.0, 0.588432),
    (0.0088, 540.0, 0.498888),
    (0.009, 540.0, 0.778688),
    (0.01, 420.0, 0.660192),
}

EXACT_UNCONSTRAINED_METRICS = {
    (0.0024, 1260.0, 0.551368),
    (0.0046, 1020.0, 0.618608),
    (0.0056, 900.0, 0.524472),
    (0.0068, 780.0, 0.694048),
    (0.0078, 660.0, 0.588432),
    (0.0088, 540.0, 0.498888),
    (0.009, 540.0, 0.778688),
    (0.01, 420.0, 0.660192),
    (0.011, 300.0, 0.559728),
    (0.012, 180.0, 0.474552),
    (0.012, 660.0, 0.829472),
    (0.015, 780.0, 0.883568),
    (0.018, 900.0, 0.941192),
}

CONSTRAINED_TASK = TaskSpecification(
    required_capabilities=["retrieval", "generation", "verification"],
    max_cost=0.01,
    max_latency=900,
)
UNCONSTRAINED_TASK = TaskSpecification(
    required_capabilities=["retrieval", "generation", "verification"],
)
INFEASIBLE_TASK = TaskSpecification(
    required_capabilities=["retrieval", "generation", "verification"],
    max_cost=0.0005,  # no 3-stage combination of demo components fits
)

# --- Synthesizers under test -----------------------------------------------------

# --- Synthesizers under test -----------------------------------------------------
# One descriptor per synthesizer: the single owner of "who gets which
# budget". ``build`` constructs it, ``budget`` is the budget its
# pinned-strength gates run with (``None`` — no cap — where there is no
# RNG to pin), and ``approximate`` selects the Pareto-quality gates
# (exact-reference synthesizers are excluded there). Adding a
# synthesizer means adding one entry here plus its measured floors.

SYNTHESIZERS = {
    "exhaustive": {
        "build": lambda reg: ExhaustiveSynthesizer(reg),
        "budget": None,
        "approximate": False,
    },
    "heuristic": {
        "build": lambda reg: HeuristicSynthesizer(reg),
        "budget": None,
        "approximate": False,
    },
    "beam": {
        "build": lambda reg: BeamSynthesizer(reg, beam_width=5),
        "budget": None,
        "approximate": True,
    },
    "pareto_partial": {
        "build": lambda reg: ParetoPartialBeamSynthesizer(reg, beam_width=5),
        "budget": None,
        "approximate": True,
    },
    "progressive": {
        "build": lambda reg: ProgressiveParetoBeamSynthesizer(reg),
        "budget": None,
        "approximate": True,
    },
    "random": {
        "build": lambda reg: RandomSynthesizer(reg),
        "budget": SynthesisBudget(max_evaluations=200, random_seed=42),
        "approximate": True,
    },
}

# The chain-family synthesizers materialize a deterministic
# per-combination evaluation count (see ``_engine_chain_count``).
CHAIN_FAMILY = ("exhaustive", "beam", "pareto_partial", "progressive")


def synthesizer(name: str, registry) :
    """Build the named synthesizer for one test call."""
    return SYNTHESIZERS[name]["build"](registry)

COVERAGE_FLOORS = {
    "beam": 4/6,          # measured 4/6 exact points
    "pareto_partial": 5/6,  # measured 5/6
    "progressive": 1.0,   # measured 1.0 (6/6)
    "random": 1.0,        # measured 1.0 (200 seeded samples)
}

# Same gate on the UNCONSTRAINED task (exact frontier = 13 points).
# Measured with the budgets in PARETO_SYNTHESIZERS; every regret was
# (0, 0, 0), so coverage is the whole gate here.
UNCONSTRAINED_COVERAGE_FLOORS = {
    "beam": 6/13,           # measured 6/13 exact points
    "pareto_partial": 12/13,  # measured 12/13
    "progressive": 1.0,     # measured 1.0 (13/13)
    "random": 1.0,          # measured 1.0 (200 seeded samples)
}

# (cost, latency, reliability) ceilings; measured values in trailing comment.
REGRET_CEILINGS = {
    "beam": (1e-6, 1e-3, 1e-6),              # measured (0, 0, 0)
    "pareto_partial": (1e-6, 1e-3, 1e-6),    # measured (0, 0, 0)
    "progressive": (1e-6, 1e-3, 1e-6),       # measured (0, 0, 0)
    "random": (1e-6, 1e-3, 1e-6),            # measured (0, 0, 0)
}

def metric_key(candidate):
    """Comparable objective tuple for a candidate, rounded for float safety."""
    return (
        round(candidate.total_cost, 6),
        round(candidate.total_latency, 6),
        round(candidate.total_reliability, 6),
    )


# --- Reference anchor: the exact frontier itself is pinned ----------------------


def test_exhaustive_frontier_matches_recorded_reference(
    exact_constrained, exact_unconstrained
):
    """If the exact frontier drifts, everything built on this file drifts too."""
    assert len(exact_constrained) > 0 and len(exact_unconstrained) > 0
    assert {metric_key(c) for c in exact_constrained} == EXACT_CONSTRAINED_METRICS
    assert {
        metric_key(c) for c in exact_unconstrained
    } == EXACT_UNCONSTRAINED_METRICS
    for c in exact_constrained:
        assert c.total_cost <= CONSTRAINED_TASK.max_cost + 1e-9
        assert c.total_latency <= CONSTRAINED_TASK.max_latency + 1e-9


@pytest.fixture(name="exact_constrained")
def fixture_exact_constrained(registry):
    return ExhaustiveSynthesizer(registry).synthesize(CONSTRAINED_TASK)


@pytest.fixture(name="exact_unconstrained")
def fixture_exact_unconstrained(registry):
    return ExhaustiveSynthesizer(registry).synthesize(UNCONSTRAINED_TASK)


# --- Pareto quality gates against the exact reference ---------------------------


@pytest.mark.parametrize(
    "name",
    sorted(n for n, s in SYNTHESIZERS.items() if s["approximate"]),
)
def test_pareto_synthesizers_cover_exact_frontier(
    name, registry, exact_constrained
):
    candidates = synthesizer(name, registry).synthesize(
        CONSTRAINED_TASK, budget=SYNTHESIZERS[name]["budget"]
    )

    assert candidates
    assert frontier_coverage(candidates, exact_constrained) >= COVERAGE_FLOORS[name]

    cost_regret, latency_regret, reliability_regret = average_regret(
        candidates, exact_constrained
    )
    ceil_cost, ceil_latency, ceil_reliability = REGRET_CEILINGS[name]
    assert cost_regret <= ceil_cost
    assert latency_regret <= ceil_latency
    assert reliability_regret <= ceil_reliability

    # An approximate set must never outperform (dominate) the exact frontier.
    for candidate in candidates:
        for exact in exact_constrained:
            assert not dominates(candidate, exact)


# Same gate on the unconstrained task: a floor the constrained frontier
# cannot imply (beam covers 4/6 there but only 6/13 here).

@pytest.mark.parametrize(
    "name",
    sorted(n for n, s in SYNTHESIZERS.items() if s["approximate"]),
)
def test_pareto_synthesizers_cover_unconstrained_frontier(
    name, registry, exact_unconstrained
):
    candidates = synthesizer(name, registry).synthesize(
        UNCONSTRAINED_TASK, budget=SYNTHESIZERS[name]["budget"]
    )

    assert candidates
    assert (
        frontier_coverage(candidates, exact_unconstrained)
        >= UNCONSTRAINED_COVERAGE_FLOORS[name]
    )


def test_progressive_pinned_stack_keeps_reliability_corner(registry):
    """Pins the headline-claim geometry at the pinned default base width:
    the exact frontier's reliability corner must survive progressive beam
    synthesis (the former lexicographic-stage pruning dropped it)."""
    candidates = ProgressiveParetoBeamSynthesizer(registry).synthesize(
        CONSTRAINED_TASK
    )
    metrics = {metric_key(c) for c in candidates}
    assert (0.009, 540.0, 0.778688) in metrics


@pytest.mark.parametrize(
    "name", sorted(n for n in SYNTHESIZERS if n != "heuristic")
)
def test_synthesizers_never_violate_hard_constraints(name, registry):
    """No synthesizer may return an architecture outside the feasible region."""
    for candidate in synthesizer(name, registry).synthesize(
        CONSTRAINED_TASK, budget=SYNTHESIZERS[name]["budget"]
    ):
        assert candidate.total_cost <= CONSTRAINED_TASK.max_cost + 1e-9
        assert candidate.total_latency <= CONSTRAINED_TASK.max_latency + 1e-9


def test_heuristic_greedy_candidate_is_on_exact_frontier(
    exact_unconstrained, registry
):
    """The greedy synthesizer returns its score-optimal chain, which happens
    to be a point of the exact frontier on this task."""
    candidates = HeuristicSynthesizer(registry).synthesize(UNCONSTRAINED_TASK)
    assert len(candidates) == 1
    for exact in exact_unconstrained:
        assert not dominates(candidates[0], exact)


# --- Determinism ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SYNTHESIZERS))
def test_synthesizer_output_is_deterministic(name, registry):
    # Two independent synthesize() calls carrying the same descriptor
    # budget (fresh instances of the same seed) must produce identical
    # metric sets.
    budget = SYNTHESIZERS[name]["budget"]
    if budget is None:
        budget_b = None
    else:
        # Fresh instance of the same budget spec — same seed, same cap.
        budget_b = SynthesisBudget(
            max_evaluations=budget.max_evaluations,
            random_seed=budget.random_seed,
        )
    first = synthesizer(name, registry).synthesize(
        UNCONSTRAINED_TASK, budget=budget
    )
    second = synthesizer(name, registry).synthesize(
        UNCONSTRAINED_TASK, budget=budget_b
    )
    assert sorted(map(metric_key, first)) == sorted(map(metric_key, second))


# --- Budget semantics -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SYNTHESIZERS))
def test_evaluation_budget_caps_returned_candidates(name, registry):
    assert synthesizer(name, registry).synthesize(
        UNCONSTRAINED_TASK, budget=SynthesisBudget(max_evaluations=0)
    ) == []
    assert len(
        synthesizer(name, registry).synthesize(
            UNCONSTRAINED_TASK, budget=SynthesisBudget(max_evaluations=7)
        )
    ) <= 7


def _engine_chain_count(registry, task):
    """Chains the chain-family synthesizers materialize for ``task`` when
    unconstrained: one per combination of stage candidates (finalize
    registers exactly one evaluation per chain)."""
    count = 1
    for capability in task.required_capabilities:
        count *= len(
            registry.get_by_capability(
                capability=capability,
                privacy_constraint=task.privacy_constraint,
            )
        )
    return count


@pytest.mark.parametrize("name", CHAIN_FAMILY)
def test_budget_cap_at_full_work_returns_the_complete_frontier(name, registry):
    """A cap at exactly the search's full-work evaluation count must return
    the identical frontier the no-budget search returns — the cap at
    exhaustion stops nothing. (Only the chain-family synthesizers materialize
    a deterministic per-combination evaluation count; the seeded sampler's
    mid-search behavior is pinned by the next test instead.)"""
    count = _engine_chain_count(registry, UNCONSTRAINED_TASK)
    full = synthesizer(name, registry).synthesize(UNCONSTRAINED_TASK)
    at_cap = synthesizer(name, registry).synthesize(
        UNCONSTRAINED_TASK, budget=SynthesisBudget(max_evaluations=count)
    )
    assert sorted(map(metric_key, at_cap)) == sorted(map(metric_key, full))


@pytest.mark.parametrize(
    "name", sorted(n for n in SYNTHESIZERS if n != "heuristic")
)
def test_small_budget_cap_provably_stops_the_search_early(name, registry):
    """Cap = 7 is smaller than every synthesizer's completed-search frontier
    on this task (13/8/12/13/13 measured), so returning at most 7 distinct
    candidates proves the search was cut before completion — fewer
    evaluations than a full run would consume."""
    random_budget = SYNTHESIZERS["random"]["budget"]
    budget = (
        SynthesisBudget(
            max_evaluations=7,
            random_seed=random_budget.random_seed,
        )
        if name == "random"
        else SynthesisBudget(max_evaluations=7)
    )
    full_sizes = {
        "exhaustive": 13,
        "beam": 8,
        "pareto_partial": 12,
        "progressive": 13,
        "random": 13,
    }
    capped = synthesizer(name, registry).synthesize(
        UNCONSTRAINED_TASK, budget=budget
    )
    assert len(capped) <= 7
    assert len(capped) < full_sizes[name]


# --- Structural invariants --------------------------------------------------------


@pytest.mark.parametrize(
    "name, budget",
    [
        (n, SYNTHESIZERS[n]["budget"])
        for n in ("exhaustive", "beam", "pareto_partial", "progressive", "random")
    ],
    ids=["exhaustive", "beam", "pareto_partial", "progressive", "random_seeded"],
)
def test_architectures_are_stage_chains(name, budget, registry):
    """Every architecture is a linear chain: one node per required capability,
    linked in capability order. The consolidation refactor must not change
    this graph shape or the node-id scheme."""
    n_stages = len(CONSTRAINED_TASK.required_capabilities)
    for candidate in synthesizer(name, registry).synthesize(
        CONSTRAINED_TASK, budget=budget
    ):
        graph = candidate.graph
        assert graph.validate()
        node_ids = graph.list_nodes()
        assert len(node_ids) == n_stages
        assert len(set(node_ids)) == n_stages
        execution_order = graph.get_execution_order()
        assert len(execution_order) == n_stages
        # Chain wiring: exactly one edge between each consecutive stage.
        edges = set(graph.edges())
        chain_edges = {
            (execution_order[i], execution_order[i + 1])
            for i in range(n_stages - 1)
        }
        assert edges == chain_edges


# --- Documented divergence: infeasible tasks ---------------------------------------


@pytest.mark.parametrize("name", sorted(SYNTHESIZERS))
def test_infeasible_tasks_return_empty_or_raise(name, registry):
    """The beam-family, random, and exhaustive synthesizers return [] for
    infeasible tasks; the heuristic raises ``SynthesisError`` instead.
    Both behaviors are current and pinned until a refactor unifies them."""
    if name == "heuristic":
        with pytest.raises(SynthesisError):
            synthesizer(name, registry).synthesize(INFEASIBLE_TASK)
        return
    assert synthesizer(name, registry).synthesize(INFEASIBLE_TASK) == []


MISSING_CAPABILITY_TASK = TaskSpecification(
    required_capabilities=["nonexistent_capability"]
)


@pytest.mark.parametrize("name", sorted(SYNTHESIZERS))
def test_missing_capability_returns_empty_or_raise(name, registry):
    """Unknown required capability: beams/exhaustive/random return []; the
    heuristic raises ``SynthesisError`` naming the capability. Same
    divergence shape as the infeasible-task gate above; re-homed from
    ``tests/test_synthesizers.py``, which accepted either outcome."""
    if name == "heuristic":
        with pytest.raises(SynthesisError):
            synthesizer(name, registry).synthesize(MISSING_CAPABILITY_TASK)
        return
    assert synthesizer(name, registry).synthesize(MISSING_CAPABILITY_TASK) == []
