"""Behavioral tests for chatcortex.loop.run_closed_loop.

The loop is only as trustworthy as its glue: these tests recompute the
expected calibration arithmetic independently of the loop and assert the
observable contract — priors move toward empirical values, the frontier
shifts as metadata changes, the run is deterministic, and mis- configured
strategies fail loudly instead of silently learning nothing.
"""

import json

import pytest

from chatcortex.execution.executor import AgentExecutor
from chatcortex.graph.node_ids import component_name_from_node_id
from chatcortex.loop import LoopIteration, run_closed_loop
from chatcortex.registry.demo_components import build_demo_registry
from chatcortex.synthesis.beam_synthesizer import BeamSynthesizer
from chatcortex.synthesis.task_specification import TaskSpecification

ALPHA = 0.3


def _demo_setup():
    capabilities = ("retrieval", "generation", "verification")
    registry = build_demo_registry(capabilities=capabilities)
    task = TaskSpecification(
        required_capabilities=list(capabilities), max_latency=1500.0
    )
    return registry, task


def _empirical_stats(traces):
    """Empirical success rate per component name, from raw records."""
    buckets = {}
    for trace in traces:
        name = component_name_from_node_id(trace["component"])
        seen = buckets.setdefault(name, {"trials": 0, "successes": 0})
        seen["trials"] += 1
        seen["successes"] += 1 if trace["success"] else 0
    return {
        name: stats["successes"] / stats["trials"]
        for name, stats in buckets.items()
    }


def _expected_reliability(alpha, traces, prior_registry):
    """Blend the loop should produce, computed independently of it."""
    stats = _empirical_stats(traces)
    expected = {}
    for name, rate in stats.items():
        prior_score = prior_registry.get(name).reliability_score
        expected[name] = alpha * rate + (1 - alpha) * prior_score
    return expected


def test_priors_move_toward_empirical_values():
    """Calibrated reliability equals the independently recomputed blend,
    priors compound across iterations, and the loop reports which
    components moved."""
    registry, task = _demo_setup()
    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=BeamSynthesizer,
        executor=AgentExecutor(mode="probabilistic", seed=42),
        iterations=3,
        alpha=ALPHA,
    )
    assert [r.iteration for r in records] == [0, 1, 2]

    prior = registry
    for record in records:
        expected = _expected_reliability(ALPHA, record.traces, prior)
        assert expected, "simulated runs must produce traces"
        for name, expected_score in expected.items():
            calibrated_score = record.registry.get(name).reliability_score
            assert calibrated_score == pytest.approx(expected_score, abs=1e-12)
        prior = record.registry

    # Priors actually moved: the final registry differs from the declared
    # one for at least one component, and the loop named the movers.
    movers = {name for record in records for name in record.updated_components}
    assert movers
    for name in movers:
        assert (
            records[-1].registry.get(name).reliability_score
            != registry.get(name).reliability_score
        )


def test_frontier_shifts_as_metadata_changes():
    """Changed reliability priors change the synthesized frontier's
    objective rows (same seed, alpha=1.0 pushes calibration hard)."""
    registry, task = _demo_setup()
    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=BeamSynthesizer,
        executor=AgentExecutor(mode="probabilistic", seed=42),
        iterations=3,
        alpha=1.0,
    )
    first_rows = [tuple(sorted(r.items())) for r in records[0].frontier_objectives]
    last_rows = [tuple(sorted(r.items())) for r in records[-1].frontier_objectives]
    assert first_rows != last_rows


def test_two_runs_are_identical_and_serializable():
    """The loop is deterministic under fixed seeds and fully JSON-safe."""
    runs = []
    for _ in range(2):
        registry, task = _demo_setup()
        records = run_closed_loop(
            registry=registry,
            task=task,
            synthesizer=BeamSynthesizer,
            executor=AgentExecutor(mode="probabilistic", seed=42),
            iterations=3,
            alpha=ALPHA,
        )
        runs.append([record.to_dict() for record in records])
    assert runs[0] == runs[1]

    # And every record survives a JSON round-trip.
    payload = json.dumps(runs[0])
    reparsed = json.loads(payload)
    assert reparsed[0]["registry_snapshot"]["components"]


def test_candidates_per_iteration_limits_executions():
    """candidates_per_iteration executes objective-sorted candidates only."""
    registry, task = _demo_setup()
    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=BeamSynthesizer,
        executor=AgentExecutor(mode="probabilistic", seed=42),
        iterations=2,
        alpha=ALPHA,
        candidates_per_iteration=1,
    )
    for record in records:
        assert record.frontier_size >= 1
        assert record.executed_candidates == 1
        assert record.trace_count <= 3  # one chain of at most 3 steps


def test_prebuilt_synthesizer_instance_is_rejected():
    """A pre-built instance would bind to stale metadata: fail loudly."""
    registry, task = _demo_setup()
    with pytest.raises(TypeError, match="pre-built instance"):
        run_closed_loop(
            registry=registry,
            task=task,
            synthesizer=BeamSynthesizer(registry),
            executor=AgentExecutor(mode="probabilistic", seed=42),
        )


def test_factory_synthesizer_is_supported():
    """Custom configurations flow through a factory(registry) callable."""
    registry, task = _demo_setup()
    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=lambda reg: BeamSynthesizer(reg, beam_width=5),
        executor=AgentExecutor(mode="probabilistic", seed=42),
        iterations=1,
    )
    assert records[0].frontier_size >= 1


def test_iterations_must_be_positive():
    registry, task = _demo_setup()
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        run_closed_loop(
            registry=registry,
            task=task,
            synthesizer=BeamSynthesizer,
            executor=AgentExecutor(mode="probabilistic", seed=42),
            iterations=0,
        )


def test_loop_iteration_type_is_exported():
    _, task = _demo_setup()
    records = run_closed_loop(
        registry=build_demo_registry(),
        task=TaskSpecification(
            required_capabilities=["retrieval", "generation", "verification"]
        ),
        synthesizer=BeamSynthesizer,
        executor=AgentExecutor(mode="probabilistic", seed=42),
        iterations=1,
    )
    assert isinstance(records[0], LoopIteration)
