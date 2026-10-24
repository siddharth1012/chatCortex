"""The closed loop: synthesize -> execute -> calibrate -> re-synthesize.

:func:`run_closed_loop` glues three already-shipped mechanisms into one
iterative drive: a synthesizer maps the *current* registry to a Pareto
frontier, an executor runs the frontier's architectures and produces
telemetry records, and :func:`chatcortex.telemetry.calibration.calibrate`
blends the measured success/latency/cost back into the registry — so the
next iteration synthesizes against a registry that remembers what actually
happened, not only what was declared.

Determinism: given a deterministic-or-seeded synthesizer, a seeded (or
deterministic) executor, and seeded synthesis budgets, two calls with the
same inputs produce iteration-for-iteration identical records. The seeded
executor's RNG advances across iterations, which is still deterministic —
it is state, not scheduling.

Component mapping: telemetry records carry graph *node ids*;
``calibrate`` resolves them through
:func:`chatcortex.graph.node_ids.component_name_from_node_id` and silently
ignores names it cannot match, so traces from graphs built from the
previous registry still calibrate the components they were produced from.

Calibration feedback is *iterative*: each iteration blends only the
latest traces into the previous iteration's calibration, so priors
compound toward empirical values across iterations at roughly
``1 - (1 - alpha) ** k`` after ``k`` iterations.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from chatcortex.execution.executor import AgentExecutor
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.task_specification import TaskSpecification
from chatcortex.telemetry.calibration import calibrate


@dataclass
class LoopIteration:
    """Everything one loop iteration observed, kept and serializable.

    ``registry`` holds the *calibrated* registry this iteration produced
    (the next iteration's synthesis input and the value users typically
    keep); ``to_dict()`` embeds it as the loader-shaped
    ``registry_snapshot`` so a full loop can be checkpointed to JSON.
    ``traces`` are the raw telemetry records behind the calibration.
    """

    iteration: int
    frontier_size: int
    frontier_objectives: List[Dict[str, float]]
    executed_candidates: int
    trace_count: int
    updated_components: List[str]
    traces: List[Dict[str, Any]]
    registry: CapabilityRegistry
    prior_registry: CapabilityRegistry

    def to_dict(self) -> dict:
        """JSON-safe record including the calibrated registry snapshot."""
        return {
            "iteration": self.iteration,
            "frontier_size": self.frontier_size,
            "frontier_objectives": self.frontier_objectives,
            "executed_candidates": self.executed_candidates,
            "trace_count": self.trace_count,
            "updated_components": list(self.updated_components),
            "traces": [dict(trace) for trace in self.traces],
            "registry_snapshot": self.registry.to_dict(),
        }


def _make_synthesizer(
    synthesizer: Union[type, Callable[[CapabilityRegistry], Synthesizer]],
    registry: CapabilityRegistry,
) -> Synthesizer:
    """Instantiate the synthesis strategy against the *current* registry.

    Accepts a synthesizer *class* (constructed with its defaults) or a
    one-argument factory such as
    ``lambda reg: BeamSynthesizer(reg, beam_width=5)``. A pre-built
    instance is rejected: its constructor bound it to the previous
    registry, so reusing it would silently synthesize against stale
    metadata and the loop would learn nothing.
    """
    if isinstance(synthesizer, type) and issubclass(synthesizer, Synthesizer):
        return synthesizer(registry)
    if callable(synthesizer):
        return synthesizer(registry)
    raise TypeError(
        "synthesizer must be a Synthesizer subclass or a "
        f"factory(registry) callable, {synthesizer!r} given; passing a "
        "pre-built instance would bind the loop to stale metadata"
    )


def run_closed_loop(
    registry: CapabilityRegistry,
    task: TaskSpecification,
    synthesizer: Union[type, Callable[[CapabilityRegistry], Synthesizer]],
    executor: AgentExecutor,
    iterations: int = 3,
    alpha: float = 0.3,
    budget: Optional[SynthesisBudget] = None,
    candidates_per_iteration: Optional[int] = None,
) -> List[LoopIteration]:
    """Run the synthesize -> execute -> calibrate cycle ``iterations`` times.

    Args:
        registry: The declared (prior) registry; iteration 0 synthesizes
            against it unchanged.
        task: The synthesis problem, as for any synthesizer.
        synthesizer: A :class:`~chatcortex.synthesis.base.Synthesizer`
            subclass or ``factory(registry) -> Synthesizer`` callable. It
            is instantiated fresh against the *current* registry every
            iteration, because calibrate returns a new registry.
        executor: A shared :class:`AgentExecutor` instance (simulated or
            backend-backed); its seeded RNG advances across iterations,
            which keeps the whole run deterministic.
        iterations: How many loop rounds to run (>= 1).
        alpha: Blend weight passed to :func:`calibrate`; validated, so
            every round only executes under a valid blend. Non-finite
            values are rejected like any out-of-range value.
        candidates_per_iteration: Execute only the first ``N`` frontier
            members (objective-sorted, deterministic) instead of all of
            them; useful when executions are expensive. Must be ``>= 1``
            when given, and is validated before anything executes.
        budget: Optional per-iteration :class:`SynthesisBudget`; it is a
            frozen dataclass, so the same constraints apply every round
            without state leaking between iterations.
        candidates_per_iteration: Execute only the first ``N`` frontier
            members (objective-sorted, deterministic) instead of all of
            them; useful when executions are expensive.

    Returns:
        One :class:`LoopIteration` per round; iteration ``k``'s
        ``frontier_size``/``frontier_objectives`` describe the frontier
        synthesized from iteration ``k-1``'s calibrated registry, and
        ``registry`` is the registry calibrated *during* round ``k``.

    Raises:
        ValueError: If ``iterations < 1``, ``alpha`` is outside ``[0, 1]``,
            or ``candidates_per_iteration`` is given below 1 — all before
            any round runs — or if a round synthesizes an empty frontier
            (nothing to execute would stall the loop).
        TypeError: If ``synthesizer`` is neither a Synthesizer subclass
            nor a factory callable.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, {iterations} given")
    if candidates_per_iteration is not None and candidates_per_iteration < 1:
        raise ValueError(
            "candidates_per_iteration must be >= 1 when given, "
            f"{candidates_per_iteration} given"
        )
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], {alpha!r} given")

    records: List[LoopIteration] = []
    current = registry
    for iteration in range(iterations):
        engine = _make_synthesizer(synthesizer, current)
        frontier = engine.synthesize(task, budget)
        if not frontier:
            raise ValueError(
                f"iteration {iteration}: the synthesizer returned an "
                "empty frontier; the task may be infeasible under the "
                "current registry and constraints"
            )

        selected = frontier
        if candidates_per_iteration is not None:
            selected = sorted(
                frontier,
                key=lambda c: (c.total_cost, c.total_latency, -c.total_reliability),
            )[:candidates_per_iteration]

        traces: List[Dict[str, Any]] = []
        for candidate in selected:
            traces.extend(
                executor.execute(candidate.graph).telemetry.records
            )

        calibrated = calibrate(current, traces, alpha=alpha)
        updated = [
            prior.name
            for prior in current.list_all()
            if _differs(calibrated.get(prior.name), prior)
        ]

        records.append(
            LoopIteration(
                iteration=iteration,
                frontier_size=len(frontier),
                frontier_objectives=[
                    {
                        "cost": c.total_cost,
                        "latency": c.total_latency,
                        "reliability": c.total_reliability,
                    }
                    for c in frontier
                ],
                executed_candidates=len(selected),
                trace_count=len(traces),
                updated_components=updated,
                traces=[dict(trace) for trace in traces],
                registry=calibrated,
                prior_registry=current,
            )
        )
        current = calibrated
    return records


def _differs(calibrated, prior) -> bool:
    """True when calibration changed any performance field.

    Exact comparison is safe: ``blend == prior`` exactly when the
    empirical value equals the prior, with no rounding involved.
    """
    return (
        calibrated.reliability_score != prior.reliability_score
        or calibrated.avg_latency_ms != prior.avg_latency_ms
        or calibrated.cost_per_call != prior.cost_per_call
    )
