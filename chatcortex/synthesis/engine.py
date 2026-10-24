"""The one chain-search engine behind the beam-family synthesizers.

Every chain synthesizer is the same three-phase search:

1. Stage expansion  : ``expand_stage`` copies each beam graph and attaches
   one registry component per stage (shared node-id scheme and chain
   wiring), optionally advancing a cumulative score. Budget is enforced
   mid-search here, so an exhausted wall-time budget stops runaway
   expansion instead of being noticed only at the very end.
2. Stage truncation : a per-synthesizer policy callable
   ``(expanded, stage_idx, keep_all) -> beam`` decides which partials
   survive. ``keep_all`` is True on the final stage, where a policy
   receives every expansion and normally keeps all of them; deliberately
   single-output strategies (the greedy synthesizer) may keep fewer.
3. Materialization  : ``finalize`` performs budget accounting, hard
   constraint checking, and final Pareto maintenance.

The public synthesizers (Beam, ParetoPartial, Progressive, Exhaustive,
Heuristic) are thin configurations of this engine and keep their public
names, constructors, and observable semantics.
"""

from typing import Callable, List, Optional, Tuple

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.graph.node_ids import node_id_for
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification

# A beam element: (partial graph, cumulative score).
BeamElement = Tuple[AgentGraph, float]

# Truncation policy: (expanded beam, stage index, keep_all) -> beam.
TruncationPolicy = Callable[[List[BeamElement], int, bool], List[BeamElement]]


def graph_to_candidate(graph: AgentGraph) -> ArchitectureCandidate:
    """Materialize a partial or complete graph into a metric candidate."""
    return ArchitectureCandidate(
        graph=graph,
        total_cost=graph.total_cost(),
        total_latency=graph.total_latency(),
        total_reliability=graph.aggregate_reliability(),
    )


def objective_key(candidate: ArchitectureCandidate) -> tuple:
    """Deterministic comparable key; reliability negated so 'better' sorts first."""
    return (
        candidate.total_cost,
        candidate.total_latency,
        -candidate.total_reliability,
    )


def crowding_select(candidates: List[ArchitectureCandidate], width: int) -> List[ArchitectureCandidate]:
    """Pareto-aware width selection in objective space.

    The extreme points of the "better" direction of each objective (min
    cost, min latency, max reliability) are always kept when the width
    allows, then remaining slots are filled by descending crowding
    distance. All tie-breaks use the objective key, so selection depends
    only on metric values, never on iteration order of the input."""
    n = len(candidates)
    distances = [0.0] * n

    for value_of in (
        lambda c: c.total_cost,
        lambda c: c.total_latency,
        lambda c: -c.total_reliability,
    ):
        order = sorted(range(n), key=lambda i: (value_of(candidates[i]), i))
        values = [value_of(candidates[i]) for i in order]
        span = values[-1] - values[0]
        if span <= 0:
            continue  # all partials equal on this objective
        for rank in range(1, n - 1):
            i = order[rank]
            distances[i] += (values[rank + 1] - values[rank - 1]) / span

    # Protected extremes guarantee the corners of the front survive.
    corners = [
        min(range(n), key=lambda i: candidates[i].total_cost),
        min(range(n), key=lambda i: candidates[i].total_latency),
        max(range(n), key=lambda i: candidates[i].total_reliability),
    ]
    corner_set = set(corners)
    rest = [i for i in range(n) if i not in corner_set]
    fill = sorted(rest, key=lambda i: (-distances[i], objective_key(candidates[i])))
    ranked = sorted(corners, key=lambda i: objective_key(candidates[i])) + fill
    return [candidates[i] for i in ranked[:width]]


def crowding_truncation(
    width_for_stage: Callable[[int], int],
) -> Callable[[List[BeamElement], int, bool], List[BeamElement]]:
    """Pareto-aware stage truncation policy over (cost, latency, reliability).

    Chosen over an epsilon-grid because it needs no scale parameters and
    is deterministic via lexicographic tie-breaks; the crowding extremes
    preserve spread across the full front, including the high-reliability
    corner that the former lexicographic sort dropped."""

    def truncate(expanded, stage_idx, keep_all):
        front = ParetoSet()
        for graph, _ in expanded:
            front.add(graph_to_candidate(graph))

        partials = list(front)

        width = width_for_stage(stage_idx)
        if not keep_all and len(partials) > width:
            partials = crowding_select(partials, width)

        return [(candidate.graph, 0.0) for candidate in partials]

    return truncate


def score_component(meta: ComponentMetadata, weights: dict) -> float:
    """Weighted scalar score (lower is better) for one component."""
    return (
        weights["cost"] * meta.cost_per_call
        + weights["latency"] * meta.avg_latency_ms
        - weights["error"] * meta.reliability_score
    )


class ChainSearchEngine:
    """Runs the shared expansion -> truncation -> materialization search."""

    def __init__(self, registry):
        self.registry = registry

    def run(
        self,
        task: TaskSpecification,
        budget=None,
        scorer: Optional[Callable[[ComponentMetadata, dict], float]] = None,
        truncation: Optional[TruncationPolicy] = None,
        on_violation: Optional[Callable[[str], None]] = None,
    ) -> List[ArchitectureCandidate]:
        """Run the full search for one task.

        ``scorer`` advances the beam's cumulative scores (score-ordered
        synthesizers); ``truncation`` prunes intermediate stages; and
        ``on_violation`` is invoked with the violated constraint name when
        a caller wants hard-constraint violations to raise instead of being
        silently skipped.
        """
        task.validate()
        context = SynthesisContext(budget)

        beam: List[BeamElement] = [(AgentGraph(), 0.0)]
        n_stages = len(task.required_capabilities)

        for stage_idx in range(n_stages):
            expanded = self.expand_stage(beam, stage_idx, task, context, scorer)

            if truncation is not None:
                beam = truncation(
                    expanded, stage_idx, keep_all=stage_idx == n_stages - 1
                )
            else:
                beam = list(expanded)

        return self.finalize([graph for graph, _ in beam], task, context, on_violation)

    def expand_stage(
        self,
        beam: List[BeamElement],
        stage_idx: int,
        task: TaskSpecification,
        context: SynthesisContext,
        scorer: Optional[Callable[[ComponentMetadata, dict], float]] = None,
    ) -> List[BeamElement]:
        """Expand every beam graph with every component available for this
        stage's capability.

        Returns ``[(graph, cumulative_score)]`` in beam order; scores only
        advance when ``scorer`` is supplied. When the budget is exhausted,
        expansion stops early and returns the candidates built so far.
        """
        expanded: List[BeamElement] = []
        capability = task.required_capabilities[stage_idx]
        components = self.registry.get_by_capability(
            capability=capability,
            privacy_constraint=task.privacy_constraint,
        )

        for graph, cumulative_score in beam:
            if not components:
                continue

            for component in components:
                if not context.can_evaluate():
                    return expanded  # budget (count or wall clock) exhausted

                new_graph = graph.copy()
                node_id = node_id_for(component.name, stage_idx)
                new_graph.chain_append(node_id, component)

                increment = scorer(component, task.objective_weights) if scorer else 0.0
                expanded.append((new_graph, cumulative_score + increment))

        return expanded

    def finalize(
        self,
        beam_graphs: List[AgentGraph],
        task: TaskSpecification,
        context: SynthesisContext,
        on_violation: Optional[Callable[[str], None]] = None,
    ) -> List[ArchitectureCandidate]:
        """Turn surviving chains into the final Pareto frontier.

        Registers one evaluation per chain (budget cuts mid-materialization
        simply stop early), checks hard constraints, and maintains the
        final non-dominated set.
        """
        pareto_set = ParetoSet()

        for graph in beam_graphs:
            try:
                context.register_evaluation()
            except BudgetExceeded:
                break

            candidate = graph_to_candidate(graph)

            if task.max_cost is not None and candidate.total_cost > task.max_cost:
                if on_violation is None:
                    continue
                on_violation("max_cost")

            if task.max_latency is not None and candidate.total_latency > task.max_latency:
                if on_violation is None:
                    continue
                on_violation("max_latency")

            pareto_set.add(candidate)

        return list(pareto_set)
