from typing import List

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisBudget, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification


class ParetoPartialBeamSynthesizerV3(Synthesizer):
    """
    Multi-objective aware beam search with improved diversity truncation

    Strategy:
        - Maintain Pareto set of partial graphs at each stage
        - If size > beam_width, apply multi-dimensional diversity selection
          using extreme-point preservation + crowding-distance selection     
    """

    def __init__(self, registry, beam_width: int = 3):
        super().__init__(registry)
        self.beam_width = beam_width
    
    def _to_candidate(self, graph: AgentGraph) -> ArchitectureCandidate:
        return ArchitectureCandidate(
            graph=graph,
            total_cost=graph.total_cost(),
            total_latency=graph.total_latency(),
            total_reliability=graph.aggregate_reliability(),
        )
    
    def _crowding_distance(self, candidates: List[ArchitectureCandidate]):
        """
        Compute crowding distance similar to NSGA-2
        Cost and latency minimized, reliability maximized
        """
        n = len(candidates)
        if n == 0:
            return []
        
        distances = [0.0] * n

        # Normalize and sort per objective
        objectives = [
            (lambda c: c.total_cost, False), # minimize
            (lambda c: c.total_latency, False), # minimize
            (lambda c: c.total_reliability, True), # maximize
        ]

        for objective_function, maximize in objectives:
            indexed = list(enumerate(candidates))
            indexed.sort(key=lambda x: objective_function(x[1]), reverse=maximize)

            distances[indexed[0][0]] = float("inf")
            distances[indexed[-1][0]] = float("inf")

            values = [objective_function(c) for _, c in indexed]
            min_val = min(values)
            max_val = max(values)
            denom = max_val - min_val if max_val != min_val else 1.0

            for i in range(1, n-1):
                prev_val = values[i - 1]
                next_val = values[i + 1]
                distances[indexed[i][0]] += (next_val - prev_val) / denom

        return distances

    def _diversity_truncate_v3(self, candidates):

        if len(candidates) <= self.beam_width:
            return candidates

        # Step 1: preserve extremes
        min_cost = min(candidates, key=lambda c: c.total_cost)
        min_latency = min(candidates, key=lambda c: c.total_latency)
        max_reliability = max(candidates, key=lambda c: c.total_reliability)

        selected = [min_cost, min_latency, max_reliability]
        selected = list({id(c): c for c in selected}.values())

        remaining = [c for c in candidates if c not in selected]

        if len(selected) >= self.beam_width:
            return selected[: self.beam_width]

        # Step 2: sort remaining by cost
        remaining.sort(key=lambda c: c.total_cost)

        # Step 3: uniform sampling across cost axis
        m = self.beam_width - len(selected)
        n = len(remaining)

        if n > 0:
            for i in range(m):
                idx = int(i * (n - 1) / max(m - 1, 1))
                candidate = remaining[idx]

                # Optional reliability diversity guard
                if all(
                    abs(candidate.total_reliability - s.total_reliability) > 1e-6
                    for s in selected
                ):
                    selected.append(candidate)

                if len(selected) >= self.beam_width:
                    break

        return selected

    def synthesize(
        self, 
        task: TaskSpecification, 
        budget: SynthesisBudget = None
    ) -> List[ArchitectureCandidate]:

        task.validate()

        context = SynthesisContext(budget)
        
        # Start with empty graph
        beam_graphs: List[AgentGraph] = [AgentGraph()]

        for stage_idx, capability in enumerate(task.required_capabilities):

            expanded_graphs: List[AgentGraph] = []

            for graph in beam_graphs:

                candidates = self.registry.get_by_capability(
                    capability=capability,
                    privacy_constraint=task.privacy_constraint,
                )

                if not candidates:
                    continue

                for component in candidates:
                    
                    new_graph = graph.copy()

                    node_id = f"{component.name}_{stage_idx}"
                    new_graph.add_component(node_id, component)

                    if stage_idx > 0:
                        prev_nodes = list(new_graph._graph.nodes)
                        if len(prev_nodes) > 1:
                            new_graph.add_edge(prev_nodes[-2], prev_nodes[-1])
                    
                    expanded_graphs.append(new_graph)
            
            # Convert expanded graphs to partial candidates
            pareto = ParetoSet()

            for graph in expanded_graphs:
                candidate = self._to_candidate(graph)
                pareto.add(candidate)
            
            partial_candidates = list(pareto)

            # If not final stage, apply beam width constraint

            if stage_idx < len(task.required_capabilities) - 1:
                partial_candidates = self._diversity_truncate_v3(partial_candidates)
                beam_graphs = [c.graph for c in partial_candidates]
            else:
                # Final stage - keep all
                beam_graphs = [c.graph for c in partial_candidates]
        
        final_pareto_set = ParetoSet()

        for graph in beam_graphs:

            try:
                context.register_evaluation()
            except BudgetExceeded:
                break

            total_cost = graph.total_cost()
            total_latency = graph.total_latency()
            total_reliability = graph.aggregate_reliability()

            if task.max_cost is not None and total_cost > task.max_cost:
                continue

            if task.max_latency is not None and total_latency > task.max_latency:
                continue

            candidate = ArchitectureCandidate(
                graph=graph,
                total_cost=total_cost,
                total_latency=total_latency,
                total_reliability=total_reliability,
            )        

            final_pareto_set.add(candidate=candidate)
        
        return list(final_pareto_set)