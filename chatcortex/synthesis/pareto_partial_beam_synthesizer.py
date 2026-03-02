from typing import List, Tuple

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisBudget, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification


class ParetoPartialBeamSynthesizer(Synthesizer):
    """
    Beam search that performs multi-objective Pareto pruning
    at each intermediate stage.
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
    
    def _diversity_truncate(
        self, candidates: List[ArchitectureCandidate]
    ) -> List[ArchitectureCandidate]:
        """
        Simple diversity truncation:
        sort by cost ascending and select evenly spaced elements.
        """

        candidates.sort(key=lambda c: c.total_cost)

        if len(candidates) <= self.beam_width:
            return candidates
        
        step = len(candidates) / self.beam_width
        selected = []

        for i in range(self.beam_width):
            idx = int(i * step)
            selected.append(candidates[idx])

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
                if len(partial_candidates) > self.beam_width:
                    partial_candidates = self._diversity_truncate(partial_candidates)
                
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