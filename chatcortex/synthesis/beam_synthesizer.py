from typing import List, Tuple

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisBudget, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification


class BeamSynthesizer(Synthesizer):
    """
    Budget-aware beam search synthesizer

    Maintains top-k partial architectures per stage
    """

    def __init__(self, registry, beam_width: int = 3):
        super().__init__(registry)
        self.beam_width = beam_width

    def _score(self, meta: ComponentMetadata, weights: dict) -> float:
        return (
            weights["cost"] * meta.cost_per_call
            + weights["latency"] * meta.avg_latency_ms
            - weights["error"] * meta.reliability_score
        )
    
    def synthesize(
        self, 
        task: TaskSpecification, 
        budget: SynthesisBudget = None
    ) -> List[ArchitectureCandidate]:

        task.validate()

        context = SynthesisContext(budget)
        
        # Each beam element = (graph, cumulative_score)
        beam: List[Tuple[AgentGraph, float]] = [(AgentGraph(), 0.0)]

        for stage_idx, capability in enumerate(task.required_capabilities):

            new_beam: List[Tuple[AgentGraph, float]] = []

            for graph, cumulative_score in beam:

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
                    
                    score_increment = self._score(
                        component, task.objective_weights
                    )

                    new_beam.append(
                        (new_graph, cumulative_score + score_increment)
                    )
            
            # Keep top-k
            new_beam.sort(key=lambda x: x[1])
            if stage_idx < len(task.required_capabilities) - 1:
                beam = new_beam[: self.beam_width]
            else:
                beam = new_beam # Keep all final candidates
        
        beam.sort(key=lambda x: x[1])
        pareto_set = ParetoSet()

        for graph, _ in beam:

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

            pareto_set.add(candidate=candidate)
        
        return list(pareto_set)