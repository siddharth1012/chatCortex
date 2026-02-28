from typing import List, Optional
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisBudget, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification
from chatcortex.graph.agent_graph import AgentGraph


class SynthesisError(Exception):
    pass


class HeuristicSynthesizer(Synthesizer):
    """
    Deterministic greedy synthesizer
    
    v0.3.0:
        - Budget-aware
        - Returns ArchitectureCandidate
        - Conforms to unified synthesis contract

    Produces exactly one architecture (if feasible)
    """

    def _score(self, meta: ComponentMetadata, weights: dict) -> float:
        """
        Weighted scoring function - Lower is better
        """

        return (
            weights["cost"] * meta.cost_per_call
            + weights["latency"] * meta.avg_latency_ms
            - weights["error"] * meta.reliability_score
        )
    
    def synthesize(
        self, 
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:
        
        task.validate()

        context = SynthesisContext(budget)

        graph = AgentGraph()
        previous_node: Optional[str] = None

        for idx, capability in enumerate(task.required_capabilities):

            candidates = self.registry.get_by_capability(
                capability=capability,
                privacy_constraint=task.privacy_constraint
            )

            if not candidates:
                raise SynthesisError(
                    f"No Components available for capability '{capability}' under given constraints"
                )
            
            candidates_sorted = sorted(
                candidates,
                key=lambda candidate: self._score(candidate, task.objective_weights)
            )

            selected_candidate = candidates_sorted[0]
            node_id = f"{selected_candidate.name}_{idx}"
            
            graph.add_component(node_id, selected_candidate)

            if previous_node:
                graph.add_edge(previous_node, node_id)
            
            previous_node = node_id
        
        try:
            context.register_evaluation()
        except BudgetExceeded:
            return []
        
        # Compute metrics

        total_cost = graph.total_cost()
        total_latency = graph.total_latency()
        total_reliability = graph.aggregate_reliability()

        # Hard constraints

        if task.max_cost is not None:
            if graph.total_cost() > task.max_cost:
                raise SynthesisError(
                    f"Constructed agent exceeds max_cost constraint"
                )
        
        if task.max_latency is not None:
            if graph.total_latency() > task.max_latency:
                raise SynthesisError(
                    f"Constructed agent exceeds max_latency constraint"
                )
        
        candidate = ArchitectureCandidate(
            graph=graph,
            total_cost=total_cost,
            total_latency=total_latency,
            total_reliability=total_reliability,
        )
        
        return [candidate]
        