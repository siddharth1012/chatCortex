from typing import Optional
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.task_specification import TaskSpecification
from chatcortex.graph.agent_graph import AgentGraph


class SynthesisError(Exception):
    pass


class HeuristicSynthesizer:
    """
    Phase 1 deterministic synthesizer
    
    Builds a linear agent graph by:
    - Matching ordered required capabilities
    - Applying hard constraints
    - Selecting lowest-scoring candidate per stage
    """

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def _score(self, meta: ComponentMetadata, weights: dict) -> float:
        """
        Weighted scoring function - Lower is better
        """

        return (
            weights["cost"] * meta.cost_per_call
            + weights["latency"] * meta.avg_latency_ms
            - weights["error"] * meta.reliability_score
        )
    
    def synthesize(self, task: TaskSpecification) -> AgentGraph:
        task.validate()

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
        
        # Hard constraint validation after graph construction
        # later can be pruned earlier for efficiency

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
        
        return graph
        