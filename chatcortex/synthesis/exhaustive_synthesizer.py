from itertools import product
from typing import List

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.synthesis.task_specification import TaskSpecification


class ExhaustiveSynthesizer:
    """
    Generate all feasible architectures via Cartesian product
    of candidate components per capability
    """

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
    
    def synthesize(self, task: TaskSpecification) -> List[ArchitectureCandidate]:
        task.validate()

        # Step 1: Collect candidates per capability
        candidate_lists = []

        for capability in task.required_capabilities:
            candidates = self.registry.get_by_capability(
                capability=capability,
                privacy_constraint=task.privacy_constraint,
            )

            if not candidates:
                return [] 

            candidate_lists.append(candidates)
        
        # Step 2: Cartesian Product
        combinations = list(product(*candidate_lists))

        architectures = []

        for combination in combinations:
            graph = AgentGraph()

            previous_node = None

            for idx, component in enumerate(combination):
                node_id = f"{component.name}_{idx}"
                graph.add_component(node_id, component)

                if previous_node:
                    graph.add_edge(previous_node, node_id)
                
                previous_node = node_id
            
            # Hard constraints check
            total_cost = graph.total_cost()
            total_latency = graph.total_latency()
            total_reliability = graph.aggregate_reliability()

            if task.max_cost is not None and total_cost > task.max_cost:
                continue

            if task.max_latency is not None and total_latency > task.max_latency:
                continue

            architectures.append(
                ArchitectureCandidate(
                    graph=graph,
                    total_cost=total_cost,
                    total_latency=total_latency,
                    total_reliability=total_reliability,
                )
            )
        
        return architectures