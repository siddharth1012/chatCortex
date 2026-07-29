from itertools import product
from typing import List, Optional

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisBudget, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification


class ExhaustiveSynthesizer(Synthesizer):
    """
    Generate all feasible architectures via Cartesian product
    of candidate components per capability

    v0.3.0:
        - Budget-aware
        - Incremental Pareto maintenance
        - Memory-efficient (no full cartesian materialization)
    """

    def synthesize(
        self,
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:

        task.validate()

        context = SynthesisContext(budget)
        pareto_set = ParetoSet()

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

        # Step 2: Cartesian Product (lazy iteration)
        for combination in product(*candidate_lists):

            graph = AgentGraph()
            previous_node = None

            for idx, component in enumerate(combination):
                node_id = f"{component.name}_{idx}"
                graph.add_component(node_id, component)

                if previous_node:
                    graph.add_edge(previous_node, node_id)

                previous_node = node_id

            try:
                context.register_evaluation()
            except BudgetExceeded:
                break

            # Hard constraints check
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

            # Incremental Pareto insertion
            pareto_set.add(candidate)

        return list(pareto_set)
