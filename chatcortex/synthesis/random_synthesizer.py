import random
from typing import List

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import BudgetExceeded, SynthesisBudget, SynthesisContext
from chatcortex.synthesis.task_specification import TaskSpecification


class RandomSynthesizer(Synthesizer):
    """
    Random architecture sampling synthesizer

    v0.3.0 baseline strategy
        - Uniformly samples architectures under evaluation budget
        - maintains incremental Pareto frontier
    """

    def synthesize(
        self, 
        task: TaskSpecification, 
        budget: SynthesisBudget = None
    ) -> List[ArchitectureCandidate]:

        task.validate()

        context = SynthesisContext(budget)
        pareto_set = ParetoSet()

        random_number_generation = random.Random(budget.random_seed if budget else None)

        candidates_list = []

        for capability in task.required_capabilities:
            candidates = self.registry.get_by_capability(
                capability=capability,
                privacy_constraint=task.privacy_constraint,
            )

            if not candidates:
                return []

            candidates_list.append(candidates)

        
        while True:

            graph = AgentGraph()
            previous_node = None

            # Randomly select one component per capability

            for idx, candidates in enumerate(candidates_list):
                component = random_number_generation.choice(candidates) 
                node_id = f"{component.name}_{idx}"
                graph.add_component(node_id, component)

                if previous_node:
                    graph.add_edge(previous_node, node_id)
                
                previous_node = node_id
            
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