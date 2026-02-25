from dataclasses import dataclass

from chatcortex.graph.agent_graph import AgentGraph


@dataclass(frozen=True)
class ArchitectureCandidate:
    """
    Represents a fully constructed agent architecture
    with cached deterministic metrics for optimization
    """

    graph: AgentGraph
    total_cost: float
    total_latency: float
    total_reliability: float

    def metrics(self):
        return {
            "cost": self.total_cost,
            "latency": self.total_latency,
            "reliability": self.total_reliability,
        }