import networkx as nx
from typing import List
from chatcortex.registry.metadata import ComponentMetadata


class AgentGraph:
    """
    Directed Acyclic Graph (DAG) representing an agent architecture

    Nodes:
        ComponentMetadata instances
    Edges:
        Execution/Data flow order
    """

    def __init__(self):
        self._graph = nx.DiGraph()
    

    # Node management
    def add_component(self, node_id: str, metadata: ComponentMetadata) -> None:
        if node_id in self._graph:
            raise ValueError(f"Node '{node_id}' already exists in graph")
        self._graph.add_node(node_id, metadata=metadata)
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        self._graph.add_edge(from_node, to_node)

        # Enforce DAG constraint - Later versions can be extended to loop modeling 
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(from_node, to_node)
            raise ValueError("Edge creates cucle. AgentGraph must remain acyclic")
    
    def copy(self) -> "AgentGraph":
        """
        Deep copy of the AgentGraph

        Copies:
            - All nodes
            - associated metadata
            - all edges

        Preserves DAG structure
        """

        new_graph = AgentGraph()

        # Copy nodes with metadata
        for node_id, data in self._graph.nodes(data=True):
            metadata = data["metadata"]
            new_graph.add_component(node_id, metadata)

        # Copy edges
        for source, target in self._graph.edges():
            new_graph.add_edge(source, target)
        
        return new_graph

    # Validation

    def validate(self) -> bool:
        return nx.is_directed_acyclic_graph(self._graph)
    
    # Introspection

    def get_execution_order(self) -> List[str]:
        return list(nx.topological_sort(self._graph))
    
    def get_metadata(self, node_id: str) -> ComponentMetadata:
        return self._graph.nodes[node_id]["metadata"]
    
    def list_nodes(self) -> List[str]:
        return list(self._graph.nodes)
    
    # Aggregate Metrics

    def total_cost(self) -> float:
        return sum(
            self.get_metadata(n).cost_per_call
            for n in self._graph.nodes
        )

    def total_latency(self) -> float:
        return sum(
            self.get_metadata(n).avg_latency_ms
            for n in self._graph.nodes
        )

    def aggregate_reliability(self) -> float:
        """
        Simple multiplicative reliability model:
        Assumes independent failure probabilities
        """

        reliability = 1.0
        for n in self._graph.nodes:
            reliability *= self.get_metadata(n).reliability_score
        return reliability
    