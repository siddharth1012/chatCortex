import pytest

from chatcortex import AgentGraph, ComponentMetadata


def make_component(name, cost=0.001, latency=100, reliability=0.9):
    return ComponentMetadata(
        name=name,
        component_type="tool",
        capabilities=["x"],
        cost_per_call=cost,
        avg_latency_ms=latency,
        reliability_score=reliability,
        privacy_level="internal",
    )


def build_chain():
    g = AgentGraph()
    g.add_component("a", make_component("a", 0.001, 100, 0.9))
    g.add_component("b", make_component("b", 0.002, 200, 0.8))
    g.add_edge("a", "b")
    return g


def test_metrics_aggregation():
    g = build_chain()
    assert g.total_cost() == pytest.approx(0.003)
    assert g.total_latency() == pytest.approx(300)
    assert g.aggregate_reliability() == pytest.approx(0.72)


def test_duplicate_node_raises():
    g = build_chain()
    with pytest.raises(ValueError):
        g.add_component("a", make_component("a"))


def test_cycle_rejected():
    g = build_chain()
    with pytest.raises(ValueError):
        g.add_edge("b", "a")
    assert g.validate()


def test_execution_order():
    g = build_chain()
    assert g.get_execution_order() == ["a", "b"]


def test_copy_is_deep():
    g = build_chain()
    g2 = g.copy()
    g2.add_component("c", make_component("c"))
    assert "c" not in g.list_nodes()
    assert g2.total_cost() > g.total_cost()


def test_iter_components():
    g = build_chain()
    pairs = list(g.iter_components())
    assert [node for node, _ in pairs] == ["a", "b"]
    assert pairs[0][1].name == "a"
