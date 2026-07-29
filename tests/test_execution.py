from chatcortex import AgentGraph, ComponentMetadata
from chatcortex.execution.executor import AgentExecutor


def build_graph():
    g = AgentGraph()
    g.add_component(
        "a",
        ComponentMetadata(
            name="a",
            component_type="tool",
            capabilities=["x"],
            cost_per_call=0.001,
            avg_latency_ms=1,
            reliability_score=0.5,
            privacy_level="internal",
        ),
    )
    return g


def test_deterministic_execution():
    result = AgentExecutor(mode="deterministic").execute(build_graph())
    assert result.summary() is not None


def test_probabilistic_execution_is_seeded():
    r1 = AgentExecutor(mode="probabilistic", seed=42).execute(build_graph())
    r2 = AgentExecutor(mode="probabilistic", seed=42).execute(build_graph())
    assert r1.summary() == r2.summary()
