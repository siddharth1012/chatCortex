import pytest

from chatcortex import AgentGraph, ComponentMetadata
from chatcortex.adapters.openai_backend import OpenAIBackend
from chatcortex.execution.executor import (
    AgentExecutor,
    ExecutionProtocol,
    ProbeResult,
)


def build_chain():
    """Two-node chain whose components have distinct simulated metrics."""
    graph = AgentGraph()
    graph.chain_append(
        "a",
        ComponentMetadata(
            name="a",
            component_type="tool",
            capabilities=["x"],
            cost_per_call=0.001,
            avg_latency_ms=10,
            reliability_score=0.5,
            privacy_level="internal",
        ),
    )
    graph.chain_append(
        "b",
        ComponentMetadata(
            name="b",
            component_type="model",
            capabilities=["y"],
            cost_per_call=0.002,
            avg_latency_ms=20,
            reliability_score=0.5,
            privacy_level="internal",
        ),
    )
    return graph


def test_probe_result_is_reachable_on_the_protocol():
    """The plan-spelled access path and the flat alias are the same type."""
    assert ExecutionProtocol.ProbeResult is ProbeResult

    probe = ExecutionProtocol.ProbeResult(
        success=True, latency_ms=1.0, cost=0.0, output="ok"
    )
    assert probe.success is True
    probe_validator = AgentExecutor(backend=lambda m, i: probe)
    result = probe_validator.execute(build_chain())
    assert result.summary()["steps"] == 2


class CannedBackend:
    """Returns pre-decided ProbeResults in execution order."""

    def __init__(self, probes, fail_at=None):
        self.probes = list(probes)
        self.calls = []
        self.fail_at = fail_at  # raise instead of returning at this index

    def __call__(self, metadata, node_input):
        index = len(self.calls)
        self.calls.append((metadata.name, node_input))
        if index == self.fail_at:
            raise RuntimeError("backend exploded")
        return self.probes[index]


def test_simulator_parity_when_no_backend():
    graph = build_chain()
    result = AgentExecutor(mode="deterministic").execute(graph)
    summary = result.summary()
    assert summary["success"] is True
    assert summary["steps"] == 2
    assert summary["total_cost"] == pytest.approx(0.003)
    assert summary["total_latency"] == pytest.approx(30)
    assert [record["latency_ms"] for record in result.telemetry.records] == [10, 20]
    assert [record["cost"] for record in result.telemetry.records] == [0.001, 0.002]


def test_backend_records_measured_values_and_chains_outputs():
    seen_probes = [
        ProbeResult(success=True, latency_ms=41.5, cost=0.017, output="first"),
        ProbeResult(success=True, latency_ms=87.25, cost=0.033, output="second"),
    ]
    backend = CannedBackend(seen_probes)
    graph = build_chain()
    result = AgentExecutor(backend=backend).execute(graph)

    # Measured numbers, not the metadata estimates, reach the telemetry.
    assert [record["latency_ms"] for record in result.telemetry.records] == [41.5, 87.25]
    assert [record["cost"] for record in result.telemetry.records] == [0.017, 0.033]
    assert [record["success"] for record in result.telemetry.records] == [True, True]

    # node_input chains each node's output into the next call.
    assert backend.calls[0] == ("a", None)
    assert backend.calls[1] == ("b", "first")


def test_backend_failure_stops_chain_and_keeps_telemetry():
    backend = CannedBackend(
        [
            ProbeResult(success=False, latency_ms=12.0, cost=0.009, output=None),
            ProbeResult(success=True, latency_ms=1.0, cost=0.0, output="never used"),
        ]
    )
    graph = build_chain()
    result = AgentExecutor(backend=backend).execute(graph)

    assert result.summary()["success"] is False
    assert result.summary()["steps"] == 1  # node b never probed
    assert result.telemetry.records[0]["latency_ms"] == 12.0


def test_backend_exception_is_recorded_as_failed_probe():
    class ExplodingBackend:
        def __call__(self, metadata, node_input):
            raise ValueError("bad key")

    graph = build_chain()
    result = AgentExecutor(backend=ExplodingBackend()).execute(graph)

    records = result.telemetry.records
    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["cost"] == 0.0
    assert result.summary()["success"] is False


# --- OpenAI adapter, exercised against a stub client (no network, no key)


class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)


class _StubCompletions:
    def __init__(self, content="stubbed reply"):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Response:
            choices = [_StubChoice(self.content)]

        return Response()


class _StubClient:
    def __init__(self, content="stubbed reply"):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _StubCompletions(content)


def test_openai_backend_with_stubbed_client():
    stub = _StubClient("hello from stub")
    backend = OpenAIBackend(model="stub-model", client=stub)
    metadata = build_chain().get_metadata("a")

    probe = backend(metadata, None)

    assert probe.success is True
    assert probe.output == "hello from stub"
    assert probe.latency_ms >= 0.0
    # Cost stays the component's declared value; metadata is never touched.
    assert probe.cost == pytest.approx(metadata.cost_per_call)
    call = stub.chat.completions.calls[0]
    assert call["model"] == "stub-model"
    assert "Perform the 'x' task as component 'a'." in call["messages"][0]["content"]


def test_openai_backend_model_for_mapping():
    stub = _StubClient()
    backend = OpenAIBackend(
        model="default-model",
        model_for=lambda metadata: f"m-{metadata.name}",
        client=stub,
    )
    metadata = build_chain().get_metadata("b")
    backend(metadata, "hi")
    assert stub.chat.completions.calls[0]["model"] == "m-b"


def test_openai_backend_failure_returns_unsuccessful_probe():
    class BrokenChat:
        def create(self, **kwargs):
            raise ConnectionError("network down")

    class BrokenClient:
        chat = type("Chat", (), {})()
        chat.completions = BrokenChat()

    backend = OpenAIBackend(model="x", client=BrokenClient())
    probe = backend(build_chain().get_metadata("a"), None)

    assert probe.success is False
    assert probe.cost == 0.0
    assert "network down" in str(probe.output)
