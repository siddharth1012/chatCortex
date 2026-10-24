import random
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.telemetry.logger import TelemetryLogger

ExecutionMode = Literal["deterministic", "probabilistic"]


class ExecutionProtocol(Protocol):
    """Optional real-execution backend protocol (see ``AgentExecutor``).

    Any callable with the signature ``backend(metadata, node_input) ->
    ProbeResult`` satisfies this protocol; a class only needs a matching
    ``__call__``. Implementations own the actual invocation (HTTP, SDK,
    stub); the executor only consumes the returned ProbeResult.
    """

    @dataclass(frozen=True)
    class ProbeResult:
        """Outcome of one real backend invocation of a graph node.

        A execution backend returns one of these per component call; the
        values are recorded verbatim in the execution's TelemetryLogger,
        so they should be *measured* values, not estimates.
        """

        success: bool
        latency_ms: float
        cost: float
        output: Any

    def __call__(
        self,
        metadata: ComponentMetadata,
        node_input: Any,
    ) -> ProbeResult:  # pragma: no cover - protocol definition only
        ...


# Flat alias so the pre-nesting import path (`from ...executor import
# ProbeResult`) keeps working alongside ``ExecutionProtocol.ProbeResult``.
ProbeResult = ExecutionProtocol.ProbeResult


class ExecutionResult:
    def __init__(self, telemetry: TelemetryLogger):
        self.telemetry = telemetry

    def summary(self):
        return self.telemetry.summary()


class AgentExecutor:
    """
    Simulated execution engine for AgentGraph
    Supports deterministic and probabilistic modes,
    plus an optional real-execution backend.

    With ``backend=None`` (default)
    behavior is identical to the pure simulator. With a backend set,
    each node is invoked through ``backend(metadata, node_input) ->
    ProbeResult`` and the *measured* latency/cost/success from the
    ProbeResult (never the metadata estimates) are recorded in the
    telemetry; ``mode`` is ignored because the backend decides success.
    The chain stops on the first failure, exactly like the simulator.
    A backend exception is recorded as a failed probe (zero cost) and
    also stops the chain, so partial telemetry survives the error.
    """
    def __init__(
            self,
            mode: ExecutionMode = "deterministic",
            seed: Optional[int] = None,
            backend: Optional[ExecutionProtocol] = None,
        ):
        self.mode = mode
        self.backend = backend
        # local random generator - NO RNG pollution, no side-effects, clean isolation
        self.random = random.Random(seed)

    def execute(self, graph: AgentGraph) -> ExecutionResult:
        telemetry = TelemetryLogger()

        node_input: Any = None
        for node_id in graph.get_execution_order():
            metadata = graph.get_metadata(node_id)
            latency_ms, cost, success, output = self._evaluate_node(
                metadata, node_input
            )

            # Shared semantics for both paths: record the probe, then
            # stop the chain on the first failure.
            telemetry.log(
                component_name=node_id,
                latency_ms=latency_ms,
                cost=cost,
                success=success,
            )

            if not success:
                break # Stop pipeline on failure
            node_input = output

        return ExecutionResult(telemetry=telemetry)

    def _evaluate_node(self, metadata: ComponentMetadata, node_input: Any):
        """Run one node through the backend (if set) or the simulator.

        Returns ``(latency_ms, cost, success, output)``; the shared
        log-then-stop-on-failure tail lives in ``execute``.
        """
        if self.backend is not None:
            probe = self._invoke_backend(metadata, node_input)
            return probe.latency_ms, probe.cost, probe.success, probe.output

        # Simulated latency
        simulated_latency = metadata.avg_latency_ms

        # Optional: actually sleep (disable if too slow)
        # time.sleep(simulated_latency / 1000)

        # Determine success
        if self.mode == "deterministic":
            success = True
        elif self.mode == "probabilistic":
            success = self.random.random() <= metadata.reliability_score
        else:
            raise ValueError("Invalid execution mode")

        return simulated_latency, metadata.cost_per_call, success, None

    def _invoke_backend(
        self, metadata: ComponentMetadata, node_input: Any
    ) -> ProbeResult:
        """Call the backend once, turning exceptions into failed probes."""
        try:
            return self.backend(metadata, node_input)
        except Exception as exc:  # backend failures are execution results
            return ProbeResult(
                success=False,
                latency_ms=0.0,
                cost=0.0,
                output=f"backend raised {type(exc).__name__}: {exc}",
            )
