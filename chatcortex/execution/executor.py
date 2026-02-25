import time
import random
from typing import Literal, Optional

from chatcortex.graph.agent_graph import AgentGraph
from chatcortex.telemetry.logger import TelemetryLogger


ExecutionMode = Literal["deterministic", "probabilistic"]


class ExecutionResult:
    def __init__(self, telemetry: TelemetryLogger):
        self.telemetry = telemetry
    
    def summary(self):
        return self.telemetry.summary()


class AgentExecutor:
    """
    Simulated execution engine for AgentGraph
    Supports deterministic and probabilistic modes
    """
    def __init__(
            self, 
            mode: ExecutionMode = "deterministic",
            seed: Optional[int] = None,    
        ):
        self.mode = mode
        # local random generator - NO RNG pollution, no side-effects, clean isolation
        self.random = random.Random(seed)
    
    def execute(self, graph: AgentGraph) -> ExecutionResult:
        telemetry = TelemetryLogger()

        for node_id in graph.get_execution_order():
            metadata = graph.get_metadata(node_id)

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
            
            telemetry.log(
                component_name=node_id,
                latency_ms=simulated_latency,
                cost=metadata.cost_per_call,
                success=success,
            )

            if not success:
                break # Stop pipeline on failure
        
        return ExecutionResult(telemetry=telemetry)