import time 
from typing import Dict, List


class TelemetryLogger:
    """
    Records execution metrics for each component invocation
    """

    def __init__(self):
        self.records: List[Dict] = []
    
    def log(
        self,
        component_name: str,
        latency_ms: float,
        cost: float,
        success: bool
    ) -> None:
        self.records.append(
            {
                "component": component_name,
                "latency_ms": latency_ms,
                "cost": cost,
                "success": success
            }
        )
    
    def summary(self) -> Dict:
        total_cost = sum(r["cost"] for r in self.records)
        total_latency = sum(r["latency_ms"] for r in self.records)
        overall_success = all(r["success"] for r in self.records)

        return {
            "total_cost": total_cost,
            "total_latency": total_latency,
            "success": overall_success,
            "steps": len(self.records),
        }
    