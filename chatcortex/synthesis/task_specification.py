# Represents formal task definition with ordered capability requirements and constraints

# To-Dos:
# 1. Extension to DAG like task definitions

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

PrivacyLevel = Literal["internal", "external", "hybrid"]


@dataclass(frozen=True)
class TaskSpecification:
    """
    Formal definition of a task for agent synthesis

    A task consists of: 
    1. An ordered sequence of required capabilities
    2. Hard constraints - cost, latency, privacy
    3. objective weighting for optimization
    """

    # Ordered capability chain
    required_capabilities: List[str]

    # Hard constraints - solutions that are allowed (feasible region)
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    privacy_constraint: Optional[PrivacyLevel] = None

    # Multi-objective weights - Allows optimization inside feasible region
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "cost": 1.0,
            "latency": 1.0,
            "error": 1.0
        }
    )

    def validate(self) -> None:
        """
        Ensure task definition is well-formed
        """

        if not self.required_capabilities:
            ValueError(f"Task must define at least one required capability")
        
        for key in self.objective_weights:
            if key not in {"cost", "latency", "error"}:
                raise ValueError(f"Invalid objective weight key: {key}")
            