from dataclasses import dataclass
from typing import List, Dict, Literal, Optional


ComponentType = Literal["model", "tool", "memory", "verification"]
PrivacyLevel = Literal["internal", "external", "hybrid"]
 

# Keep frozen to True for immutability - Metadata should not mutate during runtime
@dataclass(frozen=True)
class ComponentMetadata:
    """
    Formal metadata description of a component in the agent capability graph

    This class contains only declarative properties used for:
    - Capability matching
    - Constraint filtering
    - Optimization scoring
    """

    name: str
    component_type: ComponentType

    # Functional properties
    capabilities: List[str]

    # Performance metrics
    cost_per_call: float
    avg_latency_ms: float

    # Abstract for now, later on can introduce correlated failures, retries, fallback strategies
    
    reliability_score: float # 0.0 to 1.0

    # Governance constraints
    privacy_level: PrivacyLevel

    # Structural compability
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None

    def supports(self, capability: str) -> bool:
        """Check whether components supports a required capability"""
        return capability in self.capabilities