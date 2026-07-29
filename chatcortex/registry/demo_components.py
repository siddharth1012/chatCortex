"""
Prebuilt demo registry so new users can run a synthesis end-to-end
in seconds, without hand-writing component metadata.

Each capability stage offers components with adversarial tradeoffs
(fast-but-unreliable, cheap-but-slow, reliable-but-expensive) so that
synthesizers produce non-trivial Pareto frontiers.
"""

from .capability_registry import CapabilityRegistry
from .metadata import ComponentMetadata

# (name_suffix, cost_per_call, avg_latency_ms, reliability_score)
_PROFILES = [
    ("speedster", 0.004, 60, 0.78),
    ("budget", 0.0008, 420, 0.82),
    ("reliable", 0.006, 300, 0.98),
    ("balanced", 0.003, 180, 0.92),
]

_CAPABILITY_TYPES = {
    "retrieval": "tool",
    "generation": "model",
    "verification": "verification",
    "refinement": "model",
    "postprocessing": "tool",
}


def build_demo_registry(
    capabilities=("retrieval", "generation", "verification"),
) -> CapabilityRegistry:
    """
    Build an in-memory registry with 4 tradeoff variants per capability.

    Args:
        capabilities: capability stages to populate. Any names are
            accepted; names listed in ``_CAPABILITY_TYPES`` get a
            matching component type, others default to "tool".

    Returns:
        A populated CapabilityRegistry ready for synthesis.
    """
    registry = CapabilityRegistry()

    for capability in capabilities:
        component_type = _CAPABILITY_TYPES.get(capability, "tool")
        for suffix, cost, latency, reliability in _PROFILES:
            registry.register(
                ComponentMetadata(
                    name=f"{capability}_{suffix}",
                    component_type=component_type,
                    capabilities=[capability],
                    cost_per_call=cost,
                    avg_latency_ms=latency,
                    reliability_score=reliability,
                    privacy_level="internal",
                )
            )

    return registry
