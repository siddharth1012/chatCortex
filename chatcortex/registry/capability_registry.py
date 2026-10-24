# Stores components, filters by capability and constraints

# To-Do:
# 1. Support loading from YAML (JSON loading is available via registry.loader)
# 2. Support dynamic runtime registration


# Simple in-memory registration for v0.2.0
from typing import List, Optional

from .metadata import ComponentMetadata, PrivacyLevel


class CapabilityRegistry:
    """
    In-memory registry of agent components.

    Responsible for:
    - Storing component metadata
    - Filtering candidates by capability
    - Applying hard constraints like privacy_level
    """

    def __init__(self):
        self._components: dict[str, ComponentMetadata] = {}

    # Registration

    def register(self, metadata: ComponentMetadata) -> None:
        if metadata.name in self._components:
            raise ValueError(f"Component '{metadata.name}' already registered.")
        self._components[metadata.name] = metadata

    def update(self, name: str, metadata: ComponentMetadata) -> None:
        """Replace the metadata of an already-registered component.

        Used by trace calibration, which derives new performance metrics
        from execution records; metadata stays immutable, so callers pass
        a new ComponentMetadata rather than mutating the old one.
        """
        if name not in self._components:
            raise ValueError(f"Component '{name}' is not registered.")
        if metadata.name != name:
            raise ValueError(
                f"update('{name}') received metadata for '{metadata.name}'."
            )
        self._components[name] = metadata

    def get(self, name: str) -> ComponentMetadata:
        return self._components[name]

    def to_dict(self) -> dict:
        """Serialize the registry into the JSON loader's document shape.

        Returns ``{"components": [...]}`` where each component object
        carries exactly the loader's required fields (``name``,
        ``component_type``, ``capabilities``, ``cost_per_call``,
        ``avg_latency_ms``, ``reliability_score``, ``privacy_level``),
        in registration order. The per-component shape is owned by
        :meth:`ComponentMetadata.to_json`; optional schema fields are
        not emitted. Feed the result to
        :func:`chatcortex.registry.loader.load_registry` to rebuild an
        equivalent registry.
        """
        return {"components": [metadata.to_json() for metadata in self.list_all()]}

    def list_all(self) -> List[ComponentMetadata]:
        return list(self._components.values())

    # Capability Filtering

    def get_by_capability(
        self,
        capability: str,
        privacy_constraint: Optional[PrivacyLevel] = None,
    ) -> List[ComponentMetadata]:
        """
        Return components supporting a given capability
        optionally filtered by privacy level.
        """

        candidates = [
            cap for cap in self._components.values()
            if cap.supports(capability)
        ]

        if privacy_constraint is not None:
            candidates = [
                candidate for candidate in candidates
                if candidate.privacy_level == privacy_constraint
            ]

        return candidates
