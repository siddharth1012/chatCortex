# Stores components, filters by capability and constraints

# To-Do: 
# 1. Support loading from YAML, JSON (Experimental)
# 2. Support dynamic runtime registration


# Simple in-memory registration for v0.2.0
from typing import Optional, List
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
    
    def get(self, name: str) -> ComponentMetadata:
        return self._components[name]
    
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