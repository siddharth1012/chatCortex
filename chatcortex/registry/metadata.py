from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, List, Literal, Optional, get_args

ComponentType = Literal["model", "tool", "memory", "verification"]
PrivacyLevel = Literal["internal", "external", "hybrid"]

# Enum values derive from the type aliases above: one declaration, no drift.
COMPONENT_TYPES: tuple = get_args(ComponentType)
PRIVACY_LEVELS: tuple = get_args(PrivacyLevel)

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

    def to_json(self) -> Dict[str, Any]:
        """The registered component shape: this instance's required JSON fields.

        Derived from the dataclass fields themselves; the registry loader's
        document shape and ``to_dict`` round-trips are built on this.
        """
        return {
            field_name: (list(value) if isinstance(value, list) else value)
            for field_name, value in (
                (field.name, getattr(self, field.name))
                for field in _required_json_fields()
            )
        }


def _required_json_fields() -> tuple:
    """Dataclass fields WITHOUT defaults — exactly the required JSON keys."""
    return tuple(
        field
        for field in fields(ComponentMetadata)
        if field.default is MISSING and field.default_factory is MISSING
    )


REQUIRED_JSON_FIELDS: tuple = tuple(field.name for field in _required_json_fields())


def validate_json_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one untrusted component object against the metadata schema.

    Returns the constructor kwargs for :class:`ComponentMetadata` (the
    required fields plus any supplied optional schemas) so the JSON loader
    stays a thin wrapper. Raises ``ValueError`` with messages that name the
    offending field but NOT the document row (e.g.
    ``"reliability_score out of range (must be 0.0-1.0)"``); the loader
    prefixes the row when re-raising.
    """
    kwargs: Dict[str, Any] = {}

    for field_name in REQUIRED_JSON_FIELDS:
        if field_name not in payload:
            raise ValueError(f"{field_name} is required but missing")

    name = payload["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    kwargs["name"] = name

    component_type = payload["component_type"]
    if component_type not in COMPONENT_TYPES:
        raise ValueError(
            f"component_type must be one of {list(COMPONENT_TYPES)}, "
            f"{component_type!r} given"
        )
    kwargs["component_type"] = component_type

    capabilities = payload["capabilities"]
    if not isinstance(capabilities, list) or not all(
        isinstance(capability, str) and capability for capability in capabilities
    ):
        raise ValueError("capabilities must be a list of non-empty strings")
    kwargs["capabilities"] = list(capabilities)

    for field_name in ("cost_per_call", "avg_latency_ms"):
        value = payload[field_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field_name} must be a number >= 0")
        kwargs[field_name] = value

    reliability = payload["reliability_score"]
    if (
        not isinstance(reliability, (int, float))
        or isinstance(reliability, bool)
        or reliability < 0
        or reliability > 1
    ):
        raise ValueError("reliability_score out of range (must be 0.0-1.0)")
    kwargs["reliability_score"] = reliability

    privacy_level = payload["privacy_level"]
    if privacy_level not in PRIVACY_LEVELS:
        raise ValueError(
            f"privacy_level must be one of {list(PRIVACY_LEVELS)}, "
            f"{privacy_level!r} given"
        )
    kwargs["privacy_level"] = privacy_level

    for field_name in ("input_schema", "output_schema"):
        if field_name in payload and payload[field_name] is not None:
            kwargs[field_name] = payload[field_name]

    return kwargs
