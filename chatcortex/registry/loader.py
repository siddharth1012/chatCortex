"""Load a :class:`CapabilityRegistry` from a JSON file.

Expected document shape (all required component fields must be present)::

    {
        "components": [
            {
                "name": "retriever_reliable",
                "component_type": "tool",          # model | tool | memory | verification
                "capabilities": ["retrieval"],     # non-empty list of strings
                "cost_per_call": 0.006,
                "avg_latency_ms": 300.0,
                "reliability_score": 0.98,         # 0.0 .. 1.0
                "privacy_level": "internal",       # internal | external | hybrid
                "input_schema": null,              # optional
                "output_schema": null              # optional
            }
        ]
    }

The per-field schema lives in :mod:`chatcortex.registry.metadata`
(:func:`ComponentMetadata.to_json`, :func:`validate_json_payload`, and
``REQUIRED_JSON_FIELDS`` derive from the dataclass itself); this module
adds only document-level rules: row numbering in every ``ValueError``
message (e.g. ``components[2].reliability_score out of range``) and
duplicate-name rejection. Components are registered in file order, so
``list_all()`` preserves the document order.
"""

import json
from pathlib import Path
from typing import Any, Union

from .capability_registry import CapabilityRegistry
from .metadata import ComponentMetadata, validate_json_payload


def _validate_component(row: int, payload: Any) -> ComponentMetadata:
    if not isinstance(payload, dict):
        raise ValueError(
            f"components[{row}] must be an object, {type(payload).__name__} given"
        )

    try:
        kwargs = validate_json_payload(payload)
        return ComponentMetadata(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"components[{row}].{exc}") from exc


def load_registry(path: Union[str, Path]) -> CapabilityRegistry:
    """Build a :class:`CapabilityRegistry` from a JSON registry file.

    Args:
        path: Path to the JSON document described in the module docstring.

    Returns:
        A populated registry; component order follows file order.

    Raises:
        ValueError: On unparsable JSON, a wrong document shape, any
            per-component validation failure (message names the row and
            field), or duplicate component names.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read registry file {file_path}: {exc}") from exc

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Registry file {file_path} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict) or not isinstance(document.get("components"), list):
        raise ValueError(
            f"Registry file {file_path} must contain a top-level object with "
            "a 'components' list"
        )

    registry = CapabilityRegistry()
    seen_names: set = set()
    for row, payload in enumerate(document["components"]):
        metadata = _validate_component(row, payload)
        if metadata.name in seen_names:
            raise ValueError(
                f"components[{row}].name duplicate component '{metadata.name}' "
                "(names must be unique)"
            )
        seen_names.add(metadata.name)
        registry.register(metadata)
    return registry
