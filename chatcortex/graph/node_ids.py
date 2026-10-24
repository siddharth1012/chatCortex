"""Single owner of the chain node-id scheme.

Chain synthesizers name nodes ``f"{component.name}_{stage_idx}"``
(:func:`node_id_for`); execution records those node ids in the
TelemetryLogger, and consumers that map trace records back to
components (:mod:`chatcortex.telemetry.calibration`) decode them with
:func:`component_name_from_node_id`. Encoding and decoding live here so
a naming change can land in one place with a shared inverse, instead of
drifting apart and silently mis-mapping traces.
"""


def node_id_for(component_name: str, stage_idx: int) -> str:
    """Canonical node id for a component occupying a chain stage."""
    return f"{component_name}_{stage_idx}"


def component_name_from_node_id(node_id: str) -> str:
    """Inverse of :func:`node_id_for`.

    Strips one trailing ``_<digits>`` stage suffix; ids without one are
    returned unchanged (callers treat an unknown name as no match).
    """
    stem, _, suffix = node_id.rpartition("_")
    if suffix.isdigit():
        return stem
    return node_id
