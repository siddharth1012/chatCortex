"""Turn execution traces into calibrated component metadata.

:func:`calibrate` takes a registry and a list of TelemetryLogger-shaped
records (``{"component", "latency_ms", "cost", "success"}`` dicts) and
returns a *new* registry whose performance fields blend the empirical
values with the declared priors::

    new = alpha * empirical + (1 - alpha) * prior

Components with no matching records keep their prior values untouched.
The result is deterministic for a given input: records are grouped by
component name and components are visited in registry order (which is
registration order and therefore stable).

Component mapping: TelemetryLogger records carry graph *node ids* whose
scheme is owned by :mod:`chatcortex.graph.node_ids` (the synthesizers
encode them via ``node_id_for``). A record's component is matched to
registry metadata by exact name first, then by decoding the node id
with ``component_name_from_node_id`` from that same module — the one
place that encodes and decodes the scheme, so a naming change fails
loudly here instead of silently keeping priors. Names that match
neither way are skipped, so traces from graphs outside the registry
are ignored rather than raising.
"""

import dataclasses
import math
from collections import defaultdict
from typing import Dict, List

from ..graph.node_ids import component_name_from_node_id
from ..registry.capability_registry import CapabilityRegistry

# TelemetryLogger.record shapes: {"component", "latency_ms", "cost", "success"}
Record = Dict


def _resolve_component_name(record_component: str, name_to_index: dict) -> int:
    """Index of the component a trace record belongs to, or -1 if unknown."""
    index = name_to_index.get(record_component)
    if index is not None:
        return index
    return name_to_index.get(component_name_from_node_id(record_component), -1)


def _blend(alpha: float, empirical: float, prior: float) -> float:
    return alpha * empirical + (1.0 - alpha) * prior


def calibrate(
    registry: CapabilityRegistry,
    traces: List[Record],
    alpha: float = 0.3,
) -> CapabilityRegistry:
    """Blend empirical trace statistics into component metadata.

    Args:
        registry: Registry holding the declared (prior) metadata.
        traces: Records shaped like ``TelemetryLogger.records``: dicts
            with ``component`` (node id or component name), ``latency_ms``,
            ``cost``, and ``success`` keys.
        alpha: Blend weight on the empirical values, in ``[0, 1]``; 0
            keeps the priors, 1 replaces them, the default 0.3 leans
            toward the declared values. Non-finite values are rejected
            like any out-of-range value.

    Returns:
        A NEW registry; the input registry and its frozen metadata are
        never mutated. The clone starts as a copy of ``registry`` and
        every traced component's replacement goes through
        :meth:`CapabilityRegistry.update` — the immutable replacement
        method the plan designated for calibration. Components without
        matching traces keep prior values; their metadata instances are
        shared unchanged.

    Raises:
        ValueError: If ``alpha`` is outside ``[0.0, 1.0]`` or not finite.
    """
    # NaN passes naive range comparisons, so a missing finite check would
    # let NaN flow into metadata and propagate silently through blends.
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], {alpha!r} given")

    all_metadata = registry.list_all()
    name_to_index = {metadata.name: idx for idx, metadata in enumerate(all_metadata)}

    grouped: Dict[int, List[Record]] = defaultdict(list)
    for record in traces:
        idx = _resolve_component_name(record["component"], name_to_index)
        if idx >= 0:
            grouped[idx].append(record)

    calibrated = CapabilityRegistry()
    for prior in all_metadata:
        calibrated.register(prior)

    for idx, prior in enumerate(all_metadata):
        records = grouped.get(idx)
        if not records:
            continue

        n = len(records)
        empirical_success = sum(bool(r["success"]) for r in records) / n
        empirical_latency = sum(r["latency_ms"] for r in records) / n
        empirical_cost = sum(r["cost"] for r in records) / n

        updated = dataclasses.replace(
            prior,
            reliability_score=_blend(alpha, empirical_success, prior.reliability_score),
            avg_latency_ms=_blend(alpha, empirical_latency, prior.avg_latency_ms),
            cost_per_call=_blend(alpha, empirical_cost, prior.cost_per_call),
        )
        calibrated.update(prior.name, updated)
    return calibrated
