"""Pin the node-id scheme at its single owner and at its consumers.

The encoding format and the encode/decode inverse are pinned byte-for
byte here; the synthesis link test guards calibration against silently
drifting away from the scheme the synthesizers emit.
"""

from chatcortex.graph.node_ids import (
    component_name_from_node_id,
    node_id_for,
)
from chatcortex.telemetry.calibration import calibrate


def test_encode_format_is_pinned():
    assert node_id_for("retrieval_reliable", 2) == "retrieval_reliable_2"
    assert node_id_for("a", 0) == "a_0"


def test_decode_inverts_encode():
    assert component_name_from_node_id(node_id_for("x", 7)) == "x"
    assert component_name_from_node_id("no_stage_suffix") == "no_stage_suffix"
    # A name that itself ends in digits decodes correctly.
    assert component_name_from_node_id(node_id_for("gpt_4", 0)) == "gpt_4"


def test_registry_names_with_digit_suffix_never_confuse_decode():
    # "comp_1" and "comp" both registered: a node id comp_1_0 must map
    # to comp_1 via the shared decode, not be misattributed.
    assert component_name_from_node_id("comp_1_0") == "comp_1"


def test_synthesized_node_ids_decode_to_registered_components(registry, task):
    """The visible link: every node id a synthesizer emits decodes, via
    the shared helper, back to a component the registry knows."""
    from chatcortex import ProgressiveParetoBeamSynthesizer

    synthesizer = ProgressiveParetoBeamSynthesizer(registry)
    frontier = synthesizer.synthesize(task)
    assert frontier

    known = {metadata.name for metadata in registry.list_all()}
    for candidate in frontier:
        for node_id, _ in candidate.graph.iter_components():
            decoded = component_name_from_node_id(node_id)
            assert decoded in known, f"node id {node_id!r} no longer decodes"


def test_calibration_consumes_synthesized_node_ids(registry, task):
    """A record traced under a synthesized node id must reach its
    component through the shared decode, keeping priors untouched for
    components without traces."""
    from chatcortex import ProgressiveParetoBeamSynthesizer
    from chatcortex.graph.node_ids import node_id_for as nid

    synthesizer = ProgressiveParetoBeamSynthesizer(registry)
    frontier = synthesizer.synthesize(task)
    graph = frontier[0].graph

    node_ids = [node for node, _ in graph.iter_components()]
    traces = [
        {"component": node_ids[0], "latency_ms": 10.0, "cost": 0.5, "success": False}
    ]
    calibrated = calibrate(registry, traces, alpha=1.0)

    first_name = component_name_from_node_id(node_ids[0])
    assert calibrated.get(first_name).reliability_score == 0.0
    others = [
        m
        for m in calibrated.list_all()
        if nid(m.name, 0) not in node_ids and m.name != first_name
    ]
    for metadata in others:
        assert metadata == registry.get(metadata.name)
