import pytest

from chatcortex import CapabilityRegistry, ComponentMetadata


def make_component(name="comp", capability="retrieval", privacy="internal"):
    return ComponentMetadata(
        name=name,
        component_type="tool",
        capabilities=[capability],
        cost_per_call=0.001,
        avg_latency_ms=100,
        reliability_score=0.9,
        privacy_level=privacy,
    )


def test_register_and_get():
    reg = CapabilityRegistry()
    comp = make_component()
    reg.register(comp)
    assert reg.get("comp") is comp
    assert reg.list_all() == [comp]


def test_duplicate_registration_raises():
    reg = CapabilityRegistry()
    reg.register(make_component())
    with pytest.raises(ValueError):
        reg.register(make_component())


def test_get_by_capability_filters():
    reg = CapabilityRegistry()
    reg.register(make_component("a", "retrieval"))
    reg.register(make_component("b", "generation"))
    names = [c.name for c in reg.get_by_capability("retrieval")]
    assert names == ["a"]
    assert reg.get_by_capability("nonexistent") == []


def test_metadata_supports():
    comp = make_component(capability="retrieval")
    assert comp.supports("retrieval")
    assert not comp.supports("generation")


def test_metadata_is_immutable():
    comp = make_component()
    with pytest.raises(Exception):
        comp.cost_per_call = 5.0


def test_demo_registry_populated():
    from chatcortex import build_demo_registry

    reg = build_demo_registry(capabilities=("retrieval", "generation"))
    assert len(reg.get_by_capability("retrieval")) == 4
    assert len(reg.get_by_capability("generation")) == 4
