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


def test_update_replaces_metadata_of_registered_component():
    import dataclasses

    reg = CapabilityRegistry()
    original = make_component()
    reg.register(original)
    replacement = dataclasses.replace(original, reliability_score=0.4)

    reg.update("comp", replacement)

    assert reg.get("comp") is replacement
    assert reg.get("comp").reliability_score == 0.4


def test_update_unknown_name_raises():
    reg = CapabilityRegistry()
    reg.register(make_component())
    with pytest.raises(ValueError, match="is not registered"):
        reg.update("nonexistent", make_component())


def test_update_name_mismatch_raises():
    import dataclasses

    reg = CapabilityRegistry()
    reg.register(make_component())
    renamed = dataclasses.replace(make_component(), name="other")
    with pytest.raises(ValueError, match="received metadata for 'other'"):
        reg.update("comp", renamed)


def test_required_json_fields_derive_from_the_dataclass():
    """The JSON schema owner: required fields MUST be the dataclass
    fields without defaults, so a new metadata field is required in the
    loader the moment it is declared."""
    import dataclasses

    from chatcortex.registry.metadata import REQUIRED_JSON_FIELDS

    assert REQUIRED_JSON_FIELDS == (
        "name",
        "component_type",
        "capabilities",
        "cost_per_call",
        "avg_latency_ms",
        "reliability_score",
        "privacy_level",
    )
    assert REQUIRED_JSON_FIELDS == tuple(
        field.name
        for field in dataclasses.fields(ComponentMetadata)
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
