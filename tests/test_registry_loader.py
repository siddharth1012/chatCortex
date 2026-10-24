import json

import pytest

from chatcortex import CapabilityRegistry, load_registry


def load_document_through_loader(tmp_path, document):
    path = tmp_path / "roundtrip.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_registry(path)


def component_dict(name="comp_a", **overrides):
    payload = {
        "name": name,
        "component_type": "tool",
        "capabilities": ["x"],
        "cost_per_call": 0.001,
        "avg_latency_ms": 5.0,
        "reliability_score": 0.9,
        "privacy_level": "internal",
    }
    payload.update(overrides)
    return payload


def write_registry(tmp_path, components):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"components": components}), encoding="utf-8")
    return path


def test_load_registry_happy_path(tmp_path):
    path = write_registry(
        tmp_path,
        [component_dict("a"), component_dict("b", component_type="model")],
    )
    registry = load_registry(path)
    assert isinstance(registry, CapabilityRegistry)
    names = [metadata.name for metadata in registry.list_all()]
    # Order follows the file, not dict hashing.
    assert names == ["a", "b"]
    metadata = registry.get("a")
    assert metadata.component_type == "tool"
    assert metadata.cost_per_call == pytest.approx(0.001)
    assert metadata.supports("x")


def test_load_registry_malformed_field_names_row(tmp_path):
    path = write_registry(
        tmp_path,
        [
            component_dict("a", capabilities=["x"]),
            {"name": "b"},  # missing every other required field
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        load_registry(path)
    message = str(excinfo.value)
    assert "components[1].component_type" in message


def test_load_registry_out_of_range_reliability(tmp_path):
    path = write_registry(tmp_path, [component_dict("a", reliability_score=1.4)])
    with pytest.raises(ValueError) as excinfo:
        load_registry(path)
    assert "components[0].reliability_score out of range" in str(excinfo.value)


def test_load_registry_duplicate_name_raises(tmp_path):
    path = write_registry(tmp_path, [component_dict("dup"), component_dict("dup")])
    with pytest.raises(ValueError) as excinfo:
        load_registry(path)
    assert "components[1].name" in str(excinfo.value)


def test_load_registry_empty_file_is_invalid(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_registry(path)
    assert "not valid JSON" in str(excinfo.value)


def test_load_registry_rejects_bad_document_shape(tmp_path):
    path = tmp_path / "shape.json"
    path.write_text(json.dumps({"registry": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(path)


def test_to_dict_round_trips_through_the_loader(tmp_path):
    path = write_registry(
        tmp_path,
        [
            component_dict("first"),
            component_dict("second", component_type="model", reliability_score=0.5),
        ],
    )
    original = load_registry(path)

    document = original.to_dict()
    assert set(document) == {"components"}
    assert set(document["components"][0]) == {
        "name",
        "component_type",
        "capabilities",
        "cost_per_call",
        "avg_latency_ms",
        "reliability_score",
        "privacy_level",
    }

    rebuilt = load_document_through_loader(tmp_path, document)
    assert rebuilt.list_all() == original.list_all()
    assert [m.name for m in rebuilt.list_all()] == ["first", "second"]


def test_load_registry_tolerates_optional_schemas(tmp_path):
    path = write_registry(
        tmp_path,
        [component_dict("a", input_schema={"type": "object"})],
    )
    registry = load_registry(path)
    assert registry.get("a").input_schema == {"type": "object"}
