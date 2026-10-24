"""Behavior tests for registry packs (package.save / validate / CLI)."""

import json

import pytest

from chatcortex import (
    RegistryReport,
    build_demo_registry,
    load_registry,
    save_registry,
    validate_registry_file,
)
from chatcortex.registry.cli import main as cli_main


def test_save_round_trips_and_writes_only_loader_shape(tmp_path):
    registry = build_demo_registry()
    path = save_registry(registry, tmp_path / "demo.pack.json")

    reloaded = load_registry(path)
    assert reloaded.list_all() == registry.list_all()

    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {"components"}
    each = document["components"][0]
    assert set(each) == {
        "name",
        "component_type",
        "capabilities",
        "cost_per_call",
        "avg_latency_ms",
        "reliability_score",
        "privacy_level",
    }


def test_validate_valid_pack(tmp_path):
    registry = build_demo_registry()
    path = save_registry(registry, tmp_path / "demo.pack.json")

    report = validate_registry_file(path)
    assert isinstance(report, RegistryReport)
    assert report.valid is True
    assert report.component_count == len(registry.list_all())
    assert report.issues == []
    assert report.duplicate_names == []
    assert report.to_dict()["valid"] is True


def _write(path, rows):
    path.write_text(json.dumps({"components": rows}), encoding="utf-8")
    return path


def test_validate_collects_all_row_issues_not_first_failure(tmp_path):
    valid = build_demo_registry().get("retrieval_speedster").to_json()
    bad_reliability = {**valid, "name": "bad_one", "reliability_score": 5.0}
    missing_name = {
        k: v for k, v in valid.items() if k != "name"
    }

    path = _write(tmp_path / "corrupt.pack.json", [bad_reliability, missing_name])
    report = validate_registry_file(path)

    assert report.valid is False
    assert report.component_count == 0
    assert any(
        issue.startswith("components[0].reliability_score") for issue in report.issues
    )
    assert any(
        issue.startswith("components[1].name is required") for issue in report.issues
    )


def test_validate_reports_every_duplicate_name(tmp_path):
    triple = build_demo_registry().get("generation_reliable").to_json()
    path = _write(
        tmp_path / "dupe.pack.json",
        [triple, triple, build_demo_registry().get("retrieval_speedster").to_json()],
    )
    report = validate_registry_file(path)

    assert report.valid is False
    assert report.duplicate_names == ["generation_reliable"]
    assert report.component_count == 2  # the unique rows still validate


def test_validate_never_raises_on_unreadable_or_unshaped_files(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    not_json = tmp_path / "not_json.json"
    not_json.write_text("{components: [", encoding="utf-8")
    wrong_shape = tmp_path / "wrong_shape.json"
    wrong_shape.write_text('{"registry": []}', encoding="utf-8")
    not_object = _write(tmp_path / "rows_not_objects.json", [["oops"], "also bad"])

    for path, fragment in (
        (missing, "Cannot read"),
        (not_json, "not valid JSON"),
        (wrong_shape, "'components' list"),
        (not_object, "must be an object,"),
    ):
        report = validate_registry_file(path)
        assert report.valid is False
        assert any(fragment in issue for issue in report.issues), report.issues


def test_cli_exit_codes_and_report_lines(tmp_path, capsys):
    valid_path = save_registry(build_demo_registry(), tmp_path / "good.pack.json")
    bad_path = _write(
        tmp_path / "bad.pack.json",
        [build_demo_registry().get("retrieval_speedster").to_json()] * 2,
    )

    assert cli_main(["validate", str(valid_path)]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out

    assert cli_main(["validate", str(bad_path)]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out and "duplicate component: 'retrieval_speedster'" in out


def test_cli_argparse_errors_exit_nonzero(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli_main(["validate", str(tmp_path / "nope.json"), "--extra"])
    # argparse exits with status 2 on usage errors, never silently.
    assert exc.value.code != 0
