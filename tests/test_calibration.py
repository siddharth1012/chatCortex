import pytest

from chatcortex import CapabilityRegistry, ComponentMetadata
from chatcortex.telemetry.calibration import calibrate


def make_registry():
    registry = CapabilityRegistry()
    registry.register(
        ComponentMetadata(
            name="comp_a",
            component_type="tool",
            capabilities=["x"],
            cost_per_call=0.001,
            avg_latency_ms=100,
            reliability_score=0.5,
            privacy_level="internal",
        )
    )
    registry.register(
        ComponentMetadata(
            name="comp_b",
            component_type="model",
            capabilities=["y"],
            cost_per_call=0.002,
            avg_latency_ms=200,
            reliability_score=0.8,
            privacy_level="internal",
        )
    )
    return registry


def trace(component, latency_ms, cost, success):
    return {
        "component": component,
        "latency_ms": latency_ms,
        "cost": cost,
        "success": success,
    }


def test_no_traces_returns_prior_registry_untouched():
    registry = make_registry()
    calibrated = calibrate(registry, [])
    for original in registry.list_all():
        after = calibrated.get(original.name)
        assert after == original
    assert calibrated is not registry


def test_blending_math_is_monotone_and_exact():
    registry = make_registry()
    traces = [
        trace("comp_a_0", 50.0, 0.002, True),
        trace("comp_a_0", 60.0, 0.004, True),
        trace("comp_b", 300.0, 0.010, False),
        trace("comp_b", 500.0, 0.010, True),
    ]
    calibrated = calibrate(registry, traces, alpha=0.3)

    a = calibrated.get("comp_a")
    b = calibrated.get("comp_b")

    # comp_a: all successes, latency mean 55, cost mean 0.003.
    assert a.reliability_score == pytest.approx(0.3 * 1.0 + 0.7 * 0.5)
    assert a.avg_latency_ms == pytest.approx(0.3 * 55.0 + 0.7 * 100)
    assert a.cost_per_call == pytest.approx(0.3 * 0.003 + 0.7 * 0.001)

    # comp_b: given by exact name, not only via node ids.
    assert b.reliability_score == pytest.approx(0.3 * 0.5 + 0.7 * 0.8)
    assert b.avg_latency_ms == pytest.approx(0.3 * 400.0 + 0.7 * 200)
    assert b.cost_per_call == pytest.approx(0.3 * 0.01 + 0.7 * 0.002)

    # Empirical evidence moves priors in their empirical direction.
    assert a.reliability_score > registry.get("comp_a").reliability_score
    assert a.avg_latency_ms < registry.get("comp_a").avg_latency_ms


def test_alpha_extremes():
    registry = make_registry()
    traces = [trace("comp_a_0", 10.0, 0.005, True)]

    prior_kept = calibrate(registry, traces, alpha=0.0).get("comp_a")
    assert prior_kept.reliability_score == pytest.approx(0.5)
    assert prior_kept.avg_latency_ms == pytest.approx(100)
    assert prior_kept.cost_per_call == pytest.approx(0.001)

    prior_replaced = calibrate(registry, traces, alpha=1.0).get("comp_a")
    assert prior_replaced.reliability_score == pytest.approx(1.0)
    assert prior_replaced.avg_latency_ms == pytest.approx(10.0)
    assert prior_replaced.cost_per_call == pytest.approx(0.005)


def test_components_absent_from_traces_keep_prior():
    registry = make_registry()
    traces = [trace("comp_a_0", 40.0, 0.002, True)]
    calibrated = calibrate(registry, traces, alpha=0.9)

    assert calibrated.get("comp_b") == registry.get("comp_b")
    # And it is the same object: untouched components share metadata instances.
    assert calibrated.get("comp_b") is registry.get("comp_b")


def test_calibration_routes_replacements_through_update(monkeypatch):
    """The plan-letter mechanism: replacements go through
    CapabilityRegistry.update, not through re-registration."""
    from chatcortex.telemetry import calibration as calibration_module

    updates = []

    class SpyRegistry(calibration_module.CapabilityRegistry):
        def update(self, name, metadata):
            updates.append(name)
            return super().update(name, metadata)

    monkeypatch.setattr(calibration_module, "CapabilityRegistry", SpyRegistry)

    registry = make_registry()
    traces = [
        trace("comp_a_0", 50.0, 0.002, True),
        trace("comp_b", 300.0, 0.010, True),
    ]
    calibrated = calibrate(registry, traces, alpha=0.3)

    assert sorted(updates) == ["comp_a", "comp_b"]
    # The blend still happened through update: values changed.
    assert calibrated.get("comp_a").reliability_score != registry.get("comp_a").reliability_score


def test_unchanged_components_share_metadata_instances():
    registry = make_registry()
    traces = [trace("comp_a_0", 40.0, 0.002, True)]
    calibrated = calibrate(registry, traces, alpha=0.9)

    # Traced component: a NEW metadata instance with blended values.
    assert calibrated.get("comp_a") is not registry.get("comp_a")
    # Untraced component: the SAME shared instance, prior untouched.
    assert calibrated.get("comp_b") is registry.get("comp_b")


def test_unknown_trace_components_are_skipped():
    registry = make_registry()
    traces = [trace("totally_unrelated", 10.0, 0.0, True)]
    calibrated = calibrate(registry, traces)
    assert calibrated.get("comp_a") == registry.get("comp_a")


def test_calibration_is_deterministic_and_non_mutating():
    registry = make_registry()
    registry_before = list(registry.list_all())
    traces = [
        trace("comp_a_0", 10.0, 0.003, True),
        trace("comp_b", 999.0, 0.011, False),
        trace("comp_a_1", 30.0, 0.001, False),
    ]

    first = calibrate(registry, traces)
    second = calibrate(registry, traces)
    assert first.list_all() == second.list_all()

    # The input registry and its frozen metadata are untouched.
    for original in registry.list_all():
        assert original == registry.get(original.name)
    assert registry.list_all() == registry_before


def test_alpha_out_of_bounds_raises():
    registry = make_registry()
    with pytest.raises(ValueError):
        calibrate(registry, [], alpha=1.5)
    with pytest.raises(ValueError):
        calibrate(registry, [], alpha=-0.1)


def test_alpha_nonfinite_raises_instead_of_silent_nan_metadata():
    """NaN passes naive range comparisons (<, >); without an explicit
    finite check it would blend NaN reliability/latency/cost straight into
    the registry."""
    registry = make_registry()
    for alpha in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            calibrate(registry, [], alpha=alpha)
