"""The full ChatCortex loop with zero API keys.

Writes a small JSON registry, loads it with ``load_registry``, synthesizes
a Pareto frontier, executes one architecture with the built-in simulator,
and feeds the telemetry through ``calibrate`` so reliability and latency
become partially empirical. Run from the repo root:

    python examples/registry_from_json.py
"""

import json
import tempfile
from pathlib import Path

from chatcortex import (
    ProgressiveParetoBeamSynthesizer,
    SynthesisBudget,
    TaskSpecification,
    load_registry,
)
from chatcortex.execution.executor import AgentExecutor
from chatcortex.telemetry.calibration import calibrate

REGISTRY_JSON = {
    "components": [
        {
            "name": "retrieval_speedster",
            "component_type": "tool",
            "capabilities": ["retrieval"],
            "cost_per_call": 0.004,
            "avg_latency_ms": 60,
            "reliability_score": 0.78,
            "privacy_level": "internal",
        },
        {
            "name": "retrieval_reliable",
            "component_type": "tool",
            "capabilities": ["retrieval"],
            "cost_per_call": 0.006,
            "avg_latency_ms": 300,
            "reliability_score": 0.98,
            "privacy_level": "internal",
        },
        {
            "name": "generation_balanced",
            "component_type": "model",
            "capabilities": ["generation"],
            "cost_per_call": 0.003,
            "avg_latency_ms": 180,
            "reliability_score": 0.92,
            "privacy_level": "internal",
        },
        {
            "name": "generation_reliable",
            "component_type": "model",
            "capabilities": ["generation"],
            "cost_per_call": 0.006,
            "avg_latency_ms": 300,
            "reliability_score": 0.98,
            "privacy_level": "internal",
        },
        {
            "name": "verification_budget",
            "component_type": "verification",
            "capabilities": ["verification"],
            "cost_per_call": 0.0008,
            "avg_latency_ms": 420,
            "reliability_score": 0.82,
            "privacy_level": "internal",
        },
        {
            "name": "verification_reliable",
            "component_type": "verification",
            "capabilities": ["verification"],
            "cost_per_call": 0.006,
            "avg_latency_ms": 300,
            "reliability_score": 0.98,
            "privacy_level": "internal",
        },
    ]
}


def main():
    # 1. Write and load a JSON registry.
    registry_path = Path(tempfile.mkdtemp()) / "registry.json"
    registry_path.write_text(json.dumps(REGISTRY_JSON, indent=2), encoding="utf-8")
    registry = load_registry(registry_path)
    print(f"Loaded {len(registry.list_all())} components from {registry_path.name}")

    # 2. Synthesize the Pareto frontier for a 3-stage task.
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_latency=1200.0,
    )
    budget = SynthesisBudget(max_evaluations=500)
    synthesizer = ProgressiveParetoBeamSynthesizer(registry)
    frontier = synthesizer.synthesize(task, budget)
    print(f"Synthesized frontier: {len(frontier)} architectures")

    # 3. Execute the best architecture and calibrate from the telemetry.
    best = max(frontier, key=lambda candidate: candidate.total_reliability)
    for node_id, metadata in best.graph.iter_components():
        print(
            f"  chain: {node_id} "
            f"(cost={metadata.cost_per_call:.4f}, latency_ms={metadata.avg_latency_ms})"
        )

    result = AgentExecutor(mode="probabilistic", seed=7).execute(best.graph)
    print(f"Executed {result.summary()['steps']} steps to success={result.summary()['success']}")

    calibrated = calibrate(registry, result.telemetry.records, alpha=0.3)

    # 4. Show the calibration effect per component.
    print("\nCalibration (alpha=0.3): prior -> calibrated")
    for metadata in calibrated.list_all():
        prior = registry.get(metadata.name)
        if metadata.reliability_score != prior.reliability_score:
            print(
                f"  {metadata.name}: reliability {prior.reliability_score:.3f} -> "
                f"{metadata.reliability_score:.3f}, latency {prior.avg_latency_ms:.1f} -> "
                f"{metadata.avg_latency_ms:.1f} ms"
            )
        else:
            print(f"  {metadata.name}: no traces, prior kept")


if __name__ == "__main__":
    main()
