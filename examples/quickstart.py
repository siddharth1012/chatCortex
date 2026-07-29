"""
ChatCortex quickstart.

Synthesizes Pareto-optimal agent architectures for a 3-stage task
using the prebuilt demo registry. Runs in under a second:

    python examples/quickstart.py
"""

from chatcortex import (
    BeamSynthesizer,
    ExhaustiveSynthesizer,
    ProgressiveParetoBeamSynthesizer,
    TaskSpecification,
    build_demo_registry,
)


def main():
    # 1. A registry of components with cost/latency/reliability tradeoffs
    registry = build_demo_registry(
        capabilities=("retrieval", "generation", "verification")
    )

    # 2. The task: an ordered capability chain with hard constraints
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_cost=0.02,
        max_latency=1500,
    )

    # 3. Synthesize architectures with three strategies
    synthesizers = {
        "Beam": BeamSynthesizer(registry, beam_width=5),
        "ProgressivePareto": ProgressiveParetoBeamSynthesizer(registry),
        "Exhaustive (exact frontier)": ExhaustiveSynthesizer(registry),
    }

    for name, synth in synthesizers.items():
        candidates = synth.synthesize(task)
        print(f"\n{name}: {len(candidates)} Pareto-optimal architecture(s)")
        for c in sorted(candidates, key=lambda c: c.total_cost):
            components = " -> ".join(
                meta.name for _, meta in c.graph.iter_components()
            )
            print(
                f"  cost=${c.total_cost:.4f}  "
                f"latency={c.total_latency:.0f}ms  "
                f"reliability={c.total_reliability:.3f}  |  {components}"
            )


if __name__ == "__main__":
    main()
