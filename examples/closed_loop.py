"""The closed loop on the demo registry - no API keys, no network.

Four rounds of synthesize -> execute (simulated) -> calibrate ->
re-synthesize, printing the Pareto movement per iteration. Every number
the last rounds print was shaped by measurements the loop itself took in
earlier rounds: the exact pitch of ChatCortex, running offline.

Run from the repo root:

    python examples/closed_loop.py
"""

from chatcortex.execution.executor import AgentExecutor
from chatcortex.loop import run_closed_loop
from chatcortex.synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from chatcortex.synthesis.task_specification import TaskSpecification

ITERATIONS = 4
ALPHA = 0.4


def main():
    registry = build_demo_registry()
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_latency=1500.0,
    )

    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=lambda reg: ProgressiveParetoBeamSynthesizer(
            reg, base_beam_width=3, growth_factor=2.0
        ),
        executor=AgentExecutor(mode="probabilistic", seed=42),
        iterations=ITERATIONS,
        alpha=ALPHA,
    )

    print(f"Closed loop, {ITERATIONS} iterations (alpha={ALPHA}, seed=42)\n")
    previous = None
    for record in records:
        objectives = record.frontier_objectives
        min_cost = min(o["cost"] for o in objectives)
        max_cost = max(o["cost"] for o in objectives)
        min_lat = min(o["latency"] for o in objectives)
        max_lat = max(o["latency"] for o in objectives)
        min_rel = min(o["reliability"] for o in objectives)
        max_rel = max(o["reliability"] for o in objectives)
        steps = record.trace_count

        print(f"=== iteration {record.iteration} "
              f"(probes this round: {steps}) ===")
        print(
            f"  frontier: {record.frontier_size} architectures "
            f"({record.executed_candidates} executed)"
        )
        print(
            f"    cost         = [{min_cost:.4f}, {max_cost:.4f}]"
        )
        print(
            f"    latency_ms   = [{min_lat:.0f}, {max_lat:.0f}]"
        )
        print(
            f"    reliability  = [{min_rel:.3f}, {max_rel:.3f}]"
        )

        if previous is not None:
            d_cost = min_cost - previous[0]
            d_rel = max_rel - previous[1]
            print(
                f"    movement     : cheapest cost {d_cost:+.4f}, "
                f"best reliability {d_rel:+.3f} vs last round"
            )
        if record.updated_components:
            names = ", ".join(record.updated_components)
            truncated = names if len(names) <= 72 else names[:69] + "..."
            print(f"    calibrated   : {truncated}")
        else:
            print("    calibrated   : every measurement matched its prior")
        print()

        previous = (min_cost, max_rel)

    final = records[-1].registry
    declared = {m.name: m for m in registry.list_all()}
    print("Reliability: declared prior -> after the loop")
    for metadata in final.list_all():
        before = declared[metadata.name].reliability_score
        if metadata.reliability_score != before:
            arrow = ">" * 3
            print(
                f"  {metadata.name:28s} {before:.3f} {arrow} "
                f"{metadata.reliability_score:.3f}"
            )


if __name__ == "__main__":
    from chatcortex.registry.demo_components import build_demo_registry

    main()
