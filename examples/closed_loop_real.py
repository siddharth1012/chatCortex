"""The closed loop with REAL executions via the ``chatcortex[real]`` extra.

Same loop as ``closed_loop.py``, but every probe is an actual OpenAI
chat-completions call through the ``OpenAIBackend`` adapter, so latency
and success are *measured* and calibration learns from real behavior.
Executions cost money (the owner's credits), so this demo defaults to
two rounds with two candidates each.

Run from the repo root:

    pip install "chatcortex[real]"
    export OPENAI_API_KEY=sk-...
    python examples/closed_loop_real.py

Without the key (or the extra) this prints the setup card and exits
cleanly, so CI with no secrets stays green.
"""

import os
import sys

from chatcortex.execution.executor import AgentExecutor
from chatcortex.loop import run_closed_loop
from chatcortex.synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from chatcortex.synthesis.task_specification import TaskSpecification

ITERATIONS = 2
CANDIDATES_PER_ITERATION = 2
ALPHA = 0.5

SETUP_CARD = """
OpenAIBackend needs one thing this environment does not have yet:

  1. pip install "chatcortex[real]"     # the openai SDK is optional
  2. export OPENAI_API_KEY=sk-...       # https://freebuff.com/account/api
  3. python examples/closed_loop_real.py

Until then, run the offline flagship instead:

  python examples/closed_loop.py        # full loop, simulated probes
"""


def _backend_or_none():
    """The adapter (lazy-importing `openai`) or None when unusable."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from chatcortex.adapters.openai_backend import OpenAIBackend
    except ImportError:
        return None
    return OpenAIBackend()


def main():
    backend = _backend_or_none()
    if backend is None:
        print(SETUP_CARD, end="")
        return 0

    registry = build_demo_registry(capabilities=("retrieval", "generation"))
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation"],
    )

    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=lambda reg: ProgressiveParetoBeamSynthesizer(
            reg, base_beam_width=3, growth_factor=2.0
        ),
        executor=AgentExecutor(backend=backend),  # measured: backend decides
        iterations=ITERATIONS,
        alpha=ALPHA,
        candidates_per_iteration=CANDIDATES_PER_ITERATION,
    )

    print(f"Closed loop, {ITERATIONS} iterations with REAL OpenAI probes "
          f"({CANDIDATES_PER_ITERATION} candidates/round)\n")
    for record in records:
        print(f"=== iteration {record.iteration} "
              f"({record.executed_candidates} architectures, "
              f"{record.trace_count} real probes) ===")
        failures = sum(1 for t in record.traces if not t["success"])
        print(f"    probe failures: {failures}")
        for objective in record.frontier_objectives:
            print(
                f"    frontier row: cost=${objective['cost']:.4f} "
                f"latency={objective['latency']:.0f}ms "
                f"reliability={objective['reliability']:.3f}"
            )
        if record.updated_components:
            print(f"    calibrated   : {', '.join(record.updated_components)}")
        else:
            print("    calibrated   : every measurement matched its prior")
        print()

    final = records[-1].registry
    declared = {m.name: m for m in registry.list_all()}
    print("Measured vs declared (latency ms and reliability)")
    for metadata in final.list_all():
        before = declared[metadata.name]
        if (
            metadata.avg_latency_ms != before.avg_latency_ms
            or metadata.reliability_score != before.reliability_score
        ):
            print(
                f"  {metadata.name:28s} "
                f"latency {before.avg_latency_ms:.1f} -> {metadata.avg_latency_ms:.1f}   "
                f"reliability {before.reliability_score:.3f} -> "
                f"{metadata.reliability_score:.3f}"
            )
    return 0


if __name__ == "__main__":
    from chatcortex.registry.demo_components import build_demo_registry

    sys.exit(main())
