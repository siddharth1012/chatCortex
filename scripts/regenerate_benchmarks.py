"""Regenerate the measured benchmark tables checked into docs/BENCHMARKS.md.

Builds the 5-stage demo registry (4 tradeoff variants per capability, so
4**5 = 1024 candidate chains), computes the exact Pareto frontier with
the exhaustive synthesizer, then measures coverage, hypervolume loss,
and regret for the shipped strategies:

* Budget sweep : Greedy, Random (5 seeds), and the three beam
  synthesizers at a reference beam width, for budgets 20..180.
* Width sweep  : the three beam synthesizers for widths 3..15 at a
  fixed evaluation budget.

Per-run reports are identical to ``evaluate_approximation`` except that
the true frontier's hypervolume (same sampler, sample count, and seed)
is precomputed once and reused - halving the Monte Carlo cost without
changing any number.

Outputs JSON (``--out``, default ``docs/benchmark_results.json``) and
Markdown tables for the documentation. Run from the repo root:

    python scripts/regenerate_benchmarks.py
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import chatcortex
from chatcortex import TaskSpecification, build_demo_registry
from chatcortex.evaluation.approximation_metrics import (
    average_regret,
    compute_coverage,
)
from chatcortex.optimization.pareto import hypervolume_monte_carlo
from chatcortex.synthesis.beam_synthesizer import BeamSynthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.exhaustive_synthesizer import ExhaustiveSynthesizer
from chatcortex.synthesis.heuristic_synthesizer import HeuristicSynthesizer
from chatcortex.synthesis.pareto_partial_beam_synthesizer import (
    ParetoPartialBeamSynthesizer,
)
from chatcortex.synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from chatcortex.synthesis.random_synthesizer import RandomSynthesizer


def make_report(approx_frontier, true_frontier, reference_point, samples, hv_true):
    """Same numbers as ``evaluate_approximation``, with ``hv_true`` precomputed."""
    hv_approx = hypervolume_monte_carlo(
        approx_frontier, reference_point, num_samples=samples, seed=42
    )
    cost_r, latency_r, reliability_r = average_regret(approx_frontier, true_frontier)
    return {
        "coverage": compute_coverage(approx_frontier, true_frontier),
        "hypervolume_loss": max(0.0, hv_true - hv_approx),
        "avg_cost_regret": cost_r,
        "avg_latency_regret": latency_r,
        "avg_reliability_regret": reliability_r,
    }


def beam_algorithms(registry, width):
    """The three beam synthesizers at a given width, under their shorthands."""
    return (
        ("ScalarBeam", BeamSynthesizer(registry, beam_width=width)),
        ("ParetoPartialBeam", ParetoPartialBeamSynthesizer(registry, beam_width=width)),
        (
            "ProgressiveParetoBeam",
            ProgressiveParetoBeamSynthesizer(
                registry, base_beam_width=width, growth_factor=2.0
            ),
        ),
    )


def _row(algorithm, budget, beam_width, coverage, hypervolume_loss,
         avg_cost_regret, avg_latency_regret, avg_reliability_regret,
         coverage_std=None):
    return {
        "algorithm": algorithm,
        "budget": budget,
        "beam_width": beam_width,
        "coverage": coverage,
        "coverage_std": coverage_std if coverage_std is not None else 0.0,
        "hypervolume_loss": hypervolume_loss,
        "avg_cost_regret": avg_cost_regret,
        "avg_latency_regret": avg_latency_regret,
        "avg_reliability_regret": avg_reliability_regret,
    }


def run_budget_sweep(registry, task, true_frontier, reference_point, hv_true, args):
    """Greedy / Random(5 seeds) / beams(reference width) x budget sweep."""
    rows = []
    for budget in args.budgets:
        print(f"--- budget {budget} ---", flush=True)

        greedy = make_report(
            HeuristicSynthesizer(registry).synthesize(
                task, SynthesisBudget(max_evaluations=budget, random_seed=42)
            ),
            true_frontier, reference_point, args.samples, hv_true,
        )
        rows.append(_row("Greedy", budget, None, **greedy))

        random_reports = [
            make_report(
                RandomSynthesizer(registry).synthesize(
                    task, SynthesisBudget(max_evaluations=budget, random_seed=seed)
                ),
                true_frontier, reference_point, args.samples, hv_true,
            )
            for seed in args.seeds
        ]
        random_keys = ("hypervolume_loss", "avg_cost_regret",
                       "avg_latency_regret", "avg_reliability_regret")
        rows.append(_row(
            "Random", budget, None,
            coverage=statistics.mean(r["coverage"] for r in random_reports),
            coverage_std=statistics.stdev(r["coverage"] for r in random_reports),
            **{key: statistics.mean(r[key] for r in random_reports)
               for key in random_keys},
        ))

        for name, synthesizer in beam_algorithms(registry, args.reference_width):
            report = make_report(
                synthesizer.synthesize(
                    task, SynthesisBudget(max_evaluations=budget, random_seed=42)
                ),
                true_frontier, reference_point, args.samples, hv_true,
            )
            rows.append(_row(name, budget, args.reference_width, **report))
    return rows


def run_width_sweep(registry, task, true_frontier, reference_point, hv_true, args):
    """The three beam synthesizers x beam-width sweep at a fixed budget."""
    rows = []
    for width in args.widths:
        print(f"--- beam width {width} (budget {args.width_budget}) ---", flush=True)
        for name, synthesizer in beam_algorithms(registry, width):
            report = make_report(
                synthesizer.synthesize(
                    task, SynthesisBudget(max_evaluations=args.width_budget, random_seed=42)
                ),
                true_frontier, reference_point, args.samples, hv_true,
            )
            rows.append(_row(name, args.width_budget, width, **report))
    return rows


def metric_table(rows, column_key, metric, title, note):
    """Markdown table: one row per sweep algorithm, one column per value
    of ``column_key``, cell = the ``metric`` (with std for Random coverage)."""
    columns = sorted({row[column_key] for row in rows})
    algorithms = []
    for row in rows:
        if row["algorithm"] not in algorithms:
            algorithms.append(row["algorithm"])

    lines = [f"### {title}", "", note, ""]
    lines.append("| Algorithm | " + " | ".join(str(c) for c in columns) + " |")
    lines.append("|" + "---|" * (len(columns) + 1))
    for algorithm in algorithms:
        cells = []
        for column in columns:
            matching = [
                r for r in rows
                if r["algorithm"] == algorithm and r[column_key] == column
            ]
            if not matching:
                cells.append("-")
                continue
            row = matching[0]
            if metric == "coverage" and row["algorithm"] == "Random" and row["coverage_std"]:
                cells.append(
                    f"{row['coverage']:.2f} +/- {row['coverage_std']:.2f}"
                )
            elif metric == "coverage":
                cells.append(f"{row['coverage']:.2f}")
            else:
                cells.append(f"{row[metric]:.4f}")
        lines.append("| " + algorithm + " | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the measured benchmark tables in the docs."
    )
    parser.add_argument("--out", default="docs/benchmark_results.json")
    parser.add_argument("--budgets", default="20,40,80,120,180", type=str)
    parser.add_argument("--widths", default="3,5,7,10,15", type=str)
    parser.add_argument("--reference-width", default=7, type=int)
    parser.add_argument("--width-budget", default=120, type=int)
    parser.add_argument(
        "--samples", default=200000, type=int,
        help="Monte Carlo samples per hypervolume estimate",
    )
    args = parser.parse_args()
    args.budgets = [int(b) for b in args.budgets.split(",")]
    args.widths = [int(w) for w in args.widths.split(",")]
    args.seeds = [42, 43, 44, 45, 46]

    capabilities = (
        "retrieval", "generation", "verification", "refinement", "postprocessing",
    )
    registry = build_demo_registry(capabilities=capabilities)
    task = TaskSpecification(required_capabilities=list(capabilities))
    reference_point = (0.1, 3000, 0.0)

    print("Computing the exact Pareto frontier...", flush=True)
    true_frontier = ExhaustiveSynthesizer(registry).synthesize(task)
    print(f"Exact frontier size: {len(true_frontier)}", flush=True)

    print(f"Monte Carlo hypervolume of the true frontier ({args.samples} samples)...",
          flush=True)
    started = time.perf_counter()
    hv_true = hypervolume_monte_carlo(
        true_frontier, reference_point, num_samples=args.samples, seed=42
    )
    print(f"Done in {time.perf_counter() - started:.1f}s", flush=True)

    budget_rows = run_budget_sweep(
        registry, task, true_frontier, reference_point, hv_true, args
    )
    width_rows = run_width_sweep(
        registry, task, true_frontier, reference_point, hv_true, args
    )

    payload = {
        "meta": {
            "chatcortex_version": chatcortex.__version__,
            "python": platform.python_version(),
            "stage_setup": "5-stage demo registry, 4 variants per capability "
                           "(4**5 = 1024 candidate chains)",
            "exact_frontier_size": len(true_frontier),
            "reference_point": list(reference_point),
            "monte_carlo_samples": args.samples,
            "regeneration": "python scripts/regenerate_benchmarks.py",
            "budget_sweep": {
                "budgets": args.budgets,
                "beam_width": args.reference_width,
                "seeds": args.seeds,
            },
            "width_sweep": {
                "widths": args.widths,
                "budget": args.width_budget,
            },
        },
        "budget_sweep": budget_rows,
        "width_sweep": width_rows,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    print("\n<!--COPY BELOW INTO docs/BENCHMARKS.md-->")
    print("\n".join(metric_table(
        budget_rows, "budget", "coverage",
        "Coverage by evaluation budget (beam width "
        f"{args.reference_width}, 5-stage demo registry)",
        "Coverage of the exact Pareto frontier under a bounded evaluation "
        "budget; `Random` cells are mean +/- std over 5 seeds (42-46).",
    )))
    print("\n".join(metric_table(
        budget_rows, "budget", "hypervolume_loss",
        "Mean hypervolume loss by evaluation budget (beam width "
        f"{args.reference_width})",
        "Monte Carlo hypervolume loss against the exact frontier "
        f"({args.samples} samples, seed 42); `Random` cells are means over 5 seeds.",
    )))
    print("\n".join(metric_table(
        width_rows, "beam_width", "coverage",
        f"Coverage by beam width (evaluation budget {args.width_budget})",
        "Coverage of the exact Pareto frontier against a fixed evaluation budget.",
    )))
    print("\n".join(metric_table(
        width_rows, "beam_width", "hypervolume_loss",
        f"Mean hypervolume loss by beam width (evaluation budget {args.width_budget})",
        "Monte Carlo hypervolume loss against the exact frontier "
        f"({args.samples} samples, seed 42).",
    )))
    print("<!--COPY ABOVE INTO docs/BENCHMARKS.md-->")


if __name__ == "__main__":
    main()
