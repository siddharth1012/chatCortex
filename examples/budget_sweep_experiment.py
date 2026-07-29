from statistics import mean, stdev

from chatcortex.evaluation.approximation_metrics import evaluate_approximation
from chatcortex.synthesis.beam_synthesizer import BeamSynthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.exhaustive_synthesizer import ExhaustiveSynthesizer
from chatcortex.synthesis.heuristic_synthesizer import HeuristicSynthesizer
from chatcortex.synthesis.pareto_partial_beam_synthesizer import ParetoPartialBeamSynthesizer
from chatcortex.synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from chatcortex.synthesis.random_synthesizer import RandomSynthesizer


def run_budget_sweep(registry, task):

    exhaustive = ExhaustiveSynthesizer(registry)
    true_frontier = exhaustive.synthesize(task)

    print(f"True Pareto size: {len(true_frontier)}")

    reference_point = (0.1, 3000, 0.0)

    budgets = [20, 40, 80, 120, 180]
    beam_widths = [3, 5, 7, 10, 15]
    random_seeds = [42, 43, 44, 45, 46]  # Multi-run random baseline

    results = []

    for b in budgets:
        print(f"--- Budget {b} ---")

        greedy_budget = SynthesisBudget(max_evaluations=b, random_seed=42)
        greedy = HeuristicSynthesizer(registry)
        greedy_frontier = greedy.synthesize(task, greedy_budget)

        greedy_report = evaluate_approximation(
            approx_frontier=greedy_frontier,
            true_frontier=true_frontier,
            reference_point=reference_point,
        )

        results.append({
            "algorithm": "Greedy",
            "budget": b,
            "beam_width": None,
            "coverage_mean": greedy_report["coverage"],
            "coverage_std": 0.0,
            "hypervolume_loss_mean": greedy_report["hypervolume_loss"],
            "hypervolume_loss_std": 0.0,
            "avg_cost_regret": greedy_report["avg_cost_regret"],
            "avg_latency_regret": greedy_report["avg_latency_regret"],
            "avg_reliability_regret": greedy_report["avg_reliability_regret"],
        })

        print(
            f"Greedy: "
            f"coverage={greedy_report['coverage']:.2f}, "
            f"hv_loss={greedy_report['hypervolume_loss']:.2f}"
        )

         # --------------------------------------------------------------
        # Random (Multi-run baseline)
        # --------------------------------------------------------------
        random_coverages = []
        random_hv_losses = []
        random_cost_regrets = []
        random_latency_regrets = []
        random_reliability_regrets = []

        for seed in random_seeds:
            random_budget = SynthesisBudget(max_evaluations=b, random_seed=seed)
            random_syn = RandomSynthesizer(registry)

            random_frontier = random_syn.synthesize(task, random_budget)

            report = evaluate_approximation(
                approx_frontier=random_frontier,
                true_frontier=true_frontier,
                reference_point=reference_point,
            )

            random_coverages.append(report["coverage"])
            random_hv_losses.append(report["hypervolume_loss"])
            random_cost_regrets.append(report["avg_cost_regret"])
            random_latency_regrets.append(report["avg_latency_regret"])
            random_reliability_regrets.append(report["avg_reliability_regret"])

        results.append({
            "algorithm": "Random",
            "budget": b,
            "beam_width": None,
            "coverage_mean": mean(random_coverages),
            "coverage_std": stdev(random_coverages),
            "hypervolume_loss_mean": mean(random_hv_losses),
            "hypervolume_loss_std": stdev(random_hv_losses),
            "avg_cost_regret": mean(random_cost_regrets),
            "avg_latency_regret": mean(random_latency_regrets),
            "avg_reliability_regret": mean(random_reliability_regrets),
        })

        print(
            f"Random: "
            f"coverage={mean(random_coverages):.2f} ± {stdev(random_coverages):.2f}, "
            f"hv_loss={mean(random_hv_losses):.2f}"
        )

        # --------------------------------------------------------------
        # Beam Comparison (Scalar vs ParetoPartial vs Progressive)
        # --------------------------------------------------------------
        for w in beam_widths:

            # -------------------------
            # Scalar Beam (Phase 3)
            # -------------------------
            scalar_budget = SynthesisBudget(max_evaluations=b, random_seed=42)
            scalar_beam = BeamSynthesizer(registry, beam_width=w)
            scalar_frontier = scalar_beam.synthesize(task, scalar_budget)

            scalar_report = evaluate_approximation(
                approx_frontier=scalar_frontier,
                true_frontier=true_frontier,
                reference_point=reference_point,
            )

            results.append({
                "algorithm": "ScalarBeam",
                "budget": b,
                "beam_width": w,
                "coverage_mean": scalar_report["coverage"],
                "coverage_std": 0.0,
                "hypervolume_loss_mean": scalar_report["hypervolume_loss"],
                "hypervolume_loss_std": 0.0,
                "avg_cost_regret": scalar_report["avg_cost_regret"],
                "avg_latency_regret": scalar_report["avg_latency_regret"],
                "avg_reliability_regret": scalar_report["avg_reliability_regret"],
            })

            print(
                f"ScalarBeam (w={w}): "
                f"coverage={scalar_report['coverage']:.2f}, "
                f"hv_loss={scalar_report['hypervolume_loss']:.2f}"
            )

            # -------------------------
            # ParetoPartial Beam (Phase 4B)
            # -------------------------
            pareto_budget = SynthesisBudget(max_evaluations=b, random_seed=42)
            pareto_beam = ParetoPartialBeamSynthesizer(registry, beam_width=w)
            pareto_frontier = pareto_beam.synthesize(task, pareto_budget)

            pareto_report = evaluate_approximation(
                approx_frontier=pareto_frontier,
                true_frontier=true_frontier,
                reference_point=reference_point,
            )

            results.append({
                "algorithm": "ParetoPartialBeam",
                "budget": b,
                "beam_width": w,
                "coverage_mean": pareto_report["coverage"],
                "coverage_std": 0.0,
                "hypervolume_loss_mean": pareto_report["hypervolume_loss"],
                "hypervolume_loss_std": 0.0,
                "avg_cost_regret": pareto_report["avg_cost_regret"],
                "avg_latency_regret": pareto_report["avg_latency_regret"],
                "avg_reliability_regret": pareto_report["avg_reliability_regret"],
            })

            print(
                f"ParetoPartialBeam (w={w}): "
                f"coverage={pareto_report['coverage']:.2f}, "
                f"hv_loss={pareto_report['hypervolume_loss']:.2f}"
            )

            # Progressive synthesizer
            progressive = ProgressiveParetoBeamSynthesizer(
                registry,
                base_beam_width=w,
                growth_factor=2.0,
            )

            progressive_budget = SynthesisBudget(max_evaluations=b, random_seed=42)
            progressive_frontier = progressive.synthesize(task, progressive_budget)
            progressive_report = evaluate_approximation(
                approx_frontier=progressive_frontier,
                true_frontier=true_frontier,
                reference_point=reference_point,
            )

            results.append({
                "algorithm": "ProgressiveParetoBeam",
                "budget": b,
                "beam_width": w,
                "coverage_mean": progressive_report["coverage"],
                "coverage_std": 0.0,
                "hypervolume_loss_mean": progressive_report["hypervolume_loss"],
                "hypervolume_loss_std": 0.0,
                "avg_cost_regret": progressive_report["avg_cost_regret"],
                "avg_latency_regret": progressive_report["avg_latency_regret"],
                "avg_reliability_regret": progressive_report["avg_reliability_regret"],
            })

            print(
                f"ProgressiveParetoBeam (w={w}): "
                f"coverage={progressive_report['coverage']:.2f}, "
                f"hv_loss={progressive_report['hypervolume_loss']:.2f}"
            )

    return results


if __name__ == "__main__":
    from chatcortex import TaskSpecification, build_demo_registry

    capabilities = (
        "retrieval",
        "generation",
        "verification",
        "refinement",
        "postprocessing",
    )
    registry = build_demo_registry(capabilities=capabilities)
    task = TaskSpecification(required_capabilities=list(capabilities))
    run_budget_sweep(registry, task)
