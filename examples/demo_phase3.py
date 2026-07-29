from budget_sweep_experiment import run_budget_sweep

from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.task_specification import TaskSpecification


def main():

    registry = CapabilityRegistry()

    # -------------------------------------------------------
    # Structured Retrieval Models
    # -------------------------------------------------------

    # registry.register(ComponentMetadata(
    #     name="cheap_retrieval",
    #     component_type="tool",
    #     capabilities=["retrieval"],
    #     cost_per_call=0.001,
    #     avg_latency_ms=100,
    #     reliability_score=0.85,
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="balanced_retrieval",
    #     component_type="tool",
    #     capabilities=["retrieval"],
    #     cost_per_call=0.003,
    #     avg_latency_ms=200,
    #     reliability_score=0.93,
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="strong_retrieval",
    #     component_type="tool",
    #     capabilities=["retrieval"],
    #     cost_per_call=0.006,
    #     avg_latency_ms=400,
    #     reliability_score=0.98,
    #     privacy_level="internal",
    # ))


    # -------------------------------------------------------
    # Retrieval (Adversarial Tradeoffs)
    # -------------------------------------------------------

    registry.register(ComponentMetadata(
        name="retrieval_speedster",
        component_type="tool",
        capabilities=["retrieval"],
        cost_per_call=0.004,
        avg_latency_ms=60,
        reliability_score=0.78,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="retrieval_budget",
        component_type="tool",
        capabilities=["retrieval"],
        cost_per_call=0.0008,
        avg_latency_ms=420,
        reliability_score=0.82,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="retrieval_reliable",
        component_type="tool",
        capabilities=["retrieval"],
        cost_per_call=0.009,
        avg_latency_ms=500,
        reliability_score=1.05,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="retrieval_chaotic",
        component_type="tool",
        capabilities=["retrieval"],
        cost_per_call=0.006,
        avg_latency_ms=110,
        reliability_score=0.92,
        privacy_level="internal",
    ))

    # -------------------------------------------------------
    # Structured Generators
    # -------------------------------------------------------

    # registry.register(ComponentMetadata(
    #     name="tiny_generator",
    #     component_type="model",
    #     capabilities=["generation"],
    #     cost_per_call=0.005,
    #     avg_latency_ms=200,
    #     reliability_score=0.80,
    #     privacy_level="external",
    # ))

    # registry.register(ComponentMetadata(
    #     name="small_generator",
    #     component_type="model",
    #     capabilities=["generation"],
    #     cost_per_call=0.01,
    #     avg_latency_ms=400,
    #     reliability_score=0.88,
    #     privacy_level="external",
    # ))

    # registry.register(ComponentMetadata(
    #     name="large_generator",
    #     component_type="model",
    #     capabilities=["generation"],
    #     cost_per_call=0.02,
    #     avg_latency_ms=700,
    #     reliability_score=0.93,
    #     privacy_level="external",
    # ))

    # registry.register(ComponentMetadata(
    #     name="ultra_generator",
    #     component_type="model",
    #     capabilities=["generation"],
    #     cost_per_call=0.04,
    #     avg_latency_ms=1200,
    #     reliability_score=0.97,
    #     privacy_level="external",
    # ))

    # -------------------------------------------------------
    # Generation (Strong Conflicts)
    # -------------------------------------------------------

    registry.register(ComponentMetadata(
        name="gen_fast_unstable",
        component_type="model",
        capabilities=["generation"],
        cost_per_call=0.006,
        avg_latency_ms=120,
        reliability_score=0.70,
        privacy_level="external",
    ))

    registry.register(ComponentMetadata(
        name="gen_cheap_slow",
        component_type="model",
        capabilities=["generation"],
        cost_per_call=0.002,
        avg_latency_ms=900,
        reliability_score=0.84,
        privacy_level="external",
    ))

    registry.register(ComponentMetadata(
        name="gen_high_precision",
        component_type="model",
        capabilities=["generation"],
        cost_per_call=0.025,
        avg_latency_ms=800,
        reliability_score=1.12,
        privacy_level="external",
    ))

    registry.register(ComponentMetadata(
        name="gen_weird_tradeoff",
        component_type="model",
        capabilities=["generation"],
        cost_per_call=0.018,
        avg_latency_ms=150,
        reliability_score=0.88,
        privacy_level="external",
    ))

    # -------------------------------------------------------
    # Structured Verifiers
    # -------------------------------------------------------

    # registry.register(ComponentMetadata(
    #     name="fast_verifier",
    #     component_type="verification",
    #     capabilities=["verification"],
    #     cost_per_call=0.003,
    #     avg_latency_ms=200,
    #     reliability_score=0.90,
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="strict_verifier",
    #     component_type="verification",
    #     capabilities=["verification"],
    #     cost_per_call=0.008,
    #     avg_latency_ms=500,
    #     reliability_score=0.96,
    #     privacy_level="internal",
    # ))


    # -------------------------------------------------------
    # Verification
    # -------------------------------------------------------

    registry.register(ComponentMetadata(
        name="verify_minimal",
        component_type="verification",
        capabilities=["verification"],
        cost_per_call=0.001,
        avg_latency_ms=150,
        reliability_score=0.80,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="verify_balanced",
        component_type="verification",
        capabilities=["verification"],
        cost_per_call=0.006,
        avg_latency_ms=450,
        reliability_score=0.98,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="verify_extreme",
        component_type="verification",
        capabilities=["verification"],
        cost_per_call=0.015,
        avg_latency_ms=1000,
        reliability_score=1.20,
        privacy_level="internal",
    ))

    # -------------------------------------------------------
    # Refinement Stage (New – Enables 72 Architectures)
    # -------------------------------------------------------

    # registry.register(ComponentMetadata(
    #     name="no_refinement",
    #     component_type="refinement",
    #     capabilities=["refinement"],
    #     cost_per_call=0.0,
    #     avg_latency_ms=0,
    #     reliability_score=1.00,
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="fast_refine",
    #     component_type="refinement",
    #     capabilities=["refinement"],
    #     cost_per_call=0.004,        # asymmetric moderate cost
    #     avg_latency_ms=150,
    #     reliability_score=1.12,     # multiplicative boost
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="robust_refine",
    #     component_type="refinement",
    #     capabilities=["refinement"],
    #     cost_per_call=0.012,
    #     avg_latency_ms=350,
    #     reliability_score=1.22,
    #     privacy_level="internal",
    # ))

    # -------------------------------------------------------
    # Refinement (Geometry Distortion)
    # -------------------------------------------------------

    registry.register(ComponentMetadata(
        name="no_refine",
        component_type="refinement",
        capabilities=["refinement"],
        cost_per_call=0.0,
        avg_latency_ms=0,
        reliability_score=1.0,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="refine_fast_boost",
        component_type="refinement",
        capabilities=["refinement"],
        cost_per_call=0.005,
        avg_latency_ms=120,
        reliability_score=1.15,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="refine_heavy_boost",
        component_type="refinement",
        capabilities=["refinement"],
        cost_per_call=0.018,
        avg_latency_ms=600,
        reliability_score=1.35,
        privacy_level="internal",
    ))

    # -------------------------------------------------------
    # Post-Processing / Formatting (Decorrelated Objectives)
    # -------------------------------------------------------

    # registry.register(ComponentMetadata(
    #     name="fast_unstable_formatter",
    #     component_type="postprocess",
    #     capabilities=["postprocess"],
    #     cost_per_call=0.002,      # very cheap
    #     avg_latency_ms=100,       # fast
    #     reliability_score=0.85,   # low reliability
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="cheap_high_reliability_formatter",
    #     component_type="postprocess",
    #     capabilities=["postprocess"],
    #     cost_per_call=0.003,      # still cheap
    #     avg_latency_ms=350,       # slow
    #     reliability_score=1.08,   # very reliable
    #     privacy_level="internal",
    # ))

    # registry.register(ComponentMetadata(
    #     name="expensive_fast_low_reliability_formatter",
    #     component_type="postprocess",
    #     capabilities=["postprocess"],
    #     cost_per_call=0.012,      # expensive
    #     avg_latency_ms=80,        # very fast
    #     reliability_score=0.88,   # mediocre reliability
    #     privacy_level="internal",
    # ))

    # -------------------------------------------------------
    # Postprocess (Adversarial Final Stage)
    # -------------------------------------------------------

    registry.register(ComponentMetadata(
        name="post_fast_fragile",
        component_type="postprocess",
        capabilities=["postprocess"],
        cost_per_call=0.003,
        avg_latency_ms=70,
        reliability_score=0.82,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="post_slow_precise",
        component_type="postprocess",
        capabilities=["postprocess"],
        cost_per_call=0.010,
        avg_latency_ms=800,
        reliability_score=1.18,
        privacy_level="internal",
    ))

    registry.register(ComponentMetadata(
        name="post_expensive_low_gain",
        component_type="postprocess",
        capabilities=["postprocess"],
        cost_per_call=0.020,
        avg_latency_ms=200,
        reliability_score=0.90,
        privacy_level="internal",
    ))

    task = TaskSpecification(
        required_capabilities=[
            "retrieval",
            "generation",
            "verification",
            "refinement",
            "postprocess",
        ],
        max_cost=0.05,
        max_latency=3000,
        privacy_constraint=None,
        objective_weights={
            "cost": 1.0,
            "latency": 0.001,
            "error": 1.0,
        },
    )

    # budget = SynthesisBudget(
    #     max_evaluations=10,
    #     random_seed=42,
    # )

    # print("\n=== GREEDY ===")
    # greedy = HeuristicSynthesizer(registry)
    # greedy_frontier = greedy.synthesize(task, budget)

    # print("\n=== BEAM (width=2) ===")
    # beam = BeamSynthesizer(registry, beam_width=2)
    # beam_frontier = beam.synthesize(task, budget)

    # print("\n=== RANDOM ===")
    # random_syn = RandomSynthesizer(registry)
    # random_frontier = random_syn.synthesize(task, budget)

    # reference_point = (1.0, 5000.0, 0.0)

    # print("\n=== EVALUATION REPORTS ===")

    # for name, frontier in [
    #     ("Greedy", greedy_frontier),
    #     ("Beam", beam_frontier),
    #     ("Random", random_frontier),
    # ]:

    #     report = evaluate_approximation(
    #         approx_frontier=frontier,
    #         true_frontier=true_frontier,
    #         reference_point=reference_point,
    #     )

    #     print(f"\n{name} Results:")
    #     for k, v in report.items():
    #         print(f"  {k}: {v}")

    print("\n=== DONE ===\n")

    run_budget_sweep(registry, task)


if __name__ == "__main__":
    main()
