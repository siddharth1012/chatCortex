
from chatcortex.optimization.pareto import compute_pareto_front
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.exhaustive_synthesizer import ExhaustiveSynthesizer
from chatcortex.synthesis.task_specification import TaskSpecification


def main():
    # Create Registry
    registry = CapabilityRegistry()

    # Register Fake componenets
    registry.register(
        ComponentMetadata(
            name="local_retrieval",
            component_type="tool",
            capabilities=["retrieval"],
            cost_per_call=0.002,
            avg_latency_ms=150,
            reliability_score=0.99,
            privacy_level="internal"
        )
    )

    registry.register(
        ComponentMetadata(
            name="fast_classifier",
            component_type="model",
            capabilities=["classification"],
            cost_per_call=0.003,
            avg_latency_ms=200,
            reliability_score=0.95,
            privacy_level="internal",
        )
    )

    registry.register(
        ComponentMetadata(
            name="strong_generator",
            component_type="model",
            capabilities=["generation"],
            cost_per_call=0.02,
            avg_latency_ms=800,
            reliability_score=0.93,
            privacy_level="external",
        )
    )

    registry.register(
        ComponentMetadata(
            name="cheap_generator",
            component_type="model",
            capabilities=["generation"],
            cost_per_call=0.01,
            avg_latency_ms=600,
            reliability_score=0.85,
            privacy_level="external",
        )
    )

    registry.register(
        ComponentMetadata(
            name="output_verifier",
            component_type="verification",
            capabilities=["verification"],
            cost_per_call=0.005,
            avg_latency_ms=300,
            reliability_score=0.97,
            privacy_level="internal",
        )
    )

    # Define a 3-step task
    task = TaskSpecification(
        required_capabilities=[
            "retrieval",
            "generation",
            "verification",
        ],
        max_cost=0.05,
        max_latency=2000,
        privacy_constraint=None,
        objective_weights={
            "cost": 1.0,
            "latency": 0.001,
            "error": 1.0
        }
    )

    print("\n=== PHASE 2: EXHAUSTIVE SYNTHESIS ===")

    exhautive = ExhaustiveSynthesizer(registry=registry)
    candidates = exhautive.synthesize(task)

    print(f"\n total feasible architectures: {len(candidates)}")

    if not candidates:
        print("No feasible archtectures found")
        return

    pareto_front = compute_pareto_front(candidates)

    print(f"Pareto-optimal architectures: {len(pareto_front)}")

    # Sort pareto front by cost
    pareto_sorted = sorted(pareto_front, key=lambda c: c.total_cost)

    print("\n --- Pareto front (sorted by cost) ---")

    for idx, arch in enumerate(pareto_sorted):
        print(f"Architecture {idx+1}:")
        print(f"  Cost: {arch.total_cost}")
        print(f"  Latency: {arch.total_latency}")
        print(f"  Reliability: {arch.total_reliability}")
        print(f"  Execution Order: {arch.graph.get_execution_order()}")
        print()


if __name__ == "__main__":
    main()
