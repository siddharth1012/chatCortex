from chatcortex.execution.executor import AgentExecutor
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.synthesis.heuristic_synthesizer import HeuristicSynthesizer
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

    # Run synthesizer
    synthesizer = HeuristicSynthesizer(registry)
    candidates = synthesizer.synthesize(task)
    agent_graph = candidates[0].graph

    # Print results
    print("\n=== Synthesize Agent ===")

    print("Execution Order:")

    for node in agent_graph.get_execution_order():
        print(" -", node)

    print("\nTotal Cost: ", round(agent_graph.total_cost(), 5))
    print("Total Latency (ms): ", agent_graph.total_latency())
    print("Aggregate Reliability:", round(agent_graph.aggregate_reliability(), 5))

    print("\n=== DETERMINISTIC EXECUTION ===")
    deterministic_executor = AgentExecutor(mode="deterministic")
    deterministic_result = deterministic_executor.execute(agent_graph)
    print(deterministic_result.summary())

    print("\n=== PROBABILISTIC EXECUTION (5 RUNS, FIXED SEED) ===")
    prob_executor = AgentExecutor(mode="probabilistic", seed=42)

    for i in range(50):
        prob_executor.execute(agent_graph)
        # print(f"Run {i+1}: {result_prob.summary()}")

    from chatcortex.evaluation.harness import EvaluationHarness

    tasks = {
        "basic_task": task
    }

    synthesizers = {
        "heuristic": synthesizer
    }

    harness = EvaluationHarness(
        tasks=tasks,
        synthesizers=synthesizers,
        runs_per_experiment=20,
        execution_mode="probabilistic",
        base_seed=42,
    )

    results = harness.run()

    for res in results:
        print(res.aggregate())

if __name__ == "__main__":
    main()
