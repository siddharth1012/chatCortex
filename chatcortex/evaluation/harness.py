from typing import Dict, List, Optional

from chatcortex.execution.executor import AgentExecutor
from chatcortex.synthesis.heuristic_synthesizer import HeuristicSynthesizer
from chatcortex.synthesis.task_specification import TaskSpecification


class EvaluationResult:
    """
    Stores aggregated results for a (task, synthesizer) pair
    """

    def __init__(self, task_name: str, synthesizer_name: str):
        self.task_name = task_name
        self.synthesizer_name = synthesizer_name
        self.runs = []
    
    def add_run(self, summary: Dict):
        self.runs.append(summary)
    
    def aggregate(self) -> Dict:
        total_runs = len(self.runs)

        avg_cost = sum(r["total_cost"] for r in self.runs) / total_runs
        avg_latency = sum(r["total_latency"] for r in self.runs) / total_runs
        success_rate = sum(1 for r in self.runs if r["success"]) / total_runs

        return {
            "task": self.task_name,
            "synthesizer": self.synthesizer_name,
            "runs": total_runs,
            "avg_cost": avg_cost,
            "avg_latency": avg_latency,
            "success_rate": success_rate,
        }
    

class EvaluationHarness:
    """
    Orchestrates synthesis + execution experiments
    """

    def __init__(
        self,
        tasks: Dict[str, TaskSpecification],
        synthesizers: Dict[str, HeuristicSynthesizer],
        runs_per_experiment: int = 10,
        execution_mode: str = "probabilistic",
        base_seed: Optional[int] = 42,
    ):
        self.tasks = tasks
        self.synthesizers = synthesizers
        self.runs_per_experiment = runs_per_experiment
        self.execution_mode = execution_mode
        self.base_seed = base_seed

    def run(self) -> List[EvaluationResult]:
        
        results = []

        for task_name, task in self.tasks.items():
            for synthesizer_name, synthesizer in self.synthesizers.items():

                result = EvaluationResult(task_name, synthesizer_name)

                # Build architecture once per experiment
                graph = synthesizer.synthesize(task)

                for run_idx in range(self.runs_per_experiment):
                    
                    # Create deterministic but varying seeds
                    seed = None
                    if self.base_seed is not None:
                        seed = self.base_seed + run_idx
                    
                    executor = AgentExecutor(
                        mode=self.execution_mode,
                        seed=seed
                    )

                    execution_result = executor.execute(graph)
                    summary = execution_result.summary()

                    result.add_run(summary)
                
                results.append(result)
        
        return results


