from typing import List, Optional

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.engine import ChainSearchEngine, score_component
from chatcortex.synthesis.task_specification import TaskSpecification


class SynthesisError(Exception):
    pass


def _violation_raises(constraint: str) -> None:
    raise SynthesisError(f"Constructed agent exceeds {constraint} constraint")


class HeuristicSynthesizer(Synthesizer):
    """
    Deterministic greedy synthesizer (width-1 configuration of the engine)

    Builds the score-optimal single chain. Produces exactly one
    architecture (if feasible) or raises SynthesisError.
    """

    def synthesize(
        self,
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:

        task.validate()
        for capability in task.required_capabilities:
            if not self.registry.get_by_capability(
                capability=capability,
                privacy_constraint=task.privacy_constraint,
            ):
                raise SynthesisError(
                    f"No Components available for capability '{capability}' under given constraints"
                )

        def truncate(expanded, stage_idx, keep_all):
            # Greedy is single-output: the score-argmin is taken at every
            # stage including the final one, so keep_all is ignored.
            expanded.sort(key=lambda element: element[1])
            return expanded[:1]

        engine = ChainSearchEngine(self.registry)
        return engine.run(
            task,
            budget,
            scorer=score_component,
            truncation=truncate,
            on_violation=_violation_raises,
        )
