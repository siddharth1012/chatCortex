from typing import List, Optional

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.engine import ChainSearchEngine
from chatcortex.synthesis.task_specification import TaskSpecification


class ExhaustiveSynthesizer(Synthesizer):
    """
    Exact Pareto frontier via full enumeration

    An untruncated configuration of the shared engine: every stage keeps
    all expansions, so the beam becomes the full Cartesian product of
    candidate components per capability.
    """

    def synthesize(
        self,
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:
        return ChainSearchEngine(self.registry).run(task, budget)
