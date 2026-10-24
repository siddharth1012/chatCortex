from typing import List

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.engine import ChainSearchEngine, score_component
from chatcortex.synthesis.task_specification import TaskSpecification


class BeamSynthesizer(Synthesizer):
    """
    Budget-aware beam search synthesizer

    Maintains top-k partial architectures per stage, ranked by the
    cumulative weighted component score.
    """

    def __init__(self, registry, beam_width: int = 3):
        super().__init__(registry)
        self.beam_width = beam_width

    def synthesize(
        self,
        task: TaskSpecification,
        budget=None,
    ) -> List[ArchitectureCandidate]:
        """Beam search keeping the ``beam_width`` best-scored partials per
        stage; the final stage keeps every expansion before Pareto filtering."""

        def truncate(expanded, stage_idx, keep_all):
            expanded.sort(key=lambda element: element[1])
            return expanded if keep_all else expanded[: self.beam_width]

        engine = ChainSearchEngine(self.registry)
        return engine.run(task, budget, scorer=score_component, truncation=truncate)
