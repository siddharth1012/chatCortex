from typing import List, Optional

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.engine import ChainSearchEngine, crowding_truncation
from chatcortex.synthesis.task_specification import TaskSpecification


class ProgressiveParetoBeamSynthesizer(Synthesizer):
    """
    Progressive Beam Widening

    Beam width increases with stage depth
    """

    def __init__(
        self,
        registry,
        base_beam_width: int = 5,
        growth_factor: float = 1.8,
    ):
        super().__init__(registry)
        self.base_beam_width = base_beam_width
        self.growth_factor = growth_factor

    def synthesize(
        self,
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:
        """Per-stage Pareto filtering, a width that widens with depth
        (base * growth_factor ** stage_idx), and Pareto-aware crowding
        truncation on intermediate stages."""
        return ChainSearchEngine(self.registry).run(
            task,
            budget,
            truncation=crowding_truncation(self._stage_width),
        )

    def _stage_width(self, stage_idx: int) -> int:
        return max(1, int(self.base_beam_width * (self.growth_factor ** stage_idx)))
