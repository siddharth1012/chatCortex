"""
Experimental synthesizers.

These are research iterations retained for benchmark reproducibility
(see examples/budget_sweep_experiment.py and docs/BENCHMARKS.md).
They are not part of the stable public API and may change or be
removed without notice.

The canonical multi-objective synthesizer is
``chatcortex.ProgressiveParetoBeamSynthesizer``.
"""

from .pareto_partial_beam_synthesizer_v2 import ParetoPartialBeamSynthesizerV2
from .pareto_partial_beam_synthesizer_v3 import ParetoPartialBeamSynthesizerV3
from .system_score_beam_synthesizer import SystemScoreBeamSynthesizer

__all__ = [
    "ParetoPartialBeamSynthesizerV2",
    "ParetoPartialBeamSynthesizerV3",
    "SystemScoreBeamSynthesizer",
]
