"""
ChatCortex

Framework for automated synthesis and optimization of AI agent
architectures under multi-objective constraints.

Core capabilities include:

- Task specification
- Agent architecture synthesis
- Multi-objective optimization
- Pareto frontier analysis
"""

from .synthesis.task_specification import TaskSpecification

from .synthesis.heuristic_synthesizer import HeuristicSynthesizer
from .synthesis.random_synthesizer import RandomSynthesizer
from .synthesis.beam_synthesizer import BeamSynthesizer
from .synthesis.exhaustive_synthesizer import ExhaustiveSynthesizer

from .graph.agent_graph import AgentGraph

__all__ = [
    "TaskSpecification",
    "HeuristicSynthesizer",
    "RandomSynthesizer",
    "BeamSynthesizer",
    "ExhaustiveSynthesizer",
    "AgentGraph",
]

__version__ = "0.4.0"