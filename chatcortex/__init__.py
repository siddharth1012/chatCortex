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

from .graph.agent_graph import AgentGraph
from .loop import LoopIteration, run_closed_loop
from .registry.capability_registry import CapabilityRegistry
from .registry.demo_components import build_demo_registry
from .registry.loader import load_registry
from .registry.metadata import ComponentMetadata
from .registry.package import RegistryReport, save_registry, validate_registry_file
from .synthesis.beam_synthesizer import BeamSynthesizer
from .synthesis.budget import SynthesisBudget
from .synthesis.exhaustive_synthesizer import ExhaustiveSynthesizer
from .synthesis.heuristic_synthesizer import HeuristicSynthesizer
from .synthesis.pareto_partial_beam_synthesizer import ParetoPartialBeamSynthesizer
from .synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from .synthesis.random_synthesizer import RandomSynthesizer
from .synthesis.task_specification import TaskSpecification

__all__ = [
    "TaskSpecification",
    "HeuristicSynthesizer",
    "RandomSynthesizer",
    "BeamSynthesizer",
    "ExhaustiveSynthesizer",
    "ParetoPartialBeamSynthesizer",
    "ProgressiveParetoBeamSynthesizer",
    "SynthesisBudget",
    "AgentGraph",
    "CapabilityRegistry",
    "LoopIteration",
    "ComponentMetadata",
    "build_demo_registry",
    "load_registry",
    "run_closed_loop",
    "save_registry",
    "validate_registry_file",
    "RegistryReport",
]

__version__ = "0.7.0"
