from abc import ABC, abstractmethod
from typing import List, Optional

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.task_specification import TaskSpecification


class Synthesizer(ABC):
    """
    Abstract Base Class for all synthesis strategies

    Define the formal contract:
        S_B(Task) -> Approximate Pareto Set

    Where: 
        - Task Specification defines constraints and objectives
        - SynthesisBudget constrains computational resources
        - Output is a list of ArchitectureCandidate objects
    """

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
    
    @abstractmethod
    def synthesize(
        self,
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:
        """
        Perform synthesis under optional computational budget

        returns:
            - A list of ArchitectureCandidate objects representing 
            an approximate or exact pareto frontier
        """
        pass