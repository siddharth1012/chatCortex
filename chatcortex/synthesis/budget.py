from dataclasses import dataclass
import time
from typing import Optional


@dataclass(frozen=True)
class SynthesisBudget:
    """
    Hard constraints on the synthesis process itself

    max_evaluation: Maximum number of architecture evaluations allowed
    max_time_seconds: Maximum clock time allowed for synthesis
    random_seed: Optional deterministic seed for stochastic synthesizers
    """
    max_evaluations: Optional[int] = None
    max_time_seconds: Optional[float] = None
    random_seed: Optional[int] = None


class BudgetExceeded(Exception):
    """
    Raised when synthesis budget is exhausted
    """ 
    pass


class SynthesisContext:
    """
    Tracks evaluation count and time usage during synthesis
    All synthesizers must use this to enforce hard budget limits
    """

    def __init__(self, budget: Optional[SynthesisBudget] = None):
        self.budget = budget
        self.evaluations: int = 0
        self.start_time: float = time.time()
    
    def _check_evaluation_limit(self):
        if (
            self.budget
            and self.budget.max_evaluations is not None
            and self.evaluations >= self.budget.max_evaluations
        ):
            raise BudgetExceeded("Maximum evaluation budget reached")
    
    def _check_time_limit(self):
        if (
            self.budget
            and self.budget.max_time_seconds is not None
            and (time.time() - self.start_time) >= self.budget.max_time_seconds
        ):
            raise BudgetExceeded("Maximum time budget reached")
    
    def can_evaluate(self) -> bool:
        try:
            self._check_evaluation_limit()
            self._check_time_limit()
            return True
        except BudgetExceeded:
            return False
    
    def register_evaluation(self):
        """
        Must be called every time an architecture is evaluated
        """
        self._check_evaluation_limit()
        self._check_time_limit()
        self.evaluations += 1