import random
from typing import Iterable, List, Set, Tuple

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate


def dominates(a: ArchitectureCandidate, b: ArchitectureCandidate) -> bool:
    """
    Returns True if architecture a dominates architecture b
    Objectives:
        - Minimize total_cost
        - Minimize total_latency
        - Maximize total_reliability
    """

    better_or_equal_all = (
        a.total_cost <= b.total_cost
        and a.total_latency <= b.total_latency
        and a.total_reliability >= b.total_reliability
    )

    strictly_better_at_least_one = (
        a.total_cost < b.total_cost
        or a.total_latency < b.total_latency
        or a.total_reliability > b.total_reliability
    )

    return better_or_equal_all and strictly_better_at_least_one


def compute_pareto_front(
    candidates: List[ArchitectureCandidate],
) -> List[ArchitectureCandidate]:
    """
    Exact O(n^2) Pareto front computation.
    """
    
    pareto = []

    for candidate in candidates:
        dominated = False

        for other in candidates:
            if other is not candidate and dominates(other, candidate):
                dominated = True
                break
        
        if not dominated:
            pareto.append(candidate)
    
    return pareto


class ParetoSet:
    """
    Maintains a non-dominated architecture set incrementally

    Used by:
        - ExhaustiveSynthesizer (Optional)
        - BeamSynthesizer
        - EvolutionarySynthesizer
        - Budget-aware Synthesis
    """

    def __init__(self):
        self._set: Set[ArchitectureCandidate] = set()

    def __iter__(self):
        return iter(self._set)
    
    def __len__(self):
        return len(self._set)
    
    def as_set(self) -> Set[ArchitectureCandidate]:
        return set(self._set)
    
    def add(self, candidate: ArchitectureCandidate) -> bool:
        """
        Adds candidate if non-dominated
        Removes any member dominated by others

        Returns:
            - True if candidate is added,
            - False if candidate was dominated
        """ 

        to_remove = []

        for existing in self._set:
            if dominates(existing, candidate):
                return False # candidate dominated by existing
            if dominates(candidate, existing):
                to_remove.append(existing)
        
        for item in to_remove:
            self._set.remove(item)
        
        self._set.add(candidate)

        return True


def frontier_coverage(
    approx_frontier: Iterable[ArchitectureCandidate],
    true_frontier: Iterable[ArchitectureCandidate],
) -> float:
    """
    Metric based frontier coverage
    Coverage is computed over objective tuples:
        - (total_cost, total_latency, total_reliability)
    
    This avoids dependency on object identity or graph instance equality
    """

    true_metrics = {
        (c.total_cost, c.total_latency, c.total_reliability)
        for c in true_frontier
    }

    approx_metrics = {
        (c.total_cost, c.total_latency, c.total_reliability)
        for c in approx_frontier
    }

    if not true_metrics:
        return 1.0
    
    return len(true_metrics & approx_metrics) / len(true_metrics)

def dominance_rank(
    candidate: ArchitectureCandidate,
    population: Iterable[ArchitectureCandidate],
) -> int:
    """
    Computes dominance rank of candidate within population

    Rank 1 -> Non-dominated (Pareto)
    Rank 2 -> Dominated only by rank 1

    O(n^2) layered ranking 
    """

    remaining = list(population)
    rank = 1

    while remaining:
        current_front = compute_pareto_front(remaining)

        if candidate in current_front: 
            return rank
        
        remaining = [c for c in remaining if c not in current_front]
        rank += 1

    return rank

def hypervolume_monte_carlo(
    frontier: Iterable[ArchitectureCandidate],
    reference_point: Tuple[float, float, float],
    num_samples: int = 10000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo estimation of dominated hypervolume

    reference_point:
        (worst_case, worst_latency, worst_reliability)

    Assumes:
        - cost minimized
        - latency minimized
        - reliability maximized
    """

    frontier = list(frontier)

    if not frontier:
        return 0.0
    
    ref_cost, ref_latency, ref_reliability = reference_point

    random_number_generator = random.Random(seed)

    dominated_count = 0

    for _ in range(num_samples):

        sample_cost = random_number_generator.uniform(0, ref_cost)
        sample_latency = random_number_generator.uniform(0, ref_latency)
        sample_reliability = random_number_generator.uniform(ref_reliability, 1.0)

        for candidate in frontier:
            if (
                candidate.total_cost <= sample_cost
                and candidate.total_latency <= sample_latency
                and candidate.total_reliability >= sample_reliability
            ):
                dominated_count += 1
                break
    
    box_volume = ref_cost * ref_latency * (1.0 - ref_reliability)

    return (dominated_count / num_samples) * box_volume