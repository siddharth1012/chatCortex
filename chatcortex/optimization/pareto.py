from typing import List

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate


def dominates(a: ArchitectureCandidate, b: ArchitectureCandidate) -> bool:
    """
    Returns True if architecture a dominates architecture b
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