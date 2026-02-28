import random
from typing import Iterable, List, Tuple

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import dominates, frontier_coverage, hypervolume_monte_carlo


def compute_coverage(
    approx_frontier: Iterable[ArchitectureCandidate],
    true_frontier: Iterable[ArchitectureCandidate],
) -> float:
    return frontier_coverage(approx_frontier, true_frontier)

def compute_hypervolume(
    frontier: Iterable[ArchitectureCandidate],
    reference_point: Tuple[float, float, float],
) -> float:
    return hypervolume_monte_carlo(frontier, reference_point)
    

def hypervolume_loss(
    approx_frontier,
    true_frontier,
    reference_point,
    num_samples=200000,
    seed=42,
) -> float:
    ref_cost, ref_latency, ref_reliability = reference_point

    random_number_generator = random.Random(seed)

    dominated_true = 0
    dominated_approx = 0

    for _ in range(num_samples):

        sample_cost = random_number_generator.uniform(0, ref_cost)
        sample_latency = random_number_generator.uniform(0, ref_latency)
        sample_reliability = random_number_generator.uniform(ref_reliability, 1.0)

        for candidate in true_frontier:
            if (
                candidate.total_cost <= sample_cost
                and candidate.total_latency <= sample_latency
                and candidate.total_reliability >= sample_reliability
            ):
                dominated_true += 1
                break

        for candidate in approx_frontier:
            if (
                candidate.total_cost <= sample_cost
                and candidate.total_latency <= sample_latency
                and candidate.total_reliability >= sample_reliability
            ):
                dominated_approx += 1
                break
    
    
    box_volume = ref_cost * ref_latency * (1.0 - ref_reliability)

    hv_true = (dominated_true / num_samples) * box_volume
    hv_approx = (dominated_approx / num_samples) * box_volume

    return max(0.0, hv_true - hv_approx)

def additive_regret(
    candidate: ArchitectureCandidate,
    true_frontier: Iterable[ArchitectureCandidate],
) -> Tuple[float, float, float]:
    """
    returns
        - (cost_regret, latency_regret, reliability_regret)
    """

    dominating = [
        p for p in true_frontier if dominates(p, candidate)
    ]

    if not dominating:
        return (0.0, 0.0, 0.0)
    
    min_cost_diff = min(candidate.total_cost - p.total_cost for p in dominating)
    min_latency_diff = min(candidate.total_latency - p.total_latency for p in dominating)
    min_reliability_diff = min(p.total_reliability - candidate.total_reliability for p in dominating)

    return (
        max(0.0, min_cost_diff),
        max(0.0, min_latency_diff),
        max(0.0, min_reliability_diff),
    )

def average_regret(
    approx_frontier: Iterable[ArchitectureCandidate],
    true_frontier: Iterable[ArchitectureCandidate]
) -> Tuple[float, float, float]:
    """
    Average additive regret across approx frontier
    """

    approx_list = list(approx_frontier)

    if not approx_list:
        return (0.0, 0.0, 0.0)

    total_cost_regret = 0.0
    total_latency_regret = 0.0
    total_reliability_regret = 0.0

    for candidate in approx_list:
        cr, lr, rr = additive_regret(candidate, true_frontier)
        total_cost_regret += cr
        total_latency_regret += lr
        total_reliability_regret += rr
    
    n = len(approx_list)

    return (
        total_cost_regret / n,
        total_latency_regret / n,
        total_reliability_regret / n,
    )

def evaluate_approximation(
    approx_frontier: List[ArchitectureCandidate],
    true_frontier: List[ArchitectureCandidate],
    reference_point: Tuple[float, float, float],
) -> dict:
    """
    Returns a full approximate report
    """

    coverage = compute_coverage(approx_frontier, true_frontier)
    
    hv_loss = hypervolume_loss(approx_frontier, true_frontier, reference_point)
    avg_regret = average_regret(approx_frontier, true_frontier)

    return {
        "coverage": coverage,
        "hypervolume_loss": hv_loss,
        "avg_cost_regret": avg_regret[0],
        "avg_latency_regret": avg_regret[1],
        "avg_reliability_regret": avg_regret[2],
    }