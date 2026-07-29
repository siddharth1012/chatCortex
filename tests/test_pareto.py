from chatcortex import AgentGraph
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet, compute_pareto_front, dominates


def cand(cost, latency, reliability):
    return ArchitectureCandidate(
        graph=AgentGraph(),
        total_cost=cost,
        total_latency=latency,
        total_reliability=reliability,
    )


def test_dominates_strict():
    assert dominates(cand(1, 100, 0.9), cand(2, 200, 0.8))


def test_no_domination_on_tradeoff():
    a = cand(1, 300, 0.9)
    b = cand(2, 100, 0.9)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_equal_candidates_do_not_dominate():
    a = cand(1, 100, 0.9)
    b = cand(1, 100, 0.9)
    assert not dominates(a, b)


def test_compute_pareto_front():
    dominated = cand(3, 300, 0.5)
    front = compute_pareto_front([cand(1, 100, 0.9), cand(2, 50, 0.8), dominated])
    assert dominated not in front
    assert len(front) == 2


def test_pareto_set_incremental():
    ps = ParetoSet()
    assert ps.add(cand(2, 200, 0.8))
    assert ps.add(cand(1, 100, 0.9))  # dominates the first
    assert len(ps) == 1
    assert not ps.add(cand(3, 300, 0.5))  # dominated, rejected
    assert len(ps) == 1
