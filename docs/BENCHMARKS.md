# ChatCortex Benchmark Experiments

This document summarizes the experimental evaluation of the ChatCortex synthesis framework.

The experiments evaluate how different synthesis algorithms recover Pareto-optimal architectures under bounded evaluation budgets.

---

# Experimental Setup

Architecture search space:

5-stage agent pipeline

Capabilities:

* retrieval
* generation
* verification
* refinement
* postprocessing

Total architecture space grows combinatorially with available components.

In stress tests, the true Pareto frontier contained up to **95 architectures**.

---

# Evaluation Metrics

The framework evaluates synthesis algorithms using several multi-objective metrics.

Coverage
Fraction of the true Pareto frontier recovered by the algorithm.

Hypervolume Loss
Difference between the hypervolume of the true frontier and the approximated frontier.

Average Regret
Average performance gap across objectives:

* cost
* latency
* reliability

---

# Budget Experiments

Architecture evaluations are constrained by a synthesis budget.

Budgets tested:

20
40
80
120
180

Higher budgets allow more candidate architectures to be evaluated.

---

# Beam Width Experiments

Beam search algorithms were evaluated with different beam widths.

Widths tested:

3
5
7
10
15

Beam width controls the diversity of partial architectures retained during search.

---

# Algorithm Comparison

Algorithms evaluated:

HeuristicSynthesizer
RandomSynthesizer
BeamSynthesizer
ParetoPartialBeamSynthesizer
ProgressiveParetoBeamSynthesizer

---

# Reproducibility

The tables below are **generated, not hand-written**: they were measured by
`scripts/regenerate_benchmarks.py` on chatcortex 0.6.0
(Python 3.10.4), using the 5-stage demo registry (4 tradeoff
variants per capability, so 4**5 = 1024 candidate chains) whose exact Pareto
frontier contains 27 architectures. The earlier
"up to **95 architectures**" figures above come from larger historical stress
test registries, not this demo setup.

Verify any future claim in this document by re-running:

```bash
python scripts/regenerate_benchmarks.py
```

which rewrites `docs/benchmark_results.json` (all coverage, hypervolume-loss,
and regret rows) and prints the tables below. Random baselines use seeds
42-46; hypervolume estimates use the library defaults (200,000 Monte Carlo
samples, seed 42).

### Coverage by evaluation budget (beam width 7, 5-stage demo registry)

Coverage of the exact Pareto frontier under a bounded evaluation budget; `Random` cells are mean +/- std over 5 seeds (42-46).

| Algorithm | 20 | 40 | 80 | 120 | 180 |
|---|---|---|---|---|---|
| Greedy | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| Random | 0.04 +/- 0.03 | 0.11 +/- 0.04 | 0.20 +/- 0.06 | 0.31 +/- 0.03 | 0.39 +/- 0.11 |
| ScalarBeam | 0.15 | 0.19 | 0.19 | 0.19 | 0.19 |
| ParetoPartialBeam | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| ProgressiveParetoBeam | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 |

### Mean hypervolume loss by evaluation budget (beam width 7)

Monte Carlo hypervolume loss against the exact frontier (200000 samples, seed 42); `Random` cells are means over 5 seeds.

| Algorithm | 20 | 40 | 80 | 120 | 180 |
|---|---|---|---|---|---|
| Greedy | 115.5990 | 115.5990 | 115.5990 | 115.5990 | 115.5990 |
| Random | 43.2480 | 28.4835 | 20.1942 | 18.9936 | 16.7583 |
| ScalarBeam | 75.7605 | 68.4000 | 68.4000 | 68.4000 | 68.4000 |
| ParetoPartialBeam | 6.9060 | 6.9060 | 6.9060 | 6.9060 | 6.9060 |
| ProgressiveParetoBeam | 38.2215 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Coverage by beam width (evaluation budget 120)

Coverage of the exact Pareto frontier against a fixed evaluation budget.

| Algorithm | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|
| ScalarBeam | 0.19 | 0.19 | 0.19 | 0.26 | 0.26 |
| ParetoPartialBeam | 0.22 | 0.26 | 0.33 | 0.63 | 0.85 |
| ProgressiveParetoBeam | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |

### Mean hypervolume loss by beam width (evaluation budget 120)

Monte Carlo hypervolume loss against the exact frontier (200000 samples, seed 42).

| Algorithm | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|
| ScalarBeam | 83.9265 | 83.9265 | 68.4000 | 68.4000 | 68.4000 |
| ParetoPartialBeam | 23.2560 | 13.2660 | 6.9060 | 0.1005 | 0.0135 |
| ProgressiveParetoBeam | 1.9695 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

# Key Results

Scalar beam search suffers from **early-stage pruning bias**.

This bias causes irreversible loss of Pareto-optimal architectures.

Stage truncation is Pareto-aware: intermediate stages keep only
non-dominated partial architectures, always retain the cost, latency,
and reliability extremes, and fill remaining slots by crowding distance
in objective space. On the demo 3-stage task the reliability corner of
the exact frontier `(0.009, 540 ms, 0.7787)` — previously lost by both
Pareto-aware beams at nearly every beam width — is recovered from width
4 up, with full 6/6 frontier recovery from width 6.


Pareto-aware partial beam search improves coverage by preserving non-dominated partial architectures.

Progressive Pareto Beam Widening further improves recovery by gradually increasing search diversity with depth.

Under moderate budgets, Progressive Pareto Beam achieved **near-complete recovery of the true Pareto frontier** in dense architecture spaces.

---

# Observed Trends

Key empirical findings:

* Increasing beam width improves Pareto recovery but increases computational cost.
* Random search approximates hypervolume but converges slowly.
* Progressive beam widening balances exploration and exploitation.

---

# Conclusion

The experiments demonstrate that **depth-aware beam widening significantly improves multi-objective architecture search performance**.

This makes Progressive Pareto Beam a promising approach for scalable synthesis of AI agent architectures.

---
