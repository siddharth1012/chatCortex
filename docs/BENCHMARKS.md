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

# Key Results

Scalar beam search suffers from **early-stage pruning bias**.

This bias causes irreversible loss of Pareto-optimal architectures.

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
