# ChatCortex Architecture

ChatCortex is a research framework for **automated synthesis and evaluation of AI agent architectures** under multi-objective constraints.

The system treats agent design as an **architecture search problem**, where candidate pipelines are generated, evaluated, and compared across multiple objectives.

---

# High-Level Architecture

The framework is organized into layered components.

TaskSpecification
↓
CapabilityRegistry
↓
Synthesis Engine
↓
AgentGraph
↓
Execution Engine
↓
Telemetry
↓
Evaluation Harness
↓
Pareto Optimization

Each layer isolates a specific concern in the architecture synthesis process.

---

# TaskSpecification

Defines the **architecture synthesis problem**.

A task includes:

* required capabilities
* hard constraints (cost, latency, privacy)
* objective weights

Example:

```python
task = TaskSpecification(
    required_capabilities=[
        "retrieval",
        "generation",
        "verification"
    ],
    max_cost=0.01,
    max_latency=2000
)
```

The task definition specifies **what capabilities must be satisfied**, but does not define how they are implemented.

---

# CapabilityRegistry

The registry stores all available system components.

Each component declares:

* capability type
* cost
* latency
* reliability
* privacy level

The registry supports:

* capability lookup
* constraint filtering
* component discovery

Registries can also be materialized from a JSON document with
**`load_registry(path)`** (`chatcortex.registry.loader`). Every row is
validated — costs, latency, and the reliability range; component type,
privacy, and capability vocabulary — and validation failures name the
row and field (e.g. `components[2].reliability_score out of range`).
`CapabilityRegistry.to_dict()` emits the same document shape, so
registries round-trip through JSON. They also ship as packs:
`save_registry(registry, path)` (`chatcortex.registry.package`) writes
that same shape and proves the round-trip by reloading the file, and
`validate_registry_file(path)` returns a structured report — component
count, per-row field issues, duplicate names — without raising. The
`chatcortex-registry validate <file>` console command wraps the report,
exiting 0 only for a fully valid pack.

Importantly, the registry **does not perform optimization**. It simply exposes valid components to the synthesis engine.

---

# AgentGraph

Agent architectures are represented as **Directed Acyclic Graphs (DAGs)**.

Each node corresponds to a component instance.

Edges represent execution order.

System-level metrics are computed from node metadata:

Total Cost
Sum of component costs

Total Latency
Sequential latency aggregation

System Reliability
Multiplicative aggregation of component success probabilities

---

# Synthesis Engine

The synthesis engine generates candidate architectures.

Implemented strategies include:

HeuristicSynthesizer
Greedy deterministic architecture construction.

RandomSynthesizer
Random sampling baseline for stochastic exploration.

BeamSynthesizer
Width-constrained architecture search.

ExhaustiveSynthesizer
Enumerates the full architecture space to compute the exact Pareto frontier.

ParetoPartialBeamSynthesizer
Beam search whose stage truncation keeps only Pareto-non-dominated partials, always retaining the cost, latency, and reliability extremes and filling remaining slots by crowding distance in objective space.

ProgressiveParetoBeamSynthesizer
Introduces depth-aware beam widening to mitigate early-stage pruning bias.

---

# Execution Engine

The execution engine simulates architecture behavior.

Two execution modes are supported.

Deterministic Mode
All components succeed. Used for structural validation.

Probabilistic Mode
Each component succeeds with probability equal to its reliability score.

This allows empirical evaluation of system robustness.

Optionally, `AgentExecutor(..., backend=...)` delegates each component
call to a real execution backend — an `openai`-SDK chat-completions
adapter ships behind the `chatcortex[real]` extra (see
`ExecutionProtocol`). With a backend set, telemetry records the
backend's *measured* latency, cost, and success instead of simulated
values, and the chain still stops on the first failing component.

---

# Telemetry

Execution traces capture:

* component execution order
* latency
* cost
* success/failure status

These logs support detailed experimental analysis.

**Trace calibration** closes the loop:
`chatcortex.telemetry.calibration.calibrate(registry, records, alpha)`
blends empirical success/latency/cost from recorded traces back into a
*new* immutable registry via `CapabilityRegistry.update()`:

```
new = alpha * empirical + (1 - alpha) * prior
```

Components without traces keep their declared priors, so the calibrated
registry can feed straight back into synthesis.

---

# Evaluation Harness

The evaluation harness provides a controlled environment for comparing synthesis algorithms.

Features:

* multiple task definitions
* multiple synthesizers
* stochastic execution trials
* reproducible experiments

Metrics are aggregated across runs to estimate system performance.

---

# Pareto Optimization

ChatCortex computes **Pareto-optimal architectures** across:

* cost
* latency
* reliability

Dominance definition:

Architecture A dominates B if:

A.cost ≤ B.cost
A.latency ≤ B.latency
A.reliability ≥ B.reliability
and at least one strict improvement.

The resulting Pareto frontier represents the set of **optimal architecture trade-offs**.

---

# Future Architecture Extensions

Planned improvements include:

* graph-structured agent synthesis
* conditional execution nodes
* real LLM and tool integrations
* enterprise deployment optimization

These extensions will expand the search space beyond linear pipelines to full **agent architecture graphs**.
