# ChatCortex

[![CI](https://github.com/siddharth1012/chatCortex/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharth1012/chatCortex/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/chatcortex)](https://pypi.org/project/chatcortex/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ChatCortex** is a framework for **automated synthesis and optimization of AI agent architectures** — think *AutoML for AI agents*.

Instead of manually wiring LLM pipelines (`retrieval → LLM → verifier → tool`), ChatCortex treats agent design as a **multi-objective architecture search problem** and automatically discovers architectures that optimize **cost**, **latency**, and **reliability** simultaneously.

```
Task Specification → Architecture Search → Pareto-Optimal Architectures
```

**ChatCortex is the only agent framework that re-synthesizes your architecture from what actually happened.** Execution telemetry is calibrated back into the component registry, so every new synthesis round runs against *measured* behavior instead of declared guesses. The shipped demo makes this concrete: in four closed-loop iterations, `examples/closed_loop.py` moves a component's declared reliability from 0.78 to a corrected 0.53, lifts a budget retriever from 0.82 to 0.93, and reshapes the search frontier around the values the system actually measured.

## Installation

```bash
pip install chatcortex
```

Or from source:

```bash
git clone https://github.com/siddharth1012/chatCortex
cd chatCortex
pip install -e .
```

## Quickstart

```python
from chatcortex import (
    TaskSpecification,
    BeamSynthesizer,
    build_demo_registry,
)

# A registry of components with cost/latency/reliability tradeoffs
registry = build_demo_registry(
    capabilities=("retrieval", "generation", "verification")
)

# The task: an ordered capability chain with hard constraints
task = TaskSpecification(
    required_capabilities=["retrieval", "generation", "verification"],
    max_cost=0.02,
    max_latency=1500,
)

# Synthesize Pareto-optimal architectures
synth = BeamSynthesizer(registry, beam_width=5)
for arch in synth.synthesize(task):
    print(
        f"cost=${arch.total_cost:.4f} "
        f"latency={arch.total_latency:.0f}ms "
        f"reliability={arch.total_reliability:.3f}"
    )
```

Or run the full demo comparing three synthesis strategies:

```bash
python examples/quickstart.py
```

To see the full loop — load a JSON registry, synthesize a frontier, execute one
architecture, and calibrate the registry's metadata from real telemetry — run:

```bash
python examples/registry_from_json.py
```

Run the flagship closed loop — synthesize → execute → calibrate →
re-synthesize, four iterations of measured behavior flowing back into the
registry:

```bash
python examples/closed_loop.py
```

With an OpenAI key you can run the same loop against real backends:

```bash
pip install "chatcortex[real]"
export OPENAI_API_KEY=...   # every reply spends your credits
python examples/closed_loop_real.py
```

To search over your own components, build a `CapabilityRegistry` and register
`ComponentMetadata` describing each model, tool, or verifier — see
`examples/` for complete walkthroughs.

## Registry packs

A registry ships as a JSON pack in the loader's document shape. Saving it
proves the round-trip, and one command validates any pack offline:

```python
from chatcortex import build_demo_registry, save_registry, validate_registry_file

save_registry(build_demo_registry(), "registry.json")  # writes, then reloads to prove it
print(validate_registry_file("registry.json").valid)   # True
```

```bash
chatcortex-registry validate registry.json
#   components validated: 12
#   result: VALID          (exit 0; issues and duplicates are named when invalid)
```

## How It Works

ChatCortex is organized as a layered architecture synthesis system:

```
TaskSpecification → CapabilityRegistry → Synthesis Engine → AgentGraph (DAG)
    → Execution Engine → Telemetry → Evaluation Harness → Pareto Optimization
```

**Core concepts:**

- **ComponentMetadata** — declarative description of an agent component (model, retriever, tool, verifier, memory) with cost per call, latency, reliability score, and privacy level.
- **CapabilityRegistry** — stores components and filters candidates by capability and privacy constraints.
- **TaskSpecification** — the synthesis problem: an ordered capability chain plus hard constraints (max cost, max latency, privacy) and objective weights.
- **AgentGraph** — architectures represented as DAGs; cost is additive, latency sequential, reliability multiplicative.
- **Synthesizers** — search strategies that map a task to an (approximate) Pareto frontier.

**Available synthesizers:**

| Synthesizer | Strategy |
|---|---|
| `HeuristicSynthesizer` | Greedy deterministic construction |
| `RandomSynthesizer` | Random baseline |
| `BeamSynthesizer` | Budget-aware beam search |
| `ParetoPartialBeamSynthesizer` | Beam search with per-stage Pareto pruning |
| `ProgressiveParetoBeamSynthesizer` | Depth-aware beam widening (best Pareto recovery) |
| `ExhaustiveSynthesizer` | Exact Pareto frontier (small spaces) |

Approximation quality is measured with **Pareto coverage**, **hypervolume loss**, and **average regret** — see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system design and component layers
- [Benchmarks](docs/BENCHMARKS.md) — experimental evaluation of synthesis strategies
- [Examples](examples/) — runnable demos and budget-sweep experiments
- [Contributing](CONTRIBUTING.md) — development setup and PR guidelines
- [Changelog](CHANGELOG.md)

## Research Context

ChatCortex is a controlled experimental platform for studying automated agent architecture synthesis: multi-objective optimization, architecture search, reliability–cost tradeoffs, and AutoML-style agent design. Experiments cover 5-stage pipelines with Pareto frontiers up to 95 architectures and evaluation budgets from 20 to 180 evaluations; progressive beam widening shows improved Pareto recovery over static beam strategies.

If you use ChatCortex in your research, please cite it (see [CITATION.cff](CITATION.cff)).

## Roadmap

- [x] Component modeling, capability registry, task specification, DAG representation
- [x] Exact Pareto frontier computation (exhaustive search)
- [x] Budget-aware beam search with Pareto pruning and progressive widening
- [ ] Graph-structured (non-chain) agent synthesis
- [ ] Real model / tool integrations (OpenAI chat-completions backend already available behind the `chatcortex[real]` extra; more backends planned)
- [x] Registry loading from JSON (`load_registry`, with error messages naming the failing field); YAML planned

## Status

Research framework under active development (alpha). APIs may change between minor versions.

## License

[MIT](LICENSE) — developed by **Siddharth Saraswat**
