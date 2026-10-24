# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.0] - 2026-10-24

### Added
- `AgentGraph.edges()` public edge accessor and `AgentGraph.chain_append()` chain builder; the regression gates and the synthesis engine now use the public API instead of `graph._graph`
- `load_registry(path)` — JSON registry loader in `chatcortex.registry.loader`; validates every row and names the failing field on error. `CapabilityRegistry.to_dict()` emits the same document shape for round-trip serialization
- Optional real-execution backend protocol (`ExecutionProtocol` with nested `ProbeResult`, plus a flat `ProbeResult` alias) plus an `openai`-SDK chat-completions adapter behind a new `chatcortex[real]` extra; the zero-dependency core is unchanged
- `AgentExecutor(..., backend=...)` — with a backend set, telemetry records the backend's measured latency/cost/success and the chain stops on the first failure exactly as before
- Trace calibration (`chatcortex.telemetry.calibration.calibrate`) — blends empirical success/latency/cost from TelemetryLogger records into metadata with a documented alpha rule; components without traces keep their priors; a new `CapabilityRegistry.update()` supports the immutable replacement
- Example `examples/registry_from_json.py` runs the full loop (JSON registry → synthesis → execution → calibration) with no API keys

### Changed
- `ParetoPartialBeamSynthesizer` and `ProgressiveParetoBeamSynthesizer` stage truncation replaced with one deterministic Pareto-aware crowding-distance policy in objective space (in `chatcortex.synthesis.engine`), replacing the former cost-first lexicographic sort and the ad-hoc cost-diversity prune. The policy keeps the nondominated partials only, always retains the cost, latency, and reliability extremes, and fills remaining slots by crowding distance. Expected effect, measured on the demo 3-stage task: the reliability corner of the exact frontier `(0.009, 540ms, 0.7787)` — previously lost by both synthesizers at nearly every beam width — is recovered from partial-beam width 4 up (full 6/6 frontier from width 6), and the partial beam's regret at the pinned width drops to zero. Beam widths smaller than the number of nondominated partials may recover a different frontier subset than before (per-stage pruning is stricter about mid-distance partials).
- `ParetoSet` is deterministic: metric-identical candidates are kept once and iteration yields members sorted by objective tuple `(cost, latency, reliability)`. Synthesized frontiers no longer contain duplicate objective rows and are returned in sorted order; the regression-gate coverage floors were tightened to the exact measured values (`4/6`, `5/6`, `1.0`, `1.0`) that the duplicate-tolerant floors previously approximated from below
- `hypervolume_loss` reuses `hypervolume_monte_carlo` instead of duplicating its Monte Carlo sampling loop (two independent seeded runs replace the former shared RNG stream); the unused `compute_hypervolume` wrapper and the undeclared-anywhere `numpy` dependency were removed

## [0.5.0] - 2026-07-29

### Added
- `build_demo_registry()` helper for instant end-to-end quickstarts
- `AgentGraph.iter_components()` for public component iteration
- `examples/quickstart.py` comparing beam, progressive, and exhaustive synthesis
- Full pytest suite covering registry, graph, Pareto logic, synthesizers, and execution
- GitHub Actions CI (ruff + pytest on Python 3.9–3.12)
- Issue and PR templates, code of conduct
- Top-level exports for `ParetoPartialBeamSynthesizer`, `ProgressiveParetoBeamSynthesizer`, `SynthesisBudget`, `CapabilityRegistry`, `ComponentMetadata`

### Fixed
- `TaskSpecification.validate()` did not actually raise on empty capability lists
- `RandomSynthesizer` looped forever when called without a budget; now defaults to 200 evaluations

### Changed
- Demo scripts moved from the package to `examples/`
- Removed research-iteration synthesizers (v2, v3, system-score); their logic is superseded by `ProgressiveParetoBeamSynthesizer` and they remain available in git history
- README rewritten with a working, tested quickstart

## [0.4.0] - 2026-03-08

### Added
- `ProgressiveParetoBeamSynthesizer` — depth-aware progressive beam widening
- Multi-objective search backbone

## [0.3.0]

### Added
- Budget-aware synthesis (`SynthesisBudget`), beam search, scaling experiments

## [0.2.0]

### Added
- Telemetry, evaluation harness, base synthesizers, Pareto computation

## [0.1.0]

### Added
- Component metadata, capability registry, task specification, agent graph
