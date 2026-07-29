# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
- Research-iteration synthesizers (v2, v3, system-score) moved to `chatcortex.synthesis.experimental`
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
