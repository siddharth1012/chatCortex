# Contributing to ChatCortex

Thank you for your interest in contributing to ChatCortex.

ChatCortex is a research framework for automated synthesis and optimization of AI agent architectures.

## Development Setup

Clone the repository:

```bash
git clone https://github.com/siddharth1012/chatCortex
cd chatCortex
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install development dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Try the quickstart:

```bash
python examples/quickstart.py
```

## Code Style

ChatCortex uses:

* **black** for formatting
* **ruff** for linting

Run:

```bash
black .
ruff check chatcortex tests examples
```

## Pull Requests

When submitting a PR:

1. Create a feature branch.
2. Ensure tests pass.
3. Write clear commit messages.

Example:

```
feat: add progressive pareto beam search
fix: correct hypervolume computation
docs: update architecture documentation
```

### Commit message policy

Commit and tag messages must contain no attribution footers or trailers
— no "Generated with ..." lines, no emoji footer, and no `Co-Authored-By`
lines (including any AI tooling). A commit message ends at its body's
last line.

## Research Contributions

ChatCortex welcomes contributions related to:

* architecture search algorithms
* multi-objective optimization
* agent system evaluation
* experimental benchmarks

## Code of Conduct

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
