# Contributing to uot-bench

Thank you for your interest! This document covers the basics of setting up a
development environment and the conventions we follow.

## Setup

```bash
# Clone and install in editable mode with dev extras
git clone https://github.com/AI-Quantum-Research-Group/uot-bench
cd uot-bench
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
```

Tests that require optional dependencies (color transfer, MNIST) are skipped
automatically if the extras are not installed.

## Code style

We use [black](https://black.readthedocs.io/) and [ruff](https://docs.astral.sh/ruff/) for formatting and linting.

```bash
black uot/ tests/
ruff check uot/ tests/
```

## Type checking

```bash
pyright uot/
```

## Pull requests

1. Branch from `main`.
2. Add tests for new functionality.
3. Run `pytest`, `black`, and `ruff` before opening a PR.
4. Update `CHANGELOG.md` under `[Unreleased]`.

## Package layout: `uot.algorithms` vs `uot.solvers`

`uot.algorithms` is an **internal** package containing pure numerical kernels (Sinkhorn, raPDHG, LP, etc.).  
`uot.solvers` contains the public `BaseSolver` subclasses that wrap those kernels behind a consistent API.

When adding a new method:
- Pure numerical code goes in `uot.algorithms`.
- The user-facing `BaseSolver` subclass that calls it goes in `uot.solvers`.

Note: `uot/algorithms/rapdhg/` and `uot/solvers/pdlp_bary/` are **diverged** copies of the raPDHG kernel — one targets the two-marginal problem and one the barycenter formulation. Do not merge them without reconciling the differences.

## Writing a custom Problem or Generator

See [docs/extending.md](docs/extending.md) for complete worked examples.

## Reporting bugs

Open an issue at <https://github.com/AI-Quantum-Research-Group/uot-bench/issues>.
