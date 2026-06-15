# Installation

## Core install

```bash
pip install uot-bench
```

> **Install name vs import name**: the package is distributed as `uot-bench` on PyPI, but imported as `uot`.

## Optional extras

```bash
pip install "uot-bench[ott]"            # OTT-JAX backend (ott-jax>=0.4.7)
pip install "uot-bench[viz]"            # plotting helpers (matplotlib, plotly, seaborn)
pip install "uot-bench[image-analysis]" # Pillow, scikit-image
pip install "uot-bench[color-transfer]" # color transfer experiment
pip install "uot-bench[storage]"        # HDF5 problem store (h5py)
pip install "uot-bench[profiling]"      # GPU resource tracking
pip install "uot-bench[mnist]"          # MNIST classification (scikit-learn)
pip install "uot-bench[gurobi]"         # Gurobi LP solver
pip install "uot-bench[dev]"            # development tools (pytest, ruff, pyright)
pip install "uot-bench[docs]"           # docs build (mkdocs-material, mkdocstrings)
pip install "uot-bench[all]"            # all optional extras (includes ott)
```

## CUDA (JAX GPU support)

```bash
pip install "uot-bench[cuda12]"
```

## Development install (editable)

```bash
git clone https://github.com/AI-Quantum-Research-Group/uot-bench
cd uot-bench
pip install -e ".[dev]"
```

## Benchmarking tasks with Pixi

For running the full benchmarking pipeline with [Pixi](https://pixi.sh/):

```bash
pixi install
pixi run benchmark --config configs/runners/gaussians.yaml --folds 1 --export results.csv
```
