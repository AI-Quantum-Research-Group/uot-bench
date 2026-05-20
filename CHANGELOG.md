# Changelog

All notable changes to `uot-bench` are documented here.

## [Unreleased]

### Changed
- **Public API**: `MarginalProblem` renamed to `Problem`; `ProblemGenerator` renamed to `Generator`. Old names remain as deprecated aliases.
- `uot/__init__.py` now exposes `Problem`, `Generator`, `TwoMarginalProblem`, `BarycenterProblem`, `MultiMarginalProblem`, `BaseSolver`, `SolverConfig`, `Experiment`, `run_pipeline`, `SolverInputs`, `PointCloudInputs`, `GridInputs`, `BaseMeasure`, and `__version__` from a single import path.
- `uot/problems/__init__.py` exposes `Problem`, `Generator`, and `MultiMarginalProblem`.
- `uot/solvers/__init__.py` exposes all concrete solver classes.
- `uot/experiments/__init__.py` exposes `Experiment`, `run_pipeline`, and all measurement functions.
- `uot/data/__init__.py` exposes `BaseMeasure`.

### Added
- `py.typed` marker — downstream type-checkers now pick up annotations.
- `SolverOutput` `TypedDict` in `uot.solvers.base_solver` for typed solver return values.
- `SolveFn` `Protocol` in `uot.experiments.experiment` for typed `solve_fn` callbacks.
- `ShareMode` and `Backend` type aliases in `uot.utils.types`.
- `uot/algorithms/__init__.py` — fixes implicit namespace-package risk.
- `tests/conftest.py` with `assert_is_jax_array` helper.
- `docs/extending.md` — worked examples for custom `Problem` and `Generator`.
- `CHANGELOG.md` and `CONTRIBUTING.md`.
- `[tool.pyright]` config in `pyproject.toml`.

### Fixed
- `np.in1d` → `np.isin` (numpy 2.x compatibility).
- `BarycenterProblem.free_memory()` now implemented (was raising `NotImplementedError`).
- `BarycenterProblem.get_costs()` return type annotation corrected.
- `TwoMarginalProblem.get_exact_coupling()` return type corrected from `float` to `ArrayLike`.
- Stale `__version__ = "0.1.0.dev"` removed from `uot.algorithms.rapdhg` and `uot.solvers.pdlp_bary`.
- `ArrayLike` type alias now correctly covers `jax.Array | np.ndarray` at runtime.
- Russian profanity comment removed from `uot/experiments/measurement.py`.
- `Generator.generate()` and `ProblemGenerator.one()` signatures tightened (no more `*args, **kwargs`).

### Packaging
- Base dependencies slimmed: `Pillow`, `scikit-image`, `scikit-learn`, `h5py`, `gpu-tracker` moved to extras (`image-analysis`, `mnist`, `storage`, `profiling`).
- New extras: `storage`, `profiling`, `mnist`, `dev`, `all`.
- Added classifiers: `Development Status :: 4 - Beta`, `Intended Audience :: Science/Research`, `Topic :: Scientific/Engineering :: Mathematics`, `Operating System :: OS Independent`, `Typing :: Typed`.
- Added `[project.urls].Documentation`.

## [0.1.8] — 2025-XX-XX

Initial public release on PyPI.
