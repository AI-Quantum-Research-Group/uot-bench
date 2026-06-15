# Changelog

All notable changes to `uot-bench` are documented here.

## [Unreleased]

### Added — OTT-JAX interoperability (`feat/ott-jax`)

- **`uot.interop.ott`** — optional adapter package (requires `pip install "uot-bench[ott]"`).
  Provides `BaseSolver` wrappers for eight OTT-JAX solvers:
  `OTTSinkhornSolver`, `OTTLRSinkhornSolver`, `OTTGromovWassersteinSolver`,
  `OTTLRGromovWassersteinSolver`, `OTTSinkhornDivergence`,
  `OTTDiscreteBarycenterSolver`, `OTTGWBarycenterSolver`, `OTTUnivariateSolver`.
  All drop into the existing `run_pipeline` / YAML / SLURM / Dash pipeline without
  changing any core code.
- **`uot[ott]` optional extra** in `pyproject.toml` (`ott-jax>=0.4.7`).
- **Representation registry** (`uot.experiments.representations`).
  Registers a *kind* → *builder* mapping; the runner calls `build_representation`
  outside the timed solve region and caches the result on the problem. Built-in
  kinds: `"marginals_costs"`, `"point_cloud"`, `"grid"`.  OTT wrappers register
  `"ott_linear"`, `"ott_quadratic"`, `"ott_barycenter"`, `"ott_gw_barycenter"` at
  import time.
- **`BaseSolver.input_kind`** class attribute (default `"marginals_costs"`).
  Solvers set a non-default kind to receive a pre-built representation object as
  the first positional argument to `solve`, rather than `(marginals, costs)`.
- **`invoke_solver`** helper in `uot.experiments.measurement`.
  Routes the call correctly for both `"marginals_costs"` (unpacks `view.marginals`
  / `view.costs`) and other kinds (passes `view` positionally).
- **Post-solve hook system** (`uot.experiments.hooks`).
  `PostSolveHook` protocol, `apply_hooks` function.  Hooks run after every
  `solve_fn` call and can return `None` (no-op), `dict` (merge into row), or
  `list[dict]` (fan-out).  `Experiment` accepts an optional `hooks` list;
  `Problem.post_solve_hooks()` allows problem-scoped hooks.
- **`ColorTransferHook`** (`uot.experiments.real_data.color_transfer.hooks`).
  Replaces `ColorTransferExperiment`; works with the generic `Experiment` +
  `run_pipeline` and the `PostSolveHook` fan-out protocol.
- **`Problem._view_cache`** — cleared by `free_memory()` call added to the end of
  each problem loop iteration.
- **`Problem.post_solve_hooks()`** override point on the base class.
- **`SolverOutput.low_rank_plan`** field for low-rank coupling factors `(Q, R, g)`.
- **`TwoMarginalProblem.to_ott_linear_problem()`** and
  `to_ott_quadratic_problem()` helpers.
- **`BarycenterProblem.to_ott_barycenter_problem()`** helper.
- **`PointCloudMeasure.to_ott_geometry()`** and **`GridMeasure.to_ott_geometry()`**
  helpers.
- **`COST_REGISTRY`** in `uot.utils.costs` — maps cost names to both a uot
  callable and an OTT `CostFn` factory.
- YAML runner configs for OTT solvers: `configs/runners/cot/ott_sinkhorn.yaml`,
  `ott_lr_sinkhorn.yaml`, `ott_gw.yaml`.
- Benchmark notebook: `notebooks/ott_interop_benchmark.ipynb`.
- Interop tests: `tests/interop/test_ott_adapter.py`.

### Changed — OTT-JAX interoperability

- **`Experiment.run_on_problems`**: builds representation outside the timed
  region; solver kwargs are included in the hook `context`; calls
  `problem.free_memory()` unconditionally after each problem.
- **`run_pipeline`** (`uot.experiments.runner`): fixed JIT warm-up to draw from
  `chain(*deepcopy(all_iterators_list))` instead of `deepcopy(current_iterators)`
  (the latter raised `TypeError` because `itertools.chain` is not picklable).
- **`apply_hooks`**: fixed hook chaining — each hook now receives the **current
  accumulated row** (not always the original `base_metrics`), so fan-out and
  dict-merge hooks compose correctly across a hook list.
- **`SolveFn` protocol**: `view: Any` replaces `marginals` + `costs` parameters.
  All built-in measurement functions updated accordingly.
- **`ColorTransferExperiment`** and **`run_color_transfer_pipeline`** are
  deprecated; the modules now emit `DeprecationWarning` and proxy to the generic
  replacements via `__getattr__`.
- **`from_gw_output`** in `uot.interop.ott._outputs`: filters OTT's `-1.0`
  padding sentinels from `out.costs` before reporting the final GW cost and
  iteration count; guards all attribute access with `getattr` for OTT version
  portability.
- **`from_sinkhorn_divergence_output`**: tolerates scalar vs. tuple `converged` /
  `n_iters` / `errors` across OTT versions.

### Changed — Public API / typing refactor

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
