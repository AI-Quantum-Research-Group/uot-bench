"""Post-solve hook for colour-transfer experiments.

Moves the image-reconstruction, domain-metric computation, and image saving
logic out of the dedicated :class:`ColorTransferExperiment` and into a
:class:`~uot.experiments.hooks.PostSolveHook` so the generic
:func:`~uot.experiments.runner.run_pipeline` can drive colour-transfer runs.

Typical usage::

    from uot.experiments import Experiment, run_pipeline
    from uot.experiments.real_data.color_transfer.hooks import ColorTransferHook

    hook = ColorTransferHook(
        output_dir="output/color_transfer",
        soft_extension_modes=[False, True],
        displacement_alphas=[1.0, 0.5],
    )
    experiment = Experiment(name="CT", solve_fn=measure_time_and_output, hooks=[hook])
    df = run_pipeline(experiment, solvers, iterators)
"""

from __future__ import annotations

import datetime
import gc
import hashlib
import os
from typing import Any

import jax
import numpy as np
from PIL import Image
from jax import numpy as jnp

from uot.experiments.real_data.color_transfer.measurement import (
    _build_postprocess_modes,
    _compute_distribution_metrics,
    _compute_image_quality_metrics,
    _compute_map_quality_metrics,
    _process_transported_image,
    _build_plan_grid_map,
)
from uot.problems.base_problem import Problem
from uot.utils.logging import logger


class ColorTransferHook:
    """Post-solve hook that reconstructs transported images and computes domain metrics.

    Produces one result row per (soft_extension, displacement_alpha) combination,
    replacing the single base-metrics row with a list of expanded rows.

    Parameters
    ----------
    output_dir:
        Directory where transported images are saved.
    soft_extension_modes:
        List of booleans controlling soft-extension post-processing.
        ``None`` means no soft-extension sweep.
    displacement_alphas:
        List of alpha values for displacement interpolation.
    drop_columns:
        Column names to drop from every result row (e.g. large arrays).
    """

    def __init__(
        self,
        output_dir: str = "output/color_transfer",
        soft_extension_modes: list[bool] | None = None,
        displacement_alphas: list[float] | None = None,
        drop_columns: list[str] | None = None,
    ):
        self.output_dir = output_dir
        self.soft_extension_modes = soft_extension_modes if soft_extension_modes is not None else [False]
        self.displacement_alphas = displacement_alphas if displacement_alphas is not None else [1.0]
        self.drop_columns = drop_columns or []
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
        self,
        problem: Problem,
        view: Any,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Reconstruct and evaluate the transported image for every postprocess mode.

        Returns a list of metric dicts — one per (soft_extension, displacement_alpha)
        combination — to be emitted as individual DataFrame rows.
        """
        try:
            return self._run(problem, view, metrics, context)
        except Exception as exc:
            logger.error(f"ColorTransferHook failed: {exc}")
            return [dict(metrics)]

    def _run(
        self,
        problem: Problem,
        view: Any,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        marginals = view.marginals if hasattr(view, "marginals") else problem.get_marginals()

        axes_mu, mu_nd = marginals[0].as_grid(backend="jax", dtype=jnp.float64)
        axes_nu, nu_nd = marginals[1].as_grid(backend="jax", dtype=jnp.float64)
        target_palette, _ = marginals[1].as_point_cloud()

        plan_grid_map = None
        plan_mask = None
        if "transport_plan" in metrics:
            plan_grid_map, plan_mask = _build_plan_grid_map(
                metrics["transport_plan"],
                target_palette,
                mu_nd,
            )

        soft_extension_specified, postprocess_modes = _build_postprocess_modes(
            self.soft_extension_modes,
            self.displacement_alphas,
        )

        solver_name = context.get("solver_name", "unknown_solver")
        base_without_plan = {k: v for k, v in metrics.items() if k != "transport_plan"}

        results: list[dict[str, Any]] = []
        for use_soft_extension, alpha_value in postprocess_modes:
            active_soft = use_soft_extension if soft_extension_specified else False
            logger.info(
                f"Transporting image with soft_extension={active_soft}, "
                f"displacement_alpha={alpha_value}..."
            )
            transported_image = _process_transported_image(
                problem,
                marginals,
                metrics,
                axes_mu=axes_mu,
                mu_nd=mu_nd,
                axes_nu=axes_nu,
                nu_nd=nu_nd,
                target_palette=target_palette,
                use_soft_extension=active_soft,
                displacement_alpha=alpha_value,
                plan_grid_map=plan_grid_map,
                plan_mask=plan_mask,
            )

            entry = dict(base_without_plan)
            entry.update(_compute_distribution_metrics(transported_image, marginals[1]))
            entry.update(_compute_map_quality_metrics(
                marginals,
                metrics,
                axes_mu=axes_mu,
                mu_nd=mu_nd,
                axes_nu=axes_nu,
                nu_nd=nu_nd,
                target_palette=target_palette,
                plan_grid_map=plan_grid_map,
                plan_mask=plan_mask,
                use_soft_extension=active_soft,
                displacement_alpha=alpha_value,
            ))
            entry.update(_compute_image_quality_metrics(transported_image, problem))

            if soft_extension_specified:
                entry["soft_extension"] = bool(active_soft)
            entry["displacement_alpha"] = alpha_value

            image_params = dict(context.get("solver_kwargs", {}))
            if "soft_extension" in entry:
                image_params["soft_extension"] = "yes" if entry.get("soft_extension") else "no"
            image_params["displacement_alpha"] = f"{alpha_value:.3f}"
            image = transported_image
            if hasattr(problem, "to_rgb_image"):
                image = problem.to_rgb_image(image)  # type: ignore[attr-defined]
            filename = self._save_image(image, problem, solver_name, image_params)
            if filename:
                entry["result_image_filename"] = filename

            results.append(entry)
            gc.collect()
            logger.info(
                f"Completed metrics for soft_extension={active_soft}, "
                f"displacement_alpha={alpha_value}."
            )

        for row in results:
            for col in self.drop_columns:
                row.pop(col, None)

        return results

    def _save_image(
        self,
        image: Any,
        problem: Problem,
        solver_name: str,
        params: dict[str, Any],
    ) -> str | None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        filename = self._build_safe_filename(problem.name, solver_name, param_str, timestamp)
        save_path = os.path.join(self.output_dir, "images")
        os.makedirs(save_path, exist_ok=True)

        try:
            if isinstance(image, jax.Array):
                image = np.asarray(image)
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(save_path, filename))
            logger.info(f"Saved transported image to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")
            return None

    @staticmethod
    def _sanitize_component(text: str, max_length: int) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in text)
        return safe[:max_length] if len(safe) > max_length else safe

    def _build_safe_filename(
        self,
        problem_name: str,
        solver_name: str,
        param_str: str,
        timestamp: str,
    ) -> str:
        problem_component = self._sanitize_component(problem_name, 60)
        solver_component = self._sanitize_component(solver_name, 40)
        safe_params = self._sanitize_component(param_str, 120)
        if len(safe_params) < len(param_str):
            digest = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:10]
            safe_params = f"params_{digest}"
        base = f"{problem_component}_{solver_component}_{safe_params}_{timestamp}.png"
        max_len = 200
        if len(base) <= max_len:
            return base
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
        return f"{problem_component[:50]}_{solver_component[:30]}_{digest}_{timestamp}.png"
