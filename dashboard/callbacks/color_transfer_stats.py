import pandas as pd
import plotly.express as px
from dash import Input, Output, callback

from color_transfer.stats import (
    LOG_SCALE_METRICS,
    METRIC_GROUPS,
    filter_stats_dataframe,
    get_stats_dataframe,
)


df, _status = get_stats_dataframe()

solver_order = sorted(df["solver_base"].dropna().unique().tolist()) if "solver_base" in df.columns else []
palette = px.colors.qualitative.Plotly
color_map = {solver: palette[index % len(palette)] for index, solver in enumerate(solver_order)}


def _empty_figure(title: str):
    return px.scatter(title=f"{title} (no data)")


def _ensure_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ("bins_per_channel", "displacement_alpha", "solver_base"):
        if column not in data.columns:
            data[column] = pd.NA
    return data


def _prepare_metrics_data(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    data = _ensure_group_columns(frame)
    available = [metric for metric in metrics if metric in data.columns]
    if not available:
        return pd.DataFrame(columns=["bins_per_channel", "displacement_alpha", "solver_base", "metric", "value"])
    return data.melt(
        id_vars=["bins_per_channel", "displacement_alpha", "solver_base"],
        value_vars=available,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])


def _facet_boxplot(data: pd.DataFrame, title: str):
    if data.empty:
        return _empty_figure(title)

    metrics = data["metric"].dropna().unique().tolist()
    bins = sorted(data["bins_per_channel"].dropna().unique().tolist())

    figure = px.box(
        data,
        x="displacement_alpha",
        y="value",
        color="solver_base",
        facet_row="metric",
        facet_col="bins_per_channel",
        facet_row_spacing=0.06,
        facet_col_spacing=0.06,
        category_orders={
            "solver_base": solver_order,
            "bins_per_channel": bins,
            "metric": metrics,
        },
        color_discrete_map=color_map,
        points=False,
        title=title,
    )
    figure.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split("=")[-1].replace("_", " ").title(),
            font={"size": 16},
        )
    )
    figure.update_layout(
        legend_title_text="Solver",
        height=300 + 200 * len(metrics),
    )
    figure.update_xaxes(matches=None, title_text="displacement_alpha", showticklabels=True)
    figure.update_yaxes(matches=None, title_text="")

    for row_index, metric in enumerate(metrics, start=1):
        if metric in LOG_SCALE_METRICS:
            # Apply log scaling to all y‑axes in the given metric row.
            figure.update_yaxes(type="log", row=row_index)
    return figure


def _build_filtered_frame(solvers, alphas, bins, regs, color_spaces) -> pd.DataFrame:
    return filter_stats_dataframe(df, solvers, alphas, bins, regs, color_spaces)


def _update_metric_group(metrics: list[str], title: str, solvers, alphas, bins, regs, color_spaces):
    filtered = _build_filtered_frame(solvers, alphas, bins, regs, color_spaces)
    data = _prepare_metrics_data(filtered, metrics)
    return _facet_boxplot(data, title)


@callback(
    Output("ct-stat-general", "figure"),
    [
        Input("ct-stat-solver-filter", "value"),
        Input("ct-stat-alpha-filter", "value"),
        Input("ct-stat-bins-filter", "value"),
        Input("ct-stat-reg-filter", "value"),
        Input("ct-stat-color-space-filter", "value"),
    ],
)
def _update_general(solvers, alphas, bins, regs, color_spaces):
    filtered = _build_filtered_frame(solvers, alphas, bins, regs, color_spaces)
    if filtered.empty or "time" not in filtered.columns:
        return _empty_figure("General")

    figure = px.box(
        filtered,
        x="displacement_alpha",
        y="time",
        color="solver_base",
        color_discrete_map=color_map,
        category_orders={"solver_base": solver_order},
        points=False,
        title="General: Runtime vs displacement alpha",
    )
    figure.update_layout(legend_title_text="Solver")
    figure.update_yaxes(type="log", title_text="time (log)")
    figure.update_xaxes(title_text="displacement_alpha")
    return figure


@callback(
    Output("ct-stat-runtime", "figure"),
    [
        Input("ct-stat-solver-filter", "value"),
        Input("ct-stat-alpha-filter", "value"),
        Input("ct-stat-bins-filter", "value"),
        Input("ct-stat-reg-filter", "value"),
        Input("ct-stat-color-space-filter", "value"),
    ],
)
def _update_runtime(solvers, alphas, bins, regs, color_spaces):
    return _update_metric_group(
        METRIC_GROUPS["runtime"],
        "Runtime Basic Metrics",
        solvers,
        alphas,
        bins,
        regs,
        color_spaces,
    )


@callback(
    Output("ct-stat-distance", "figure"),
    [
        Input("ct-stat-solver-filter", "value"),
        Input("ct-stat-alpha-filter", "value"),
        Input("ct-stat-bins-filter", "value"),
        Input("ct-stat-reg-filter", "value"),
        Input("ct-stat-color-space-filter", "value"),
    ],
)
def _update_distance(solvers, alphas, bins, regs, color_spaces):
    return _update_metric_group(
        METRIC_GROUPS["distance"],
        "Distribution Distance Metrics",
        solvers,
        alphas,
        bins,
        regs,
        color_spaces,
    )


@callback(
    Output("ct-stat-map", "figure"),
    [
        Input("ct-stat-solver-filter", "value"),
        Input("ct-stat-alpha-filter", "value"),
        Input("ct-stat-bins-filter", "value"),
        Input("ct-stat-reg-filter", "value"),
        Input("ct-stat-color-space-filter", "value"),
    ],
)
def _update_map(solvers, alphas, bins, regs, color_spaces):
    return _update_metric_group(
        METRIC_GROUPS["map"],
        "Map Structure Metrics",
        solvers,
        alphas,
        bins,
        regs,
        color_spaces,
    )


@callback(
    Output("ct-stat-image-quality", "figure"),
    [
        Input("ct-stat-solver-filter", "value"),
        Input("ct-stat-alpha-filter", "value"),
        Input("ct-stat-bins-filter", "value"),
        Input("ct-stat-reg-filter", "value"),
        Input("ct-stat-color-space-filter", "value"),
    ],
)
def _update_image_quality(solvers, alphas, bins, regs, color_spaces):
    return _update_metric_group(
        METRIC_GROUPS["image_quality"],
        "Image Quality Metrics",
        solvers,
        alphas,
        bins,
        regs,
        color_spaces,
    )


@callback(
    Output("ct-stat-marginal-error", "figure"),
    [
        Input("ct-stat-solver-filter", "value"),
        Input("ct-stat-alpha-filter", "value"),
        Input("ct-stat-bins-filter", "value"),
        Input("ct-stat-reg-filter", "value"),
        Input("ct-stat-color-space-filter", "value"),
    ],
)
def _update_marginal_error(solvers, alphas, bins, regs, color_spaces):
    filtered = _build_filtered_frame(solvers, alphas, bins, regs, color_spaces)
    if filtered.empty:
        return _empty_figure("Marginal Error")

    data = filtered.copy()
    if "marginal_error_L2" in data.columns:
        data["marginal_error"] = data["marginal_error_L2"]
        if "error" in data.columns:
            data["marginal_error"] = data["marginal_error"].fillna(data["error"])
    elif "error" in data.columns:
        data["marginal_error"] = data["error"]
    else:
        data["marginal_error"] = pd.NA

    if "marginal_error" not in data.columns or data["marginal_error"].dropna().empty:
        return _empty_figure("Marginal Error")

    # Ensure numeric values for log scaling; drop non‑convertible rows.
    data["marginal_error"] = pd.to_numeric(
        data["marginal_error"],
        errors="coerce",
    )
    data = data.dropna(subset=["marginal_error"])
    if data.empty:
        return _empty_figure("Marginal Error")

    prepared = _prepare_metrics_data(data, ["marginal_error"])
    if prepared.empty:
        return _empty_figure("Marginal Error")
    figure = _facet_boxplot(prepared, "Marginal Error (L2)")
    figure.update_yaxes(type="log")
    return figure
