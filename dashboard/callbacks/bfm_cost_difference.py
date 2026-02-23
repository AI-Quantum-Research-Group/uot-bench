import dash
from dash import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.bfm import (
    filtered_bfm_dataframe,
    empty_figure,
    BFM_CONFIG_COLOR_MAP,
)
from utils.data import df_master

SIMPLEX_SOLVER = "Simplex"
FALLBACK_COLORS = px.colors.qualitative.Dark24

_simplex_cost_lookup = (
    df_master[df_master["solver"] == SIMPLEX_SOLVER][["dataset", "cost"]]
    .dropna()
    .groupby("dataset")["cost"]
    .median()
)


def _config_key(row):
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name
    push = row.get("pushforward_fn") or "Back-and-forth"
    step = row.get("stepsize")
    return f"{push}|{step}"


def _config_label(row):
    push = row.get("pushforward_fn")
    push_label = push.replace("_", " ").title() if isinstance(push, str) else "Back-and-Forth"
    stepsize = row.get("stepsize")
    if pd.notna(stepsize):
        return f"{push_label} (σ={stepsize:g})"
    return push_label


@dash.callback(
    Output("bfm-cost-diff", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_cost_difference(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if df.empty:
        return empty_figure("Transport Cost Difference (Back-and-Forth − Simplex)")

    df = df.copy()
    df["simplex_cost"] = df["dataset"].map(_simplex_cost_lookup)
    df = df.dropna(subset=["simplex_cost", "cost", "size"])
    if df.empty:
        return empty_figure("Transport Cost Difference (Back-and-Forth − Simplex)")

    df["cost_difference"] = df["cost"] - df["simplex_cost"]
    df["config_key"] = df.apply(_config_key, axis=1)
    df["config_label"] = df.apply(_config_label, axis=1)

    summary = (
        df.groupby(["config_key", "config_label", "size"])["cost_difference"]
        .agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values("size")
    )
    if summary.empty:
        return empty_figure("Transport Cost Difference (Back-and-Forth − Simplex)")

    fig = go.Figure()
    fallback_idx = 0
    tick_values = sorted(summary["size"].unique().tolist())

    for config_key, group in summary.groupby("config_key"):
        color = BFM_CONFIG_COLOR_MAP.get(config_key)
        if color is None:
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1

        label = group["config_label"].iloc[0]
        median = group["median"]
        upper = group["q3"] - median
        lower = median - group["q1"]

        fig.add_trace(
            go.Scatter(
                x=group["size"],
                y=median,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=8),
                error_y=dict(
                    type="data",
                    array=upper,
                    arrayminus=lower,
                    visible=True,
                    thickness=1,
                    width=4,
                ),
                hovertemplate=(
                    "Config: %{text}<br>"
                    "Grid size: %{x}<br>"
                    "Median Δcost: %{y:.4g}<br>"
                    "P25: %{customdata[0]:.4g}<br>"
                    "P75: %{customdata[1]:.4g}<extra></extra>"
                ),
                text=[label] * len(group),
                customdata=group[["q1", "q3"]].to_numpy(),
            )
        )

    fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"))

    fig.update_layout(
        template="plotly_white",
        title="Transport Cost Difference (Back-and-Forth − Simplex)",
        yaxis_title="Cost Difference",
        legend_title="Configuration",
        hovermode="closest",
        xaxis=dict(
            title="Grid Size (# points)",
            type="log",
            tickmode="array",
            tickvals=tick_values,
            ticktext=[f"{int(val)}" for val in tick_values],
        ),
    )
    return fig
