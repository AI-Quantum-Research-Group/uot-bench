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

TV_METRIC = "tv_mu_to_nu"
FALLBACK_COLORS = px.colors.qualitative.Dark24


def _config_key(row):
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name
    push = row.get("pushforward_fn") or "Back-and-forth"
    stepsize = row.get("stepsize")
    return f"{push}|{stepsize}"


def _config_label(row):
    push = row.get("pushforward_fn")
    push_label = push.replace("_", " ").title() if isinstance(push, str) else "Back-and-Forth"
    stepsize = row.get("stepsize")
    if pd.notna(stepsize):
        return f"{push_label} (σ={stepsize:g})"
    return push_label


@dash.callback(
    Output("bfm-tv-vs-size", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_tv_vs_size(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if df.empty or TV_METRIC not in df.columns:
        return empty_figure("TV Distance vs Grid Size (median ± IQR)")

    df = df.copy()
    df["tv_metric"] = pd.to_numeric(df[TV_METRIC], errors="coerce")
    df = df.dropna(subset=["tv_metric", "size"])
    if df.empty:
        return empty_figure("TV Distance vs Grid Size (median ± IQR)")

    df["config_key"] = df.apply(_config_key, axis=1)
    df["config_label"] = df.apply(_config_label, axis=1)

    summary = (
        df.groupby(["config_key", "config_label", "size"])["tv_metric"]
        .agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values("size")
    )

    if summary.empty:
        return empty_figure("TV Distance vs Grid Size (median ± IQR)")

    fig = go.Figure()
    fallback_idx = 0

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
                    "Median TV: %{y:.4g}<br>"
                    "P25: %{customdata[0]:.4g}<br>"
                    "P75: %{customdata[1]:.4g}<extra></extra>"
                ),
                text=[label] * len(group),
                customdata=group[["q1", "q3"]].to_numpy(),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="TV Distance vs Grid Size (median ± IQR)",
        xaxis_title="Grid Size (# points)",
        yaxis_title="TV Distance",
        legend_title="Configuration",
        hovermode="closest",
    )
    return fig
