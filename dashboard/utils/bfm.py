import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset import (
    load_bfm_dataframe,
    BFM_PERFORMANCE_COLUMNS,
    BFM_MONGE_COLUMNS,
)
from utils.filters import coerce_list

df_bfm = load_bfm_dataframe()
_palette = px.colors.qualitative.Safe
_config_keys = sorted(df_bfm["name"].dropna().unique().tolist())
BFM_CONFIG_COLOR_MAP = {
    key: _palette[idx % len(_palette)]
    for idx, key in enumerate(_config_keys)
}


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[dict(text="No data for current filters", x=0.5, y=0.5, showarrow=False)],
        template="plotly_white",
    )
    return fig


def filtered_bfm_dataframe(dims, dists, stepsizes, sizes=None, pushforwards=None):
    dims = coerce_list(dims)
    dists = coerce_list(dists)
    stepsizes = coerce_list(stepsizes)
    sizes = coerce_list(sizes)
    pushforwards = coerce_list(pushforwards)
    if len(dims) == 0 or len(dists) == 0 or len(stepsizes) == 0:
        return dims, dists, stepsizes, sizes, pushforwards, df_bfm.iloc[0:0]
    df = df_bfm[
        df_bfm["dim"].isin(dims)
        & df_bfm["distribution"].isin(dists)
        & df_bfm["stepsize"].isin(stepsizes)
    ]
    if len(sizes) == 0:
        return dims, dists, stepsizes, sizes, pushforwards, df.iloc[0:0]
    if len(pushforwards) == 0:
        return dims, dists, stepsizes, sizes, pushforwards, df.iloc[0:0]
    if sizes:
        df = df[df["size"].isin(sizes)]
    if pushforwards:
        df = df[df["pushforward_fn"].isin(pushforwards)]
    return dims, dists, stepsizes, sizes, pushforwards, df


def stepsize_metric_figure(df, metrics, title: str) -> go.Figure:
    if df.empty or not metrics:
        return empty_figure(title)
    has_pushforward = "pushforward_fn" in df.columns and df["pushforward_fn"].notna().any()
    pushforward_values = (
        sorted(df["pushforward_fn"].dropna().unique().tolist()) if has_pushforward else [None]
    )

    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
        horizontal_spacing=0.05,
    )
    for idx, metric in enumerate(metrics, start=1):
        for pushforward in pushforward_values:
            sub_df = df if pushforward is None else df[df["pushforward_fn"] == pushforward]
            if sub_df.empty:
                continue
            summary = (
                sub_df.groupby("stepsize")[metric]
                .agg(["mean", "std"])
                .sort_index()
            )
            if summary.empty:
                continue
            steps = summary.index.astype(float)
            mean = summary["mean"]
            std = summary["std"].fillna(0.0)
            down_err = np.minimum(std.to_numpy(), mean.to_numpy())
            label = "adaptive" if pushforward is None else str(pushforward)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean,
                    mode="lines+markers",
                    error_y=dict(
                        type="data",
                        array=std,
                        arrayminus=down_err,
                        visible=True,
                    ),
                    name=f"{metric}: {label}",
                    legendgroup=label,
                    showlegend=True,
                ),
                row=1,
                col=idx,
            )
        fig.update_xaxes(
            title_text="Stepsize",
            type="log",
            row=1,
            col=idx,
        )
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=420,
    )
    return fig


__all__ = [
    "df_bfm",
    "filtered_bfm_dataframe",
    "stepsize_metric_figure",
    "empty_figure",
    "BFM_PERFORMANCE_COLUMNS",
    "BFM_MONGE_COLUMNS",
    "BFM_CONFIG_COLOR_MAP",
]
