import dash
from dash import Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.bfm import filtered_bfm_dataframe
from dataset import BFM_PERFORMANCE_COLUMNS, BFM_MONGE_COLUMNS

ALL_VIOLIN_METRICS = BFM_PERFORMANCE_COLUMNS + BFM_MONGE_COLUMNS


@dash.callback(
    Output("bfm-violin-metric", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_violin_grid(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if (
        len(dims) == 0
        or len(dists) == 0
        or len(stepsizes) == 0
        or len(sizes) == 0
        or len(pushforwards) == 0
        or df.empty
    ):
        fig = go.Figure()
        fig.update_layout(
            title="Metric distributions across stepsize/pushforward",
            annotations=[dict(text="No data for these filters", x=0.5, y=0.5, showarrow=False)],
            template="plotly_white",
        )
        return fig

    metrics = [m for m in ALL_VIOLIN_METRICS if m in df.columns]
    n_metrics = len(metrics)
    if n_metrics == 0:
        return go.Figure()

    # Step + pushforward grouping
    df_sorted = df.sort_values(["stepsize", "pushforward_fn"])
    # df_sorted = df.sort_values(["pushforward_fn", "stepsize"])
    df_sorted["group_label"] = df_sorted.apply(
        lambda row: f"σ={row['stepsize']}, {row['pushforward_fn']}", axis=1
    )

    n_cols = min(3, n_metrics)
    n_rows = int((n_metrics + n_cols - 1) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )

    min_positive = 1e-9

    for idx, metric in enumerate(metrics, start=1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1
        series = df_sorted[metric].copy()
        series = series.mask(series <= 0, min_positive)
        fig.add_trace(
            go.Violin(
                x=df_sorted["group_label"],
                y=series,
                box_visible=True,
                meanline_visible=True,
                # points="outliers",
                points=False,
                name=metric,
                legendgroup=metric,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Stepsize / Pushforward", row=row, col=col)
        axis_kwargs = {
            # "type": "log"
            }
        if metric == "phi_is_convex":
            axis_kwargs["autorange"] = "reversed"
        fig.update_yaxes(
            title_text=metric.replace("_", " ").title(),
            row=row,
            col=col,
            **axis_kwargs,
        )

    fig.update_layout(
        template="plotly_white",
        height=350 * n_rows,
        title="Metric distributions per stepsize & pushforward type",
    )
    return fig
