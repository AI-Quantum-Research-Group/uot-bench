import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from dataset import (
    load_bfm_dataframe,
    BFM_PERFORMANCE_COLUMNS,
    BFM_MONGE_COLUMNS,
)
from components import filters

dash.register_page(__name__, path="/back-and-forth", name="Back & Forth")

df_bfm = load_bfm_dataframe()
dim_options = sorted(df_bfm["dim"].dropna().unique().tolist())
distribution_options = sorted(df_bfm["distribution"].dropna().unique().tolist())
stepsize_options = sorted(df_bfm["stepsize"].dropna().unique().tolist())
size_options = sorted(df_bfm["size"].dropna().unique().tolist())
pushforward_options = sorted(df_bfm["pushforward_fn"].dropna().unique().tolist())

layout = dbc.Container(
    [
        html.H2("Back-and-Forth Solver Analysis", className="mb-3"),
        html.P(
            "Explore aggregated diagnostics for the Back-and-Forth solver on grid benchmarks. "
            "Use the filters below to focus on specific dimensions, distributions, or stepsizes.",
            className="text-muted",
        ),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    filters.switch_filter(
                        label="Dimensions",
                        filter_id="bfm-dim-filter",
                        options=[{"label": f"{d}D", "value": d} for d in dim_options],
                        value=dim_options,
                        width=3,
                    ),
                    filters.dropdown_filter(
                        label="Distributions",
                        filter_id="bfm-dist-filter",
                        options=[{"label": d, "value": d} for d in distribution_options],
                        value=distribution_options,
                        width=3,
                        placeholder="Select distributions",
                    ),
                    filters.switch_filter(
                        label="Stepsizes",
                        filter_id="bfm-stepsize-filter",
                        options=[{"label": str(s), "value": s} for s in stepsize_options],
                        value=stepsize_options,
                        width=3,
                    ),
                ], class_name="g-3"),
                dbc.Row([
                    filters.switch_filter(
                        label="Problem Sizes",
                        filter_id="bfm-size-filter",
                        options=[{"label": str(s), "value": s} for s in size_options],
                        value=size_options,
                        width=3,
                    ),
                    filters.switch_filter(
                        label="Pushforward",
                        filter_id="bfm-pushforward-filter",
                        options=[{"label": pf, "value": pf} for pf in pushforward_options],
                        value=pushforward_options,
                        width=3,
                    ),
                ], class_name="g-3 mt-2"),
            ]),
            className="mb-4",
        ),
        dbc.Row([
            dbc.Col(dcc.Graph(id="bfm-performance-summary"), width=12, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="bfm-monge-summary"), width=12, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    dcc.Graph(id="bfm-stepsize-heatmaps"),
                    style={"maxHeight": "800px", "overflowY": "auto"},
                ),
                width=12,
                className="mb-4",
            ),
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    dcc.Graph(id="bfm-monge-heatmaps"),
                    style={"maxHeight": "900px", "overflowY": "auto"},
                ),
                width=12,
                className="mb-4",
            ),
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    dcc.Graph(id="bfm-monge-core-heatmaps"),
                    style={"maxHeight": "700px", "overflowY": "auto"},
                ),
                width=12,
                className="mb-4",
            ),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="bfm-cost-diff"), width=12, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="bfm-tv-vs-size"), width=12, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="bfm-violin-metric"), width=12, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="bfm-summary-table"), width=12),
        ]),
    ],
    fluid=True,
    className="p-4",
)
