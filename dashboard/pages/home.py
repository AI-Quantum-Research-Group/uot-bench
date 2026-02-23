import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from dataset import load_all_df, preprocess

dash.register_page(__name__, path="/", name="Home", title="Home")

# --- Data prep ----------------------------------
df = load_all_df()
df = preprocess(df)

solvers = sorted(df["solver"].unique())
datasets = sorted(df["distribution"].unique())
dims = sorted(df["dim"].dropna().astype(int).unique())
sizes = sorted(df["size"].dropna().astype(int).unique())
regs = sorted(df["reg"].dropna().astype(float).unique())


def format_reg_label(value: float) -> str:
    if pd.isna(value):
        return "–"
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:.3g}"

# -- Hero Section ------------------------------------------------
hero = html.Div(
    [
        html.H1("Optimal Transport Dashboard", className="display-4"),
        html.P(
            "Explore performance and statistical comparisons of OT solvers.",
            className="lead",
        ),
        dbc.Button("Descriptive Analysis", href="/descriptive", color="primary", className="me-2"),
        dbc.Button("Inferential Analysis", href="/inferential", color="secondary", className="me-2"),
        dbc.Button("Color Transfer Showcase", href="/color_transfer", color="info", className="me-2"),
        dbc.Button("Color Transfer Statistics", href="/color_transfer_stats", color="secondary"),
    ],
    className="p-5 mb-4 bg-light rounded-3",
)

# -- Summary Cards ------------------------------------------------
stats = [
    ("Datasets", len(df['dataset'].unique())),
    ("Solvers",  len(df['solver'].unique())),
    ("Total Runs", len(df)),
]

cards = [
    dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(title),
                dbc.CardBody(html.H2(value, className="card-title")),
            ],
            className="text-center",
        ),
        width=4,
    )
    for title, value in stats
]

# -- Layout -------------------------------------------------------
layout = dbc.Container(
    [
        hero,
        dbc.Row(cards, className="g-4"),
        html.H2("Method Coverage Explorer", className="mt-5"),
        html.P(
            "Filter by solver, dataset, dimension, size, or ε to see which combinations have runs.",
            className="mb-3",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Methods"),
                                    dcc.Dropdown(
                                        id="home-coverage-solver-filter",
                                        options=[{"label": s, "value": s} for s in solvers],
                                        value=solvers,
                                        multi=True,
                                        placeholder="Select methods",
                                        persistence=True,
                                    ),
                                ],
                                lg=3,
                                md=6,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Datasets"),
                                    dcc.Dropdown(
                                        id="home-coverage-dataset-filter",
                                        options=[{"label": d, "value": d} for d in datasets],
                                        value=datasets,
                                        multi=True,
                                        placeholder="Select datasets",
                                        persistence=True,
                                    ),
                                ],
                                lg=3,
                                md=6,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Dimensions"),
                                    dcc.Dropdown(
                                        id="home-coverage-dim-filter",
                                        options=[{"label": f"{d}D", "value": d} for d in dims],
                                        value=dims,
                                        multi=True,
                                        placeholder="Select dimensions",
                                        persistence=True,
                                    ),
                                ],
                                lg=2,
                                md=6,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Problem Sizes"),
                                    dcc.Dropdown(
                                        id="home-coverage-size-filter",
                                        options=[{"label": f"{s:,}", "value": s} for s in sizes],
                                        value=sizes,
                                        multi=True,
                                        placeholder="Select sizes",
                                        persistence=True,
                                    ),
                                ],
                                lg=2,
                                md=6,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Regularization ε"),
                                    dcc.Dropdown(
                                        id="home-coverage-reg-filter",
                                        options=[
                                            {"label": format_reg_label(r), "value": r} for r in regs
                                        ],
                                        value=regs,
                                        multi=True,
                                        placeholder="Select ε",
                                        persistence=True,
                                    ),
                                ],
                                lg=2,
                                md=6,
                                sm=12,
                            ),
                        ],
                        className="gy-3 gx-3",
                    ),
                    html.Hr(),
                    dbc.Spinner(
                        html.Div(id="home-coverage-matrix"),
                        color="primary",
                    ),
                    html.Small(
                        "Each cell summarizes the dimensions, problem sizes, and regularizations present for that method × dataset combination.",
                        className="text-muted d-block mt-3",
                    ),
                ]
            ),
            className="mb-5",
        ),
    ],
    fluid=True,
)
