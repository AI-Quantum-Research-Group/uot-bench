import dash
from dash import Input, Output, html
import pandas as pd

from utils.data import df_master
from utils.filters import coerce_list

ALL_SOLVERS = sorted(df_master["solver"].unique())
ALL_DATASETS = sorted(df_master["distribution"].unique())
ALL_DIMS = sorted(df_master["dim"].dropna().astype(int).unique())
ALL_SIZES = sorted(df_master["size"].dropna().astype(int).unique())
ALL_REGS = sorted(df_master["reg"].dropna().astype(float).unique())


def _normalize_selection(selection, universe):
    selected = set(coerce_list(selection))
    return [item for item in universe if item in selected]


def _format_reg(value: float) -> str:
    if pd.isna(value):
        return "–"
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def _format_dim(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"{int(value)}D"


def _format_size(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"{int(value):,}"


def _build_cell_table(cell_df: pd.DataFrame):
    if cell_df is None or cell_df.empty:
        return html.Div("—", className="text-muted small text-center")

    cell_df = (
        cell_df[["dim", "size", "reg"]]
        .drop_duplicates()
        .sort_values(["dim", "size", "reg"])
    )
    rows = []
    for (dim, size), chunk in cell_df.groupby(["dim", "size"], sort=False):
        regs = sorted(chunk["reg"].dropna().unique().tolist())
        reg_labels = ", ".join(_format_reg(r) for r in regs) if regs else "—"
        rows.append(
            html.Tr(
                [
                    html.Td(_format_dim(dim), className="small text-nowrap"),
                    html.Td(_format_size(size), className="small"),
                    html.Td(reg_labels, className="small"),
                ]
            )
        )

    return html.Table(
        html.Tbody(rows),
        className="table table-sm table-borderless mb-0 coverage-cell-table",
    )


@dash.callback(
    Output("home-coverage-matrix", "children"),
    Input("home-coverage-solver-filter", "value"),
    Input("home-coverage-dataset-filter", "value"),
    Input("home-coverage-dim-filter", "value"),
    Input("home-coverage-size-filter", "value"),
    Input("home-coverage-reg-filter", "value"),
)
def update_home_coverage_matrix(solvers, datasets, dims, sizes, regs):
    solvers = _normalize_selection(solvers, ALL_SOLVERS)
    datasets = _normalize_selection(datasets, ALL_DATASETS)
    dims = _normalize_selection(dims, ALL_DIMS)
    sizes = _normalize_selection(sizes, ALL_SIZES)
    regs = _normalize_selection(regs, ALL_REGS)

    if not all([solvers, datasets, dims, sizes, regs]):
        return html.Div(
            "Select at least one option in every filter to render the matrix.",
            className="text-muted fst-italic",
        )

    filtered = df_master[
        df_master["solver"].isin(solvers)
        & df_master["distribution"].isin(datasets)
        & df_master["dim"].isin(dims)
        & df_master["size"].isin(sizes)
        & df_master["reg"].isin(regs)
    ]

    if filtered.empty:
        return html.Div(
            "No runs match these filters.",
            className="text-muted fst-italic",
        )

    grouped = {
        (solver, dataset): group
        for (solver, dataset), group in filtered.groupby(["solver", "distribution"])
    }

    header = html.Thead(
        html.Tr(
            [html.Th("Method / Dataset", className="text-nowrap")]
            + [
                html.Th(dataset, className="text-center text-nowrap")
                for dataset in datasets
            ]
        )
    )

    rows = []
    for solver in solvers:
        row_cells = [
            html.Th(
                solver,
                scope="row",
                className="text-nowrap align-top",
            )
        ]
        for dataset in datasets:
            cell_df = grouped.get((solver, dataset))
            row_cells.append(
                html.Td(
                    _build_cell_table(cell_df),
                    className="align-top",
                )
            )
        rows.append(html.Tr(row_cells))

    table = html.Table(
        [header, html.Tbody(rows)],
        className="table table-bordered table-hover align-top mb-0",
    )

    return html.Div(
        [
            html.Div(
                f"Showing {len(solvers)} methods × {len(datasets)} datasets",
                className="text-muted small mb-2",
            ),
            html.Div(
                table,
                className="table-responsive",
            ),
        ]
    )
