from dash.dependencies import Input

FILTER_INPUTS = [
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
]


def coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]
