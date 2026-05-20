import pandas as pd

from dashboard.color_transfer import stats as ct_stats
from dashboard.color_transfer import state as ct_state


def test_filter_stats_dataframe_basic():
    df = pd.DataFrame(
        {
            "solver_base": ["a", "b", "a", "b"],
            "displacement_alpha": [0.0, 0.0, 1.0, 1.0],
            "bins_per_channel": [8, 16, 8, 16],
            "reg": [0.0, 0.1, 0.0, 0.1],
            "color_space": ["rgb", "lab", "rgb", "lab"],
        }
    )
    filtered = ct_stats.filter_stats_dataframe(
        df,
        solvers=["a"],
        alphas=[0.0],
        bins=[8],
        regs=[0.0],
        color_spaces=["rgb"],
    )
    assert len(filtered) == 1
    row = filtered.iloc[0]
    assert row["solver_base"] == "a"
    assert row["displacement_alpha"] == 0.0
    assert row["bins_per_channel"] == 8
    assert row["reg"] == 0.0
    assert row["color_space"] == "rgb"


def test_build_problem_list_and_step_index():
    df = pd.DataFrame(
        {
            "source_image_name": ["s1", "s1", "s2"],
            "target_image_name": ["t1", "t1", "t2"],
        }
    )
    problems = ct_state.build_problem_list(df)
    assert problems == [("s1", "t1"), ("s2", "t2")]

    # Index stepping / clamping
    assert ct_state.step_problem_index(None, "ct-prev-btn", max_index=1) == 0
    assert ct_state.step_problem_index(0, "ct-prev-btn", max_index=1) == 0
    assert ct_state.step_problem_index(0, "ct-next-btn", max_index=1) == 1
    assert ct_state.step_problem_index(1, "ct-next-btn", max_index=1) == 1


