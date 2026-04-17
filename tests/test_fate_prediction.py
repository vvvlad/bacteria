import numpy as np
import pandas as pd

from cell_analysis.matching import predict_fate_from_frame0


def _make_data():
    rng = np.random.default_rng(42)
    n = 100
    died = rng.choice([True, False], size=n, p=[0.5, 0.5])
    tracked = pd.DataFrame({
        "track_id": range(n),
        "frame": [0] * n,
        "area": rng.normal(800, 100, n) - 50 * died,
        "cv": rng.normal(0.5, 0.1, n) + 0.15 * died,
        "nnrm": rng.normal(0.1, 0.03, n) + 0.05 * died,
    })
    track_stats = pd.DataFrame({
        "track_id": range(n),
        "first_frame": [0] * n,
        "disappeared": died,
    })
    return tracked, track_stats


def test_output_structure():
    tracked, track_stats = _make_data()
    result_df, summary = predict_fate_from_frame0(tracked, track_stats)
    assert "predicted_prob" in result_df.columns
    assert "predicted_class" in result_df.columns
    assert "disappeared" in result_df.columns
    assert "auc" in summary
    assert "accuracy" in summary
    assert "feature_importance" in summary


def test_auc_above_random():
    tracked, track_stats = _make_data()
    _, summary = predict_fate_from_frame0(tracked, track_stats)
    assert summary["auc"] > 0.5


def test_custom_features():
    tracked, track_stats = _make_data()
    result_df, summary = predict_fate_from_frame0(
        tracked, track_stats, features=["area"],
    )
    assert summary["feature_names"] == ["area"]
    assert len(summary["feature_importance"]) == 1


def test_prediction_count_matches_frame0():
    tracked, track_stats = _make_data()
    result_df, summary = predict_fate_from_frame0(tracked, track_stats)
    assert len(result_df) == summary["n_cells"]
    assert summary["n_died"] + summary["n_survived"] == summary["n_cells"]
