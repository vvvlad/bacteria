import numpy as np
import pandas as pd

from cell_analysis.matching import (
    compare_frame0_features_by_fate,
    predict_fate_from_frame0,
)


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
        "mean_edge_distance_norm": rng.normal(1.0, 0.2, n) + 0.1 * died,
        "gaussian_sigma_norm": rng.normal(0.5, 0.1, n) - 0.1 * died,
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


def test_compare_output_structure():
    tracked, track_stats = _make_data()
    df = compare_frame0_features_by_fate(tracked, track_stats)
    expected_cols = {
        "feature", "n_survived", "n_died",
        "median_survived", "median_died", "U", "p_value",
    }
    assert expected_cols.issubset(df.columns)
    assert list(df["feature"]) == [
        "area", "cv", "nnrm",
        "mean_edge_distance_norm", "gaussian_sigma_norm",
    ]


def test_compare_cohort_counts_match_frame0():
    tracked, track_stats = _make_data()
    df = compare_frame0_features_by_fate(tracked, track_stats)
    n_total_expected = int(track_stats["disappeared"].notna().sum())
    for _, row in df.iterrows():
        assert row["n_survived"] + row["n_died"] == n_total_expected


def test_compare_custom_features():
    tracked, track_stats = _make_data()
    df = compare_frame0_features_by_fate(
        tracked, track_stats, features=["area"],
    )
    assert list(df["feature"]) == ["area"]
    assert (df["p_value"].between(0, 1)).all()


def test_summary_includes_per_class_metrics():
    tracked, track_stats = _make_data()
    _, summary = predict_fate_from_frame0(tracked, track_stats)
    assert "confusion_matrix" in summary
    cm = summary["confusion_matrix"]
    assert set(cm.keys()) == {"TN", "FP", "FN", "TP"}
    assert "per_class" in summary
    for label in ("survived", "disappeared"):
        assert label in summary["per_class"]
        for metric in ("precision", "recall", "f1", "support"):
            assert metric in summary["per_class"][label]


def test_confusion_matrix_totals_match_cohort():
    tracked, track_stats = _make_data()
    pred_df, summary = predict_fate_from_frame0(tracked, track_stats)
    cm = summary["confusion_matrix"]
    total = cm["TN"] + cm["FP"] + cm["FN"] + cm["TP"]
    assert total == len(pred_df)
    # Row totals = actual class counts (positive class = disappeared)
    assert cm["TN"] + cm["FP"] == summary["n_survived"]
    assert cm["FN"] + cm["TP"] == summary["n_died"]


def test_print_fate_comparison_uses_both_summaries(capsys):
    from cell_analysis.pipeline import print_fate_comparison

    tracked, track_stats = _make_data()
    _, summary_full = predict_fate_from_frame0(tracked, track_stats)
    _, summary_no_area = predict_fate_from_frame0(
        tracked, track_stats,
        features=["cv", "nnrm", "mean_edge_distance_norm",
                  "gaussian_sigma_norm"],
    )

    print_fate_comparison(summary_full, summary_no_area)
    out = capsys.readouterr().out

    # Both AUC values appear (.3f format)
    assert f"{summary_full['auc']:.3f}" in out
    assert f"{summary_no_area['auc']:.3f}" in out
    # Headers for the two columns
    assert "full" in out
    assert "no-area" in out
    # Conclusion line is one of the two known strings
    assert ("consider permanently swapping" in out
            or "differences small" in out)
