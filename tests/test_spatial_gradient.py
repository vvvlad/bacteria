import numpy as np
import pandas as pd

from cell_analysis.matching import analyze_spatial_gradient


def _make_gradient_data():
    rng = np.random.default_rng(42)
    n = 200
    x_pos = rng.uniform(0, 1400, n)
    died = x_pos > 700 + rng.normal(0, 100, n)
    tracked = pd.DataFrame({
        "track_id": range(n),
        "frame": [0] * n,
        "centroid_x": x_pos,
        "centroid_y": rng.uniform(0, 1040, n),
        "area": rng.normal(800, 100, n),
    })
    track_stats = pd.DataFrame({
        "track_id": range(n),
        "first_frame": [0] * n,
        "disappeared": died,
    })
    return tracked, track_stats


def test_output_structure():
    tracked, track_stats = _make_gradient_data()
    gradient_df, summary = analyze_spatial_gradient(tracked, track_stats)
    assert "gradient_quartile" in gradient_df.columns
    assert "centroid_x" in gradient_df.columns
    assert "centroid_y" in gradient_df.columns
    assert "disappeared" in gradient_df.columns
    assert summary["gradient_axis"] in ("centroid_x", "centroid_y")
    assert "axes_results" in summary
    assert "quartile_death_rates" in summary
    assert "auc_position_only" in summary


def test_detects_x_gradient():
    tracked, track_stats = _make_gradient_data()
    _, summary = analyze_spatial_gradient(tracked, track_stats)
    assert summary["gradient_axis"] == "centroid_x"


def test_quartile_death_rate_increases():
    tracked, track_stats = _make_gradient_data()
    _, summary = analyze_spatial_gradient(tracked, track_stats)
    rates = [summary["quartile_death_rates"][q]["death_rate"] for q in range(1, 5)]
    assert rates[-1] > rates[0]


def test_auc_above_random():
    tracked, track_stats = _make_gradient_data()
    _, summary = analyze_spatial_gradient(tracked, track_stats)
    assert summary["auc_position_only"] > 0.6


def test_quartile_counts_sum():
    tracked, track_stats = _make_gradient_data()
    gradient_df, summary = analyze_spatial_gradient(tracked, track_stats)
    total = sum(summary["quartile_death_rates"][q]["n_cells"] for q in range(1, 5))
    assert total == len(gradient_df)
