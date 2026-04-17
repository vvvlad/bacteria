import numpy as np
import pandas as pd
import pytest

from cell_analysis.matching import compute_death_clustering


def _make_track_stats_clustered():
    rng = np.random.default_rng(42)
    n_dis = 20
    n_surv = 30
    records = []
    for i in range(n_dis):
        records.append({
            "track_id": i,
            "disappeared": True,
            "last_frame": rng.integers(5, 10),
            "last_y": 100 + rng.normal(0, 5),
            "last_x": 100 + rng.normal(0, 5),
        })
    for i in range(n_surv):
        records.append({
            "track_id": n_dis + i,
            "disappeared": False,
            "last_frame": 24,
            "last_y": rng.uniform(0, 1000),
            "last_x": rng.uniform(0, 1000),
        })
    return pd.DataFrame(records)


def _make_track_stats_uniform():
    rng = np.random.default_rng(42)
    records = []
    for i in range(50):
        records.append({
            "track_id": i,
            "disappeared": i < 20,
            "last_frame": rng.integers(5, 10) if i < 20 else 24,
            "last_y": rng.uniform(0, 1000),
            "last_x": rng.uniform(0, 1000),
        })
    return pd.DataFrame(records)


def test_output_structure():
    stats = _make_track_stats_clustered()
    result = compute_death_clustering(stats)
    assert "mean_nn_distance_deaths" in result
    assert "mean_nn_distance_random" in result
    assert "clustering_ratio" in result
    assert "p_value" in result


def test_clustered_deaths_detected():
    stats = _make_track_stats_clustered()
    result = compute_death_clustering(stats, n_permutations=200)
    assert result["clustering_ratio"] < 1.0, "Clustered deaths should have ratio < 1"


def test_uniform_deaths_not_clustered():
    stats = _make_track_stats_uniform()
    result = compute_death_clustering(stats, n_permutations=200)
    assert result["clustering_ratio"] > 0.5, "Uniform deaths should have ratio near 1"


def test_too_few_deaths_returns_nan():
    stats = pd.DataFrame({
        "track_id": [1, 2, 3],
        "disappeared": [True, False, False],
        "last_frame": [5, 24, 24],
        "last_y": [100.0, 200.0, 300.0],
        "last_x": [100.0, 200.0, 300.0],
    })
    result = compute_death_clustering(stats)
    assert np.isnan(result["clustering_ratio"])
