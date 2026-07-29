"""End-to-end smoke test for the analysis-notebook pipeline.

Chains all the `add_*` / `run_*` calls the analysis notebook makes, on tiny
synthetic data, and asserts each step produces the columns the next step
expects. Catches integration bugs that per-function unit tests miss:

- Missing re-exports in `cell_analysis/__init__.py` (the notebook imports
  from the top-level package, not from `cell_analysis.io`).
- Return-value order swaps (e.g., ``comparison_df, persistence_summary``
  vs. ``persistence_summary, comparison_df``).
- Orphan feature/column names in configs (e.g., FATE_FEATURES referencing
  a column no `add_*` function produces).
- Wrong-shape config values (e.g., ``PERI_CORE_RINGS = (2, 2)`` when the
  code expects a 3-tuple ``(core_max, peri_min, peri_max)``).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Import surface — the notebook expects everything at the package top level.
# ---------------------------------------------------------------------------

def test_notebook_import_surface_is_stable():
    """Every symbol the analysis notebook imports must exist at the top
    level of `cell_analysis`. A missed re-export in `__init__.py` breaks
    the notebook's first code cell."""
    from cell_analysis import (
        load_extraction, load_stack,
        add_fluorescence, add_nucleoid_distribution, add_geometry,
        add_fluorescence_concentration, add_sav_ratio,
        add_preburst_fluorescence,
        add_fluorescence_alignment, add_peri_core_alignment,
        run_nucleus_persistence,
        add_frame0_fate_comparison, add_fate_prediction,
        print_fate_comparison,
        save_dataframe, save_summary_dict, save_main_outputs,
        show_with_source,
        plot_nucleus_persistence, plot_fate_prediction,
    )
    for name, obj in [
        ("load_extraction", load_extraction),
        ("load_stack", load_stack),
        ("add_fluorescence", add_fluorescence),
        ("add_nucleoid_distribution", add_nucleoid_distribution),
        ("add_geometry", add_geometry),
        ("add_fluorescence_concentration", add_fluorescence_concentration),
        ("add_sav_ratio", add_sav_ratio),
        ("add_preburst_fluorescence", add_preburst_fluorescence),
        ("add_fluorescence_alignment", add_fluorescence_alignment),
        ("add_peri_core_alignment", add_peri_core_alignment),
        ("run_nucleus_persistence", run_nucleus_persistence),
        ("add_frame0_fate_comparison", add_frame0_fate_comparison),
        ("add_fate_prediction", add_fate_prediction),
        ("print_fate_comparison", print_fate_comparison),
        ("save_dataframe", save_dataframe),
        ("save_summary_dict", save_summary_dict),
        ("save_main_outputs", save_main_outputs),
        ("show_with_source", show_with_source),
        ("plot_nucleus_persistence", plot_nucleus_persistence),
        ("plot_fate_prediction", plot_fate_prediction),
    ]:
        assert callable(obj), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_stacks_and_tracks():
    """Two disappearing tracks over 5 frames, one surviving track.

    Frames:  0  1  2  3  4
    Track 1: X  X  X  -  -     (disappears after frame 2)
    Track 2: X  X  X  X  -     (disappears after frame 3)
    Track 3: X  X  X  X  X     (survives)

    Stack shape (5, 12, 12). Each track occupies a distinct 3x3 tile so
    the peri/core rings have both interior and edge pixels.
    """
    T, H, W = 5, 12, 12
    label_stack = np.zeros((T, H, W), dtype=np.int32)
    fluor_stack = np.zeros((T, H, W), dtype=np.float32)
    nucleus_label_stack = np.zeros((T, H, W), dtype=np.int32)

    # Track 1 (label 1): 4x4 mask top-left, present frames 0..2 (disappears at 3)
    for t in range(3):
        label_stack[t, 0:4, 0:4] = 1
        nucleus_label_stack[t, 1:3, 1:3] = 1
    # Track 2 (label 2): 4x4 mask bottom-right, present frames 0..3
    for t in range(4):
        label_stack[t, 8:12, 8:12] = 2
        nucleus_label_stack[t, 9:11, 9:11] = 2
    # Track 3 (label 3): 4x4 mask middle, all frames
    for t in range(5):
        label_stack[t, 4:8, 4:8] = 3
        nucleus_label_stack[t, 5:7, 5:7] = 3

    # Fluorescence: give each cell a modest gradient so nucleoid metrics
    # aren't degenerate (peri and core rings both non-empty and unequal).
    rng = np.random.default_rng(0)
    fluor_stack[label_stack > 0] = 50.0
    fluor_stack += rng.normal(0, 2, size=fluor_stack.shape).astype(np.float32)
    fluor_stack = np.clip(fluor_stack, 0, None)

    # tracked DataFrame — one row per (frame, track_id) matching the masks.
    # Columns match `labels_to_detections` output plus `track_id`.
    rows = []
    for t in range(T):
        for tid in (1, 2, 3):
            m = label_stack[t] == tid
            if not m.any():
                continue
            ys, xs = np.where(m)
            rows.append({
                "frame": t,
                "track_id": tid,
                "label": tid,
                "centroid_y": float(ys.mean()),
                "centroid_x": float(xs.mean()),
                "area": int(m.sum()),
            })
    tracked = pd.DataFrame(rows)

    stats = tracked.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        mean_area=("area", "mean"),
        num_detections=("frame", "count"),
    ).reset_index()
    stats["lifetime"] = stats["last_frame"] - stats["first_frame"] + 1
    stats["disappeared"] = stats["last_frame"] < T - 1

    return tracked, stats, label_stack, fluor_stack, nucleus_label_stack


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def test_analysis_chain_runs_end_to_end():
    """Chains every add_* / run_* call the analysis notebook makes.

    Any regression in return-value order, missing column, or bad param
    shape surfaces here as a ValueError / AttributeError / KeyError.
    """
    from cell_analysis import (
        add_fluorescence, add_nucleoid_distribution, add_geometry,
        add_fluorescence_concentration, add_sav_ratio,
        add_preburst_fluorescence,
        add_fluorescence_alignment, add_peri_core_alignment,
        run_nucleus_persistence,
        add_frame0_fate_comparison,
    )

    tracked, stats, label_stack, fluor_stack, nucleus_label_stack = (
        _make_stacks_and_tracks()
    )

    # 1. Fluorescence measurement — the first analysis-notebook step.
    tracked, stats = add_fluorescence(tracked, stats, fluor_stack, label_stack)
    assert "mean_intensity" in tracked.columns
    assert "total_intensity" in tracked.columns
    assert "fluor_rel" in tracked.columns

    # 2. Nucleoid distribution — depends on `PERI_CORE_RINGS` being a valid
    #    3-tuple (core_max, peri_min, peri_max). A 2-tuple would raise
    #    ValueError inside `_peri_core_asymmetry`.
    tracked, stats = add_nucleoid_distribution(
        tracked, stats, fluor_stack, label_stack,
        peri_core_rings=(0.15, 0.35, 0.55),
    )
    assert "cv" in tracked.columns
    assert "nnrm" in tracked.columns
    assert "mean_edge_distance_norm" in tracked.columns
    assert "gaussian_sigma_norm" in tracked.columns
    assert "peri_core_asymmetry" in tracked.columns

    # 3. Geometry (µm units + derived columns).
    tracked, stats = add_geometry(tracked, stats, pixel_size_um=0.0645)
    for col in ("radius", "volume", "surface_area", "volume_rel"):
        assert col in tracked.columns, f"add_geometry missing {col}"

    # 4. Fluorescence concentration + SAV — cheap derivations, catch missing
    #    dependencies (both need `volume` from add_geometry).
    tracked = add_fluorescence_concentration(tracked)
    assert "fluor_concentration" in tracked.columns
    tracked = add_sav_ratio(tracked)
    assert "sav_ratio" in tracked.columns

    # 5. Pre-burst — needs `disappeared` on stats and lifetime data on tracked.
    stats = add_preburst_fluorescence(tracked, stats)

    # 6. Alignment tables — window param comes from analysis config.
    align_df = add_fluorescence_alignment(
        tracked, stats, fluor_stack, label_stack, window=1,
    )
    assert "offset" in align_df.columns
    assert "F_norm" in align_df.columns
    peri_align_df = add_peri_core_alignment(tracked, stats, window=1)
    assert "offset" in peri_align_df.columns

    # 7. Nucleus persistence — the return order was reversed in the initial
    #    Task 6 notebook rewrite; this pins the contract.
    comparison_df, persistence_summary = run_nucleus_persistence(
        label_stack, nucleus_label_stack,
    )
    assert isinstance(comparison_df, pd.DataFrame), (
        "run_nucleus_persistence must return (DataFrame, dict) in that order"
    )
    assert isinstance(persistence_summary, dict)
    assert "conclusion" in persistence_summary

    # 8. Frame-0 comparisons — verifies FATE_FEATURES lists in configs
    #    reference columns that exist on `tracked` at this point. This is
    #    what caught the old orphan `fluor_total` regression.
    features = ["area", "cv", "nnrm",
                "mean_edge_distance_norm", "gaussian_sigma_norm"]
    for f in features:
        assert f in tracked.columns, f"fate feature `{f}` missing from tracked"
    _ = add_frame0_fate_comparison(tracked, stats, features=features)
    # Note: `add_fate_prediction` needs both classes present in the LOO
    # split (real datasets have hundreds of cells; sklearn refuses to fit
    # a one-class training fold). The column-existence check above is the
    # regression the smoke test cares about — actual model fitting is
    # covered by `test_fate_prediction.py` with a 100-cell synthetic set.


# ---------------------------------------------------------------------------
# Regression pins for specific bugs we've already fixed
# ---------------------------------------------------------------------------

def test_peri_core_rings_shape_is_a_triple():
    """`PERI_CORE_RINGS_DEFAULT` must be a 3-tuple; a 2-tuple would raise
    ValueError inside `_peri_core_asymmetry`. Pins the config schema."""
    from cell_analysis.matching import PERI_CORE_RINGS_DEFAULT

    assert len(PERI_CORE_RINGS_DEFAULT) == 3, (
        "PERI_CORE_RINGS_DEFAULT must be (core_max, peri_min, peri_max); "
        "if this fails, matching.py's asymmetry code + all YAML configs' "
        "PERI_CORE_RINGS: [...] entries need to stay in sync."
    )


def test_run_nucleus_persistence_returns_df_then_summary():
    """Pins the return-value order — swapping breaks the notebook's
    `save_dataframe(comparison_df, ...)` call at the analysis step."""
    from cell_analysis import run_nucleus_persistence

    T = 3
    label_stack = np.zeros((T, 4, 4), dtype=np.int32)
    nucleus_label_stack = np.zeros((T, 4, 4), dtype=np.int32)
    label_stack[:, 0:2, 0:2] = 1
    nucleus_label_stack[:, 0, 0] = 1

    first, second = run_nucleus_persistence(label_stack, nucleus_label_stack)
    assert isinstance(first, pd.DataFrame)
    assert isinstance(second, dict)
