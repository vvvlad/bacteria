"""Match phase-contrast cells to fluorescence nuclei."""

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial.distance import cdist


def match_cells_to_nuclei(
    cell_labels: np.ndarray,
    nucleus_labels: np.ndarray,
) -> pd.DataFrame:
    """Match cells (phase-contrast) to nuclei (fluorescence) per frame.

    Uses two strategies:
    1. Overlap: if a nucleus overlaps with exactly one cell, they match.
    2. Nearest centroid: for unmatched nuclei, find the nearest cell centroid.

    Parameters
    ----------
    cell_labels : np.ndarray
        Cell label stack (T, Y, X) from phase-contrast segmentation.
    nucleus_labels : np.ndarray
        Nucleus label stack (T, Y, X) from fluorescence segmentation.

    Returns
    -------
    pd.DataFrame
        Columns: frame, cell_id, nucleus_id, match_method
    """
    matches = []

    for t in range(cell_labels.shape[0]):
        cells = cell_labels[t]
        nuclei = nucleus_labels[t]

        cell_ids = set(np.unique(cells)) - {0}
        nuc_ids = set(np.unique(nuclei)) - {0}

        matched_nucs = set()

        # Strategy 1: overlap-based matching
        for nuc_id in nuc_ids:
            nuc_mask = nuclei == nuc_id
            overlapping_cells = np.unique(cells[nuc_mask])
            overlapping_cells = overlapping_cells[overlapping_cells != 0]

            if len(overlapping_cells) == 1:
                matches.append({
                    "frame": t,
                    "cell_id": int(overlapping_cells[0]),
                    "nucleus_id": int(nuc_id),
                    "match_method": "overlap",
                })
                matched_nucs.add(nuc_id)

        # Strategy 2: nearest centroid for remaining nuclei
        unmatched_nucs = nuc_ids - matched_nucs
        if unmatched_nucs and cell_ids:
            cell_centroids = np.array([
                ndimage.center_of_mass(cells == cid) for cid in sorted(cell_ids)
            ])
            cell_id_list = sorted(cell_ids)

            for nuc_id in unmatched_nucs:
                nuc_centroid = np.array(ndimage.center_of_mass(nuclei == nuc_id)).reshape(1, -1)
                dists = cdist(nuc_centroid, cell_centroids).flatten()
                nearest_idx = np.argmin(dists)
                matches.append({
                    "frame": t,
                    "cell_id": int(cell_id_list[nearest_idx]),
                    "nucleus_id": int(nuc_id),
                    "match_method": "nearest_centroid",
                    "distance": float(dists[nearest_idx]),
                })

    return pd.DataFrame(matches)


def measure_fluorescence(
    fluor_stack: np.ndarray,
    cell_labels: np.ndarray,
) -> pd.DataFrame:
    """Measure fluorescence intensity and distribution metrics per cell per frame.

    Parameters
    ----------
    fluor_stack : np.ndarray
        Fluorescence image stack (T, Y, X).
    cell_labels : np.ndarray
        Cell label stack (T, Y, X).

    Returns
    -------
    pd.DataFrame
        Columns: frame, cell_id, mean_intensity, total_intensity,
        min_intensity, max_intensity, std_intensity, cv, skewness,
        kurtosis, nnrm
    """
    records = []
    for t in range(fluor_stack.shape[0]):
        frame = fluor_stack[t]
        labels = cell_labels[t]
        slices = ndimage.find_objects(labels)

        for cid_idx, sl in enumerate(slices):
            if sl is None:
                continue
            cid = cid_idx + 1
            crop_labels = labels[sl]
            crop_frame = frame[sl]
            mask = crop_labels == cid
            pixels = crop_frame[mask].astype(np.float64)
            if pixels.size == 0:
                continue
            mean_val = pixels.mean()
            std_val = pixels.std()

            # CV: coefficient of variation (nucleoid heterogeneity)
            cv = float(std_val / mean_val) if mean_val > 0 else 0.0

            # nNRM: KS statistic vs normal with same mean/std
            # Measures how non-Gaussian the pixel distribution is
            # (Gough et al. 2014, PLOS ONE)
            if std_val > 0:
                ks_stat, _ = stats.kstest(pixels, "norm", args=(mean_val, std_val))
                nnrm = float(ks_stat)
            else:
                nnrm = 0.0

            if std_val > 0:
                z = (pixels - mean_val) / std_val
                skewness = float(np.mean(z ** 3))
                kurtosis = float(np.mean(z ** 4) - 3.0)
            else:
                skewness = kurtosis = 0.0

            records.append({
                "frame": t,
                "cell_id": int(cid),
                "mean_intensity": float(mean_val),
                "total_intensity": float(pixels.sum()),
                "min_intensity": float(pixels.min()),
                "max_intensity": float(pixels.max()),
                "std_intensity": float(std_val),
                "cv": cv,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "nnrm": nnrm,
            })

    return pd.DataFrame(records)


def detect_fluorescence_disappearance(
    tracked: pd.DataFrame,
    threshold: float = -0.5,
    drop_window: int = 1,
) -> pd.DataFrame:
    """Detect per-track fluorescence disappearance frame.

    For each track, finds the frame where total fluorescence intensity
    has the largest relative drop over *drop_window* consecutive frames.
    If the drop exceeds *threshold* (negative), that frame is flagged as
    the fluorescence disappearance frame.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain columns: track_id, frame, total_intensity.
    threshold : float
        Minimum relative change to count as disappearance (e.g., -0.5
        means a 50% drop). Tuned empirically.
    drop_window : int
        Number of frames over which to measure the cumulative drop.
        1 = single-frame drop (default), 2 = two-frame cumulative drop.

    Returns
    -------
    pd.DataFrame
        One row per track with columns: track_id, fluor_disappearance_frame,
        max_drop (the relative change at that frame). Tracks with no drop
        exceeding the threshold have NaN values.
    """
    results = []
    for tid, grp in tracked.groupby("track_id"):
        ts = grp.sort_values("frame")[["frame", "total_intensity"]].copy()
        if len(ts) < drop_window + 1:
            results.append({"track_id": tid, "fluor_disappearance_frame": np.nan,
                            "max_drop": np.nan})
            continue

        ts["prev"] = ts["total_intensity"].shift(drop_window)
        ts["rel_change"] = (ts["total_intensity"] - ts["prev"]) / ts["prev"]
        ts = ts.dropna(subset=["rel_change"])

        if ts.empty:
            results.append({"track_id": tid, "fluor_disappearance_frame": np.nan,
                            "max_drop": np.nan})
            continue

        worst = ts.loc[ts["rel_change"].idxmin()]
        drop_val = float(worst["rel_change"])
        results.append({
            "track_id": tid,
            "fluor_disappearance_frame": int(worst["frame"]) if drop_val <= threshold else np.nan,
            "max_drop": drop_val,
        })

    return pd.DataFrame(results)


def compute_death_clustering(track_stats, n_permutations=1000, min_deaths=5):
    """Test whether cell deaths are spatially clustered.

    Compares mean nearest-neighbor distance among cells that disappeared
    to a null distribution generated by randomly relabeling which cells
    "died" among all cells' last-observed positions.

    Parameters
    ----------
    track_stats : pd.DataFrame
        Must contain: track_id, disappeared, last_y, last_x.
    n_permutations : int
        Number of random permutations for the null distribution.
    min_deaths : int
        Minimum disappeared cells required for analysis.

    Returns
    -------
    dict
        Keys: mean_nn_distance_deaths, mean_nn_distance_random,
        clustering_ratio (observed/random), p_value, null_distribution.
    """
    dead = track_stats[track_stats["disappeared"]]
    if len(dead) < min_deaths:
        return {
            "mean_nn_distance_deaths": np.nan,
            "mean_nn_distance_random": np.nan,
            "clustering_ratio": np.nan,
            "p_value": np.nan,
            "null_distribution": np.array([]),
        }

    all_positions = track_stats[["last_y", "last_x"]].values
    dead_positions = dead[["last_y", "last_x"]].values
    n_dead = len(dead)

    def _mean_nn_dist(positions):
        if len(positions) < 2:
            return np.nan
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        return float(dists.min(axis=1).mean())

    observed = _mean_nn_dist(dead_positions)

    rng = np.random.default_rng(42)
    null_dists = np.empty(n_permutations)
    for i in range(n_permutations):
        idx = rng.choice(len(all_positions), size=n_dead, replace=False)
        null_dists[i] = _mean_nn_dist(all_positions[idx])

    mean_random = float(null_dists.mean())
    p_value = float((null_dists <= observed).sum() / n_permutations)

    return {
        "mean_nn_distance_deaths": observed,
        "mean_nn_distance_random": mean_random,
        "clustering_ratio": observed / mean_random if mean_random > 0 else np.nan,
        "p_value": p_value,
        "null_distribution": null_dists,
    }


def compute_preburst_fluorescence(tracked, track_stats, n_frames=5):
    """Measure fluorescence behavior in the final N frames before burst.

    For disappeared tracks, fits a linear slope to mean_intensity over
    the window [last_frame - n_frames, last_frame - 1] (excluding the
    burst frame itself). A positive slope indicates a pre-burst spike.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain: track_id, frame, mean_intensity.
    track_stats : pd.DataFrame
        Must contain: track_id, disappeared, last_frame.
    n_frames : int
        Number of frames before burst to analyze.

    Returns
    -------
    pd.DataFrame
        One row per disappeared track: track_id, preburst_slope,
        preburst_spike (bool: True if slope > 0 and at least one frame
        in the window exceeds the track's earlier baseline mean).
    """
    disappeared = track_stats[track_stats["disappeared"]]
    grouped = tracked.groupby("track_id")
    records = []

    for _, row in disappeared.iterrows():
        tid = row["track_id"]
        last_f = int(row["last_frame"])
        grp = grouped.get_group(tid).sort_values("frame")
        window = grp[(grp["frame"] >= last_f - n_frames) & (grp["frame"] < last_f)]
        before = grp[grp["frame"] < last_f - n_frames]

        if len(window) < 2:
            records.append({
                "track_id": tid, "preburst_slope": np.nan,
                "preburst_spike": False,
            })
            continue

        frames_arr = window["frame"].values.astype(np.float64)
        intensity = window["mean_intensity"].values.astype(np.float64)
        slope = np.polyfit(frames_arr, intensity, 1)[0]

        baseline_mean = before["mean_intensity"].mean() if len(before) > 0 else intensity[0]
        has_spike = slope > 0 and float(intensity.max()) > baseline_mean

        records.append({
            "track_id": tid,
            "preburst_slope": float(slope),
            "preburst_spike": bool(has_spike),
        })

    return pd.DataFrame(records)


def _get_frame0_with_fate(tracked, track_stats, columns=None):
    frame0_ids = track_stats[track_stats["first_frame"] == 0]["track_id"]
    cols = ["track_id"] + (columns or [])
    frame0 = tracked[
        (tracked["track_id"].isin(frame0_ids)) & (tracked["frame"] == 0)
    ][cols].copy()
    return frame0.merge(
        track_stats[["track_id", "disappeared"]], on="track_id",
    )


def predict_fate_from_frame0(tracked, track_stats, features=None):
    """Logistic regression predicting cell death from frame-0 features.

    Uses leave-one-out cross-validation (appropriate for ~300-400 cells).
    Features are z-scored before fitting.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain: track_id, frame, and columns listed in *features*.
    track_stats : pd.DataFrame
        Must contain: track_id, first_frame, disappeared.
    features : list of str or None
        Column names to use as predictors. Default: area, cv, nnrm.

    Returns
    -------
    pd.DataFrame
        One row per frame-0 cell: track_id, disappeared (actual),
        predicted_prob, predicted_class, plus each feature value.
    dict
        Summary: auc, accuracy, n_cells, feature_importance (coefficients),
        feature_names.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    if features is None:
        features = ["area", "cv", "nnrm"]

    frame0_data = _get_frame0_with_fate(tracked, track_stats, columns=features)
    frame0_data = frame0_data.dropna(subset=features)

    X = frame0_data[features].values
    y = frame0_data["disappeared"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X_scaled):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X_scaled[test_idx])[:, 1]

    full_model = LogisticRegression(max_iter=1000)
    full_model.fit(X_scaled, y)

    result_df = frame0_data[["track_id"] + features].copy()
    result_df["disappeared"] = y.astype(bool)
    result_df["predicted_prob"] = probs
    result_df["predicted_class"] = (probs >= 0.5).astype(bool)

    auc = roc_auc_score(y, probs)
    accuracy = (result_df["predicted_class"] == result_df["disappeared"]).mean()

    coefs = dict(zip(features, full_model.coef_[0]))

    summary = {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "n_cells": len(y),
        "n_died": int(y.sum()),
        "n_survived": int((1 - y).sum()),
        "feature_importance": coefs,
        "feature_names": features,
    }

    return result_df, summary


def analyze_spatial_gradient(tracked, track_stats):
    """Test whether cell fate correlates with spatial position.

    For each axis (centroid_x, centroid_y), computes:
    - Mann-Whitney U test (died vs survived position)
    - Point-biserial correlation (position vs disappeared)
    - Logistic regression AUC (position alone predicting fate)

    Identifies the dominant gradient axis and bins cells into spatial
    quartiles along that axis for stratified analysis.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain: track_id, frame, centroid_x, centroid_y.
    track_stats : pd.DataFrame
        Must contain: track_id, first_frame, disappeared.

    Returns
    -------
    pd.DataFrame
        Frame-0 cells with columns: track_id, centroid_x, centroid_y,
        disappeared, gradient_quartile (1-4 along dominant axis).
    dict
        Summary: gradient_axis, axes_results (per-axis stats),
        quartile_death_rates, auc_position_only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    frame0 = _get_frame0_with_fate(
        tracked, track_stats, columns=["centroid_x", "centroid_y"],
    )

    y = frame0["disappeared"].astype(int).values
    axes_results = {}

    for axis in ["centroid_x", "centroid_y"]:
        pos = frame0[axis].values
        died_pos = pos[y == 1]
        surv_pos = pos[y == 0]

        u_stat, u_p = stats.mannwhitneyu(died_pos, surv_pos, alternative="two-sided")
        r_pb, r_p = stats.pointbiserialr(y, pos)

        axes_results[axis] = {
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": float(u_p),
            "point_biserial_r": float(r_pb),
            "point_biserial_p": float(r_p),
            "died_median": float(np.median(died_pos)),
            "survived_median": float(np.median(surv_pos)),
        }

    x_p = axes_results["centroid_x"]["mann_whitney_p"]
    y_p = axes_results["centroid_y"]["mann_whitney_p"]
    gradient_axis = "centroid_x" if x_p < y_p else "centroid_y"

    pos_col = frame0[gradient_axis].values.reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(pos_col, y)
    probs = model.predict_proba(pos_col)[:, 1]
    auc = roc_auc_score(y, probs)

    frame0["gradient_quartile"] = pd.qcut(
        frame0[gradient_axis], q=4, labels=[1, 2, 3, 4],
    ).astype(int)

    quartile_rates = {}
    for q in range(1, 5):
        mask = frame0["gradient_quartile"] == q
        n_q = mask.sum()
        n_died = frame0.loc[mask, "disappeared"].sum()
        quartile_rates[q] = {
            "n_cells": int(n_q),
            "n_died": int(n_died),
            "death_rate": float(n_died / n_q) if n_q > 0 else 0.0,
        }

    summary = {
        "gradient_axis": gradient_axis,
        "axes_results": axes_results,
        "quartile_death_rates": quartile_rates,
        "auc_position_only": float(auc),
    }

    return frame0, summary
