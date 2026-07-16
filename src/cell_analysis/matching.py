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
                nuc_centroid = np.array(ndimage.center_of_mass(
                    nuclei == nuc_id)).reshape(1, -1)
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


PERI_CORE_RINGS_DEFAULT = (0.15, 0.35, 0.55)


def _peri_core_asymmetry(pixels, coords, R, rings=PERI_CORE_RINGS_DEFAULT):
    """Signed peri/core asymmetry index for one cell.

    *rings* is a triple ``(core_max, peri_min, peri_max)`` giving the ring
    boundaries as fractions of R (the mask's equivalent radius). Defaults
    to ``(0.15, 0.35, 0.55)`` — narrow inner disk and a peripheral annulus
    bracketed to the donut-peak zone identified from control-cell radial
    profiles. See ``docs/peri_core_asymmetry_definition_history.md`` for
    prior parameter sets and the sweep that motivated the current default.

    Returns ``(mean_peri - mean_core) / (mean_peri + mean_core)``. The
    intermediate band (core_max <= r/R <= peri_min) and the rim
    (r/R >= peri_max) are ignored. Areas are unequal, so mean rather than
    total intensity is used. Returns np.nan if either ring is empty or the
    summed mean intensity is non-positive.
    """
    core_max, peri_min, peri_max = rings
    cy, cx = coords.mean(axis=0)
    dy = coords[:, 0] - cy
    dx = coords[:, 1] - cx
    r2 = dy * dy + dx * dx
    core_pix = r2 < (core_max * R) ** 2
    peri_pix = (r2 > (peri_min * R) ** 2) & (r2 < (peri_max * R) ** 2)
    if not core_pix.any() or not peri_pix.any():
        return np.nan
    m_core = float(pixels[core_pix].mean())
    m_peri = float(pixels[peri_pix].mean())
    total = m_core + m_peri
    if total <= 0:
        return np.nan
    return (m_peri - m_core) / total


def measure_nucleoid_metrics(pixels, crop_mask, flat_threshold_ratio,
                             peri_core_rings=PERI_CORE_RINGS_DEFAULT):
    """Per-cell nucleoid spatial-distribution metrics.

    Returns a triple ``(edge_norm, sigma_norm, asym)``. Any element may be
    np.nan if that specific metric could not be computed for this cell.

    See :func:`cell_analysis.pipeline.add_nucleoid_distribution` for the
    metric definitions and threshold rule.
    """
    R = np.sqrt(crop_mask.sum() / np.pi)
    coords = np.argwhere(crop_mask)
    asym = _peri_core_asymmetry(pixels, coords, R, rings=peri_core_rings)

    cell_mean = pixels.mean()
    if pixels.max() / cell_mean < flat_threshold_ratio:
        threshold = 0.5 * cell_mean
    else:
        threshold = cell_mean
    keep = pixels > threshold
    if not keep.any():
        return np.nan, np.nan, asym

    edt = ndimage.distance_transform_edt(crop_mask)
    edt_vals = edt[crop_mask][keep]
    edge_norm = edt_vals.mean() / (R / 3.0)

    kept_coords = coords[keep]
    weights = pixels[keep]
    wsum = weights.sum()
    if wsum <= 0:
        return edge_norm, np.nan, asym
    cy = (kept_coords[:, 0] * weights).sum() / wsum
    cx = (kept_coords[:, 1] * weights).sum() / wsum
    dy = kept_coords[:, 0] - cy
    dx = kept_coords[:, 1] - cx
    cov_yy = (weights * dy * dy).sum() / wsum
    cov_xx = (weights * dx * dx).sum() / wsum
    cov_yx = (weights * dy * dx).sum() / wsum
    cov = np.array([[cov_yy, cov_yx], [cov_yx, cov_xx]])
    eigvals = np.linalg.eigvalsh(cov)
    sigma = float(np.sqrt(max(0.0, eigvals.mean())))
    sigma_norm = sigma / R

    return edge_norm, sigma_norm, asym


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
                ks_stat, _ = stats.kstest(
                    pixels, stats.norm(mean_val, std_val).cdf)
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


def measure_post_disappearance_fluorescence(track_stats, last_labels,
                                            fluor_stack, label_stack,
                                            window):
    """Measure mean fluorescence in a cell's last-known phase mask
    applied to up to ``window`` subsequent fluorescence frames.

    Parameters
    ----------
    track_stats : DataFrame
        Must have columns ``track_id``, ``last_frame``, ``disappeared``.
    last_labels : DataFrame
        Two columns: ``track_id``, ``last_frame_label`` (the label-id the
        cell carried in ``label_stack[last_frame]``).
    fluor_stack, label_stack : ndarray
        Shape (T, Y, X). label_stack[t] holds integer cell labels.
    window : int
        Number of post-disappearance frames to measure.

    Returns
    -------
    DataFrame with columns ``track_id``, ``frame``, ``mean_intensity``.
    Rows for which ``last_frame + k`` >= T are omitted.
    """
    T = fluor_stack.shape[0]
    disappeared = track_stats[track_stats["disappeared"]]
    merged = disappeared.merge(last_labels, on="track_id", how="left")

    rows = []
    for _, r in merged.iterrows():
        tid = r["track_id"]
        last_frame = int(r["last_frame"])
        last_label = r["last_frame_label"]
        if pd.isna(last_label):
            continue
        mask = label_stack[last_frame] == int(last_label)
        if not mask.any():
            continue
        for k in range(1, window + 1):
            t = last_frame + k
            if t >= T:
                break
            mean_i = float(fluor_stack[t][mask].mean())
            rows.append({"track_id": tid, "frame": t,
                         "mean_intensity": mean_i})

    return pd.DataFrame(rows, columns=["track_id", "frame", "mean_intensity"])


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
        window = grp[(grp["frame"] >= last_f - n_frames)
                     & (grp["frame"] < last_f)]
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

        baseline_mean = before["mean_intensity"].mean() if len(
            before) > 0 else intensity[0]
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


def compare_frame0_features_by_fate(tracked, track_stats, features=None):
    """Mann-Whitney U on each frame-0 feature, survived vs disappeared.

    Univariate counterpart to :func:`predict_fate_from_frame0`: tests whether
    each initial feature differs between cells that eventually die and cells
    that survive. The logistic regression model in
    :func:`predict_fate_from_frame0` combines these features jointly.

    Parameters
    ----------
    tracked, track_stats : pd.DataFrame
        Same inputs as :func:`predict_fate_from_frame0`.
    features : list of str or None
        Columns to test. Default: area, cv, nnrm, mean_edge_distance_norm,
        gaussian_sigma_norm.

    Returns
    -------
    pd.DataFrame
        One row per feature with: n_survived, n_died,
        median_survived, median_died, U, p_value.
    """
    if features is None:
        features = ["area", "cv", "nnrm",
                    "mean_edge_distance_norm", "gaussian_sigma_norm"]

    frame0_data = _get_frame0_with_fate(tracked, track_stats, columns=features)

    rows = []
    for feat in features:
        sub = frame0_data[["disappeared", feat]].dropna()
        survived = sub.loc[~sub["disappeared"], feat].values
        died = sub.loc[sub["disappeared"], feat].values
        u_stat, p = stats.mannwhitneyu(died, survived, alternative="two-sided")
        rows.append({
            "feature": feat,
            "n_survived": int(len(survived)),
            "n_died": int(len(died)),
            "median_survived": float(np.median(survived)) if len(survived) else float("nan"),
            "median_died": float(np.median(died)) if len(died) else float("nan"),
            "U": float(u_stat),
            "p_value": float(p),
        })

    return pd.DataFrame(rows)


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
        Column names to use as predictors. Default: area, cv, nnrm,
        mean_edge_distance_norm, gaussian_sigma_norm.

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
    from sklearn.metrics import (
        roc_auc_score,
        confusion_matrix as sk_confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    if features is None:
        features = ["area", "cv", "nnrm",
                    "mean_edge_distance_norm", "gaussian_sigma_norm"]

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
    accuracy = (result_df["predicted_class"] ==
                result_df["disappeared"]).mean()

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

    y_pred = result_df["predicted_class"].astype(int).values
    cm = sk_confusion_matrix(y, y_pred, labels=[0, 1])
    # Positive class = disappeared (label 1); negative class = survived (label 0).
    summary["confusion_matrix"] = {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], zero_division=0.0,
    )
    summary["per_class"] = {
        "survived": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "disappeared": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }

    return result_df, summary
