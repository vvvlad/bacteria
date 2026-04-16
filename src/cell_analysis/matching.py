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
) -> pd.DataFrame:
    """Detect per-track fluorescence disappearance frame.

    For each track, finds the frame where total fluorescence intensity
    has the largest single-frame relative drop. If the drop exceeds
    *threshold* (negative), that frame is flagged as the fluorescence
    disappearance frame.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain columns: track_id, frame, total_intensity.
    threshold : float
        Minimum relative change to count as disappearance (e.g., -0.5
        means a 50% drop). Tuned empirically.

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
        if len(ts) < 2:
            results.append({"track_id": tid, "fluor_disappearance_frame": np.nan,
                            "max_drop": np.nan})
            continue

        ts["prev"] = ts["total_intensity"].shift(1)
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
