"""Match phase-contrast cells to fluorescence nuclei."""

import numpy as np
import pandas as pd
from scipy import ndimage
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
    """Measure fluorescence intensity per cell per frame.

    Parameters
    ----------
    fluor_stack : np.ndarray
        Fluorescence image stack (T, Y, X).
    cell_labels : np.ndarray
        Cell label stack (T, Y, X).

    Returns
    -------
    pd.DataFrame
        Columns: frame, cell_id, mean_intensity, total_intensity, max_intensity
    """
    records = []
    for t in range(fluor_stack.shape[0]):
        frame = fluor_stack[t]
        labels = cell_labels[t]
        cell_ids = np.unique(labels)
        cell_ids = cell_ids[cell_ids != 0]

        for cid in cell_ids:
            mask = labels == cid
            pixels = frame[mask]
            records.append({
                "frame": t,
                "cell_id": int(cid),
                "mean_intensity": float(pixels.mean()),
                "total_intensity": float(pixels.sum()),
                "max_intensity": float(pixels.max()),
            })

    return pd.DataFrame(records)
