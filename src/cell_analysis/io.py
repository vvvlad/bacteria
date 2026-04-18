"""Loading and saving image stacks and results."""

from pathlib import Path

import numpy as np
import tifffile


def load_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF image stack.

    Parameters
    ----------
    path : str or Path
        Path to a .tif/.tiff file. Expected shapes:
        - Single channel time-lapse: (T, Y, X)
        - Multi-channel time-lapse: (T, C, Y, X)

    Returns
    -------
    np.ndarray
        The image stack.
    """
    return tifffile.imread(str(path))


def load_paired_stacks(
    phase_path: str | Path,
    fluor_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load paired phase-contrast and fluorescence stacks.

    Validates that both stacks have matching T, Y, X dimensions.

    Returns
    -------
    tuple of (phase_stack, fluor_stack)
        Both as np.ndarray with shape (T, Y, X).
    """
    phase = load_stack(phase_path)
    fluor = load_stack(fluor_path)

    # If multi-channel, take first channel
    if phase.ndim == 4:
        phase = phase[:, 0]
    if fluor.ndim == 4:
        fluor = fluor[:, 0]

    if phase.shape != fluor.shape:
        raise ValueError(
            f"Shape mismatch: phase {phase.shape} vs fluorescence {fluor.shape}. "
            "Both stacks must have identical (T, Y, X) dimensions."
        )
    return phase, fluor


def save_results(df, path: str | Path) -> None:
    """Save a pandas DataFrame of results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary(data: dict, path: str | Path) -> None:
    """Flatten a nested summary dict and save as a single-row CSV.

    Recursively flattens nested dicts into underscore-joined column names.
    Skips array-like values (e.g. null distributions) that don't fit a
    single-row tabular format.
    """
    import pandas as pd

    def _flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_flatten(v, prefix=f"{key}_"))
            elif hasattr(v, "__len__") and not isinstance(v, str):
                continue
            else:
                flat[key] = v
        return flat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_flatten(data)]).to_csv(path, index=False)
