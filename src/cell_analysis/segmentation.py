"""Cell detection and segmentation for phase-contrast microscopy.

Two methods available:
- Cellpose (default): deep-learning segmentation, much more accurate for
  touching/adjacent cells and ignoring halos/horns.
- Classical: invert + blur + peak detection + watershed + circularity filter.
  Faster, no model download, but more false positives.

See docs/detection_tuning.md for parameter tuning notes, method comparison,
benchmarks, and known limitations.
"""

import logging
import platform
import warnings

import numpy as np
from skimage import measure

# Suppress noisy Cellpose/PyTorch warnings
logging.getLogger("cellpose.models").setLevel(logging.ERROR)
logging.getLogger("cellpose.dynamics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Sparse invariant checks")


def _resolve_gpu(gpu: bool) -> bool:
    """Force CPU on Intel Macs.

    Cellpose 4.x (SAM backbone) uses bfloat16, which is unsupported on the
    Intel-Mac MPS backend (AMD GPU exposed as MPS) and raises
    "BFloat16 is not supported on MPS". Apple Silicon MPS supports bfloat16
    in torch>=2.5, so we only opt out on Darwin x86_64.
    """
    if gpu and platform.system() == "Darwin" and platform.machine() == "x86_64":
        warnings.warn(
            "Forcing gpu=False: Intel Mac MPS lacks bfloat16 needed by Cellpose 4.x",
            stacklevel=2,
        )
        return False
    return gpu


# ---------------------------------------------------------------------------
# Cellpose-based detection (recommended)
# ---------------------------------------------------------------------------

def _make_cellpose_model(gpu: bool, model_type: str | None):
    from cellpose.models import CellposeModel

    kwargs = {"pretrained_model": model_type} if model_type else {}
    return CellposeModel(gpu=_resolve_gpu(gpu), **kwargs)


def detect_cells_frame(
    frame: np.ndarray,
    diameter: float | None = None,
    min_area: int = 200,
    min_circularity: float = 0.7,
    min_contrast: float = 1550,
    min_bg_contrast: float = 0.0,
    exclude_edges: bool = True,
    gpu: bool = False,
    resample: bool = False,
    model_type: str | None = None,
    _model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect round cells in a single phase-contrast frame using Cellpose.

    Dark cells on a lighter background with bright halos are expected.
    The image is inverted before passing to Cellpose so cells appear bright.

    Parameters
    ----------
    frame : np.ndarray
        Single 2D frame (Y, X), uint8 or uint16.
    diameter : float or None
        Expected cell diameter in pixels. None = auto-detect.
    min_area : int
        Reject regions smaller than this (debris).
    min_circularity : float
        Reject non-round shapes (0-1, where 1.0 = perfect circle).
    min_contrast : float
        Minimum intra-mask intensity std-dev — a texture filter, NOT a
        cell-vs-background contrast filter (name kept for backwards
        compatibility). Rejects featureless/smooth false positives that
        Cellpose sometimes returns on flat background regions.
    min_bg_contrast : float
        Minimum |cell_mean − background_median| — the actual cell-vs-
        background contrast. Background median is computed from all
        non-mask pixels in the frame. Default 0 = disabled. Set to a
        positive value to reject cells whose mean intensity is too close
        to the frame background (typical false-positive mode).
    exclude_edges : bool
        Reject cells whose bounding box touches the frame border.
    gpu : bool
        Use GPU acceleration if available (routes to MPS on Apple Silicon).
    resample : bool
        If True, resample masks to original image size (slower but more
        precise boundaries). False is much faster with minimal quality loss.
    _model : CellposeModel or None
        Pre-loaded model instance (avoids reloading per frame).

    Returns
    -------
    centroids : np.ndarray, shape (N, 2)
        (y, x) coordinates of detected cell centers.
    labels : np.ndarray, shape (Y, X)
        Label image (0 = background, >0 = cell ID).
    """
    if _model is None:
        _model = _make_cellpose_model(gpu, model_type)

    # Invert: dark cells become bright for Cellpose
    inverted = frame.max() - frame

    masks, _flows, _styles = _model.eval(inverted, diameter=diameter, resample=resample)

    h, w = frame.shape

    # resample=False returns masks at the model's internal resolution;
    # resize back to original frame size so masks and frame align.
    if masks.shape != frame.shape:
        from skimage.transform import resize
        masks = resize(
            masks, (h, w), order=0, preserve_range=True, anti_aliasing=False,
        ).astype(masks.dtype)

    bg_median = float(np.median(frame[masks == 0])) if min_bg_contrast > 0 else 0.0

    props = measure.regionprops(masks, intensity_image=frame)
    good_centroids = []
    good_labels = set()
    for p in props:
        # Skip cells cut off at the frame border
        if exclude_edges:
            r0, c0, r1, c1 = p.bbox
            if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
                continue

        circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        cell_pixels = frame[masks == p.label]
        texture = float(cell_pixels.std())
        bg_contrast = abs(float(cell_pixels.mean()) - bg_median)

        if (p.area >= min_area
                and circ >= min_circularity
                and texture >= min_contrast
                and bg_contrast >= min_bg_contrast):
            good_centroids.append(p.centroid)
            good_labels.add(p.label)

    # Clean label image
    clean = masks.copy()
    for label_id in np.unique(masks):
        if label_id != 0 and label_id not in good_labels:
            clean[masks == label_id] = 0

    centroids = np.array(good_centroids) if good_centroids else np.empty((0, 2))
    return centroids, clean


def detect_cells_stack(
    stack: np.ndarray,
    diameter: float | None = None,
    min_area: int = 200,
    min_circularity: float = 0.7,
    min_contrast: float = 1550,
    min_bg_contrast: float = 0.0,
    exclude_edges: bool = True,
    gpu: bool = False,
    resample: bool = False,
    model_type: str | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Detect cells in every frame of a time-lapse stack using Cellpose.

    The model is loaded once and reused across frames.

    Parameters
    ----------
    stack : np.ndarray
        Image stack with shape (T, Y, X).
    diameter, min_area, min_circularity, min_contrast, min_bg_contrast,
    exclude_edges, gpu, resample
        See detect_cells_frame.

    Returns
    -------
    centroids_per_frame : list of np.ndarray
        Each element is (N_t, 2) array of (y, x) centroids for that frame.
    label_stack : np.ndarray
        Label stack with shape (T, Y, X).
    """
    model = _make_cellpose_model(gpu, model_type)

    centroids_per_frame = []
    label_stack = np.zeros_like(stack, dtype=np.int32)

    for t in range(stack.shape[0]):
        centroids, labels = detect_cells_frame(
            stack[t],
            diameter=diameter,
            min_area=min_area,
            min_circularity=min_circularity,
            min_contrast=min_contrast,
            min_bg_contrast=min_bg_contrast,
            exclude_edges=exclude_edges,
            gpu=gpu,
            resample=resample,
            _model=model,
        )
        centroids_per_frame.append(centroids)
        label_stack[t] = labels
        print(f"  Frame {t:2d}: {len(centroids)} cells")

    return centroids_per_frame, label_stack


def detect_nuclei_stack(
    fluor_stack: np.ndarray,
    diameter: float = 25,
    min_area: int = 100,
    gpu: bool = False,
    resample: bool = False,
    model_type: str | None = None,
) -> np.ndarray:
    """Segment fluorescent nuclei across all frames using Cellpose.

    Parameters
    ----------
    fluor_stack : np.ndarray
        Fluorescence image stack (T, Y, X).
    diameter : float
        Expected nucleus diameter in pixels.
    min_area : int
        Reject regions smaller than this.
    gpu : bool
        Use GPU acceleration if available.
    resample : bool
        Resample masks to original resolution.

    Returns
    -------
    nucleus_label_stack : np.ndarray
        Label stack (T, Y, X) with integer nucleus IDs.
    """
    model = _make_cellpose_model(gpu, model_type)
    T, H, W = fluor_stack.shape
    label_stack = np.zeros((T, H, W), dtype=np.int32)

    for t in range(T):
        # No inversion: fluorescence nuclei are already bright on dark background,
        # unlike phase-contrast cells which require inversion in detect_cells_frame.
        masks, _, _ = model.eval(
            fluor_stack[t], diameter=diameter, resample=resample,
        )
        if masks.shape != (H, W):
            from skimage.transform import resize
            masks = resize(
                masks, (H, W), order=0, preserve_range=True,
                anti_aliasing=False,
            ).astype(masks.dtype)

        if min_area > 0:
            props = measure.regionprops(masks)
            reject = [p.label for p in props if p.area < min_area]
            if reject:
                masks[np.isin(masks, reject)] = 0

        label_stack[t] = masks
        n = len(np.unique(masks)) - 1
        print(f"  Frame {t:2d}: {n} nuclei")

    return label_stack


def profile_nucleus_detection(
    fluor_stack: np.ndarray,
    model_type: str | None = None,
    diameter: float = 25,
    gpu: bool = True,
    resample: bool = False,
) -> dict:
    """Time one nucleus inference on the fluorescence stack (frame 0)."""
    import time

    print("  Loading model (downloads on first use, can take minutes)...", flush=True)
    model = _make_cellpose_model(gpu, model_type)
    print("  Running inference (cpsam on CPU is 5-15 min/frame at full res)...", flush=True)
    t0 = time.perf_counter()
    model.eval(fluor_stack[0], diameter=diameter, resample=resample)
    dt = time.perf_counter() - t0

    T = fluor_stack.shape[0]
    label = model_type or "default"
    print(f"  Nucleus inference: {dt:.1f} s/frame  (model_type={label})")
    print(f"  Nucleus pass over {T} frames: ~{dt * T / 60:.1f} min")
    return {"sec_per_frame": dt, "n_frames": T, "est_minutes": dt * T / 60}


def profile_detection(
    stack: np.ndarray,
    model_type: str | None = None,
    nucleus_pass: bool = True,
    **detect_params,
) -> dict:
    """Time one detection call to estimate full-stack Cellpose runtime.

    Loads the model once (excluded from the timing), runs one inference on
    frame 0, then prints projected totals. Use this on slow hardware to
    decide whether to switch model_type or skip nucleus segmentation.
    """
    import time

    gpu = detect_params.pop("gpu", True)
    print("  Loading model (downloads on first use, can take minutes)...", flush=True)
    model = _make_cellpose_model(gpu, model_type)
    print("  Running inference (cpsam on CPU is 5-15 min/frame at full res)...", flush=True)

    t0 = time.perf_counter()
    detect_cells_frame(stack[0], _model=model, **detect_params)
    dt = time.perf_counter() - t0

    T = stack.shape[0]
    passes = 2 if nucleus_pass else 1
    label = model_type or "default"
    print(f"  Inference: {dt:.1f} s/frame  (model_type={label})")
    print(f"  Phase pass over {T} frames: ~{dt * T / 60:.1f} min")
    if nucleus_pass:
        print(f"  With nucleus pass: ~{dt * T * passes / 60:.1f} min")
    return {"sec_per_frame": dt, "n_frames": T, "est_minutes": dt * T * passes / 60}


# ---------------------------------------------------------------------------
# Classical detection (fallback -- no deep learning needed)
# ---------------------------------------------------------------------------

def detect_cells_frame_classical(
    frame: np.ndarray,
    blur_sigma: float = 8.0,
    min_distance: int = 18,
    peak_threshold: float = 0.12,
    min_area: int = 400,
    max_area: int = 6000,
    min_circularity: float = 0.45,
    intensity_ratio: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect round cells using classical image processing (no ML).

    Inverts image, blurs to merge horns, finds peaks, runs watershed,
    then filters by circularity + area + intensity.

    Parameters
    ----------
    frame : np.ndarray
        Single 2D frame (Y, X), uint8 or uint16.
    blur_sigma : float
        Gaussian sigma. Larger values merge horns into the cell body.
    min_distance : int
        Minimum pixel distance between detected cell centers.
    peak_threshold : float
        Minimum peak intensity (0-1 normalized) to count as a cell.
    min_area, max_area : int
        Area bounds (pixels) for valid cells.
    min_circularity : float
        Minimum circularity (4*pi*area/perimeter^2). 1.0 = perfect circle.
    intensity_ratio : float
        Cell mean intensity must be below background_median * this ratio.

    Returns
    -------
    centroids : np.ndarray, shape (N, 2)
        (y, x) coordinates of detected cell centers.
    labels : np.ndarray, shape (Y, X)
        Watershed label image (0 = background, >0 = cell ID).
    """
    from scipy import ndimage
    from skimage import feature, filters, morphology, segmentation

    f = frame.astype(np.float64)

    p1, p99 = np.percentile(f, [1, 99])
    normed = np.clip((f - p1) / (p99 - p1), 0, 1)
    inverted = 1.0 - normed
    smoothed = filters.gaussian(inverted, sigma=blur_sigma)

    coords = feature.peak_local_max(
        smoothed, min_distance=min_distance, threshold_abs=peak_threshold,
    )

    markers = np.zeros(f.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    thresh = filters.threshold_otsu(smoothed)
    foreground = smoothed > thresh
    foreground = morphology.opening(foreground, morphology.disk(3))
    foreground = ndimage.binary_fill_holes(foreground)

    labels = segmentation.watershed(-smoothed, markers, mask=foreground)

    median_bg = np.median(f[labels == 0])
    props = measure.regionprops(labels, intensity_image=f)

    good_centroids = []
    good_labels = set()
    for p in props:
        circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        if (
            circ >= min_circularity
            and min_area <= p.area <= max_area
            and p.intensity_mean < median_bg * intensity_ratio
        ):
            good_centroids.append(p.centroid)
            good_labels.add(p.label)

    clean_labels = labels.copy()
    for label_id in np.unique(labels):
        if label_id != 0 and label_id not in good_labels:
            clean_labels[labels == label_id] = 0

    centroids = np.array(good_centroids) if good_centroids else np.empty((0, 2))
    return centroids, clean_labels
