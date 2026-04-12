#!/usr/bin/env python3
"""Generate diagnostic overlays showing accepted vs rejected cell detections.

Produces two images in the results/ directory:
  - diagnostic_full.png  : full frame with accepted (red X) and rejected (cyan O)
  - diagnostic_crops.png : six zoomed crops with rejection reason labels

Usage:
  # Run with current defaults
  python scripts/diagnostic_overlay.py

  # Override detection parameters
  python scripts/diagnostic_overlay.py --min_contrast 1400 --min_area 200

  # Use a different frame
  python scripts/diagnostic_overlay.py --frame 5

  # Custom stack path
  python scripts/diagnostic_overlay.py --stack path/to/stack.tif

  # Custom crop regions (Y0:Y1:X0:X1, up to 6)
  python scripts/diagnostic_overlay.py --crops 150:400:250:550 400:650:700:1000

All detection parameters are optional and default to the values in DETECT_PARAMS
in the notebook (see segmentation.py docstring for tuning notes).
"""

import argparse
import logging
import warnings
from pathlib import Path

# Suppress noisy Cellpose/PyTorch warnings
logging.getLogger("cellpose.models").setLevel(logging.ERROR)
logging.getLogger("cellpose.dynamics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Sparse invariant checks")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage import measure


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic overlay: accepted vs rejected cells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--stack", default="data/raw/Gradient-0011.zvi  Ch0.tif",
                    help="Path to the TIF stack (default: %(default)s)")
    p.add_argument("--frame", type=int, default=0,
                    help="Frame index to analyse (default: 0)")
    p.add_argument("--outdir", default="results",
                    help="Output directory (default: %(default)s)")

    # Detection parameters
    g = p.add_argument_group("detection parameters")
    g.add_argument("--diameter", type=float, default=32)
    g.add_argument("--min_area", type=int, default=300)
    g.add_argument("--min_circularity", type=float, default=0.7)
    g.add_argument("--min_contrast", type=float, default=1550)
    g.add_argument("--exclude_edges", type=bool, default=True)
    g.add_argument("--no_exclude_edges", action="store_true",
                    help="Include edge cells")
    g.add_argument("--gpu", action="store_true", default=True)
    g.add_argument("--no_gpu", action="store_true")
    g.add_argument("--resample", action="store_true", default=False)

    # Crop regions
    p.add_argument("--crops", nargs="*", metavar="Y0:Y1:X0:X1",
                   help="Custom crop regions (up to 6). Format: Y0:Y1:X0:X1")

    return p.parse_args()


def classify_cells(frame, masks, min_area, min_circularity, min_contrast,
                   exclude_edges):
    """Classify Cellpose detections into accepted/rejected with reasons."""
    props = measure.regionprops(masks, intensity_image=frame)
    h, w = frame.shape

    accepted = []
    rejected = []

    for p in props:
        r0, c0, r1, c1 = p.bbox
        edge = r0 == 0 or c0 == 0 or r1 == h or c1 == w
        circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        contrast = float(frame[masks == p.label].std())
        cy, cx = p.centroid

        reasons = []
        if exclude_edges and edge:
            reasons.append("edge")
        if p.area < min_area:
            reasons.append(f"area={p.area}")
        if circ < min_circularity:
            reasons.append(f"circ={circ:.2f}")
        if contrast < min_contrast:
            reasons.append(f"c={contrast:.0f}")

        if reasons:
            rejected.append((cy, cx, ", ".join(reasons)))
        else:
            accepted.append((cy, cx))

    return accepted, rejected


def plot_full(frame, accepted, rejected, outdir, frame_idx):
    """Full-frame overlay with accepted (red X) and rejected (cyan O)."""
    p1, p99 = np.percentile(frame, [1, 99])
    acc = np.array(accepted) if accepted else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(frame, cmap="gray", vmin=p1, vmax=p99)

    if len(acc):
        ax.plot(acc[:, 1], acc[:, 0], "rx", markersize=6, markeredgewidth=1.2)

    for cy, cx, reasons in rejected:
        ax.plot(cx, cy, "co", markersize=8, markeredgewidth=1.5, fillstyle="none")
        ax.text(cx + 8, cy, reasons, color="cyan", fontsize=5, va="center")

    ax.set_title(
        f"Frame {frame_idx} — Red X = accepted ({len(accepted)}), "
        f"Cyan O = rejected ({len(rejected)})",
        fontsize=14,
    )
    ax.axis("off")
    plt.tight_layout()

    path = Path(outdir) / "diagnostic_full.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_crops(frame, accepted, rejected, outdir, frame_idx, crop_specs=None):
    """Zoomed crops with rejection reason labels."""
    p1, p99 = np.percentile(frame, [1, 99])
    h, w = frame.shape

    if crop_specs:
        crops = []
        for spec in crop_specs[:6]:
            parts = spec.split(":")
            y0, y1, x0, x1 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            crops.append((f"Y{y0}-{y1} X{x0}-{x1}", slice(y0, y1), slice(x0, x1)))
    else:
        # Auto-generate 6 crops spread across the frame
        cy, cx = h // 2, w // 2
        ch, cw = min(200, h // 3), min(350, w // 3)
        crops = [
            ("top-left",     slice(80, 80 + ch),     slice(400, 400 + cw)),
            ("top-center",   slice(80, 80 + ch),     slice(cx - cw // 2, cx + cw // 2)),
            ("mid-left",     slice(cy - ch // 2, cy + ch // 2), slice(100, 100 + cw)),
            ("mid-center",   slice(cy - ch // 2, cy + ch // 2), slice(cx - cw // 2, cx + cw // 2)),
            ("bot-left",     slice(h - 80 - ch, h - 80), slice(150, 150 + cw)),
            ("bot-right",    slice(h - 80 - ch, h - 80), slice(w - 100 - cw, w - 100)),
        ]

    ncols = min(3, len(crops))
    nrows = (len(crops) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (name, ys, xs) in enumerate(crops):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(
            frame[ys, xs], cmap="gray", vmin=p1, vmax=p99,
            extent=[xs.start, xs.stop, ys.stop, ys.start],
        )
        # Accepted cells in crop
        for cy, cx in accepted:
            if ys.start <= cy < ys.stop and xs.start <= cx < xs.stop:
                ax.plot(cx, cy, "rx", markersize=10, markeredgewidth=2)
        # Rejected cells in crop
        for cy, cx, reasons in rejected:
            if ys.start <= cy < ys.stop and xs.start <= cx < xs.stop:
                ax.plot(cx, cy, "co", markersize=12, markeredgewidth=2,
                        fillstyle="none")
                ax.text(cx + 8, cy, reasons, color="cyan", fontsize=7,
                        va="center")
        ax.set_title(name, fontsize=12)
        ax.axis("off")

    # Hide unused axes
    for idx in range(len(crops), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Frame {frame_idx} — accepted: {len(accepted)}, "
        f"rejected: {len(rejected)}",
        fontsize=14,
    )
    plt.tight_layout()

    path = Path(outdir) / "diagnostic_crops.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    args = parse_args()

    exclude_edges = args.exclude_edges and not args.no_exclude_edges
    gpu = args.gpu and not args.no_gpu

    # Print parameters for reproducibility
    print(f"Stack:  {args.stack}")
    print(f"Frame:  {args.frame}")
    print(f"Params: diameter={args.diameter}, min_area={args.min_area}, "
          f"min_circularity={args.min_circularity}, "
          f"min_contrast={args.min_contrast}, "
          f"exclude_edges={exclude_edges}, gpu={gpu}, "
          f"resample={args.resample}")
    print()

    # Load
    stack = tifffile.imread(args.stack)
    if stack.ndim == 2:
        frame = stack
    else:
        frame = stack[args.frame]
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    # Run Cellpose
    from cellpose.models import CellposeModel

    print("Running Cellpose...")
    model = CellposeModel(gpu=gpu)
    inverted = frame.max() - frame
    masks, _, _ = model.eval(inverted, diameter=args.diameter,
                             resample=args.resample)

    if masks.shape != frame.shape:
        from skimage.transform import resize
        masks = resize(
            masks, frame.shape, order=0, preserve_range=True,
            anti_aliasing=False,
        ).astype(masks.dtype)

    raw_count = len(np.unique(masks)) - 1  # exclude background
    print(f"Cellpose raw detections: {raw_count}")

    # Classify
    accepted, rejected = classify_cells(
        frame, masks, args.min_area, args.min_circularity,
        args.min_contrast, exclude_edges,
    )
    print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
    print()

    # Plot
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    plot_full(frame, accepted, rejected, args.outdir, args.frame)
    plot_crops(frame, accepted, rejected, args.outdir, args.frame,
               crop_specs=args.crops)

    # Summary table of rejected reasons
    from collections import Counter
    reason_counts = Counter()
    for _, _, reasons in rejected:
        for r in reasons.split(", "):
            tag = r.split("=")[0]
            reason_counts[tag] += 1
    print("Rejection reasons (cells can have multiple):")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
