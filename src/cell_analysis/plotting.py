"""Visualization functions for the cell analysis pipeline.

Each public function produces a self-contained figure with printed
summary statistics, designed to be called as one-liners from the
analysis notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_contrast(frame):
    return np.percentile(frame, [1, 99])


def _frame0_track_ids(tracked):
    return tracked.loc[tracked["frame"] == 0, "track_id"].unique()


def _survival_split(tracked):
    frame0_tracks = _frame0_track_ids(tracked)
    last_frame = tracked["frame"].max()
    per_track = (
        tracked[tracked["track_id"].isin(frame0_tracks)]
        .groupby("track_id")
        .agg(last_frame=("frame", "max"))
        .reset_index()
    )
    per_track["survived"] = per_track["last_frame"] == last_frame
    survived = per_track.loc[per_track["survived"], "track_id"]
    disappeared = per_track.loc[~per_track["survived"], "track_id"]
    return survived, disappeared


def _outcome_split_panel(ax, tracked, metric, survived_ids, disappeared_ids):
    for label, ids, color in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = tracked[tracked["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")[metric].agg(["mean", "sem"])
        ax.fill_between(
            g.index, g["mean"] - g["sem"], g["mean"] + g["sem"],
            color=color, alpha=0.2,
        )
        ax.plot(
            g.index, g["mean"], "o-", color=color,
            linewidth=2, markersize=4, label=f"{label} (n={len(ids)})",
        )
    ax.set_xticks(
        range(int(tracked["frame"].min()), int(tracked["frame"].max()) + 1)
    )
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_frame_preview(stack, frame=0):
    """Display a single frame with auto-contrast stretch."""
    f = stack[frame]
    p1, p99 = _auto_contrast(f)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(f, cmap="gray", vmin=p1, vmax=p99)
    ax.set_title(f"Frame {frame}")
    ax.axis("off")
    plt.tight_layout()


def plot_detections(frame, centroids, zoom=None):
    """Show detected cell centroids overlaid on a frame.

    If *zoom* is provided as ``(y_slice, x_slice)``, shows a zoomed crop
    instead of the full frame.
    """
    p1, p99 = _auto_contrast(frame)

    if zoom is None:
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.imshow(frame, cmap="gray", vmin=p1, vmax=p99)
        ax.plot(
            centroids[:, 1], centroids[:, 0],
            "rx", markersize=7, markeredgewidth=1.5,
        )
        ax.set_title(f"{len(centroids)} cells detected")
        ax.axis("off")
    else:
        y_slice, x_slice = zoom
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(
            frame[y_slice, x_slice], cmap="gray", vmin=p1, vmax=p99,
            extent=[x_slice.start, x_slice.stop, y_slice.stop, y_slice.start],
        )
        in_crop = (
            (centroids[:, 0] >= y_slice.start)
            & (centroids[:, 0] < y_slice.stop)
            & (centroids[:, 1] >= x_slice.start)
            & (centroids[:, 1] < x_slice.stop)
        )
        c = centroids[in_crop]
        ax.plot(c[:, 1], c[:, 0], "rx", markersize=12, markeredgewidth=2)
        ax.set_title(f"Zoomed \u2014 {in_crop.sum()} cells in crop")
        ax.axis("off")
    plt.tight_layout()


def plot_frame_gating(diagnostics, bad_frames):
    """Show per-frame detection counts with flagged frames highlighted."""
    from IPython.display import display, Markdown

    if bad_frames:
        display(Markdown(f"**Dropped {len(bad_frames)} frame(s):** {bad_frames}"))
        dropped = diagnostics[diagnostics["flagged"]]
        display(
            dropped[["frame", "cell_count", "mean_area", "iqr_area", "reasons"]]
        )
    else:
        print("No frames dropped \u2014 all frames passed quality gating.")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        diagnostics["frame"], diagnostics["cell_count"],
        "o-", color="steelblue",
    )
    if bad_frames:
        bad_diag = diagnostics[diagnostics["flagged"]]
        ax.scatter(
            bad_diag["frame"], bad_diag["cell_count"],
            color="red", s=100, zorder=5, label="Dropped",
        )
        ax.legend()
    ax.set(
        xlabel="Frame", ylabel="Cell count",
        title="Detection count per frame (red = dropped)",
    )
    ax.set_xticks(diagnostics["frame"])
    plt.tight_layout()


def plot_cells_per_frame(tracked):
    """Plot tracked cell count over time with 50% disappearance marker."""
    cells_per_frame = tracked.groupby("frame")["track_id"].nunique()
    initial_count = cells_per_frame.iloc[0]
    final_count = cells_per_frame.iloc[-1]
    half_count = initial_count / 2
    fraction_disappeared = (initial_count - final_count) / initial_count

    frames_below_half = cells_per_frame[cells_per_frame <= half_count]
    frame_50pct = (
        int(frames_below_half.index[0]) if len(frames_below_half) > 0 else None
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        cells_per_frame.index, cells_per_frame.values,
        "o-", color="steelblue", linewidth=2,
    )
    ax.axhline(
        half_count, color="red", linestyle="--", alpha=0.6,
        label=f"50% of initial ({half_count:.0f})",
    )
    if frame_50pct is not None:
        ax.axvline(frame_50pct, color="red", linestyle=":", alpha=0.6)
        ax.annotate(
            f"50% at frame {frame_50pct}",
            xy=(frame_50pct, half_count),
            xytext=(frame_50pct + 1, half_count + 15),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.6),
            fontsize=10, color="red",
        )
    ax.set(xlabel="Frame", ylabel="Cell count", title="Cells detected per frame")
    ax.set_xticks(cells_per_frame.index)
    ax.legend()
    plt.tight_layout()

    print(f"Initial count (frame 0): {initial_count}")
    print(f"Final count (frame {cells_per_frame.index[-1]}): {final_count}")
    print(f"Fraction disappeared: {fraction_disappeared:.1%}")
    if frame_50pct is not None:
        print(f"50% disappearance frame: {frame_50pct}")
    else:
        print("50% disappearance not reached within the stack")


def plot_lifetime_distribution(track_stats):
    """Show track lifetime histogram and disappearances per frame."""
    last_frame = track_stats["last_frame"].max()
    disappeared = track_stats[track_stats["disappeared"]].sort_values("last_frame")
    survivors = track_stats[~track_stats["disappeared"]]
    fraction_survived = len(survivors) / len(track_stats)

    print(f"Total tracks: {len(track_stats)}")
    print(f"Survived to final frame: {len(survivors)} ({fraction_survived:.1%})")
    print(f"Disappeared: {len(disappeared)} ({1 - fraction_survived:.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(
        track_stats["lifetime"], bins=range(1, last_frame + 3),
        ax=axes[0], color="steelblue", edgecolor="white",
    )
    axes[0].axvline(
        track_stats["lifetime"].median(), color="red", linestyle="--",
        label=f"Median: {track_stats['lifetime'].median():.0f}",
    )
    axes[0].set(
        xlabel="Lifetime (frames)", ylabel="Count",
        title="Track lifetime distribution",
    )
    axes[0].legend()

    disappearances = disappeared.groupby("last_frame").size()
    disappearances = disappearances.reindex(range(last_frame + 1), fill_value=0)
    ax = axes[1]
    ax.bar(
        disappearances.index, disappearances.values,
        color="steelblue", edgecolor="white",
    )
    peak_frame = int(disappearances.idxmax())
    peak_count = int(disappearances.max())
    ax.annotate(
        f"Peak: frame {peak_frame} ({peak_count} cells)",
        xy=(peak_frame, peak_count),
        xytext=(peak_frame + 1.5, peak_count + 2),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10, color="red",
    )
    ax.set(
        xlabel="Frame", ylabel="Cells disappeared",
        title="Disappearances per frame",
    )
    ax.set_xticks(range(0, last_frame + 1))

    plt.tight_layout()

    print(f"\nFrame of maximum disappearance: {peak_frame}")
    print(f"Cells lost at peak: {peak_count}")


def plot_area_distribution(tracked):
    """Show cell area histogram with equivalent volume secondary axis."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(
        tracked["area"], bins=30, ax=ax,
        color="steelblue", edgecolor="white",
    )
    ax.axvline(
        tracked["area"].median(), color="red", linestyle="--",
        label=f"Median: {tracked['area'].median():.0f} px\u00b2",
    )
    ax.set(
        xlabel="Area (px\u00b2)", ylabel="Count",
        title="Cell area distribution (all frames)",
    )
    ax.legend()

    ax_vol = ax.twiny()
    area_ticks = ax.get_xticks()
    vol_ticks = (
        (4 / 3) * np.pi * (np.sqrt(np.maximum(area_ticks, 0) / np.pi)) ** 3
    )
    ax_vol.set_xlim(ax.get_xlim())
    ax_vol.set_xticks(area_ticks)
    ax_vol.set_xticklabels([f"{v:.0f}" for v in vol_ticks], fontsize=8)
    ax_vol.set_xlabel("Equivalent volume (px\u00b3)", fontsize=9)

    plt.tight_layout()

    print(f"Median area:         {tracked['area'].median():.0f} px\u00b2")
    print(f"Median volume:       {tracked['volume'].median():.0f} px\u00b3")
    print(f"Median surface area: {tracked['surface_area'].median():.0f} px\u00b2")


def plot_swelling_dynamics(tracked):
    """Plot V(t)/V(0) and S(t)/S(0) for the frame-0 cohort."""
    frame0_tracks = _frame0_track_ids(tracked)
    cohort = tracked[tracked["track_id"].isin(frame0_tracks)].copy()
    print(f"Tracks starting at frame 0: {len(frame0_tracks)}")

    first_obs = cohort.sort_values("frame").groupby("track_id").first()
    cohort = cohort.merge(first_obs["volume"].rename("V0"), on="track_id")
    cohort = cohort.merge(
        first_obs["surface_area"].rename("S0"), on="track_id",
    )
    cohort["V_rel"] = cohort["volume"] / cohort["V0"]
    cohort["S_rel"] = cohort["surface_area"] / cohort["S0"]

    v_stats = cohort.groupby("frame")["V_rel"].agg(["mean", "sem", "count"])
    s_stats = cohort.groupby("frame")["S_rel"].agg(["mean", "sem", "count"])

    rng = np.random.default_rng(42)
    subset_ids = rng.choice(
        frame0_tracks, size=min(20, len(frame0_tracks)), replace=False,
    )
    subset = cohort[cohort["track_id"].isin(subset_ids)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, stats, sub_col, title, ylabel in [
        (axes[0], v_stats, "V_rel",
         "Relative volume  V(t) / V(0)", "V(t) / V(0)"),
        (axes[1], s_stats, "S_rel",
         "Relative surface area  S(t) / S(0)", "S(t) / S(0)"),
    ]:
        for tid in subset_ids:
            t = subset[subset["track_id"] == tid]
            ax.plot(
                t["frame"], t[sub_col],
                color="steelblue", alpha=0.1, linewidth=0.8,
            )

        frames = stats.index
        ax.fill_between(
            frames,
            stats["mean"] - stats["sem"],
            stats["mean"] + stats["sem"],
            color="steelblue", alpha=0.3, label="Mean \u00b1 SEM",
        )
        ax.plot(
            frames, stats["mean"], "o-", color="steelblue",
            linewidth=2, markersize=4,
            label=f"Mean (n={len(frame0_tracks)} initial tracks)",
        )
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set(xlabel="Frame", title=title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(int(frames.min()), int(frames.max()) + 1))
        ax.legend(fontsize=9)

        final_mean = stats["mean"].iloc[-1]
        final_frame = int(frames[-1])
        ax.annotate(
            f"{final_mean:.2f}x",
            xy=(final_frame, final_mean),
            xytext=(final_frame - 3, final_mean + 0.05),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10, color="red", fontweight="bold",
        )

    plt.suptitle(
        "Cell swelling dynamics (frame-0 cohort, spherical assumption)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    print(
        f"\nFinal V(t)/V(0) at frame {int(v_stats.index[-1])}: "
        f"{v_stats['mean'].iloc[-1]:.3f} \u00b1 {v_stats['sem'].iloc[-1]:.3f}"
    )
    print(
        f"Final S(t)/S(0) at frame {int(s_stats.index[-1])}: "
        f"{s_stats['mean'].iloc[-1]:.3f} \u00b1 {s_stats['sem'].iloc[-1]:.3f}"
    )
    print(f"Cells contributing at final frame: "
          f"{int(v_stats['count'].iloc[-1])}")


def plot_swelling_vs_survival(tracked):
    """Scatter of max swelling vs initial volume, and V(t)/V(0) by outcome."""
    from numpy.polynomial.polynomial import polyfit

    frame0_tracks = _frame0_track_ids(tracked)
    cohort = tracked[tracked["track_id"].isin(frame0_tracks)].copy()

    first_obs = cohort.sort_values("frame").groupby("track_id").first()
    cohort = cohort.merge(first_obs["volume"].rename("V0"), on="track_id")
    cohort["V_rel"] = cohort["volume"] / cohort["V0"]

    per_track = cohort.groupby("track_id").agg(
        V0=("V0", "first"),
        V_rel_max=("V_rel", "max"),
        V_rel_final=("V_rel", "last"),
        last_frame=("frame", "max"),
    ).reset_index()
    per_track["survived"] = per_track["last_frame"] == cohort["frame"].max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: scatter
    ax = axes[0]
    ax.scatter(
        per_track["V0"], per_track["V_rel_max"],
        c=per_track["survived"].map({True: "steelblue", False: "tomato"}),
        alpha=0.5, s=25, edgecolors="none",
    )
    ax.set(
        xlabel="Initial volume V(0) (px\u00b3)",
        ylabel="Max V(t) / V(0)",
        title="Swelling extent vs. initial cell size",
    )

    mask = np.isfinite(per_track["V0"]) & np.isfinite(per_track["V_rel_max"])
    if mask.sum() > 2:
        coeffs = polyfit(
            per_track.loc[mask, "V0"],
            per_track.loc[mask, "V_rel_max"], 1,
        )
        x_fit = np.linspace(
            per_track["V0"].min(), per_track["V0"].max(), 100,
        )
        ax.plot(
            x_fit, coeffs[0] + coeffs[1] * x_fit,
            "r--", alpha=0.7, linewidth=1.5,
            label=f"Linear fit (slope={coeffs[1]:.2e})",
        )
    ax.scatter([], [], c="steelblue", s=25, label="Survived")
    ax.scatter([], [], c="tomato", s=25, label="Disappeared")
    ax.legend(fontsize=9)

    # Right: outcome split
    ax = axes[1]
    survived_ids = per_track.loc[per_track["survived"], "track_id"]
    disappeared_ids = per_track.loc[~per_track["survived"], "track_id"]

    for label, ids, color in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = cohort[cohort["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")["V_rel"].agg(["mean", "sem"])
        ax.fill_between(
            g.index, g["mean"] - g["sem"], g["mean"] + g["sem"],
            color=color, alpha=0.2,
        )
        ax.plot(
            g.index, g["mean"], "o-", color=color,
            linewidth=2, markersize=4, label=f"{label} (n={len(ids)})",
        )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set(
        xlabel="Frame", ylabel="V(t) / V(0)",
        title="Swelling dynamics: disappeared vs. survived",
    )
    ax.set_xticks(
        range(int(cohort["frame"].min()), int(cohort["frame"].max()) + 1)
    )
    ax.legend(fontsize=9)

    plt.tight_layout()

    n_surv = per_track["survived"].sum()
    n_dis = (~per_track["survived"]).sum()
    print(f"Frame-0 cohort: {n_surv} survived, {n_dis} disappeared")
    print(
        f"Median max swelling (survived):    "
        f"{per_track.loc[per_track['survived'], 'V_rel_max'].median():.3f}x"
    )
    print(
        f"Median max swelling (disappeared): "
        f"{per_track.loc[~per_track['survived'], 'V_rel_max'].median():.3f}x"
    )


def plot_channels_preview(phase_stack, fluor_stack, frame=0):
    """Show phase-contrast and fluorescence channels side by side."""
    f_phase = phase_stack[frame]
    f_fluor = fluor_stack[frame]
    p1, p99 = _auto_contrast(f_phase)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].imshow(f_phase, cmap="gray", vmin=p1, vmax=p99)
    axes[0].set_title(f"Phase-contrast (frame {frame})")
    axes[0].axis("off")

    fp1, fp99 = np.percentile(f_fluor[f_fluor > 0], [1, 99])
    axes[1].imshow(f_fluor, cmap="hot", vmin=0, vmax=fp99)
    axes[1].set_title(
        f"Fluorescence (frame {frame}, mean={f_fluor.mean():.0f})"
    )
    axes[1].axis("off")

    plt.tight_layout()


def plot_fluorescence_per_frame(tracked):
    """Plot population mean fluorescence intensity over time."""
    fluor_per_frame = tracked.groupby("frame")["mean_intensity"].agg(
        ["mean", "sem"],
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        fluor_per_frame.index,
        fluor_per_frame["mean"] - fluor_per_frame["sem"],
        fluor_per_frame["mean"] + fluor_per_frame["sem"],
        color="darkorange", alpha=0.3,
    )
    ax.plot(
        fluor_per_frame.index, fluor_per_frame["mean"], "o-",
        color="darkorange", linewidth=2, markersize=4, label="Mean +/- SEM",
    )
    ax.set(
        xlabel="Frame", ylabel="Mean fluorescence intensity",
        title="Population mean fluorescence per frame",
    )
    ax.set_xticks(fluor_per_frame.index)
    ax.legend()
    plt.tight_layout()

    print(f"Frame 0 mean fluorescence: "
          f"{fluor_per_frame['mean'].iloc[0]:.0f}")
    print(f"Frame {int(fluor_per_frame.index[-1])} mean fluorescence: "
          f"{fluor_per_frame['mean'].iloc[-1]:.0f}")
    print(f"Change: "
          f"{fluor_per_frame['mean'].iloc[-1] / fluor_per_frame['mean'].iloc[0]:.2f}x")


def plot_relative_fluorescence(tracked):
    """Plot F(t)/F(0) for the frame-0 cohort, overall and by outcome."""
    frame0_tracks = _frame0_track_ids(tracked)
    cohort = tracked[tracked["track_id"].isin(frame0_tracks)].copy()

    f0_vals = (
        cohort.sort_values("frame")
        .groupby("track_id")["mean_intensity"]
        .first()
        .rename("F0")
    )
    cohort = cohort.merge(f0_vals, on="track_id")
    cohort["F_rel"] = np.where(
        cohort["F0"] > 0,
        cohort["mean_intensity"] / cohort["F0"],
        np.nan,
    )

    f_stats = cohort.groupby("frame")["F_rel"].agg(["mean", "sem", "count"])
    survived_ids, disappeared_ids = _survival_split(tracked)

    rng = np.random.default_rng(42)
    subset_ids = rng.choice(
        frame0_tracks, size=min(20, len(frame0_tracks)), replace=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: population F(t)/F(0)
    ax = axes[0]
    for tid in subset_ids:
        t = cohort[cohort["track_id"] == tid]
        ax.plot(
            t["frame"], t["F_rel"],
            color="darkorange", alpha=0.1, linewidth=0.8,
        )
    ax.fill_between(
        f_stats.index,
        f_stats["mean"] - f_stats["sem"],
        f_stats["mean"] + f_stats["sem"],
        color="darkorange", alpha=0.3, label="Mean +/- SEM",
    )
    ax.plot(
        f_stats.index, f_stats["mean"], "o-", color="darkorange",
        linewidth=2, markersize=4,
        label=f"Mean (n={len(frame0_tracks)} tracks)",
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set(
        xlabel="Frame", ylabel="F(t) / F(0)",
        title="Relative fluorescence  F(t) / F(0)",
    )
    ax.set_xticks(
        range(int(f_stats.index.min()), int(f_stats.index.max()) + 1)
    )
    ax.legend(fontsize=9)

    final_f = f_stats["mean"].iloc[-1]
    ax.annotate(
        f"{final_f:.2f}x",
        xy=(int(f_stats.index[-1]), final_f),
        xytext=(int(f_stats.index[-1]) - 3, final_f + 0.05),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        fontsize=10, color="red", fontweight="bold",
    )

    # Right: outcome split
    ax = axes[1]
    for label, ids, color in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = cohort[cohort["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")["F_rel"].agg(["mean", "sem"])
        ax.fill_between(
            g.index, g["mean"] - g["sem"], g["mean"] + g["sem"],
            color=color, alpha=0.2,
        )
        ax.plot(
            g.index, g["mean"], "o-", color=color,
            linewidth=2, markersize=4, label=f"{label} (n={len(ids)})",
        )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set(
        xlabel="Frame", ylabel="F(t) / F(0)",
        title="Fluorescence dynamics: disappeared vs. survived",
    )
    ax.set_xticks(
        range(int(cohort["frame"].min()), int(cohort["frame"].max()) + 1)
    )
    ax.legend(fontsize=9)

    plt.tight_layout()

    print(
        f"Final F(t)/F(0): {f_stats['mean'].iloc[-1]:.3f} "
        f"+/- {f_stats['sem'].iloc[-1]:.3f}"
    )
    print(
        f"Cells contributing at final frame: {int(f_stats['count'].iloc[-1])}"
    )


def plot_fluorescence_vs_volume(tracked):
    """Scatter total and mean fluorescence vs cell volume."""
    sample = tracked.dropna(subset=["total_intensity", "volume"]).sample(
        n=min(3000, len(tracked)), random_state=42,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, y_col, y_label, title in [
        (axes[0], "total_intensity", "Total fluorescence intensity",
         "Total fluorescence vs. cell volume"),
        (axes[1], "mean_intensity", "Mean fluorescence intensity",
         "Mean fluorescence vs. cell volume"),
    ]:
        ax.scatter(
            sample["volume"], sample[y_col],
            alpha=0.15, s=10, color="darkorange", edgecolors="none",
        )
        ax.set(xlabel="Volume (px^3)", ylabel=y_label, title=title)

        mask = np.isfinite(sample["volume"]) & np.isfinite(sample[y_col])
        if mask.sum() > 2:
            r = np.corrcoef(
                sample.loc[mask, "volume"], sample.loc[mask, y_col],
            )[0, 1]
            ax.annotate(
                f"r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=12, fontweight="bold", va="top",
            )

    plt.tight_layout()


def plot_metric_dynamics(tracked, track_stats, metric, label, color):
    """3-panel plot: per-frame, lifespan-normalized, and outcome-split.

    Generic for any per-cell metric (CV, nNRM, etc.).
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Left: per-frame
    ax = axes[0]
    per_frame = tracked.groupby("frame")[metric].agg(["mean", "sem"])
    ax.fill_between(
        per_frame.index,
        per_frame["mean"] - per_frame["sem"],
        per_frame["mean"] + per_frame["sem"],
        color=color, alpha=0.3,
    )
    ax.plot(
        per_frame.index, per_frame["mean"], "o-", color=color,
        linewidth=2, markersize=4, label="Mean +/- SEM",
    )
    ax.set(
        xlabel="Frame", ylabel=label,
        title=f"Population mean {label} per frame",
    )
    ax.set_xticks(per_frame.index)
    ax.legend(fontsize=9)

    # Center: lifespan-normalized
    ax = axes[1]
    n_bins = 20
    norm_parts = []
    for tid, grp in tracked.groupby("track_id"):
        grp = grp.sort_values("frame")
        if len(grp) < 2:
            continue
        first, last = grp["frame"].iloc[0], grp["frame"].iloc[-1]
        if first == last:
            continue
        grp = grp.copy()
        grp["t_norm"] = (grp["frame"] - first) / (last - first)
        norm_parts.append(grp[["track_id", "t_norm", metric]])

    norm_df = pd.concat(norm_parts, ignore_index=True)
    norm_df["bin"] = pd.cut(norm_df["t_norm"], bins=n_bins, labels=False)
    bin_centers = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    bin_stats = norm_df.groupby("bin")[metric].agg(["mean", "sem"])

    ax.fill_between(
        bin_centers,
        bin_stats["mean"] - bin_stats["sem"],
        bin_stats["mean"] + bin_stats["sem"],
        color=color, alpha=0.3,
    )
    ax.plot(
        bin_centers, bin_stats["mean"], "o-", color=color,
        linewidth=2, markersize=4,
        label=f"Mean +/- SEM (n={norm_df['track_id'].nunique()} tracks)",
    )
    ax.set(
        xlabel="Relative lifespan (0=start, 1=end)", ylabel=label,
        title=f"{label} over normalized lifespan",
    )
    ax.legend(fontsize=9)

    # Right: outcome split (frame-0 cohort)
    ax = axes[2]
    survived_ids, disappeared_ids = _survival_split(tracked)
    _outcome_split_panel(ax, tracked, metric, survived_ids, disappeared_ids)
    ax.set(
        xlabel="Frame", ylabel=label,
        title=f"{label} dynamics: disappeared vs. survived",
    )

    plt.tight_layout()

    print(f"{label} at frame 0: {per_frame['mean'].iloc[0]:.3f}")
    print(
        f"{label} at frame {int(per_frame.index[-1])}: "
        f"{per_frame['mean'].iloc[-1]:.3f}"
    )


def plot_fluorescence_disappearance(tracked, track_stats, threshold=-0.3):
    """3-panel: drop distribution, disappearance timing, phase comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Left: drop distribution
    ax = axes[0]
    all_drops = track_stats["max_drop"].dropna()
    sns.histplot(all_drops, bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(
        threshold, color="red", linestyle="--",
        label=f"Threshold: {threshold:.0%}",
    )
    ax.set(
        xlabel="Largest single-frame relative change", ylabel="Count",
        title="Distribution of max fluorescence drops per track",
    )
    ax.legend(fontsize=9)

    # Center: disappearance per frame
    ax = axes[1]
    fd_frames = track_stats["fluor_disappearance_frame"].dropna()
    if len(fd_frames) > 0:
        fd_counts = fd_frames.value_counts().sort_index()
        fd_counts = fd_counts.reindex(
            range(int(tracked["frame"].max()) + 1), fill_value=0,
        )
        ax.bar(
            fd_counts.index, fd_counts.values,
            color="darkorange", edgecolor="white",
        )
        peak_fd = int(fd_counts.idxmax())
        ax.annotate(
            f"Peak: frame {peak_fd}",
            xy=(peak_fd, fd_counts.max()),
            xytext=(peak_fd + 1.5, fd_counts.max() + 1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )
    ax.set(
        xlabel="Frame", ylabel="Count",
        title="Fluorescence disappearance per frame",
    )
    ax.set_xticks(range(int(tracked["frame"].max()) + 1))

    # Right: fluor vs phase timing
    ax = axes[2]
    both = track_stats.dropna(subset=["fluor_disappearance_frame"])
    both_dis = both[both["disappeared"]]
    if len(both_dis) > 0:
        ax.scatter(
            both_dis["last_frame"], both_dis["fluor_disappearance_frame"],
            alpha=0.4, s=20, color="tomato", edgecolors="none",
        )
        lims = [0, tracked["frame"].max()]
        ax.plot(lims, lims, "k--", alpha=0.3, label="Same frame")
        ax.set(
            xlabel="Phase disappearance frame",
            ylabel="Fluorescence disappearance frame",
            title="Fluorescence vs. phase disappearance timing",
        )
        ax.legend(fontsize=9)

        before = (
            both_dis["fluor_disappearance_frame"] < both_dis["last_frame"]
        ).sum()
        same = (
            both_dis["fluor_disappearance_frame"] == both_dis["last_frame"]
        ).sum()
        after = (
            both_dis["fluor_disappearance_frame"] > both_dis["last_frame"]
        ).sum()
        print("\nFor cells that disappeared from phase:")
        print(f"  Fluor drop BEFORE phase disappearance: {before}")
        print(f"  Fluor drop AT same frame: {same}")
        print(f"  Fluor drop AFTER (shouldn't happen): {after}")

    plt.tight_layout()
