"""Visualization functions for the cell analysis pipeline.

Each public function produces a self-contained figure with printed
summary statistics, designed to be called as one-liners from the
analysis notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Source CSV registry
# ---------------------------------------------------------------------------

PLOT_SOURCES: dict[str, list[str]] = {
    "plot_frame_gating": ["frame_diagnostics.csv", "dropped_frames.csv"],
    "plot_cells_per_frame": ["tracked_cells.csv"],
    "plot_cells_per_concenration": ["tracked_cells.csv"],
    "plot_lifetime_distribution": ["track_statistics.csv"],
    "plot_area_distribution": ["tracked_cells.csv"],
    "plot_burst_conc_all_metrics": ["tracked_cells.csv", "track_statistics.csv"],
    "check_frames_quality": ["tracked_cells.csv"],
    "plot_swelling_dynamics": ["tracked_cells.csv"],
    "plot_swelling_per_conc": ["tracked_cells.csv", "track_statistics.csv"],
    "plot_swelling_vs_survival": ["tracked_cells.csv"],
    "plot_fluorescence_per_frame": ["tracked_cells.csv"],
    "plot_relative_fluorescence": ["tracked_cells.csv"],
    "plot_fluorescence_alignment": ["fluorescence_alignment.csv"],
    "plot_peri_core_alignment": ["peri_core_alignment.csv"],
    "plot_fluorescence_vs_volume": ["tracked_cells.csv"],
    "plot_metric_dynamics": ["tracked_cells.csv", "track_statistics.csv"],
    "plot_fluorescence_concentration": ["tracked_cells.csv"],
    "plot_sav_ratio": ["tracked_cells.csv", "track_statistics.csv"],
    "plot_preburst_fluorescence": ["tracked_cells.csv", "track_statistics.csv"],
    "plot_fate_prediction": ["fate_predictions.csv", "fate_prediction_summary.csv"],
    "plot_nucleus_persistence": [
        "nucleus_persistence.csv", "nucleus_persistence_summary.csv",
    ],
    "plot_initial_features_vs_lifespan": ["tracked_cells.csv", "track_statistics.csv"],
    "plot_tension": ["tracked_cells.csv", "track_statistics.csv"],
    "plot_tension_comparison": ["tracked_cells.csv", "track_statistics.csv", "tracked_cells.csv", "track_statistics.csv"],
}


def show_with_source(plot_func, *args, **kwargs):
    """Render *plot_func* and a markdown line linking to its source CSVs.

    Looks up ``PLOT_SOURCES[plot_func.__name__]`` and renders one link per
    source CSV below the figure. Plots not in the registry render no link
    (raw-image previews, etc.). Links are relative filenames, so they
    resolve when the exported HTML report sits next to the CSVs in the
    results directory.
    """
    from IPython.display import display, Markdown

    plot_func(*args, **kwargs)
    plt.show()

    sources = PLOT_SOURCES.get(plot_func.__name__)
    if not sources:
        return

    links = ", ".join(f"[`{name}`]({name})" for name in sources)
    label = "Source" if len(sources) == 1 else "Sources"
    display(Markdown(f"_{label}: {links}_"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_contrast(frame):
    return np.percentile(frame, [1, 99])


def _frame0_track_ids(tracked):
    return tracked.loc[tracked["frame"] == 0, "track_id"].unique()


def _survival_split(tracked, cohort="all"):
    """Split tracks by fate (present at the last frame vs not).

    cohort="all" (default) splits every track; cohort="frame0" restricts
    to tracks present at frame 0 (use this only when the metric needs a
    frame-0 baseline, e.g. F(t)/F(0)).
    """
    last_frame = tracked["frame"].max()
    scope = (
        tracked[tracked["track_id"].isin(_frame0_track_ids(tracked))]
        if cohort == "frame0"
        else tracked
    )
    per_track = (
        scope.groupby("track_id")
        .agg(last_frame=("frame", "max"))
        .reset_index()
    )
    per_track["survived"] = per_track["last_frame"] == last_frame
    survived = per_track.loc[per_track["survived"], "track_id"]
    disappeared = per_track.loc[~per_track["survived"], "track_id"]
    return survived, disappeared


def _scatter_scope(sample, full):
    """Annotation describing how many points the scatter shows vs the full set."""
    if len(sample) == len(full):
        return f"scatter: all {len(full)} points"
    return f"scatter: {len(sample)} of {len(full)} (random)"


_DEFAULT_MARGIN = dict(t=80, b=50, l=60, r=30)


def _rgba(color, alpha):
    """Convert a matplotlib named colour to an 'rgba(r,g,b,a)' string for Plotly."""
    r, g, b, _ = mcolors.to_rgba(color)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def _frame_xaxis(fig, frames, row=None, col=None, title="Frame"):
    """Apply a 1-per-frame integer-tick policy to a subplot's x-axis."""
    if hasattr(frames, "min") and hasattr(frames, "max"):
        tickvals = list(range(int(frames.min()), int(frames.max()) + 1))
    else:
        tickvals = list(frames)
    if row is None:
        fig.update_xaxes(title_text=title, tickmode="array", tickvals=tickvals)
    else:
        fig.update_xaxes(title_text=title, tickmode="array", tickvals=tickvals,
                         row=row, col=col)


def _add_mean_sem_band(
    fig, x, mean, sem, color, name, legendgroup, row, col, y_label="value",
):
    """Plotly equivalent of fill_between + line+markers for a mean+/-SEM band."""
    x = list(x)
    upper = (mean + sem).tolist()
    lower = (mean - sem).tolist()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor=_rgba(color, 0.25),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
            legendgroup=legendgroup,
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=mean.tolist(), mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=5),
            name=name, legendgroup=legendgroup,
            hovertemplate=(
                f"Frame %{{x}}<br>{y_label} %{{y:.4f}}<extra>{name}</extra>"
            ),
        ),
        row=row, col=col,
    )


def _finalize_plotly(
    fig, *, log_menu=True, bin_menu_traces=None, height=None, margin=None,
):
    """Add per-figure updateMenus and render the figure with editable text.

    log_menu adds a Linear/Log y-axis button group (top-right above the figure)
    that switches ALL y-axes in the figure at once. bin_menu_traces is a list
    of trace indices that should get a 10/25/50/100 bins dropdown (top-left).
    margin defaults to a shared house style; pass a dict to override.
    The figure is rendered via fig.show() with config={"editable": True} so
    titles, axis labels, legend names, and annotation text are click-to-edit.
    """
    menus = []
    y_keys = [k for k in fig.layout if str(k).startswith("yaxis")]
    if log_menu and y_keys:
        menus.append(dict(
            type="buttons", direction="right",
            x=1.0, y=1.08, xanchor="right", yanchor="bottom",
            showactive=True, pad=dict(t=0, r=0),
            buttons=[
                dict(label="Linear", method="relayout",
                     args=[{f"{k}.type": "linear" for k in y_keys}]),
                dict(label="Log", method="relayout",
                     args=[{f"{k}.type": "log" for k in y_keys}]),
            ],
        ))
    if bin_menu_traces:
        menus.append(dict(
            type="dropdown", direction="down",
            x=0.0, y=1.08, xanchor="left", yanchor="bottom",
            showactive=True, pad=dict(t=0, l=0),
            buttons=[
                dict(label=f"{n} bins", method="restyle",
                     args=[{"nbinsx": n}, list(bin_menu_traces)])
                for n in (10, 25, 50, 100)
            ],
        ))
    update = dict(updatemenus=menus, margin=margin or _DEFAULT_MARGIN)
    if height is not None:
        update["height"] = height
    fig.update_layout(**update)
    fig.show(config={"editable": True, "displaylogo": False, "responsive": True})


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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diagnostics["frame"], y=diagnostics["cell_count"],
        mode="lines+markers", line=dict(color="steelblue", width=2),
        marker=dict(size=6),
        name="Cell count",
        hovertemplate="Frame %{x}<br>Count %{y}<extra></extra>",
    ))
    if bad_frames:
        bad_diag = diagnostics[diagnostics["flagged"]]
        fig.add_trace(go.Scatter(
            x=bad_diag["frame"], y=bad_diag["cell_count"],
            mode="markers", marker=dict(color="red", size=12),
            name="Dropped",
            hovertemplate="Frame %{x}<br>Count %{y}<extra>Dropped</extra>",
        ))
    fig.update_layout(
        title="Detection count per frame (red = dropped)",
        xaxis=dict(title="Frame", tickmode="array",
                   tickvals=list(diagnostics["frame"])),
        yaxis=dict(title="Cell count"),
    )
    _finalize_plotly(fig, height=400)


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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(cells_per_frame.index), y=list(cells_per_frame.values),
        mode="lines+markers", line=dict(color="steelblue", width=2),
        marker=dict(size=6), name="Cell count",
        hovertemplate="Frame %{x}<br>Count %{y}<extra></extra>",
    ))
    fig.add_hline(
        y=half_count, line=dict(color="red", dash="dash", width=1),
        opacity=0.6,
        annotation_text=f"50% of initial ({half_count:.0f})",
        annotation_position="top right",
    )
    if frame_50pct is not None:
        fig.add_vline(
            x=frame_50pct, line=dict(color="red", dash="dot", width=1),
            opacity=0.6,
        )
        fig.add_annotation(
            x=frame_50pct, y=half_count,
            ax=frame_50pct + 1, ay=half_count + 15,
            xref="x", yref="y", axref="x", ayref="y",
            text=f"50% at frame {frame_50pct}",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12),
        )
    fig.update_layout(
        title="Cells detected per frame",
        xaxis=dict(title="Frame", tickmode="array",
                   tickvals=list(cells_per_frame.index)),
        yaxis=dict(title="Cell count"),
    )
    _finalize_plotly(fig, height=500)

    print(f"Initial count (frame 0): {initial_count}")
    print(f"Final count (frame {cells_per_frame.index[-1]}): {final_count}")
    print(f"Fraction disappeared: {fraction_disappeared:.1%}")
    if frame_50pct is not None:
        print(f"50% disappearance frame: {frame_50pct}")
    else:
        print("50% disappearance not reached within the stack")

def plot_cells_per_concenration(tracked):
    """Plot tracked cell count over concenration with 50% disappearance marker."""
    max_conc = tracked["Sucr %"].max()
    frames_with_max_conc = tracked[tracked["Sucr %"] == max_conc]["frame"].unique()
    start_frame = frames_with_max_conc.max()

    tracked_filtered = tracked[tracked["frame"] >= start_frame]

    frame_stats = tracked_filtered.groupby("frame").agg(
        cell_count=("track_id", "nunique"),
        sucr_conc=("Sucr %", "mean") 
    ).reset_index()

    initial_count = frame_stats["cell_count"].iloc[0]
    final_count = frame_stats["cell_count"].iloc[-1]
    half_count = initial_count / 2
    fraction_disappeared = (initial_count - final_count) / initial_count

    frames_below_half = frame_stats[frame_stats["cell_count"] <= half_count]
    conc_50pct = (
        frames_below_half["sucr_conc"].iloc[0] if len(frames_below_half) > 0 else None
    )
    frame_stats = frame_stats.sort_values(by="sucr_conc", ascending=False)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frame_stats["sucr_conc"], 
        y=frame_stats["cell_count"],
        customdata=frame_stats["frame"], 
        mode="lines+markers", line=dict(color="steelblue", width=2),
        marker=dict(size=6), name="Cell count",
        hovertemplate="Frame: %{customdata}<br>Sucr: %{x}%<br>Count: %{y}<extra></extra>",
    ))
    
    fig.add_hline(
        y=half_count, line=dict(color="red", dash="dash", width=1),
        opacity=0.6,
        annotation_text=f"50% of initial ({half_count:.0f})",
        annotation_position="top right",
    )
    
    if conc_50pct is not None:
        fig.add_vline(
            x=conc_50pct, line=dict(color="red", dash="dot", width=1),
            opacity=0.6,
        )
        fig.add_annotation(
            x=conc_50pct, y=half_count,
            ax=conc_50pct + 1, ay=half_count + 15,
            xref="x", yref="y", axref="x", ayref="y",
            text=f"50% at ~{conc_50pct:.1f}% conc",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12),
        )
        
    fig.update_layout(
        title="Cells detected vs Sucrose Concentration",
        xaxis=dict(title="Sucrose Concentration (%)", autorange="reversed"),
        yaxis=dict(title="Cell count"),
    )
    
    try:
        _finalize_plotly(fig, height=500)
    except NameError:
        fig.show()

    print(f"Gradient starts decreasing after frame: {start_frame}")
    print(f"Initial count (at {frame_stats['sucr_conc'].iloc[0]:.1f}%): {initial_count}")
    print(f"Final count (at {frame_stats['sucr_conc'].iloc[-1]:.1f}%): {final_count}")
    print(f"Fraction disappeared: {fraction_disappeared:.1%}")
    
    if conc_50pct is not None:
        print(f"50% disappearance at concentration: {conc_50pct:.1f}%")
    else:
        print("50% disappearance not reached within the stack")


def plot_lifetime_distribution(track_stats, min_frame = 0, max_frame = None):
    """Show track lifetime histogram and disappearances per frame."""

    if max_frame is None:
        max_frame = track_stats["last_frame"].max()

    track_stats = track_stats[
        (track_stats["first_frame"] >= min_frame) & 
        (track_stats["last_frame"] <= max_frame)
    ].copy()
    
    disappeared = track_stats[track_stats["disappeared"]].sort_values("last_frame")
    survivors = track_stats[~track_stats["disappeared"]]
    fraction_survived = len(survivors) / len(track_stats)

    print(f"Total tracks: {len(track_stats)}")
    print(f"Survived to final frame: {len(survivors)} ({fraction_survived:.1%})")
    print(f"Disappeared: {len(disappeared)} ({1 - fraction_survived:.1%})")
    if len(disappeared) > 0:
        d_life = disappeared["lifetime"]
        print(f"Disappeared lifetime — median: {d_life.median():.0f}, "
              f"mean: {d_life.mean():.1f} ± {d_life.std():.1f} frames")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Track lifetime distribution", "Disappearances per frame"),
        horizontal_spacing=0.1,
    )
    if len(track_stats) > 0:
        median_life = track_stats["lifetime"].median()
        fig.add_trace(go.Histogram(
            x=track_stats["lifetime"], nbinsx=int(max_frame - min_frame + 2),
            marker_color="steelblue", marker_line_color="white",
            marker_line_width=1, name="Lifetime",
            hovertemplate="Lifetime %{x}<br>Count %{y}<extra></extra>",
        ), row=1, col=1)
        fig.add_vline(
            x=median_life, line=dict(color="red", dash="dash"),
            annotation_text=f"Median: {median_life:.0f}",
            annotation_position="top right",
            row=1, col=1,
        )

    if len(disappeared) > 0:
        disappearances = disappeared.groupby("last_frame").size()
        disappearances = disappearances.reindex(range(min_frame, max_frame + 1), fill_value=0)
        fig.add_trace(go.Bar(
            x=list(disappearances.index), y=list(disappearances.values),
            marker_color="steelblue", marker_line_color="white",
            marker_line_width=1, name="Disappeared",
            hovertemplate="Frame %{x}<br>Lost %{y}<extra></extra>",
        ), row=1, col=2)
        peak_frame = int(disappearances.idxmax())
        peak_count = int(disappearances.max())
        fig.add_annotation(
            x=peak_frame, y=peak_count,
            ax=peak_frame + 1.5, ay=peak_count + 2,
            xref="x2", yref="y2", axref="x2", ayref="y2",
            text=f"Peak: frame {peak_frame} ({peak_count} cells)",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12),
        )

    fig.update_xaxes(title_text="Lifetime (frames)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2, tickmode="array",
                     tickvals=list(range(min_frame, max_frame + 1)))
    fig.update_yaxes(title_text="Cells disappeared", row=1, col=2)
    fig.update_layout(showlegend=False)

    _finalize_plotly(fig, bin_menu_traces=[0], height=450)

    print(f"\nFrame of maximum disappearance: {peak_frame}")
    print(f"Cells lost at peak: {peak_count}")


def plot_area_distribution(tracked, baseline_frames=3):
    """Show cell area histogram pooled over the first ``baseline_frames`` frames.

    The window defaults to 3 (the cohort before medium changes drive swelling).
    Set ``baseline_frames=1`` to restrict to frame 0 only. Hover reveals each
    cell's equivalent spherical volume.
    """

    cohort = tracked[tracked["frame"] < baseline_frames]
    areas = cohort["area"]
    median_area = areas.median()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=areas, nbinsx=30,
        marker_color="steelblue", marker_line_color="white",
        marker_line_width=1, name="Area",
        hovertemplate="Area %{x:.2f} \u00b5m\u00b2<br>Count %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=median_area, line=dict(color="red", dash="dash"),
        annotation_text=f"Median: {median_area:.2f} \u00b5m\u00b2",
        annotation_position="top right",
    )
    fig.update_layout(
        title=(
            f"Cell area distribution (first {baseline_frames} frame"
            f"{'s' if baseline_frames > 1 else ''} pooled)"
        ),
        xaxis=dict(title="Area (\u00b5m\u00b2)"),
        yaxis=dict(title="Count"),
        showlegend=False,
    )
    _finalize_plotly(fig, bin_menu_traces=[0], height=450)

    print(f"Median area:         {median_area:.2f} \u00b5m\u00b2")
    print(f"Median volume:       {cohort['volume'].median():.2f} \u00b5m\u00b3")
    print(f"Median surface area: {cohort['surface_area'].median():.2f} \u00b5m\u00b2")


def plot_burst_conc_all_metrics(tracked, track_stats, min_frame, max_frame):
    """Plot ACTUAL log10 of sucrose concentration vs log10 of V0, V_final, expnsion ratio"""

    burst_tracks = track_stats[track_stats["disappeared"] == True]["track_id"]
    cohort = tracked[tracked["track_id"].isin(burst_tracks)].copy()
    
    if len(cohort) == 0:
        print("No burst cells found in tracked data.")
        return
        
    first_obs = cohort[cohort["frame"] == min_frame].set_index("track_id")
    last_obs = cohort.sort_values("frame").groupby("track_id").last()
    
    last_obs = last_obs[last_obs["frame"] <= max_frame]
    
    cohort = cohort.merge(first_obs["volume"].rename("V0"), on="track_id")
    cohort = cohort.merge(last_obs["Sucr %"].rename("burst_conc"), on="track_id")
    cohort = cohort.merge(last_obs["volume"].rename("V_final"), on="track_id")

    plot_df = cohort.groupby("track_id").first()
    plot_df["V0"] = pd.to_numeric(plot_df["V0"], errors="coerce")
    plot_df["burst_conc"] = pd.to_numeric(plot_df["burst_conc"], errors="coerce")
    plot_df["V_final"] = pd.to_numeric(plot_df["V_final"], errors="coerce")
    plot_df = plot_df[(plot_df["V0"] > 0) & (plot_df["burst_conc"] > 0) & (plot_df["V_final"] > 0)].copy()
    plot_df["expansion_ratio"] = plot_df["V_final"] / plot_df["V0"]

    # convert to log
    plot_df["log10_V0"] = np.log10(plot_df["V0"])
    plot_df["log10_burst_conc"] = np.log10(plot_df["burst_conc"])
    plot_df["log10_V_final"] = np.log10(plot_df["V_final"])
    plot_df["log10_expansion"] = np.log10(plot_df["expansion_ratio"])
    
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=False,
        subplot_titles=(
            "vs. Initial Vol",
            "vs. Final Vol",
            "vs. Expansion Ratio"
        ),
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Scatter(
        x=plot_df["log10_V0"], 
        y=plot_df["log10_burst_conc"],
        mode="markers",
        marker=dict(size=8, color="crimson", opacity=0.6, line=dict(width=1, color="black")),
        showlegend=False,
        customdata=plot_df[["V0", "burst_conc"]],
        hovertemplate="<b>Log10(V0): %{x:.2f}</b><br><b>Log10(Burst Conc): %{y:.2f}</b><br><br>Original Volume: %{customdata[0]:.1f} µm³<br>Original Conc: %{customdata[1]:.1f}%<extra></extra>"
    ), row=1, col=1)

    """
    df_clean = plot_df.dropna(subset=["log10_V0", "log10_V_final", "log10_expansion", "log10_burst_conc"])
    x_data = df_clean["log10_V0"]
    y_data = df_clean["log10_burst_conc"]
    z = np.polyfit(x_data, y_data, 1)
    p= np.poly1d(z)
    x_trend = np.linspace(x_data.min(), x_data.max(), 50)
    y_trend = p(x_trend)

    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode="lines",
        line=dict(color="blue", width=3, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ), row=1, col=1)"""

    fig.add_trace(go.Scatter(
            x=plot_df["log10_V_final"], 
            y=plot_df["log10_burst_conc"],
            mode="markers",
            marker=dict(size=8, color="crimson", opacity=0.6, line=dict(width=1, color="black")),
            showlegend=False,
            customdata=plot_df[["V_final", "burst_conc"]],
            hovertemplate="<b>Log10(V_final): %{x:.2f}</b><br><b>Log10(Burst Conc): %{y:.2f}</b><br><br>Original Volume: %{customdata[0]:.1f} µm³<br>Original Conc: %{customdata[1]:.1f}%<extra></extra>"
        ), row=1, col=2)

    """
    x_data = df_clean["log10_V_final"]
    z = np.polyfit(x_data, y_data, 1)
    p= np.poly1d(z)
    x_trend = np.linspace(x_data.min(), x_data.max(), 50)
    y_trend = p(x_trend)

    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode="lines",
        line=dict(color="blue", width=3, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ), row=1, col=2)"""

    fig.add_trace(go.Scatter(
            x=plot_df["log10_expansion"], 
            y=plot_df["log10_burst_conc"],
            mode="markers",
            marker=dict(size=8, color="crimson", opacity=0.6, line=dict(width=1, color="black")),
            showlegend=False,
            customdata=plot_df[["V0", "burst_conc"]],
            hovertemplate="<b>Log10(Ratio): %{x:.2f}</b><br><b>Log10(Burst Conc): %{y:.2f}</b><br><br>Original Volume: %{customdata[0]:.1f} µm³<br>Original Conc: %{customdata[1]:.1f}%<extra></extra>"
        ), row=1, col=3)

    """
    x_data = df_clean["log10_expansion"]
    z = np.polyfit(x_data, y_data, 1)
    p= np.poly1d(z)
    x_trend = np.linspace(x_data.min(), x_data.max(), 50)
    y_trend = p(x_trend)

    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode="lines",
        line=dict(color="blue", width=3, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ), row=1, col=3)"""
    
    fig.update_layout(
        title="Osmotic Burst Limits",
        height=500, width=1200, plot_bgcolor="white"
    )
    fig.update_xaxes(title_text="Log10( V0 )" ,showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=1)
    fig.update_xaxes(title_text="Log10( V_final )" ,showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=2)
    fig.update_xaxes(title_text="Log10( V_final/V0 )" ,showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=3)

    fig.update_yaxes(title_text="Log10( Burst Concentration )" ,showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=1)
    fig.update_yaxes(title_text="Log10( Burst Concentration )" ,showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=2)
    fig.update_yaxes(title_text="Log10( Burst Concentration )" ,showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=3)

    try:
        _finalize_plotly(fig, height=500)
    except:
        fig.show()

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

def check_frames_quality(tracked):
    """Plot diagnostics to find bad frames (focus loss, tracking errors)."""
    
    frame_stats = tracked.groupby('frame').agg(
        cell_count=('track_id', 'count'),     
        mean_volume=('volume', 'mean'),       
        median_volume=('volume', 'median')    
    ).reset_index()

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "1. Number of Tracked Cells (Drops indicate lost focus/tracking)", 
            "2. Average Cell Volume (Spikes indicate blur/focus shift)"
        )
    )

    fig.add_trace(go.Scatter(
        x=frame_stats['frame'], y=frame_stats['cell_count'],
        mode='lines+markers',
        name='Cell Count',
        marker=dict(size=8, color="blue"),
        hovertemplate="Frame: %{x}<br>Cells: %{y}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=frame_stats['frame'], y=frame_stats['mean_volume'],
        mode='lines+markers',
        name='Mean Volume',
        marker=dict(size=8, color="red"),
        hovertemplate="Frame: %{x}<br>Mean Vol: %{y:.1f} µm³<extra></extra>"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=frame_stats['frame'], y=frame_stats['median_volume'],
        mode='lines',
        name='Median Volume',
        line=dict(color="orange", dash="dash"),
        hoverinfo="skip"
    ), row=2, col=1)

    fig.update_layout(
        title="Frame Quality Diagnostics",
        height=600, width=800, 
        plot_bgcolor="white",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Frame Number", showgrid=True, gridcolor='LightGray', row=2, col=1)
    fig.update_yaxes(title_text="Total Cells", showgrid=True, gridcolor='LightGray', row=1, col=1)
    fig.update_yaxes(title_text="Volume (µm³)", showgrid=True, gridcolor='LightGray', row=2, col=1)

    fig.update_xaxes(dtick=1) 

    fig.show()



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

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(
            "Relative volume  V(t) / V(0)",
            "Relative surface area  S(t) / S(0)",
        ),
        horizontal_spacing=0.06,
    )

    track_color = _rgba("steelblue", 0.18)
    grouped_subset = subset.groupby("track_id")
    for col, stats, sub_col, group, y_label in [
        (1, v_stats, "V_rel", "panel1", "V(t)/V(0)"),
        (2, s_stats, "S_rel", "panel2", "S(t)/S(0)"),
    ]:
        for tid in subset_ids:
            t = grouped_subset.get_group(tid)
            fig.add_trace(go.Scatter(
                x=t["frame"], y=t[sub_col], mode="lines",
                line=dict(color=track_color, width=1),
                hoverinfo="skip", showlegend=False,
                legendgroup=group,
            ), row=1, col=col)

        _add_mean_sem_band(
            fig, stats.index, stats["mean"], stats["sem"],
            color="steelblue", name=f"Mean (n={len(frame0_tracks)})",
            legendgroup=group, row=1, col=col, y_label=y_label,
        )
        fig.add_hline(
            y=1.0, line=dict(color="gray", dash="dash", width=1),
            opacity=0.5, row=1, col=col,
        )

        final_mean = stats["mean"].iloc[-1]
        final_frame = int(stats.index[-1])
        xref = "x" if col == 1 else "x2"
        yref = "y" if col == 1 else "y2"
        fig.add_annotation(
            x=final_frame, y=final_mean,
            ax=final_frame - 3, ay=final_mean + 0.05,
            xref=xref, yref=yref, axref=xref, ayref=yref,
            text=f"{final_mean:.2f}x",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12),
        )

    _frame_xaxis(fig, v_stats.index, row=1, col=1)
    _frame_xaxis(fig, s_stats.index, row=1, col=2)
    fig.update_yaxes(title_text="V(t) / V(0)", row=1, col=1)
    fig.update_yaxes(title_text="S(t) / S(0)", row=1, col=2)
    fig.update_layout(
        title="Cell swelling dynamics (frame-0 cohort, spherical assumption)",
        showlegend=False,
    )
    _finalize_plotly(fig, height=520, margin=dict(t=100, b=50, l=60, r=30))

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


def plot_swelling_per_conc(tracked, track_stats, min_frame, max_frame):
    """Plot [V(t)-V(0)]/V(0) and [S(t)-S(0)]/S(0) vs. sucrose concentration"""

    burst_tracks = track_stats[track_stats["disappeared"] == True]["track_id"]
    cohort = tracked[tracked["track_id"].isin(burst_tracks)].copy()
    
    if len(cohort) == 0:
        print("No burst cells found in tracked data.")
        return
        
    first_obs = cohort[cohort["frame"] == min_frame].set_index("track_id")
    cohort = cohort[(cohort["frame"] >= min_frame) & (cohort["frame"] <= max_frame)]

    cohort = cohort.merge(first_obs["volume"].rename("V0"), on="track_id")
    cohort = cohort.merge(
        first_obs["surface_area"].rename("S0"), on="track_id",
    )
    cohort["V_rel"] = (cohort["volume"] - cohort["V0"]) / cohort["V0"]
    cohort["S_rel"] = (cohort["surface_area"] - cohort["S0"]) / cohort["S0"]

    stats = cohort.groupby("frame").agg(
        mean_V=("V_rel", "mean"), sem_V=("V_rel", "sem"),
        mean_S=("S_rel", "mean"), sem_S=("S_rel", "sem"),
        count=("V_rel", "count"),
        sucr_conc=("Sucr %", "mean")
    ).reset_index()

    valid_tracks = cohort["track_id"].unique()
    rng = np.random.default_rng(42)
    subset_ids = rng.choice(
        valid_tracks, size=min(20, len(valid_tracks)), replace=False,
    )
    subset = cohort[cohort["track_id"].isin(subset_ids)]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(
            "Sucrose Concentration vs. [V(t) - V(0)] / V(0)",
            "Sucrose Concentration vs. [S(t) - S(0)] / S(0)",
        ),
        horizontal_spacing=0.06,
    )

    track_color = _rgba("steelblue", 0.18)
    grouped_subset = subset.groupby("track_id")
    for col, mean_col, sem_col, sub_col, group, y_label in [
        (1, "mean_V", "sem_V", "V_rel", "panel1", "[V(t) - V(0)] / V(0)"),
        (2, "mean_S", "sem_S", "S_rel", "panel2", "[S(t) - S(0)] / S(0)"),
    ]:
        for tid in subset_ids:
            t = grouped_subset.get_group(tid)
            fig.add_trace(go.Scatter(
                x=t[sub_col], y=t["Sucr %"], mode="lines", 
                line=dict(color=track_color, width=1),
                hoverinfo="skip", showlegend=False,
                legendgroup=group,
            ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=stats[mean_col], y=stats["sucr_conc"], mode="lines",
            line=dict(color="steelblue", width=3),
            name=f"Mean (n={len(valid_tracks)})",
            legendgroup=group, showlegend=False
        ), row=1, col=col)
        fig.add_vline(
            x=0.0, line=dict(color="gray", dash="dash", width=1),
            opacity=0.5, row=1, col=col,
        )

        final_mean = stats[mean_col].iloc[-1]
        final_conc = stats["sucr_conc"].iloc[-1]
        xref = "x" if col == 1 else "x2"
        yref = "y" if col == 1 else "y2"
        fig.add_annotation(
            x=final_mean, y=final_conc,
            ax=final_mean + 0.1, ay=0,
            xref=xref, yref=yref, axref=xref, ayref=yref,
            text=f"{final_mean:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12),
        )

    fig.update_yaxes(title_text="Sucrose Concentration (%)", row=1, col=1, autorange="reversed")
    fig.update_yaxes(title_text="Sucrose Concentration (%)", row=1, col=2, autorange="reversed")
    fig.update_xaxes(title_text="[V(t)-V(0)]/V(0)", row=1, col=1)
    fig.update_xaxes(title_text="[S(t)-S(0)]/S(0)", row=1, col=2)
    fig.update_layout(
        title="Sucrose concentration vs. Cell swelling dynamics",
        showlegend=False,
    )
    _finalize_plotly(fig, height=520, margin=dict(t=100, b=50, l=60, r=30))
    """
    print(
        f"\nFinal [V(t)-V(0)]/V(0) at {stats['sucr_conc']}: "
        f"{v_stats['mean'].iloc[-1]:.3f} \u00b1 {v_stats['sem'].iloc[-1]:.3f}"
    )
    print(
        f"Final S(t)/S(0) at frame {int(s_stats.index[-1])}: "
        f"{s_stats['mean'].iloc[-1]:.3f} \u00b1 {s_stats['sem'].iloc[-1]:.3f}"
    )
    print(f"Cells contributing at final frame: "
          f"{int(v_stats['count'].iloc[-1])}")"""


def plot_tension(tracked, track_stats, min_frame, max_frame):
    """Plot [V(t)-V(0)]/V(0) and [S(t)-S(0)]/S(0) vs. sucrose concentration * R0"""

    burst_tracks = track_stats[track_stats["disappeared"] == True]["track_id"]
    cohort = tracked[tracked["track_id"].isin(burst_tracks)].copy()
    
    if len(cohort) == 0:
        print("No burst cells found in tracked data.")
        return
        
    first_obs = cohort[cohort["frame"] == min_frame].set_index("track_id")
    cohort = cohort[(cohort["frame"] >= min_frame) & (cohort["frame"] <= max_frame)]

    cohort = cohort.merge(first_obs["volume"].rename("V0"), on="track_id")
    cohort = cohort.merge(
        first_obs["surface_area"].rename("S0"), on="track_id",
    )
    cohort["V_rel"] = (cohort["volume"] - cohort["V0"]) / cohort["V0"]
    cohort["S_rel"] = (cohort["surface_area"] - cohort["S0"]) / cohort["S0"]
    cohort["Sucr_R"] = cohort["Sucr %"] * cohort["radius"]

    stats = cohort.groupby("frame").agg(
        mean_V=("V_rel", "mean"), sem_V=("V_rel", "sem"),
        mean_S=("S_rel", "mean"), sem_S=("S_rel", "sem"),
        count=("V_rel", "count"),
        mean_sucr_R=("Sucr_R", "mean") 
    ).reset_index()

    valid_tracks = cohort["track_id"].unique()
    rng = np.random.default_rng(42)
    subset_ids = rng.choice(
        valid_tracks, size=min(20, len(valid_tracks)), replace=False,
    )
    subset = cohort[cohort["track_id"].isin(subset_ids)]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(
            "Sucrose Concentration × R vs. [V(t) - V(0)] / V(0)",
            "Sucrose Concentration × R vs. [S(t) - S(0)] / S(0)",
        ),
        horizontal_spacing=0.06,
    )

    track_color = _rgba("steelblue", 0.18)
    grouped_subset = subset.groupby("track_id")
    
    for col, mean_col, sem_col, sub_col, group, y_label in [
        (1, "mean_V", "sem_V", "V_rel", "panel1", "[V(t) - V(0)] / V(0)"),
        (2, "mean_S", "sem_S", "S_rel", "panel2", "[S(t) - S(0)] / S(0)"),
    ]:
        for tid in subset_ids:
            t = grouped_subset.get_group(tid)
            fig.add_trace(go.Scatter(
                x=t[sub_col], y=t["Sucr_R"], mode="lines", 
                line=dict(color=track_color, width=1),
                hoverinfo="skip", showlegend=False,
                legendgroup=group,
            ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=stats[mean_col], 
            y=stats["mean_sucr_R"], 
            mode="lines",
            line=dict(color="steelblue", width=3),
            name=f"Mean (n={len(valid_tracks)})",
            legendgroup=group, showlegend=False
        ), row=1, col=col)
        
        fig.add_vline(
            x=0.0, line=dict(color="gray", dash="dash", width=1),
            opacity=0.5, row=1, col=col,
        )

        final_mean = stats[mean_col].iloc[-1]
        final_conc = stats["mean_sucr_R"].iloc[-1] # This works correctly now
        
        xref = "x" if col == 1 else "x2"
        yref = "y" if col == 1 else "y2"
        fig.add_annotation(
            x=final_mean, y=final_conc,
            ax=final_mean + 0.1, ay=0,
            xref=xref, yref=yref, axref=xref, ayref=yref,
            text=f"{final_mean:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12),
        )

    fig.update_yaxes(title_text="Sucrose Concentration (%) × R", row=1, col=1, autorange="reversed")
    fig.update_yaxes(title_text="Sucrose Concentration (%) × R", row=1, col=2, autorange="reversed")
    fig.update_xaxes(title_text="[V(t)-V(0)]/V(0)", row=1, col=1)
    fig.update_xaxes(title_text="[S(t)-S(0)]/S(0)", row=1, col=2)
    fig.update_layout(
        title="Sucrose concentration × R vs. Cell swelling dynamics",
        showlegend=False,
    )
    
    _finalize_plotly(fig, height=520, margin=dict(t=100, b=50, l=60, r=30))

def plot_tension_comparison(tracked_ctrl, stats_ctrl, tracked_arr, stats_arr, 
                            ctrl_min_frame=3, ctrl_max_frame=30, 
                            arr_min_frame=3, arr_max_frame=30):
    """Plot average [V(t)-V(0)]/V(0) and [S(t)-S(0)]/S(0) vs. sucrose concentration * R(t) for Control vs. Arrested."""

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(
            "Sucrose Concentration × R vs. \u0394V / V0",
            "Sucrose Concentration × R vs. \u0394S / S0",
        ),
        horizontal_spacing=0.06,
    )

    datasets = {
        "Control": {
            "tracked": tracked_ctrl, "stats": stats_ctrl, 
            "color": "steelblue", 
            "min_f": ctrl_min_frame, "max_f": ctrl_max_frame
        },
        "Arrested": {
            "tracked": tracked_arr, "stats": stats_arr, 
            "color": "darkorange", 
            "min_f": arr_min_frame, "max_f": arr_max_frame
        }
    }

    for condition_name, data in datasets.items():
        trk = data["tracked"]
        stt = data["stats"]
        line_color = data["color"]
        min_f = data["min_f"] 
        max_f = data["max_f"] 

        burst_tracks = stt[stt["disappeared"] == True]["track_id"]
        cohort = trk[trk["track_id"].isin(burst_tracks)].copy()
        
        if len(cohort) == 0:
            print(f"No burst cells found in {condition_name}.")
            continue
            
        first_obs = cohort[cohort["frame"] == min_f].set_index("track_id")
        cohort = cohort[(cohort["frame"] >= min_f) & (cohort["frame"] <= max_f)]

        cohort = cohort.merge(first_obs["volume"].rename("V0"), on="track_id")
        cohort = cohort.merge(first_obs["surface_area"].rename("S0"), on="track_id")
        
        cohort["V_rel"] = (cohort["volume"] - cohort["V0"]) / cohort["V0"]
        cohort["S_rel"] = (cohort["surface_area"] - cohort["S0"]) / cohort["S0"]
        cohort["Sucr_R"] = cohort["Sucr %"] * cohort["radius"]

        stats = cohort.groupby("frame").agg(
            mean_V=("V_rel", "mean"),
            mean_S=("S_rel", "mean"),
            mean_sucr_R=("Sucr_R", "mean") 
        ).reset_index()

        valid_tracks = cohort["track_id"].unique()

        for col, mean_col in [(1, "mean_V"), (2, "mean_S")]:
            fig.add_trace(go.Scatter(
                x=stats[mean_col], 
                y=stats["mean_sucr_R"], 
                mode="lines",
                line=dict(color=line_color, width=3),
                name=f"{condition_name} (n={len(valid_tracks)})",
                legendgroup=condition_name, 
                showlegend=(col == 1) 
            ), row=1, col=col)
            
    for col in [1, 2]:
        fig.add_vline(
            x=0.0, line=dict(color="gray", dash="dash", width=1), 
            opacity=0.5, row=1, col=col
        )
        fig.update_yaxes(title_text="Sucrose Concentration (%) × R", row=1, col=col, autorange="reversed")

    fig.update_xaxes(title_text="\u0394V / V0", row=1, col=1)
    fig.update_xaxes(title_text="\u0394S / S0", row=1, col=2)
    
    fig.update_layout(
        title="Comparison: Tension (Sucrose × R) vs. Cell Swelling Dynamics",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    try:
        _finalize_plotly(fig, height=520, margin=dict(t=100, b=50, l=60, r=30))
    except NameError:
        fig.show()


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

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Swelling extent vs. initial cell size",
            "Swelling dynamics: disappeared vs. survived (frame-0 cohort)",
        ),
        horizontal_spacing=0.1,
    )

    for label, mask_val, color in [
        ("Survived", True, "steelblue"),
        ("Disappeared", False, "tomato"),
    ]:
        sub = per_track[per_track["survived"] == mask_val]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["V0"], y=sub["V_rel_max"], mode="markers",
            marker=dict(color=color, size=6, opacity=0.55, line=dict(width=0)),
            name=label, legendgroup="panel1",
            hovertemplate=(
                "V(0) %{x:.2f} \u00b5m\u00b3<br>Max V(t)/V(0) %{y:.2f}"
                f"<extra>{label}</extra>"
            ),
        ), row=1, col=1)

    mask = np.isfinite(per_track["V0"]) & np.isfinite(per_track["V_rel_max"])
    if mask.sum() > 2:
        coeffs = polyfit(
            per_track.loc[mask, "V0"],
            per_track.loc[mask, "V_rel_max"], 1,
        )
        x_fit = np.linspace(
            per_track["V0"].min(), per_track["V0"].max(), 100,
        )
        fig.add_trace(go.Scatter(
            x=x_fit, y=coeffs[0] + coeffs[1] * x_fit, mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            opacity=0.7,
            name=f"Linear fit (slope={coeffs[1]:.2e})",
            legendgroup="panel1",
            hovertemplate="V(0) %{x:.2f}<br>fit %{y:.2f}<extra></extra>",
        ), row=1, col=1)

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
        _add_mean_sem_band(
            fig, g.index, g["mean"], g["sem"],
            color=color, name=f"{label} (n={len(ids)})",
            legendgroup="panel2", row=1, col=2, y_label="V(t)/V(0)",
        )
    fig.add_hline(
        y=1.0, line=dict(color="gray", dash="dash", width=1),
        opacity=0.5, row=1, col=2,
    )

    fig.update_xaxes(title_text="Initial volume V(0) (\u00b5m\u00b3)", row=1, col=1)
    fig.update_yaxes(title_text="Max V(t) / V(0)", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2, tickmode="array",
                     tickvals=list(range(int(cohort["frame"].min()),
                                         int(cohort["frame"].max()) + 1)))
    fig.update_yaxes(title_text="V(t) / V(0)", row=1, col=2)
    fig.update_layout(
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(fig, height=520)

    n_surv = per_track["survived"].sum()
    n_dis = (~per_track["survived"]).sum()
    print(f"Frame-0 cohort: {n_surv} survived, {n_dis} disappeared")
    v0_surv = per_track.loc[per_track["survived"], "V0"]
    v0_dis = per_track.loc[~per_track["survived"], "V0"]
    print(f"Mean initial volume (survived):    {v0_surv.mean():.2f} ± {v0_surv.std():.2f} µm³")
    print(f"Mean initial volume (disappeared): {v0_dis.mean():.2f} ± {v0_dis.std():.2f} µm³")
    print(
        f"Median max swelling (survived):    "
        f"{per_track.loc[per_track['survived'], 'V_rel_max'].median():.3f}x"
    )
    print(
        f"Median max swelling (disappeared): "
        f"{per_track.loc[~per_track['survived'], 'V_rel_max'].median():.3f}x"
    )

    per_track["lifetime"] = per_track["last_frame"] - cohort.groupby("track_id")["frame"].min().values + 1
    per_track["swelling_rate"] = (per_track["V_rel_max"] - 1) / per_track["lifetime"]
    surv_rate = per_track.loc[per_track["survived"], "swelling_rate"]
    dis_rate = per_track.loc[~per_track["survived"], "swelling_rate"]
    print(f"Median swelling rate (survived):    {surv_rate.median():.4f}x per frame")
    print(f"Median swelling rate (disappeared): {dis_rate.median():.4f}x per frame")


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

    fig = make_subplots(rows=1, cols=1)
    _add_mean_sem_band(
        fig, fluor_per_frame.index,
        fluor_per_frame["mean"], fluor_per_frame["sem"],
        color="darkorange", name="Mean +/- SEM", legendgroup="main",
        row=1, col=1, y_label="Fluorescence",
    )
    fig.update_xaxes(title_text="Frame", tickmode="array",
                     tickvals=list(fluor_per_frame.index))
    fig.update_yaxes(title_text="Mean fluorescence intensity")
    fig.update_layout(
        title="Population mean fluorescence per frame",
    )
    _finalize_plotly(fig, height=450)

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
    survived_ids, disappeared_ids = _survival_split(tracked, cohort="frame0")

    rng = np.random.default_rng(42)
    subset_ids = rng.choice(
        frame0_tracks, size=min(20, len(frame0_tracks)), replace=False,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Relative fluorescence F(t) / F(0)",
            "Fluorescence dynamics: disappeared vs. survived (frame-0)",
        ),
        horizontal_spacing=0.1,
    )

    track_color = _rgba("darkorange", 0.18)
    for tid in subset_ids:
        t = cohort[cohort["track_id"] == tid]
        fig.add_trace(go.Scatter(
            x=t["frame"], y=t["F_rel"], mode="lines",
            line=dict(color=track_color, width=1),
            hoverinfo="skip", showlegend=False,
            legendgroup="panel1",
        ), row=1, col=1)
    _add_mean_sem_band(
        fig, f_stats.index, f_stats["mean"], f_stats["sem"],
        color="darkorange",
        name=f"Mean (n={len(frame0_tracks)} tracks)",
        legendgroup="panel1", row=1, col=1, y_label="F(t)/F(0)",
    )
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1),
                  opacity=0.5, row=1, col=1)
    final_f = f_stats["mean"].iloc[-1]
    final_frame = int(f_stats.index[-1])
    fig.add_annotation(
        x=final_frame, y=final_f,
        ax=final_frame - 3, ay=final_f + 0.05,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"{final_f:.2f}x",
        showarrow=True, arrowhead=2, arrowcolor="red",
        font=dict(color="red", size=12),
    )

    for label, ids, color in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = cohort[cohort["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")["F_rel"].agg(["mean", "sem"])
        _add_mean_sem_band(
            fig, g.index, g["mean"], g["sem"],
            color=color, name=f"{label} (n={len(ids)})",
            legendgroup="panel2", row=1, col=2, y_label="F(t)/F(0)",
        )
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1),
                  opacity=0.5, row=1, col=2)

    fig.update_xaxes(title_text="Frame", row=1, col=1, tickmode="array",
                     tickvals=list(f_stats.index))
    fig.update_yaxes(title_text="F(t) / F(0)", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2, tickmode="array",
                     tickvals=list(range(int(cohort["frame"].min()),
                                         int(cohort["frame"].max()) + 1)))
    fig.update_yaxes(title_text="F(t) / F(0)", row=1, col=2)
    fig.update_layout(
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(fig, height=520)

    print(
        f"Final F(t)/F(0): {f_stats['mean'].iloc[-1]:.3f} "
        f"+/- {f_stats['sem'].iloc[-1]:.3f}"
    )
    print(
        f"Cells contributing at final frame: {int(f_stats['count'].iloc[-1])}"
    )


def plot_fluorescence_alignment(alignment_df):
    """Plot +/-window-frame fluorescence dynamics aligned to phase disappearance.

    Per-cell normalized traces (low alpha) + population mean +/- SEM band.
    Vertical line at offset=0 marks the phase mask vanishing frame.
    """
    if alignment_df.empty:
        print("No disappeared cells to align.")
        return

    fig = make_subplots(rows=1, cols=1)

    for _, grp in alignment_df.groupby("track_id"):
        grp = grp.sort_values("offset")
        fig.add_trace(go.Scatter(
            x=grp["offset"], y=grp["F_norm"], mode="lines",
            line=dict(color="rgba(70,130,180,0.15)", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    pop = (alignment_df.dropna(subset=["F_norm"])
           .groupby("offset")["F_norm"].agg(["mean", "sem", "count"])
           .reset_index())

    _add_mean_sem_band(
        fig, pop["offset"], pop["mean"], pop["sem"],
        color="steelblue", name="Mean +/- SEM", legendgroup="pop",
        row=1, col=1, y_label="F(t)/F(-window)",
    )

    fig.add_vline(
        x=0, line=dict(color="red", dash="dash"),
        annotation_text="phase mask vanishes",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Fluorescence aligned to phase disappearance",
        xaxis=dict(title="Offset (frames relative to last detection)"),
        yaxis=dict(title="F(offset) / F(-window)"),
    )
    _finalize_plotly(fig, height=480)

    n = alignment_df["track_id"].nunique()
    window = int(alignment_df["offset"].abs().max())
    medians = (alignment_df.dropna(subset=["F_norm"])
               .groupby("offset")["F_norm"].median())
    print(f"Aligned {n} disappeared tracks (window=+/-{window})")
    for off in sorted(medians.index):
        print(f"  offset {int(off):+d}: median F_norm = {medians.loc[off]:.3f}")


def plot_peri_core_alignment(alignment_df, track_stats=None):
    """Plot peri/core asymmetry aligned to phase disappearance.

    Two-panel figure:
      - Left: per-cell asymmetry trajectories (low-alpha lines) + population
        median + IQR band. Offset 0 = last-detected frame. Optionally
        split by fate group when *track_stats* is passed (uses
        ``fluor_disappearance_mode`` if present, else all disappeared
        are one group).
      - Right: frame-baseline (offset=-window) vs. offset=0 scatter,
        one point per cell. Reveals whether donut/sparse initial states
        converge to a common pre-burst state.
    """
    if alignment_df.empty:
        print("No disappeared cells to align.")
        return

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=(
            "Per-cell trajectory aligned to last frame",
            "Baseline vs. pre-burst asymmetry (per cell)",
        ),
    )

    for _, grp in alignment_df.groupby("track_id"):
        grp = grp.sort_values("offset")
        fig.add_trace(go.Scatter(
            x=grp["offset"], y=grp["peri_core_asymmetry"], mode="lines",
            line=dict(color="rgba(70,130,180,0.12)", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    pop = (alignment_df.dropna(subset=["peri_core_asymmetry"])
           .groupby("offset")["peri_core_asymmetry"]
           .agg(["median",
                 lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.75),
                 "count"])
           .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
           .reset_index())

    fig.add_trace(go.Scatter(
        x=list(pop["offset"]) + list(pop["offset"])[::-1],
        y=list(pop["q75"]) + list(pop["q25"])[::-1],
        fill="toself", fillcolor="rgba(180,80,60,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name="IQR (25–75%)",
        legendgroup="pop", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pop["offset"], y=pop["median"], mode="lines+markers",
        line=dict(color="rgb(180,80,60)", width=2.5),
        marker=dict(size=7), name="Median", legendgroup="pop",
    ), row=1, col=1)

    fig.add_vline(
        x=0, line=dict(color="red", dash="dash"),
        annotation_text="phase mask vanishes",
        annotation_position="top right",
        row=1, col=1,
    )

    window = int(alignment_df["offset"].abs().max())
    per_track = (alignment_df.pivot(index="track_id", columns="offset",
                                    values="peri_core_asymmetry"))
    if -window in per_track.columns and 0 in per_track.columns:
        base = per_track[-window]
        last = per_track[0]
        finite = base.notna() & last.notna()
        base = base[finite]
        last = last[finite]
        fig.add_trace(go.Scatter(
            x=base, y=last, mode="markers",
            marker=dict(size=6, color="rgba(70,130,180,0.55)"),
            name=f"cells (n={len(base)})",
        ), row=1, col=2)
        lo = min(base.min(), last.min())
        hi = max(base.max(), last.max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)
        fig.update_xaxes(
            title_text=f"Asymmetry at offset -{window} (baseline)",
            row=1, col=2,
        )
        fig.update_yaxes(
            title_text="Asymmetry at offset 0 (last frame)",
            row=1, col=2,
        )

    fig.update_xaxes(
        title_text="Offset (frames relative to last detection)",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="peri_core_asymmetry", row=1, col=1)
    fig.update_layout(
        title="Peri/core asymmetry aligned to phase disappearance",
    )
    _finalize_plotly(fig, height=520)

    n = alignment_df["track_id"].nunique()
    medians = (alignment_df.dropna(subset=["peri_core_asymmetry"])
               .groupby("offset")["peri_core_asymmetry"].median())
    print(f"Aligned {n} disappeared tracks (window=-{window}..0)")
    for off in sorted(medians.index):
        print(f"  offset {int(off):+d}: median asymmetry = {medians.loc[off]:+.3f}")



def plot_nucleus_persistence(comparison_df):
    """3-panel: counts over time, offset stability, and count correlation."""
    from scipy.stats import pearsonr

    r, p = pearsonr(comparison_df["phase_cells"], comparison_df["fluor_nuclei"])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Phase cells vs. fluorescent nuclei per frame",
            "Offset between fluorescence and phase-contrast counts",
            f"Count correlation (r = {r:.3f}, p = {p:.2e})",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=comparison_df["frame"], y=comparison_df["phase_cells"],
        mode="lines+markers", line=dict(color="steelblue", width=2),
        marker=dict(size=6), name="Phase cells", legendgroup="panel1",
        hovertemplate="Frame %{x}<br>Cells %{y}<extra>Phase</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=comparison_df["frame"], y=comparison_df["fluor_nuclei"],
        mode="lines+markers", line=dict(color="darkorange", width=2),
        marker=dict(size=6, symbol="square"),
        name="Fluor nuclei", legendgroup="panel1",
        hovertemplate="Frame %{x}<br>Nuclei %{y}<extra>Fluor</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=comparison_df["frame"], y=comparison_df["difference"],
        marker_color="gray", marker_line_color="white",
        marker_line_width=1, opacity=0.7, name="Offset",
        legendgroup="panel2",
        hovertemplate="Frame %{x}<br>Δ %{y}<extra></extra>",
    ), row=1, col=2)
    mean_diff = comparison_df["difference"].mean()
    fig.add_hline(
        y=mean_diff, line=dict(color="tomato", dash="dash", width=1.5),
        annotation_text=f"Mean: {mean_diff:.0f}",
        annotation_position="top right",
        row=1, col=2,
    )

    fig.add_trace(go.Scatter(
        x=comparison_df["phase_cells"], y=comparison_df["fluor_nuclei"],
        mode="markers",
        marker=dict(
            color=comparison_df["frame"], colorscale="Viridis",
            size=10, line=dict(color="black", width=0.5),
            colorbar=dict(title="Frame", x=1.02, len=0.8),
        ),
        name="Frames", showlegend=False, legendgroup="panel3",
        hovertemplate=(
            "Phase %{x}<br>Fluor %{y}<br>Frame %{marker.color}<extra></extra>"
        ),
    ), row=1, col=3)
    lo = min(comparison_df["phase_cells"].min(),
             comparison_df["fluor_nuclei"].min())
    hi = max(comparison_df["phase_cells"].max(),
             comparison_df["fluor_nuclei"].max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="black", dash="dash", width=1), opacity=0.3,
        name="1:1 line", legendgroup="panel3", hoverinfo="skip",
    ), row=1, col=3)

    frame_ticks = list(comparison_df["frame"])
    fig.update_xaxes(title_text="Frame", row=1, col=1, tickmode="array",
                     tickvals=frame_ticks)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2, tickmode="array",
                     tickvals=frame_ticks)
    fig.update_yaxes(title_text="Fluor nuclei - Phase cells",
                     row=1, col=2)
    fig.update_xaxes(title_text="Phase-contrast cell count", row=1, col=3)
    fig.update_yaxes(title_text="Fluorescence nucleus count",
                     row=1, col=3)
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    _finalize_plotly(fig, height=500, margin=dict(t=80, b=50, l=60, r=80))
    print(f"Phase vs fluor count: Pearson r = {r:.3f}, p = {p:.2e}")


def plot_fluorescence_concentration(tracked):
    """3-panel: concentration over time, outcome split, concentration vs volume."""

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Fluorescence concentration over time",
            "Concentration: disappeared vs. survived (all tracks)",
            "Concentration vs. cell volume",
        ),
        horizontal_spacing=0.08,
    )

    per_frame = tracked.groupby("frame")["fluor_concentration"].agg(["mean", "sem"])
    _add_mean_sem_band(
        fig, per_frame.index, per_frame["mean"], per_frame["sem"],
        color="purple", name="Mean +/- SEM", legendgroup="panel1",
        row=1, col=1, y_label="Concentration",
    )

    survived_ids, disappeared_ids = _survival_split(tracked)
    for label, ids, color in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = tracked[tracked["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")["fluor_concentration"].agg(["mean", "sem"])
        _add_mean_sem_band(
            fig, g.index, g["mean"], g["sem"],
            color=color, name=f"{label} (n={len(ids)})",
            legendgroup="panel2", row=1, col=2, y_label="Concentration",
        )

    full = tracked.dropna(subset=["fluor_concentration", "volume"])
    sample = full.sample(n=min(3000, len(full)), random_state=42)
    fig.add_trace(go.Scatter(
        x=sample["volume"], y=sample["fluor_concentration"],
        mode="markers",
        marker=dict(color="purple", size=4, opacity=0.25,
                    line=dict(width=0)),
        showlegend=False, legendgroup="panel3",
        hovertemplate="V %{x:.0f}<br>C %{y:.3f}<extra></extra>",
    ), row=1, col=3)
    if len(full) > 2:
        r = np.corrcoef(full["volume"], full["fluor_concentration"])[0, 1]
        fig.add_annotation(
            x=0.05, y=0.95, xref="x3 domain", yref="y3 domain",
            text=f"r = {r:.3f} (n={len(full)})",
            showarrow=False, font=dict(size=12, color="black"),
            xanchor="left", yanchor="top",
        )

    frame_ticks = list(per_frame.index)
    fig.update_xaxes(title_text="Frame", row=1, col=1, tickmode="array",
                     tickvals=frame_ticks)
    fig.update_yaxes(title_text="F_total / Volume (a.u. / µm³)", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2, tickmode="array",
                     tickvals=list(range(int(tracked["frame"].min()),
                                         int(tracked["frame"].max()) + 1)))
    fig.update_yaxes(title_text="F_total / Volume (a.u. / µm³)", row=1, col=2)
    scope_note = _scatter_scope(sample, full)
    fig.update_xaxes(title_text=f"Volume (µm³) — {scope_note}", row=1, col=3)
    fig.update_yaxes(title_text="F_total / Volume (a.u. / µm³)", row=1, col=3)
    fig.update_layout(
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(fig, height=500)

    print(f"Frame 0 mean concentration: {per_frame['mean'].iloc[0]:.3f}")
    print(f"Frame {int(per_frame.index[-1])} mean concentration: "
          f"{per_frame['mean'].iloc[-1]:.3f}")
    change = per_frame['mean'].iloc[-1] / per_frame['mean'].iloc[0]
    print(f"Change: {change:.2f}x")


def plot_sav_ratio(tracked, track_stats):
    """3-panel: SA:V over time, outcome split, SA:V at death vs survival.

    Plotly version (interactive): hover for exact values, legend-click to
    toggle traces, box-zoom, pan. Returns nothing; the figure is rendered
    via ``fig.show()`` so callers (including ``show_with_source``) work
    unchanged.
    """

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Surface-area-to-volume ratio over time",
            "SA:V ratio: disappeared vs. survived (all tracks)",
            "SA:V at disappearance vs. final frame (survived)",
        ),
        horizontal_spacing=0.08,
    )

    per_frame = tracked.groupby("frame")["sav_ratio"].agg(["mean", "sem"])
    _add_mean_sem_band(
        fig, per_frame.index, per_frame["mean"], per_frame["sem"],
        color="darkgreen", name="Mean +/- SEM", legendgroup="panel1",
        row=1, col=1, y_label="SA:V",
    )

    survived_ids, disappeared_ids = _survival_split(tracked)
    for label, ids, color in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = tracked[tracked["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")["sav_ratio"].agg(["mean", "sem"])
        _add_mean_sem_band(
            fig, g.index, g["mean"], g["sem"],
            color=color, name=f"{label} (n={len(ids)})",
            legendgroup="panel2", row=1, col=2, y_label="SA:V",
        )

    last_obs = tracked.sort_values("frame").groupby("track_id").last().reset_index()
    last_obs = last_obs.merge(
        track_stats[["track_id", "disappeared"]], on="track_id",
    )
    dis_final = last_obs.loc[last_obs["disappeared"], "sav_ratio"].dropna()
    surv_final = last_obs.loc[~last_obs["disappeared"], "sav_ratio"].dropna()
    hist_indices = []
    for vals, label, color in [
        (dis_final, "Disappeared (final frame)", "tomato"),
        (surv_final, "Survived (final frame)", "steelblue"),
    ]:
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Histogram(
                x=vals, nbinsx=25, histnorm="probability density",
                marker_color=color, opacity=0.6, name=label,
                legendgroup="panel3",
                hovertemplate="SA:V %{x:.3f}<br>Density %{y:.2f}<extra></extra>",
            ),
            row=1, col=3,
        )
        hist_indices.append(len(fig.data) - 1)

    frame_ticks = list(range(int(tracked["frame"].min()),
                             int(tracked["frame"].max()) + 1))
    for col in (1, 2):
        fig.update_xaxes(title_text="Frame", tickmode="array",
                         tickvals=frame_ticks, row=1, col=col)
        fig.update_yaxes(title_text="SA / V (µm⁻¹)", row=1, col=col)
    fig.update_xaxes(title_text="SA:V at last observation", row=1, col=3)
    fig.update_yaxes(title_text="Density", row=1, col=3)
    fig.update_layout(
        barmode="overlay",
        legend=dict(groupclick="toggleitem"),
    )

    _finalize_plotly(fig, bin_menu_traces=hist_indices, height=500)

    print(f"SA:V at frame 0: {per_frame['mean'].iloc[0]:.4f}")
    print(f"SA:V at frame {int(per_frame.index[-1])}: "
          f"{per_frame['mean'].iloc[-1]:.4f}")
    if len(dis_final) > 0:
        print(f"Median SA:V at death: {dis_final.median():.4f}")
    if len(surv_final) > 0:
        print(f"Median SA:V at end (survived): {surv_final.median():.4f}")


def plot_preburst_fluorescence(tracked, track_stats, n_frames=5):
    """3-panel: aligned pre-burst curves, slope distribution, spike vs no-spike."""

    disappeared = track_stats[track_stats["disappeared"]].copy()
    if disappeared.empty:
        print("No disappeared tracks for pre-burst analysis.")
        return

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"Pre-burst fluorescence (last {n_frames} frames)",
            "Distribution of pre-burst fluorescence slopes",
            "Pre-burst spike classification",
        ),
        horizontal_spacing=0.08,
    )

    spike_mask = disappeared["preburst_spike"].fillna(False).astype(bool)
    spike_tracks = disappeared[spike_mask]
    no_spike_tracks = disappeared[~spike_mask]

    for subset, color, label_prefix in [
        (spike_tracks, "tomato", "Spike"),
        (no_spike_tracks, "steelblue", "No spike"),
    ]:
        aligned_parts = []
        track_color = _rgba(color, 0.12)
        grouped = tracked.groupby("track_id")
        for _, row in subset.iterrows():
            tid = row["track_id"]
            last_f = int(row["last_frame"])
            grp = grouped.get_group(tid).sort_values("frame")
            window = grp[(grp["frame"] >= last_f - n_frames) & (grp["frame"] <= last_f)]
            if len(window) < 2:
                continue
            f0_intensity = grp["mean_intensity"].iloc[0]
            if f0_intensity == 0:
                continue
            rel_frame = window["frame"] - last_f
            rel_intensity = window["mean_intensity"] / f0_intensity
            aligned_parts.append(pd.DataFrame({
                "rel_frame": rel_frame, "rel_intensity": rel_intensity,
            }))
            if len(aligned_parts) <= 15:
                fig.add_trace(go.Scatter(
                    x=rel_frame, y=rel_intensity, mode="lines",
                    line=dict(color=track_color, width=1),
                    hoverinfo="skip", showlegend=False,
                    legendgroup="panel1",
                ), row=1, col=1)

        if aligned_parts:
            all_aligned = pd.concat(aligned_parts, ignore_index=True)
            stats = all_aligned.groupby("rel_frame")["rel_intensity"].agg(["mean", "sem"])
            fig.add_trace(go.Scatter(
                x=stats.index, y=stats["mean"], mode="lines+markers",
                line=dict(color=color, width=2), marker=dict(size=5),
                name=f"{label_prefix} (n={len(subset)})",
                legendgroup="panel1",
                hovertemplate=("rel %{x}<br>F %{y:.2f}"
                               f"<extra>{label_prefix}</extra>"),
            ), row=1, col=1)

    fig.add_vline(x=0, line=dict(color="black", dash="dot", width=1),
                  opacity=0.5, row=1, col=1,
                  annotation_text="Burst", annotation_position="top right")

    slopes = disappeared["preburst_slope"].dropna()
    hist_idx = None
    if len(slopes) > 0:
        fig.add_trace(go.Histogram(
            x=slopes, nbinsx=25, marker_color="darkorange",
            marker_line_color="white", marker_line_width=1,
            name="Slope", legendgroup="panel2", showlegend=False,
            hovertemplate="Slope %{x:.2f}<br>Count %{y}<extra></extra>",
        ), row=1, col=2)
        hist_idx = len(fig.data) - 1
        fig.add_vline(
            x=0, line=dict(color="black", dash="dash", width=1),
            opacity=0.5, row=1, col=2,
        )
        n_pos = (slopes > 0).sum()
        n_neg = (slopes <= 0).sum()
        fig.add_annotation(
            x=0.95, y=0.95, xref="x2 domain", yref="y2 domain",
            text=f"Positive: {n_pos}<br>Negative: {n_neg}",
            showarrow=False, font=dict(size=11),
            xanchor="right", yanchor="top",
        )

    n_spike = len(spike_tracks)
    n_no = len(no_spike_tracks)
    if n_spike + n_no > 0:
        fig.add_trace(go.Bar(
            x=["Spike", "No spike"], y=[n_spike, n_no],
            marker_color=["tomato", "steelblue"],
            marker_line_color="white", marker_line_width=1,
            text=[str(n_spike), str(n_no)], textposition="outside",
            name="Counts", legendgroup="panel3", showlegend=False,
            hovertemplate="%{x}: %{y}<extra></extra>",
        ), row=1, col=3)

    fig.update_xaxes(title_text="Frames relative to burst", row=1, col=1)
    fig.update_yaxes(title_text="F(t) / F(0)", row=1, col=1)
    fig.update_xaxes(title_text="Pre-burst intensity slope", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    fig.update_layout(
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(
        fig,
        bin_menu_traces=[hist_idx] if hist_idx is not None else None,
        height=500,
    )

    if len(slopes) > 0:
        print(f"Median pre-burst slope: {slopes.median():.2f}")
        print(f"Cells with pre-burst spike: {n_spike}/{n_spike + n_no}")


def plot_fate_prediction(prediction_df, summary):
    """3-panel: ROC curve, feature importance, probability distribution."""
    from sklearn.metrics import roc_curve

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "ROC curve (leave-one-out CV)",
            "Feature importance",
            "Prediction distribution by actual outcome",
        ),
        horizontal_spacing=0.08,
    )

    fpr, tpr, _ = roc_curve(prediction_df["disappeared"], prediction_df["predicted_prob"])
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color="darkorange", width=2),
        name=f"AUC = {summary['auc']:.3f}", legendgroup="panel1",
        hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="black", dash="dash", width=1), opacity=0.3,
        name="Random", legendgroup="panel1", hoverinfo="skip",
    ), row=1, col=1)

    features = summary["feature_names"]
    coefs = [summary["feature_importance"][f] for f in features]
    colors = ["tomato" if c > 0 else "steelblue" for c in coefs]
    fig.add_trace(go.Bar(
        x=coefs, y=features, orientation="h",
        marker_color=colors, marker_line_color="white",
        marker_line_width=1, name="Importance",
        legendgroup="panel2", showlegend=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ), row=1, col=2)
    fig.add_vline(x=0, line=dict(color="black", width=0.5), row=1, col=2)

    died = prediction_df[prediction_df["disappeared"]]
    survived = prediction_df[~prediction_df["disappeared"]]
    hist_indices = []
    for vals, label, color in [
        (survived["predicted_prob"], "Survived", "steelblue"),
        (died["predicted_prob"], "Died", "tomato"),
    ]:
        if len(vals) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=25, marker_color=color,
            marker_line_color="white", marker_line_width=1,
            opacity=0.6, name=label, legendgroup="panel3",
            hovertemplate=("p %{x:.2f}<br>Count %{y}"
                           f"<extra>{label}</extra>"),
        ), row=1, col=3)
        hist_indices.append(len(fig.data) - 1)
    fig.add_vline(
        x=0.5, line=dict(color="black", dash="dash", width=1),
        opacity=0.5, row=1, col=3,
        annotation_text="Threshold", annotation_position="top right",
    )

    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Coefficient (z-scored)", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Predicted death probability", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    fig.update_layout(
        barmode="overlay",
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(
        fig, bin_menu_traces=hist_indices, height=500,
        margin=dict(t=80, b=50, l=80, r=30),
    )


def plot_fluorescence_vs_volume(tracked):
    """Scatter total and mean fluorescence vs cell volume."""

    full = tracked.dropna(subset=["total_intensity", "volume"])
    sample = full.sample(n=min(3000, len(full)), random_state=42)
    scope_note = _scatter_scope(sample, full)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Total fluorescence vs. cell volume — {scope_note}",
            f"Mean fluorescence vs. cell volume — {scope_note}",
        ),
        horizontal_spacing=0.1,
    )

    for col, y_col, y_label in [
        (1, "total_intensity", "Total fluorescence intensity"),
        (2, "mean_intensity", "Mean fluorescence intensity"),
    ]:
        fig.add_trace(go.Scatter(
            x=sample["volume"], y=sample[y_col], mode="markers",
            marker=dict(color="darkorange", size=4, opacity=0.25,
                        line=dict(width=0)),
            name=y_label, showlegend=False,
            hovertemplate="V %{x:.0f}<br>F %{y:.0f}<extra></extra>",
        ), row=1, col=col)
        fig.update_xaxes(title_text="Volume (µm³)", row=1, col=col)
        fig.update_yaxes(title_text=y_label, row=1, col=col)
        if len(full) > 2:
            r = np.corrcoef(full["volume"], full[y_col])[0, 1]
            xref = "x domain" if col == 1 else "x2 domain"
            yref = "y domain" if col == 1 else "y2 domain"
            fig.add_annotation(
                x=0.05, y=0.95, xref=xref, yref=yref,
                text=f"r = {r:.3f} (n={len(full)})",
                showarrow=False, font=dict(size=12, color="black"),
                xanchor="left", yanchor="top",
            )

    
    _finalize_plotly(fig, height=520)


def plot_metric_dynamics(tracked, track_stats, metric, label, color):
    """3-panel plot: per-frame, lifespan-normalized, and outcome-split.

    Generic for any per-cell metric (CV, nNRM, etc.).
    """

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"Population mean {label} per frame",
            f"{label} over normalized lifespan (all tracks)",
            f"{label} dynamics: disappeared vs. survived (all tracks)",
        ),
        horizontal_spacing=0.08,
    )

    per_frame = tracked.groupby("frame")[metric].agg(["mean", "sem"])
    _add_mean_sem_band(
        fig, per_frame.index, per_frame["mean"], per_frame["sem"],
        color=color, name="Mean +/- SEM", legendgroup="panel1",
        row=1, col=1, y_label=label,
    )

    n_bins = 20
    survived_ids, disappeared_ids = _survival_split(tracked)
    bin_centers = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)

    for grp_label, ids, clr in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        norm_parts = []
        for tid, grp in tracked[tracked["track_id"].isin(ids)].groupby("track_id"):
            grp = grp.sort_values("frame")
            if len(grp) < 2:
                continue
            first, last = grp["frame"].iloc[0], grp["frame"].iloc[-1]
            if first == last:
                continue
            grp = grp.copy()
            grp["t_norm"] = (grp["frame"] - first) / (last - first)
            norm_parts.append(grp[["track_id", "t_norm", metric]])
        if not norm_parts:
            continue
        norm_df = pd.concat(norm_parts, ignore_index=True)
        norm_df["bin"] = pd.cut(norm_df["t_norm"], bins=n_bins, labels=False)
        bin_stats = norm_df.groupby("bin")[metric].agg(["mean", "sem"])
        n_tracks = norm_df["track_id"].nunique()
        _add_mean_sem_band(
            fig, pd.Series(bin_centers, index=bin_stats.index),
            bin_stats["mean"], bin_stats["sem"],
            color=clr, name=f"{grp_label} (n={n_tracks})",
            legendgroup="panel2", row=1, col=2, y_label=label,
        )

    for grp_label, ids, clr in [
        ("Survived", survived_ids, "steelblue"),
        ("Disappeared", disappeared_ids, "tomato"),
    ]:
        sub = tracked[tracked["track_id"].isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("frame")[metric].agg(["mean", "sem"])
        _add_mean_sem_band(
            fig, g.index, g["mean"], g["sem"],
            color=clr, name=f"{grp_label} (n={len(ids)})",
            legendgroup="panel3", row=1, col=3, y_label=label,
        )

    frame_ticks = list(range(int(tracked["frame"].min()),
                             int(tracked["frame"].max()) + 1))
    fig.update_xaxes(title_text="Frame", row=1, col=1, tickmode="array",
                     tickvals=list(per_frame.index))
    fig.update_yaxes(title_text=label, row=1, col=1)
    fig.update_xaxes(title_text="Relative lifespan (0=start, 1=end)",
                     row=1, col=2)
    fig.update_yaxes(title_text=label, row=1, col=2)
    fig.update_xaxes(title_text="Frame", row=1, col=3, tickmode="array",
                     tickvals=frame_ticks)
    fig.update_yaxes(title_text=label, row=1, col=3)
    fig.update_layout(
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(fig, height=500)

    print(f"{label} at frame 0: {per_frame['mean'].iloc[0]:.3f}")
    print(
        f"{label} at frame {int(per_frame.index[-1])}: "
        f"{per_frame['mean'].iloc[-1]:.3f}"
    )


def plot_initial_features_vs_lifespan(tracked, track_stats):
    """3-panel scatter: frame-0 fluorescence, CV, nNRM vs track lifetime."""
    from scipy.stats import spearmanr

    frame0_tracks = _frame0_track_ids(tracked)
    cohort = tracked[tracked["track_id"].isin(frame0_tracks)].copy()
    first_obs = cohort.sort_values("frame").groupby("track_id").first()

    df = first_obs[["mean_intensity", "cv", "nnrm"]].copy()
    df = df.merge(
        track_stats[["track_id", "lifetime", "disappeared"]].set_index("track_id"),
        left_index=True, right_index=True,
    )
    df = df.dropna(subset=["mean_intensity", "cv", "nnrm"])

    metrics = [
        ("mean_intensity", "Initial fluorescence"),
        ("cv", "Initial CV"),
        ("nnrm", "Initial nNRM"),
    ]

    corr_results = []
    titles = []
    for col, title in metrics:
        r, p = spearmanr(df[col], df["lifetime"])
        corr_results.append((title, r, p))
        titles.append(f"{title} vs. lifespan (ρ={r:.3f}, p={p:.2e})")

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=tuple(titles),
        horizontal_spacing=0.08,
    )

    for i, (col, title) in enumerate(metrics):
        group = f"panel{i + 1}"
        for label, mask_val, color in [
            ("Survived", False, "steelblue"),
            ("Disappeared", True, "tomato"),
        ]:
            sub = df[df["disappeared"] == mask_val]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub[col], y=sub["lifetime"], mode="markers",
                marker=dict(color=color, size=6, opacity=0.55,
                            line=dict(width=0)),
                name=label, legendgroup=group,
                showlegend=(i == 0),
                hovertemplate=(f"{title} %{{x:.2f}}<br>Lifetime %{{y}}"
                               f"<extra>{label}</extra>"),
            ), row=1, col=i + 1)
        fig.update_xaxes(title_text=title, row=1, col=i + 1)
        fig.update_yaxes(title_text="Lifetime (frames)", row=1, col=i + 1)

    fig.update_layout(
        legend=dict(groupclick="toggleitem"),
    )
    _finalize_plotly(fig, height=500)

    for title, r, p in corr_results:
        print(f"{title} vs lifetime: Spearman rho = {r:.3f}, p = {p:.2e}")
