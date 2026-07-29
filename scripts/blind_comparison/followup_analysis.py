#!/usr/bin/env python3
"""
Follow-up analysis comparing temporal dynamics and fate-stratified metrics
between control and protein synthesis arrested conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Data locations
CONTROL_DIR = Path("/Users/vvetshte/Projects/temp/experiments_image/results/control_experiment")
ARRESTED_DIR = Path("/Users/vvetshte/Projects/temp/experiments_image/results/protein_synthesis_arrested")

def load_data(results_dir):
    """Load tracked_cells and track_statistics for a condition."""
    tracked = pd.read_csv(results_dir / "tracked_cells.csv")
    tracks = pd.read_csv(results_dir / "track_statistics.csv")
    return tracked, tracks

def analyze_relative_fluorescence(tracked_df, condition_name):
    """
    Analysis 1: F(t)/F(0) for all tracked cells.
    """
    print(f"\n{'='*80}")
    print(f"1. RELATIVE FLUORESCENCE F(t)/F(0) — {condition_name}")
    print(f"{'='*80}")

    # Get first frame intensity for each track
    first_intensities = tracked_df.groupby('track_id')['mean_intensity'].first()

    # Normalize each track's intensities
    normalized_data = []
    for track_id in tracked_df['track_id'].unique():
        track_data = tracked_df[tracked_df['track_id'] == track_id].copy()
        f0 = first_intensities[track_id]
        if f0 > 0:  # Avoid division by zero
            track_data['f_normalized'] = track_data['mean_intensity'] / f0
            normalized_data.append(track_data[['frame', 'track_id', 'f_normalized']])

    normalized_df = pd.concat(normalized_data, ignore_index=True)

    # Compute mean F(t)/F(0) per frame across all alive tracks
    per_frame = normalized_df.groupby('frame').agg(
        mean_f_norm=('f_normalized', 'mean'),
        sem_f_norm=('f_normalized', 'sem'),
        n_cells=('track_id', 'count')
    ).reset_index()

    print(f"\nFrame-by-frame F(t)/F(0):")
    print(f"{'Frame':<8} {'Mean F/F0':<12} {'SEM':<12} {'N cells':<10}")
    print("-" * 45)
    for _, row in per_frame.iterrows():
        print(f"{row['frame']:<8} {row['mean_f_norm']:<12.3f} {row['sem_f_norm']:<12.3f} {row['n_cells']:<10.0f}")

    # Final frame summary
    final = per_frame.iloc[-1]
    print(f"\nFinal frame ({int(final['frame'])}):")
    print(f"  Mean F(t)/F(0) = {final['mean_f_norm']:.3f} ± {final['sem_f_norm']:.3f}")
    print(f"  N cells contributing = {int(final['n_cells'])}")

    return per_frame

def analyze_swelling_by_fate(tracked_df, tracks_df, condition_name):
    """
    Analysis 2: Swelling metrics stratified by fate (survived vs disappeared).
    """
    print(f"\n{'='*80}")
    print(f"2. SWELLING ANALYSIS BY FATE — {condition_name}")
    print(f"{'='*80}")

    survived = tracks_df[tracks_df['disappeared'] == False]
    disappeared = tracks_df[tracks_df['disappeared'] == True]

    for fate_label, subset in [("SURVIVED", survived), ("DISAPPEARED", disappeared)]:
        print(f"\n{fate_label} (N={len(subset)}):")
        print(f"  Mean initial area: {subset['area_initial'].mean():.3f} ± {subset['area_initial'].std():.3f} px²")
        print(f"  Median max relative swelling: {subset['area_rel_max'].median():.3f}")

        # Swelling rate
        subset = subset.copy()
        subset['swelling_rate'] = (subset['area_max'] - subset['area_initial']) / (subset['frame_of_max_area'] - subset['first_frame'])
        subset['swelling_rate'] = subset['swelling_rate'].replace([np.inf, -np.inf], np.nan)
        valid_rates = subset['swelling_rate'].dropna()
        print(f"  Median swelling rate: {valid_rates.median():.3f} px²/frame (N={len(valid_rates)})")
        print(f"  Median frame of max area: {subset['frame_of_max_area'].median():.1f}")

    # Statistical comparison
    print(f"\nMann-Whitney U test (survived vs disappeared):")

    # Initial area
    u_stat, p_val = stats.mannwhitneyu(survived['area_initial'], disappeared['area_initial'], alternative='two-sided')
    print(f"  Initial area: U={u_stat:.1f}, p={p_val:.4e}")

    # Max relative swelling
    u_stat, p_val = stats.mannwhitneyu(survived['area_rel_max'], disappeared['area_rel_max'], alternative='two-sided')
    print(f"  Max relative swelling: U={u_stat:.1f}, p={p_val:.4e}")

    # Swelling rate
    survived_copy = survived.copy()
    disappeared_copy = disappeared.copy()
    survived_copy['swelling_rate'] = (survived_copy['area_max'] - survived_copy['area_initial']) / (survived_copy['frame_of_max_area'] - survived_copy['first_frame'])
    disappeared_copy['swelling_rate'] = (disappeared_copy['area_max'] - disappeared_copy['area_initial']) / (disappeared_copy['frame_of_max_area'] - disappeared_copy['first_frame'])

    survived_rates = survived_copy['swelling_rate'].replace([np.inf, -np.inf], np.nan).dropna()
    disappeared_rates = disappeared_copy['swelling_rate'].replace([np.inf, -np.inf], np.nan).dropna()

    if len(survived_rates) > 0 and len(disappeared_rates) > 0:
        u_stat, p_val = stats.mannwhitneyu(survived_rates, disappeared_rates, alternative='two-sided')
        print(f"  Swelling rate: U={u_stat:.1f}, p={p_val:.4e}")

    return {
        'survived': survived,
        'disappeared': disappeared,
        'survived_rates': survived_rates if len(survived_rates) > 0 else None,
        'disappeared_rates': disappeared_rates if len(disappeared_rates) > 0 else None
    }

def analyze_sav_temporal(tracked_df, tracks_df, condition_name):
    """
    Analysis 3: SA:V ratio temporal dynamics.
    """
    print(f"\n{'='*80}")
    print(f"3. SA:V RATIO TEMPORAL DYNAMICS — {condition_name}")
    print(f"{'='*80}")

    # Median SA:V per frame
    sav_per_frame = tracked_df.groupby('frame')['sav_ratio'].median()
    print(f"\nMedian SA:V per frame:")
    print(f"  Start (frame {sav_per_frame.index[0]}): {sav_per_frame.iloc[0]:.3f}")
    print(f"  End (frame {sav_per_frame.index[-1]}): {sav_per_frame.iloc[-1]:.3f}")
    print(f"  Change: {sav_per_frame.iloc[-1] - sav_per_frame.iloc[0]:.3f}")

    # Linear slope
    frames = sav_per_frame.index.values
    values = sav_per_frame.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(frames, values)
    print(f"  Linear slope: {slope:.6f} per frame (R²={r_value**2:.3f}, p={p_value:.4e})")

    # SA:V at death/end by fate
    print(f"\nSA:V at final observation by fate:")

    # Get last observation for each track
    last_obs = tracked_df.groupby('track_id').last().reset_index()
    last_obs = last_obs.merge(tracks_df[['track_id', 'disappeared']], on='track_id')

    disappeared_sav = last_obs[last_obs['disappeared'] == True]['sav_ratio']
    survived_sav = last_obs[last_obs['disappeared'] == False]['sav_ratio']

    print(f"  Disappeared cells (N={len(disappeared_sav)}): median SA:V = {disappeared_sav.median():.3f}")
    print(f"  Survived cells (N={len(survived_sav)}): median SA:V = {survived_sav.median():.3f}")

    return sav_per_frame

def analyze_cv_nnrm_temporal(tracked_df, condition_name):
    """
    Analysis 4: CV and nNRM temporal trajectories.
    """
    print(f"\n{'='*80}")
    print(f"4. CV AND nNRM TEMPORAL TRAJECTORIES — {condition_name}")
    print(f"{'='*80}")

    # CV trajectory
    cv_per_frame = tracked_df.groupby('frame')['cv'].median()
    print(f"\nCV (coefficient of variation):")
    print(f"  Start (frame {cv_per_frame.index[0]}): {cv_per_frame.iloc[0]:.3f}")
    print(f"  End (frame {cv_per_frame.index[-1]}): {cv_per_frame.iloc[-1]:.3f}")
    print(f"  Change: {cv_per_frame.iloc[-1] - cv_per_frame.iloc[0]:.3f}")

    # Linear fit
    frames = cv_per_frame.index.values
    values = cv_per_frame.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(frames, values)
    print(f"  Linear slope: {slope:.6f} per frame (R²={r_value**2:.3f}, p={p_value:.4e})")

    direction = "increasing" if slope > 0 else "decreasing"
    print(f"  Direction: {direction}")

    # Check for non-monotonic pattern
    diff = np.diff(values)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    if sign_changes > len(values) * 0.3:
        shape = "non-monotonic (multiple inflections)"
    elif abs(r_value) > 0.7:
        shape = "monotonic"
    else:
        shape = "weak trend"
    print(f"  Shape: {shape}")

    # nNRM trajectory
    print(f"\nnNRM (normalized nuclear-to-rim):")
    nnrm_per_frame = tracked_df.groupby('frame')['nnrm'].median()
    print(f"  Start (frame {nnrm_per_frame.index[0]}): {nnrm_per_frame.iloc[0]:.3f}")
    print(f"  End (frame {nnrm_per_frame.index[-1]}): {nnrm_per_frame.iloc[-1]:.3f}")
    print(f"  Change: {nnrm_per_frame.iloc[-1] - nnrm_per_frame.iloc[0]:.3f}")

    # Linear fit
    frames = nnrm_per_frame.index.values
    values = nnrm_per_frame.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(frames, values)
    print(f"  Linear slope: {slope:.6f} per frame (R²={r_value**2:.3f}, p={p_value:.4e})")

    direction = "increasing" if slope > 0 else "decreasing"
    print(f"  Direction: {direction}")

    # Check for non-monotonic pattern
    diff = np.diff(values)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    if sign_changes > len(values) * 0.3:
        shape = "non-monotonic (multiple inflections)"
    elif abs(r_value) > 0.7:
        shape = "monotonic"
    else:
        shape = "weak trend"
    print(f"  Shape: {shape}")

    return cv_per_frame, nnrm_per_frame

def analyze_lifetime_distribution(tracks_df, condition_name):
    """
    Analysis 5: Lifetime distribution details.
    """
    print(f"\n{'='*80}")
    print(f"5. LIFETIME DISTRIBUTION — {condition_name}")
    print(f"{'='*80}")

    disappeared = tracks_df[tracks_df['disappeared'] == True]

    print(f"\nDisappeared cells (N={len(disappeared)}):")
    print(f"  Median lifetime: {disappeared['lifetime'].median():.1f} frames")
    print(f"  Mean lifetime: {disappeared['lifetime'].mean():.3f} ± {disappeared['lifetime'].std():.3f} frames")

    # Frame of maximum disappearance
    disappearance_counts = disappeared['last_frame'].value_counts().sort_index()
    peak_frame = disappearance_counts.idxmax()
    peak_count = disappearance_counts.max()
    print(f"\nFrame of maximum disappearance: {peak_frame}")
    print(f"  Cells lost at peak frame: {peak_count}")

    # 50% disappearance frame
    total_disappeared = len(disappeared)
    cumulative = disappearance_counts.sort_index().cumsum()
    fifty_pct_frame = cumulative[cumulative >= total_disappeared * 0.5].index[0]
    print(f"\n50% disappearance frame: {fifty_pct_frame}")

    return disappearance_counts

def analyze_cell_count(tracked_df, tracks_df, condition_name):
    """
    Analysis 6: Cell count per frame.
    """
    print(f"\n{'='*80}")
    print(f"6. CELL COUNT PER FRAME — {condition_name}")
    print(f"{'='*80}")

    count_per_frame = tracked_df.groupby('frame')['track_id'].nunique()

    initial_count = count_per_frame.iloc[0]
    final_count = count_per_frame.iloc[-1]
    fraction_disappeared = 1 - (final_count / initial_count)

    print(f"\nInitial count (frame {count_per_frame.index[0]}): {initial_count}")
    print(f"Final count (frame {count_per_frame.index[-1]}): {final_count}")
    print(f"Fraction disappeared: {fraction_disappeared:.3f}")

    # 50% disappearance frame
    fifty_pct_threshold = initial_count * 0.5
    below_threshold = count_per_frame[count_per_frame <= fifty_pct_threshold]
    if len(below_threshold) > 0:
        fifty_pct_frame = below_threshold.index[0]
        print(f"50% disappearance frame: {fifty_pct_frame}")
    else:
        print(f"50% disappearance frame: not reached (>50% survived)")

    return count_per_frame

def analyze_temporal_correlations(tracked_df, tracks_df, condition_name):
    """
    Analysis 7: Temporal correlation analysis.
    """
    print(f"\n{'='*80}")
    print(f"7. TEMPORAL CORRELATION ANALYSIS — {condition_name}")
    print(f"{'='*80}")

    # Get first frame data for each track
    first_frame_data = tracked_df.groupby('track_id').first().reset_index()
    first_frame_data = first_frame_data.merge(tracks_df[['track_id', 'lifetime', 'disappeared']], on='track_id')

    # Initial CV vs lifetime (using CV from first frame)
    print(f"\nInitial fluorescence heterogeneity (CV) vs lifetime:")
    rho, p_val = stats.spearmanr(first_frame_data['cv'], first_frame_data['lifetime'])
    print(f"  Spearman ρ = {rho:.3f}, p = {p_val:.4e}")

    # Initial area vs lifetime
    print(f"\nInitial area vs lifetime:")
    rho, p_val = stats.spearmanr(first_frame_data['area'], first_frame_data['lifetime'])
    print(f"  Spearman ρ = {rho:.3f}, p = {p_val:.4e}")

    # Swelling rate vs fluorescence loss rate
    # Calculate fluorescence loss rate for each track
    fluor_rates = []
    swelling_rates = []

    for track_id in tracks_df['track_id'].unique():
        track_data = tracked_df[tracked_df['track_id'] == track_id].sort_values('frame')
        track_stats = tracks_df[tracks_df['track_id'] == track_id].iloc[0]

        if len(track_data) > 1:
            # Fluorescence loss rate (linear fit)
            frames = track_data['frame'].values
            intensities = track_data['mean_intensity'].values
            if len(frames) > 1:
                slope_f, _, _, _, _ = stats.linregress(frames, intensities)
                fluor_rates.append(slope_f)

                # Swelling rate
                swelling_rate = (track_stats['area_max'] - track_stats['area_initial']) / (track_stats['frame_of_max_area'] - track_stats['first_frame'])
                if not np.isnan(swelling_rate) and not np.isinf(swelling_rate):
                    swelling_rates.append(swelling_rate)
                else:
                    fluor_rates.pop()  # Remove the last fluorescence rate to keep arrays aligned

    print(f"\nSwelling rate vs fluorescence loss rate:")
    if len(fluor_rates) > 5 and len(swelling_rates) > 5:
        rho, p_val = stats.spearmanr(swelling_rates, fluor_rates)
        print(f"  Spearman ρ = {rho:.3f}, p = {p_val:.4e}")
    else:
        print(f"  Insufficient data (N={len(fluor_rates)})")

    # Changepoint frame vs fluor disappearance frame
    print(f"\nChangepoint frame vs fluorescence disappearance frame:")
    valid_data = tracks_df.dropna(subset=['changepoint_frame', 'fluor_disappearance_frame'])
    if len(valid_data) > 5:
        rho, p_val = stats.spearmanr(valid_data['changepoint_frame'], valid_data['fluor_disappearance_frame'])
        print(f"  Spearman ρ = {rho:.3f}, p = {p_val:.4e}")
        print(f"  N = {len(valid_data)}")
    else:
        print(f"  Insufficient data (N={len(valid_data)})")

def main():
    """Run all analyses for both conditions."""

    print("="*80)
    print("TEMPORAL DYNAMICS AND FATE-STRATIFIED ANALYSIS")
    print("="*80)

    # Load data
    control_tracked, control_tracks = load_data(CONTROL_DIR)
    arrested_tracked, arrested_tracks = load_data(ARRESTED_DIR)

    print(f"\nData loaded:")
    print(f"  Control: {len(control_tracks)} tracks, {len(control_tracked)} observations")
    print(f"  Arrested: {len(arrested_tracks)} tracks, {len(arrested_tracked)} observations")

    # Run all analyses for each condition
    for condition_name, tracked_df, tracks_df in [
        ("CONTROL", control_tracked, control_tracks),
        ("PROTEIN SYNTHESIS ARRESTED", arrested_tracked, arrested_tracks)
    ]:
        analyze_relative_fluorescence(tracked_df, condition_name)
        analyze_swelling_by_fate(tracked_df, tracks_df, condition_name)
        analyze_sav_temporal(tracked_df, tracks_df, condition_name)
        analyze_cv_nnrm_temporal(tracked_df, condition_name)
        analyze_lifetime_distribution(tracks_df, condition_name)
        analyze_cell_count(tracked_df, tracks_df, condition_name)
        analyze_temporal_correlations(tracked_df, tracks_df, condition_name)

    # Cross-condition comparisons
    print(f"\n{'='*80}")
    print(f"CROSS-CONDITION COMPARISONS")
    print(f"{'='*80}")

    # Compare survived vs disappeared gap between conditions
    print(f"\nSwelling metrics — survived vs disappeared gap comparison:")

    control_swelling = analyze_swelling_by_fate(control_tracked, control_tracks, "CONTROL")
    arrested_swelling = analyze_swelling_by_fate(arrested_tracked, arrested_tracks, "ARRESTED")

    # Effect sizes (median differences)
    control_gap = control_swelling['disappeared']['area_rel_max'].median() - control_swelling['survived']['area_rel_max'].median()
    arrested_gap = arrested_swelling['disappeared']['area_rel_max'].median() - arrested_swelling['survived']['area_rel_max'].median()

    print(f"\nMax relative swelling gap (disappeared - survived):")
    print(f"  Control: {control_gap:.3f}")
    print(f"  Arrested: {arrested_gap:.3f}")
    print(f"  Difference: {arrested_gap - control_gap:.3f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
