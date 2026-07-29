#!/usr/bin/env python3
"""
Comprehensive blind comparison between two experimental conditions.
Reports all statistically significant differences across all metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, kstest, ks_2samp
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Paths
CONDITION_A = "/Users/vvetshte/Projects/temp/experiments_image/results/control_experiment"
CONDITION_B = "/Users/vvetshte/Projects/temp/experiments_image/results/protein_synthesis_arrested"

def load_datasets():
    """Load all CSV files from both conditions."""
    data = {}
    files = [
        'tracked_cells.csv',
        'track_statistics.csv',
        'frame_diagnostics.csv',
        'fate_predictions.csv',
        'fate_prediction_summary.csv',
        'clustering_summary.csv',
        'spatial_gradient.csv',
        'spatial_gradient_summary.csv',
        'nucleus_persistence.csv',
        'nucleus_persistence_summary.csv',
        'merge_log.csv'
    ]

    for fname in files:
        try:
            data[f'a_{fname}'] = pd.read_csv(f"{CONDITION_A}/{fname}")
            data[f'b_{fname}'] = pd.read_csv(f"{CONDITION_B}/{fname}")
            print(f"✓ Loaded {fname}")
        except FileNotFoundError:
            print(f"✗ Missing {fname}")
            data[f'a_{fname}'] = None
            data[f'b_{fname}'] = None

    return data

def cohen_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def effect_size_interpretation(d):
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

class ComparisonResults:
    """Store all comparison results for multiple testing correction."""
    def __init__(self):
        self.tests = []

    def add(self, category, metric, stat_name, stat_value, p_value, a_val, b_val, effect_size=None, notes=""):
        self.tests.append({
            'category': category,
            'metric': metric,
            'stat_name': stat_name,
            'stat_value': stat_value,
            'p_value': p_value,
            'a_value': a_val,
            'b_value': b_val,
            'effect_size': effect_size,
            'notes': notes
        })

    def correct_and_report(self, alpha=0.05):
        """Apply Benjamini-Hochberg correction and report significant findings."""
        df = pd.DataFrame(self.tests)

        # Multiple testing correction
        reject, p_adj, _, _ = multipletests(df['p_value'], alpha=alpha, method='fdr_bh')
        df['p_adjusted'] = p_adj
        df['significant'] = reject

        # Sort by category and p-value
        df = df.sort_values(['category', 'p_adjusted'])

        return df

def analyze_population(data, results):
    """1. Population-level differences."""
    print("\n" + "="*80)
    print("1. POPULATION-LEVEL DIFFERENCES")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']
    a_cells = data['a_tracked_cells.csv']
    b_cells = data['b_tracked_cells.csv']

    # Total unique tracks
    a_n_tracks = len(a_tracks)
    b_n_tracks = len(b_tracks)
    print(f"\nTotal tracks: A={a_n_tracks}, B={b_n_tracks}")

    # Total cell detections
    a_n_detections = len(a_cells)
    b_n_detections = len(b_cells)
    print(f"Total detections: A={a_n_detections}, B={b_n_detections}")

    # Disappearance rate
    a_disappeared = a_tracks['disappeared'].sum()
    b_disappeared = b_tracks['disappeared'].sum()
    a_disappear_rate = a_disappeared / a_n_tracks
    b_disappear_rate = b_disappeared / b_n_tracks

    print(f"\nDisappearance counts: A={a_disappeared}/{a_n_tracks}, B={b_disappeared}/{b_n_tracks}")
    print(f"Disappearance rates: A={a_disappear_rate:.3f}, B={b_disappear_rate:.3f}")

    # Chi-squared test for disappearance
    contingency = np.array([
        [a_disappeared, a_n_tracks - a_disappeared],
        [b_disappeared, b_n_tracks - b_disappeared]
    ])
    chi2, p_chi = chi2_contingency(contingency)[:2]
    results.add('Population', 'Disappearance rate', 'Chi-squared', chi2, p_chi,
                a_disappear_rate, b_disappear_rate, notes=f"A: {a_disappeared}/{a_n_tracks}, B: {b_disappeared}/{b_n_tracks}")

    # Detections per track
    a_detections_per_track = a_tracks['num_detections'].values
    b_detections_per_track = b_tracks['num_detections'].values
    u_stat, p_mw = mannwhitneyu(a_detections_per_track, b_detections_per_track, alternative='two-sided')
    d = cohen_d(a_detections_per_track, b_detections_per_track)
    print(f"\nDetections per track: A={np.median(a_detections_per_track):.1f} (median), B={np.median(b_detections_per_track):.1f}")
    results.add('Population', 'Detections per track', 'Mann-Whitney U', u_stat, p_mw,
                np.median(a_detections_per_track), np.median(b_detections_per_track), d)

    # Lifetime distribution
    a_lifetime = a_tracks['lifetime'].values
    b_lifetime = b_tracks['lifetime'].values
    u_stat, p_mw = mannwhitneyu(a_lifetime, b_lifetime, alternative='two-sided')
    d = cohen_d(a_lifetime, b_lifetime)
    print(f"Lifetime (frames): A={np.median(a_lifetime):.1f} (median), B={np.median(b_lifetime):.1f}")
    results.add('Population', 'Track lifetime', 'Mann-Whitney U', u_stat, p_mw,
                np.median(a_lifetime), np.median(b_lifetime), d)

    # Cells per frame over time
    a_frame = data['a_frame_diagnostics.csv']
    b_frame = data['b_frame_diagnostics.csv']

    if 'num_cells' in a_frame.columns:
        a_cells_per_frame = a_frame['num_cells'].values
        b_cells_per_frame = b_frame['num_cells'].values
        u_stat, p_mw = mannwhitneyu(a_cells_per_frame, b_cells_per_frame, alternative='two-sided')
        d = cohen_d(a_cells_per_frame, b_cells_per_frame)
        print(f"Cells per frame: A={np.mean(a_cells_per_frame):.1f}±{np.std(a_cells_per_frame):.1f}, B={np.mean(b_cells_per_frame):.1f}±{np.std(b_cells_per_frame):.1f}")
        results.add('Population', 'Cells per frame', 'Mann-Whitney U', u_stat, p_mw,
                    np.mean(a_cells_per_frame), np.mean(b_cells_per_frame), d)

def analyze_morphology(data, results):
    """2. Morphological differences."""
    print("\n" + "="*80)
    print("2. MORPHOLOGICAL DIFFERENCES")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']
    a_cells = data['a_tracked_cells.csv']
    b_cells = data['b_tracked_cells.csv']

    # Initial area (first frame of each track)
    a_initial = a_cells.groupby('track_id').first()
    b_initial = b_cells.groupby('track_id').first()

    for metric in ['area', 'volume', 'surface_area', 'sav_ratio', 'radius']:
        if metric in a_initial.columns:
            a_vals = a_initial[metric].dropna().values
            b_vals = b_initial[metric].dropna().values
            u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            d = cohen_d(a_vals, b_vals)
            print(f"\nInitial {metric}: A={np.median(a_vals):.2f}, B={np.median(b_vals):.2f} (median)")
            print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
            results.add('Morphology', f'Initial {metric}', 'Mann-Whitney U', u_stat, p_mw,
                        np.median(a_vals), np.median(b_vals), d)

    # Mean values over lifetime
    for metric in ['mean_area', 'mean_volume', 'mean_surface_area']:
        if metric in a_tracks.columns:
            a_vals = a_tracks[metric].dropna().values
            b_vals = b_tracks[metric].dropna().values
            u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            d = cohen_d(a_vals, b_vals)
            print(f"\n{metric}: A={np.median(a_vals):.2f}, B={np.median(b_vals):.2f} (median)")
            print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
            results.add('Morphology', metric, 'Mann-Whitney U', u_stat, p_mw,
                        np.median(a_vals), np.median(b_vals), d)

    # Area dynamics: growth rate
    if 'growth_rate_px_per_frame' in a_tracks.columns:
        a_gr = a_tracks['growth_rate_px_per_frame'].dropna().values
        b_gr = b_tracks['growth_rate_px_per_frame'].dropna().values
        u_stat, p_mw = mannwhitneyu(a_gr, b_gr, alternative='two-sided')
        d = cohen_d(a_gr, b_gr)
        print(f"\nGrowth rate (px/frame): A={np.median(a_gr):.3f}, B={np.median(b_gr):.3f}")
        print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
        results.add('Morphology', 'Growth rate', 'Mann-Whitney U', u_stat, p_mw,
                    np.median(a_gr), np.median(b_gr), d)

def analyze_fluorescence(data, results):
    """3. Fluorescence differences."""
    print("\n" + "="*80)
    print("3. FLUORESCENCE DIFFERENCES")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']
    a_cells = data['a_tracked_cells.csv']
    b_cells = data['b_tracked_cells.csv']

    # Initial fluorescence (first frame of each track)
    a_initial = a_cells.groupby('track_id').first()
    b_initial = b_cells.groupby('track_id').first()

    fluor_metrics = ['mean_intensity', 'total_intensity', 'min_intensity', 'max_intensity',
                     'std_intensity', 'cv', 'skewness', 'kurtosis', 'nnrm', 'fluor_concentration']

    for metric in fluor_metrics:
        if metric in a_initial.columns:
            a_vals = a_initial[metric].dropna().values
            b_vals = b_initial[metric].dropna().values
            if len(a_vals) > 0 and len(b_vals) > 0:
                u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                d = cohen_d(a_vals, b_vals)
                print(f"\nInitial {metric}: A={np.median(a_vals):.2f}, B={np.median(b_vals):.2f}")
                print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
                results.add('Fluorescence', f'Initial {metric}', 'Mann-Whitney U', u_stat, p_mw,
                            np.median(a_vals), np.median(b_vals), d)

    # Mean over lifetime
    for metric in ['mean_fluor_intensity', 'mean_fluor_total', 'mean_cv', 'mean_nnrm']:
        if metric in a_tracks.columns:
            a_vals = a_tracks[metric].dropna().values
            b_vals = b_tracks[metric].dropna().values
            if len(a_vals) > 0 and len(b_vals) > 0:
                u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                d = cohen_d(a_vals, b_vals)
                print(f"\n{metric}: A={np.median(a_vals):.2f}, B={np.median(b_vals):.2f}")
                print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
                results.add('Fluorescence', metric, 'Mann-Whitney U', u_stat, p_mw,
                            np.median(a_vals), np.median(b_vals), d)

    # Fluorescence disappearance frame
    if 'fluor_disappearance_frame' in a_tracks.columns:
        a_fd = a_tracks[a_tracks['disappeared'] == 1]['fluor_disappearance_frame'].dropna().values
        b_fd = b_tracks[b_tracks['disappeared'] == 1]['fluor_disappearance_frame'].dropna().values
        if len(a_fd) > 0 and len(b_fd) > 0:
            u_stat, p_mw = mannwhitneyu(a_fd, b_fd, alternative='two-sided')
            d = cohen_d(a_fd, b_fd)
            print(f"\nFluorescence disappearance frame (disappeared tracks only):")
            print(f"  A={np.median(a_fd):.1f}, B={np.median(b_fd):.1f}")
            print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
            results.add('Fluorescence', 'Fluor disappearance frame', 'Mann-Whitney U', u_stat, p_mw,
                        np.median(a_fd), np.median(b_fd), d)

    # Max fluorescence drop
    if 'max_drop' in a_tracks.columns:
        a_md = a_tracks['max_drop'].dropna().values
        b_md = b_tracks['max_drop'].dropna().values
        if len(a_md) > 0 and len(b_md) > 0:
            u_stat, p_mw = mannwhitneyu(a_md, b_md, alternative='two-sided')
            d = cohen_d(a_md, b_md)
            print(f"\nMax fluorescence drop: A={np.median(a_md):.2f}, B={np.median(b_md):.2f}")
            print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
            results.add('Fluorescence', 'Max fluor drop', 'Mann-Whitney U', u_stat, p_mw,
                        np.median(a_md), np.median(b_md), d)

    # Relative fluorescence dynamics F(t)/F(0) for surviving tracks
    print("\n--- Relative fluorescence dynamics (surviving tracks) ---")
    a_survived = a_cells[a_cells['track_id'].isin(a_tracks[a_tracks['disappeared'] == 0]['track_id'])]
    b_survived = b_cells[b_cells['track_id'].isin(b_tracks[b_tracks['disappeared'] == 0]['track_id'])]

    if 'mean_intensity' in a_survived.columns:
        a_rel = []
        for track_id, grp in a_survived.groupby('track_id'):
            f0 = grp.iloc[0]['mean_intensity']
            if f0 > 0:
                a_rel.extend((grp['mean_intensity'] / f0).values)

        b_rel = []
        for track_id, grp in b_survived.groupby('track_id'):
            f0 = grp.iloc[0]['mean_intensity']
            if f0 > 0:
                b_rel.extend((grp['mean_intensity'] / f0).values)

        if len(a_rel) > 0 and len(b_rel) > 0:
            u_stat, p_mw = mannwhitneyu(a_rel, b_rel, alternative='two-sided')
            d = cohen_d(np.array(a_rel), np.array(b_rel))
            print(f"Relative fluorescence F(t)/F(0): A={np.median(a_rel):.3f}, B={np.median(b_rel):.3f}")
            print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
            results.add('Fluorescence', 'Relative F(t)/F(0) survived', 'Mann-Whitney U', u_stat, p_mw,
                        np.median(a_rel), np.median(b_rel), d)

def analyze_kinetics(data, results):
    """4. Kinetic differences."""
    print("\n" + "="*80)
    print("4. KINETIC DIFFERENCES")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']

    kinetic_metrics = ['mean_speed', 'max_speed', 'speed_std', 'total_displacement', 'net_displacement']

    for metric in kinetic_metrics:
        if metric in a_tracks.columns:
            a_vals = a_tracks[metric].dropna().values
            b_vals = b_tracks[metric].dropna().values
            if len(a_vals) > 0 and len(b_vals) > 0:
                u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                d = cohen_d(a_vals, b_vals)
                print(f"\n{metric}: A={np.median(a_vals):.2f}, B={np.median(b_vals):.2f}")
                print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
                results.add('Kinetics', metric, 'Mann-Whitney U', u_stat, p_mw,
                            np.median(a_vals), np.median(b_vals), d)

def analyze_swelling(data, results):
    """5. Swelling analysis."""
    print("\n" + "="*80)
    print("5. SWELLING ANALYSIS")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']

    # Overall swelling metrics
    for metric in ['area_initial', 'area_max', 'area_rel_max', 'frame_of_max_area']:
        if metric in a_tracks.columns:
            a_vals = a_tracks[metric].dropna().values
            b_vals = b_tracks[metric].dropna().values
            if len(a_vals) > 0 and len(b_vals) > 0:
                u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                d = cohen_d(a_vals, b_vals)
                print(f"\n{metric}: A={np.median(a_vals):.2f}, B={np.median(b_vals):.2f}")
                print(f"  Cohen's d={d:.3f} ({effect_size_interpretation(d)})")
                results.add('Swelling', metric, 'Mann-Whitney U', u_stat, p_mw,
                            np.median(a_vals), np.median(b_vals), d)

    # Split by fate
    print("\n--- Split by disappeared vs survived ---")
    for fate in [0, 1]:
        fate_label = "survived" if fate == 0 else "disappeared"
        a_fate = a_tracks[a_tracks['disappeared'] == fate]
        b_fate = b_tracks[b_tracks['disappeared'] == fate]

        print(f"\n{fate_label.upper()}:")
        for metric in ['area_rel_max', 'growth_rate_px_per_frame']:
            if metric in a_tracks.columns:
                a_vals = a_fate[metric].dropna().values
                b_vals = b_fate[metric].dropna().values
                if len(a_vals) > 0 and len(b_vals) > 0:
                    u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                    d = cohen_d(a_vals, b_vals)
                    print(f"  {metric}: A={np.median(a_vals):.3f}, B={np.median(b_vals):.3f}, d={d:.3f}")
                    results.add('Swelling', f'{metric} ({fate_label})', 'Mann-Whitney U', u_stat, p_mw,
                                np.median(a_vals), np.median(b_vals), d)

def analyze_fate_prediction(data, results):
    """6. Fate prediction comparison."""
    print("\n" + "="*80)
    print("6. FATE PREDICTION MODEL PERFORMANCE")
    print("="*80)

    a_summary = data['a_fate_prediction_summary.csv']
    b_summary = data['b_fate_prediction_summary.csv']

    if a_summary is not None and b_summary is not None:
        print("\nCondition A:")
        print(a_summary.to_string(index=False))
        print("\nCondition B:")
        print(b_summary.to_string(index=False))

        # Extract scalar metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in a_summary.columns:
                a_val = a_summary[metric].values[0]
                b_val = b_summary[metric].values[0]
                print(f"\n{metric}: A={a_val:.3f}, B={b_val:.3f}, diff={b_val-a_val:.3f}")
                # Note: no statistical test here (single values per condition)
                results.add('Fate prediction', metric, 'Descriptive', np.nan, np.nan, a_val, b_val,
                            notes="Single-value comparison, no test")

    # Feature importances (if available in predictions file)
    a_preds = data['a_fate_predictions.csv']
    b_preds = data['b_fate_predictions.csv']

    if a_preds is not None and b_preds is not None:
        print("\n--- Feature distributions in predictions ---")
        feature_cols = [c for c in a_preds.columns if c not in ['track_id', 'true_label', 'predicted_label', 'predicted_prob']]

        for col in feature_cols[:10]:  # Limit to first 10 features
            if col in a_preds.columns:
                a_vals = a_preds[col].dropna().values
                b_vals = b_preds[col].dropna().values
                if len(a_vals) > 0 and len(b_vals) > 0:
                    u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                    d = cohen_d(a_vals, b_vals)
                    print(f"{col}: A={np.median(a_vals):.3f}, B={np.median(b_vals):.3f}, d={d:.3f}")
                    results.add('Fate prediction', f'Feature: {col}', 'Mann-Whitney U', u_stat, p_mw,
                                np.median(a_vals), np.median(b_vals), d)

def analyze_spatial(data, results):
    """7. Spatial pattern differences."""
    print("\n" + "="*80)
    print("7. SPATIAL PATTERNS")
    print("="*80)

    a_clust = data['a_clustering_summary.csv']
    b_clust = data['b_clustering_summary.csv']

    if a_clust is not None and b_clust is not None:
        print("\nCondition A clustering:")
        print(a_clust.to_string(index=False))
        print("\nCondition B clustering:")
        print(b_clust.to_string(index=False))

        for metric in a_clust.columns:
            if metric not in ['frame']:
                a_val = a_clust[metric].values[0] if len(a_clust) > 0 else np.nan
                b_val = b_clust[metric].values[0] if len(b_clust) > 0 else np.nan
                if not np.isnan(a_val) and not np.isnan(b_val):
                    print(f"\n{metric}: A={a_val:.3f}, B={b_val:.3f}")
                    results.add('Spatial', metric, 'Descriptive', np.nan, np.nan, a_val, b_val,
                                notes="Single-value comparison")

    # Spatial gradient
    a_grad = data['a_spatial_gradient.csv']
    b_grad = data['b_spatial_gradient.csv']

    if a_grad is not None and b_grad is not None:
        print("\n--- Spatial gradient (position vs fate) ---")
        for metric in ['last_y', 'last_x']:
            if metric in a_grad.columns:
                a_vals = a_grad[metric].dropna().values
                b_vals = b_grad[metric].dropna().values
                if len(a_vals) > 0 and len(b_vals) > 0:
                    u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                    d = cohen_d(a_vals, b_vals)
                    print(f"{metric}: A={np.median(a_vals):.1f}, B={np.median(b_vals):.1f}, d={d:.3f}")
                    results.add('Spatial', metric, 'Mann-Whitney U', u_stat, p_mw,
                                np.median(a_vals), np.median(b_vals), d)

def analyze_nucleus_persistence(data, results):
    """8. Nucleus persistence analysis."""
    print("\n" + "="*80)
    print("8. NUCLEUS PERSISTENCE")
    print("="*80)

    a_nuc = data['a_nucleus_persistence.csv']
    b_nuc = data['b_nucleus_persistence.csv']

    if a_nuc is not None and b_nuc is not None:
        print("\n--- Phase vs fluorescence cell counts ---")

        # Phase counts
        a_phase = a_nuc['phase_cells'].values
        b_phase = b_nuc['phase_cells'].values
        u_stat, p_mw = mannwhitneyu(a_phase, b_phase, alternative='two-sided')
        d = cohen_d(a_phase, b_phase)
        print(f"Phase cell count: A={np.mean(a_phase):.1f}, B={np.mean(b_phase):.1f}, d={d:.3f}")
        results.add('Nucleus', 'Phase cell count', 'Mann-Whitney U', u_stat, p_mw,
                    np.mean(a_phase), np.mean(b_phase), d)

        # Fluorescence counts
        a_fluor = a_nuc['fluor_nuclei'].values
        b_fluor = b_nuc['fluor_nuclei'].values
        u_stat, p_mw = mannwhitneyu(a_fluor, b_fluor, alternative='two-sided')
        d = cohen_d(a_fluor, b_fluor)
        print(f"Fluor nucleus count: A={np.mean(a_fluor):.1f}, B={np.mean(b_fluor):.1f}, d={d:.3f}")
        results.add('Nucleus', 'Fluor nucleus count', 'Mann-Whitney U', u_stat, p_mw,
                    np.mean(a_fluor), np.mean(b_fluor), d)

        # Count difference (phase - fluor)
        a_diff = a_nuc['difference'].values
        b_diff = b_nuc['difference'].values
        u_stat, p_mw = mannwhitneyu(a_diff, b_diff, alternative='two-sided')
        d = cohen_d(a_diff, b_diff)
        print(f"Count difference (phase-fluor): A={np.mean(a_diff):.1f}, B={np.mean(b_diff):.1f}, d={d:.3f}")
        results.add('Nucleus', 'Count difference', 'Mann-Whitney U', u_stat, p_mw,
                    np.mean(a_diff), np.mean(b_diff), d)

    a_nuc_summary = data['a_nucleus_persistence_summary.csv']
    b_nuc_summary = data['b_nucleus_persistence_summary.csv']

    if a_nuc_summary is not None and b_nuc_summary is not None:
        print("\n--- Nucleus persistence summary ---")
        print("Condition A:")
        print(a_nuc_summary.to_string(index=False))
        print("\nCondition B:")
        print(b_nuc_summary.to_string(index=False))

def analyze_changepoints(data, results):
    """9. Changepoint analysis."""
    print("\n" + "="*80)
    print("9. CHANGEPOINT ANALYSIS")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']

    if 'changepoint_frame' in a_tracks.columns:
        # Fraction with detected changepoint
        a_has_cp = (~a_tracks['changepoint_frame'].isna()).sum()
        b_has_cp = (~b_tracks['changepoint_frame'].isna()).sum()
        a_frac = a_has_cp / len(a_tracks)
        b_frac = b_has_cp / len(b_tracks)

        print(f"\nTracks with changepoint: A={a_has_cp}/{len(a_tracks)} ({a_frac:.2%}), B={b_has_cp}/{len(b_tracks)} ({b_frac:.2%})")

        # Only run chi-squared if there's variation
        if a_has_cp < len(a_tracks) and b_has_cp < len(b_tracks):
            contingency = np.array([
                [a_has_cp, len(a_tracks) - a_has_cp],
                [b_has_cp, len(b_tracks) - b_has_cp]
            ])
            chi2, p_chi = chi2_contingency(contingency)[:2]
            results.add('Changepoint', 'Has changepoint', 'Chi-squared', chi2, p_chi, a_frac, b_frac)
        else:
            print("  (All tracks have changepoints, no test performed)")

        # Changepoint frame (for tracks that have one)
        a_cp_frame = a_tracks['changepoint_frame'].dropna().values
        b_cp_frame = b_tracks['changepoint_frame'].dropna().values
        if len(a_cp_frame) > 0 and len(b_cp_frame) > 0:
            u_stat, p_mw = mannwhitneyu(a_cp_frame, b_cp_frame, alternative='two-sided')
            d = cohen_d(a_cp_frame, b_cp_frame)
            print(f"Changepoint frame: A={np.median(a_cp_frame):.1f}, B={np.median(b_cp_frame):.1f}, d={d:.3f}")
            results.add('Changepoint', 'Changepoint frame', 'Mann-Whitney U', u_stat, p_mw,
                        np.median(a_cp_frame), np.median(b_cp_frame), d)

        # Slope before/after/ratio
        for metric in ['slope_before', 'slope_after', 'slope_ratio']:
            if metric in a_tracks.columns:
                a_vals = a_tracks[metric].dropna().values
                b_vals = b_tracks[metric].dropna().values
                if len(a_vals) > 0 and len(b_vals) > 0:
                    u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                    d = cohen_d(a_vals, b_vals)
                    print(f"{metric}: A={np.median(a_vals):.3f}, B={np.median(b_vals):.3f}, d={d:.3f}")
                    results.add('Changepoint', metric, 'Mann-Whitney U', u_stat, p_mw,
                                np.median(a_vals), np.median(b_vals), d)

def analyze_preburst(data, results):
    """10. Pre-burst analysis."""
    print("\n" + "="*80)
    print("10. PRE-BURST ANALYSIS")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']

    # Pre-burst metrics (disappeared tracks only)
    a_disap = a_tracks[a_tracks['disappeared'] == 1]
    b_disap = b_tracks[b_tracks['disappeared'] == 1]

    for metric in ['preburst_slope', 'preburst_spike']:
        if metric in a_tracks.columns:
            a_vals = pd.to_numeric(a_disap[metric], errors='coerce').dropna().values
            b_vals = pd.to_numeric(b_disap[metric], errors='coerce').dropna().values
            if len(a_vals) > 0 and len(b_vals) > 0:
                u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                d = cohen_d(a_vals, b_vals)
                print(f"\n{metric} (disappeared only): A={np.median(a_vals):.3f}, B={np.median(b_vals):.3f}, d={d:.3f}")
                results.add('Pre-burst', metric, 'Mann-Whitney U', u_stat, p_mw,
                            np.median(a_vals), np.median(b_vals), d)

def analyze_correlations(data, results):
    """11. Additional patterns: correlations."""
    print("\n" + "="*80)
    print("11. CORRELATION PATTERNS")
    print("="*80)

    a_tracks = data['a_track_statistics.csv']
    b_tracks = data['b_track_statistics.csv']

    # Key pairs to check
    pairs = [
        ('area_initial', 'lifetime'),
        ('mean_fluor_intensity', 'lifetime'),
        ('growth_rate_px_per_frame', 'area_rel_max'),
        ('mean_speed', 'total_displacement'),
    ]

    print("\nSpearman correlations (key metric pairs):")
    for x, y in pairs:
        if x in a_tracks.columns and y in a_tracks.columns:
            a_data = a_tracks[[x, y]].dropna()
            b_data = b_tracks[[x, y]].dropna()

            if len(a_data) > 0 and len(b_data) > 0:
                a_rho, a_p = stats.spearmanr(a_data[x], a_data[y])
                b_rho, b_p = stats.spearmanr(b_data[x], b_data[y])

                print(f"\n{x} vs {y}:")
                print(f"  A: rho={a_rho:.3f} (p={a_p:.2e})")
                print(f"  B: rho={b_rho:.3f} (p={b_p:.2e})")
                print(f"  Δrho={b_rho - a_rho:.3f}")

                # Note: Fisher z-transformation for correlation comparison
                # (not implemented here for brevity, but worth noting)

def analyze_temporal_trends(data, results):
    """12. Temporal trends in per-frame diagnostics."""
    print("\n" + "="*80)
    print("12. TEMPORAL TRENDS (per-frame diagnostics)")
    print("="*80)

    a_frame = data['a_frame_diagnostics.csv']
    b_frame = data['b_frame_diagnostics.csv']

    if a_frame is not None and b_frame is not None:
        # Test for different slopes over time
        metrics = [c for c in a_frame.columns if c not in ['frame', 'time_minutes']]

        print("\nLinear trends (frame vs metric):")
        for metric in metrics[:10]:  # Limit output
            if metric in a_frame.columns:
                a_vals = a_frame[metric].dropna().values
                b_vals = b_frame[metric].dropna().values

                if len(a_vals) > 0 and len(b_vals) > 0:
                    a_frames = np.arange(len(a_vals))
                    b_frames = np.arange(len(b_vals))

                    a_slope, a_intercept, a_r, a_p, a_se = stats.linregress(a_frames, a_vals)
                    b_slope, b_intercept, b_r, b_p, b_se = stats.linregress(b_frames, b_vals)

                    print(f"\n{metric}:")
                    print(f"  A: slope={a_slope:.4f}, r={a_r:.3f}, p={a_p:.2e}")
                    print(f"  B: slope={b_slope:.4f}, r={b_r:.3f}, p={b_p:.2e}")

def main():
    print("="*80)
    print("BLIND COMPARISON: CONDITION A vs CONDITION B")
    print("="*80)

    # Load data
    data = load_datasets()

    # Initialize results collector
    results = ComparisonResults()

    # Run all analyses
    analyze_population(data, results)
    analyze_morphology(data, results)
    analyze_fluorescence(data, results)
    analyze_kinetics(data, results)
    analyze_swelling(data, results)
    analyze_fate_prediction(data, results)
    analyze_spatial(data, results)
    analyze_nucleus_persistence(data, results)
    analyze_changepoints(data, results)
    analyze_preburst(data, results)
    analyze_correlations(data, results)
    analyze_temporal_trends(data, results)

    # Correct for multiple testing and report
    print("\n" + "="*80)
    print("MULTIPLE TESTING CORRECTION (Benjamini-Hochberg FDR)")
    print("="*80)

    df_results = results.correct_and_report(alpha=0.05)

    # Save full results
    output_path = "/Users/vvetshte/Projects/temp/experiments_image/results/blind_comparison_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    # Report significant findings
    sig = df_results[df_results['significant']]
    print(f"\n{len(sig)} / {len(df_results)} tests significant after FDR correction (q < 0.05)")

    print("\n" + "="*80)
    print("SIGNIFICANT DIFFERENCES (q < 0.05)")
    print("="*80)

    for cat in sig['category'].unique():
        cat_sig = sig[sig['category'] == cat]
        print(f"\n{cat.upper()} ({len(cat_sig)} significant):")
        print("-" * 80)

        for _, row in cat_sig.iterrows():
            effect_str = f", d={row['effect_size']:.3f}" if not pd.isna(row['effect_size']) else ""
            print(f"  {row['metric']:40s} | A={row['a_value']:.3f}, B={row['b_value']:.3f} | q={row['p_adjusted']:.2e}{effect_str}")
            if row['notes']:
                print(f"    Note: {row['notes']}")

    # Summary statistics
    print("\n" + "="*80)
    print("EFFECT SIZE SUMMARY (all significant tests)")
    print("="*80)

    sig_with_effect = sig.dropna(subset=['effect_size'])
    if len(sig_with_effect) > 0:
        for size in ['large', 'medium', 'small', 'negligible']:
            count = sum(sig_with_effect['effect_size'].apply(lambda d: effect_size_interpretation(d) == size))
            print(f"{size.capitalize():12s}: {count}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
