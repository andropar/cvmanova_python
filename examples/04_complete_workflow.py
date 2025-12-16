"""
Complete workflow example: From data loading to publication-ready results.

This example demonstrates a full analysis pipeline:
1. Load data from SPM
2. Run searchlight analysis
3. Run complementary region analysis
4. Statistical testing with permutations
5. Visualization and export
6. Generate publication-ready figures
"""

from pathlib import Path
import numpy as np
from cvmanova import (
    SearchlightCvManova,
    RegionCvManova,
    SPMLoader,
    SearchlightConfig,
    RegionConfig,
    AnalysisConfig,
    ContrastSpec,
)

# ============================================================================
# PART 1: Configuration
# ============================================================================

# Paths
spm_dir = Path("/path/to/your/spm/directory")
output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)

# Analysis parameters
SEARCHLIGHT_RADIUS = 3.0  # mm
N_JOBS = -1  # Use all cores
PERMUTATIONS = 5000  # For significance testing
REGULARIZATION = 0.0  # No regularization (recommended)

# ROI masks
roi_paths = [
    Path("/path/to/V1_left.nii.gz"),
    Path("/path/to/V1_right.nii.gz"),
    Path("/path/to/FFA_left.nii.gz"),
    Path("/path/to/FFA_right.nii.gz"),
]
roi_names = ["V1_L", "V1_R", "FFA_L", "FFA_R"]

# Experimental design: 2×2 factorial
# Factor 1: Face vs House (2 levels)
# Factor 2: Scrambled vs Intact (2 levels)
factors = ["Stimulus", "Scrambling"]
levels = [2, 2]

print("=" * 80)
print("cvManova Analysis Pipeline")
print("=" * 80)
print(f"SPM directory: {spm_dir}")
print(f"Output directory: {output_dir}")
print(f"Searchlight radius: {SEARCHLIGHT_RADIUS} mm")
print(f"Permutations: {PERMUTATIONS}")
print(f"ROIs: {', '.join(roi_names)}")
print(f"Experimental design: {factors[0]} ({levels[0]}) × {factors[1]} ({levels[1]})")
print("=" * 80)

# ============================================================================
# PART 2: Load Data
# ============================================================================

print("\n[1/6] Loading data from SPM.mat...")

loader = SPMLoader(
    spm_dir,
    whiten=True,  # Apply whitening (recommended)
    high_pass_filter=True,  # Apply high-pass filtering
)

data, design = loader.load()

print(f"✓ Loaded {data.n_sessions} sessions")
print(f"  - Voxels per session: {data.n_voxels}")
print(f"  - Scans per session: {data.sessions[0].shape[0]}")
print(f"  - Regressors per session: {design.matrices[0].shape[1]}")
print(f"  - Degrees of freedom: {data.degrees_of_freedom}")

# ============================================================================
# PART 3: Define Contrasts
# ============================================================================

print("\n[2/6] Generating contrasts...")

# Auto-generate all contrasts (main effects + interactions)
contrast_spec = ContrastSpec(factors=factors, levels=levels, effects="all")

# Get contrast matrices and names
Cs, contrast_names = contrast_spec.to_matrices()

print(f"✓ Generated {len(Cs)} contrasts:")
for i, name in enumerate(contrast_names):
    print(f"  {i+1}. {name}")

# ============================================================================
# PART 4: Searchlight Analysis
# ============================================================================

print("\n[3/6] Running whole-brain searchlight analysis...")

# Configure searchlight
sl_config = SearchlightConfig(
    radius=SEARCHLIGHT_RADIUS,
    n_jobs=N_JOBS,
    show_progress=True,
    checkpoint_dir=output_dir / "checkpoints",
    checkpoint_name="searchlight",
)

# Configure analysis
analysis_config = AnalysisConfig(
    regularization=REGULARIZATION,
    permute=True,  # Enable permutation testing
    max_permutations=PERMUTATIONS,
    verbose=1,
)

# Create estimator and run
sl_estimator = SearchlightCvManova(
    searchlight_config=sl_config, contrasts=Cs, analysis_config=analysis_config
)

sl_result = sl_estimator.fit_score(data, design)

print(f"✓ Searchlight analysis complete")
print(f"  - Result shape: {sl_result.discriminability.shape}")
print(f"  - Mean searchlight size: {np.mean(sl_result.n_voxels):.1f} voxels")

# Save searchlight results
print("\n  Saving searchlight maps...")
for contrast_name in contrast_names:
    nifti_path = output_dir / f"searchlight_{contrast_name}.nii.gz"
    sl_result.to_nifti(contrast_name, nifti_path)
    print(f"  ✓ {nifti_path.name}")

# Find and save peaks
print("\n  Finding peaks...")
for contrast_name in contrast_names:
    peaks = sl_result.get_peaks(contrast_name, n=10, min_distance=8.0)
    peak_file = output_dir / f"peaks_{contrast_name}.txt"

    with open(peak_file, "w") as f:
        f.write(f"Top 10 peaks for {contrast_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Rank':<6} {'X':>6} {'Y':>6} {'Z':>6} {'D':>10}\n")
        f.write("-" * 60 + "\n")

        for i, peak in enumerate(peaks, 1):
            x, y, z = peak["mni_coords"]
            D = peak["value"]
            f.write(f"{i:<6} {x:>6.1f} {y:>6.1f} {z:>6.1f} {D:>10.4f}\n")

    print(f"  ✓ {peak_file.name}")

# ============================================================================
# PART 5: Region Analysis
# ============================================================================

print("\n[4/6] Running region-based analysis...")

# Configure regions
region_config = RegionConfig(
    regions=roi_paths, region_names=roi_names, min_voxels=10, allow_overlap=False
)

# Create estimator and run
roi_estimator = RegionCvManova(
    region_config=region_config, contrasts=Cs, analysis_config=analysis_config
)

roi_result = roi_estimator.fit_score(data, design)

print(f"✓ Region analysis complete")

# Export to DataFrame
roi_df = roi_result.to_dataframe()
roi_csv = output_dir / "region_results.csv"
roi_df.to_csv(roi_csv, index=False)
print(f"  ✓ Saved to {roi_csv.name}")

# Print summary
print("\n  Region summary:")
for region_name in roi_names:
    n_voxels = roi_df[roi_df["region"] == region_name]["n_voxels"].iloc[0]
    print(f"  - {region_name}: {n_voxels} voxels")

# ============================================================================
# PART 6: Statistical Testing
# ============================================================================

print("\n[5/6] Performing permutation-based significance testing...")

# For each region and contrast, compute p-values
results_table = []

for region_name in roi_names:
    for contrast_name in contrast_names:
        # Get discriminability values
        D_values = roi_result.get_region_contrast(region_name, contrast_name)

        # D_values shape: (n_permutations,)
        observed_D = D_values[0]  # First is observed
        null_dist = D_values[1:]  # Rest are null distribution

        # Compute p-value (one-tailed: observed > null)
        p_value = np.mean(null_dist >= observed_D)

        # Significance at different thresholds
        sig_05 = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        results_table.append(
            {
                "region": region_name,
                "contrast": contrast_name,
                "D_observed": observed_D,
                "D_null_mean": np.mean(null_dist),
                "D_null_std": np.std(null_dist),
                "p_value": p_value,
                "significant": sig_05,
            }
        )

# Save results table
import pandas as pd

stats_df = pd.DataFrame(results_table)
stats_csv = output_dir / "statistical_results.csv"
stats_df.to_csv(stats_csv, index=False)
print(f"✓ Saved to {stats_csv.name}")

# Print summary
print("\n  Significant effects (p < 0.05):")
sig_results = stats_df[stats_df["p_value"] < 0.05].sort_values("p_value")

if len(sig_results) > 0:
    for _, row in sig_results.iterrows():
        print(
            f"  - {row['region']:8s} {row['contrast']:20s}: "
            f"D = {row['D_observed']:6.3f}, p = {row['p_value']:.4f} {row['significant']}"
        )
else:
    print("  None found")

# ============================================================================
# PART 7: Visualization
# ============================================================================

print("\n[6/6] Generating visualizations...")

try:
    from nilearn import plotting
    import matplotlib.pyplot as plt

    # Glass brain plots for each contrast
    for contrast_name in contrast_names:
        nifti_path = output_dir / f"searchlight_{contrast_name}.nii.gz"

        fig = plotting.plot_glass_brain(
            str(nifti_path),
            threshold=0.1,
            colorbar=True,
            title=f"{contrast_name} Discriminability",
            plot_abs=False,
        )
        fig.savefig(
            output_dir / f"glass_brain_{contrast_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print("  ✓ Glass brain plots saved")

    # Stat maps with anatomical underlay
    for contrast_name in contrast_names[:2]:  # Only first 2 to save time
        nifti_path = output_dir / f"searchlight_{contrast_name}.nii.gz"

        fig = plotting.plot_stat_map(
            str(nifti_path),
            threshold=0.1,
            display_mode="z",
            cut_coords=10,
            title=contrast_name,
            colorbar=True,
        )
        fig.savefig(
            output_dir / f"stat_map_{contrast_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print("  ✓ Statistical maps saved")

    # Region bar plot
    import seaborn as sns

    fig, axes = plt.subplots(
        1, len(contrast_names), figsize=(4 * len(contrast_names), 5)
    )

    if len(contrast_names) == 1:
        axes = [axes]

    for ax, contrast_name in zip(axes, contrast_names):
        # Get mean D for each region
        region_means = []
        region_stds = []
        region_labels = []

        for region_name in roi_names:
            D_values = roi_result.get_region_contrast(region_name, contrast_name)
            region_means.append(D_values[0])  # Observed value
            region_stds.append(np.std(D_values[1:]))  # Null distribution std
            region_labels.append(region_name)

        # Bar plot with error bars
        x = np.arange(len(region_labels))
        ax.bar(x, region_means, yerr=region_stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(region_labels, rotation=45, ha="right")
        ax.set_ylabel("Discriminability (D)")
        ax.set_title(contrast_name)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "region_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  ✓ Region bar plot saved")

except ImportError:
    print("  ! Skipping visualization (install nilearn, matplotlib, seaborn)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print(f"\nResults saved to: {output_dir}")
print("\nGenerated files:")
print("  Searchlight maps:")
for contrast_name in contrast_names:
    print(f"    - searchlight_{contrast_name}.nii.gz")
print("  Peak coordinates:")
for contrast_name in contrast_names:
    print(f"    - peaks_{contrast_name}.txt")
print("  Region results:")
print("    - region_results.csv")
print("    - statistical_results.csv")
print("  Visualizations:")
print("    - glass_brain_*.png")
print("    - stat_map_*.png")
print("    - region_barplot.png")

print("\nNext steps:")
print("  1. Load results in FSLeyes, AFNI, or other viewer")
print("  2. Threshold maps based on permutation p-values")
print("  3. Extract effect sizes from significant peaks/regions")
print("  4. Create publication figures from visualization outputs")

print("\n" + "=" * 80)
