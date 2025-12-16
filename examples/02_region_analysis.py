"""
Region-based (ROI) analysis with the modern cvManova API.

This example demonstrates:
1. Loading data from NIfTI files
2. Defining regions of interest (ROIs)
3. Running region-based cvManova analysis
4. Exporting results to pandas DataFrame
"""

from pathlib import Path
import numpy as np
from cvmanova import (
    RegionCvManova,
    NiftiLoader,
    RegionConfig,
    ContrastSpec,
    AnalysisConfig,
)

# ============================================================================
# 1. Load Data from NIfTI Files
# ============================================================================

# Define input files
bold_files = [
    Path("/path/to/run1.nii.gz"),
    Path("/path/to/run2.nii.gz"),
]

mask_file = Path("/path/to/mask.nii.gz")

# Create design matrices manually
# Each matrix is (n_scans × n_regressors)
n_scans = 200
X1 = np.column_stack(
    [
        np.ones(n_scans),  # Intercept
        np.tile([1, -1, 1, -1], n_scans // 4)[
            :n_scans
        ],  # Condition A vs B
        np.tile([1, 1, -1, -1], n_scans // 4)[:n_scans],  # Condition C vs D
    ]
)

X2 = X1.copy()  # Same design for second run

design_matrices = [X1, X2]

# Load data
loader = NiftiLoader(
    bold_files=bold_files,
    mask_file=mask_file,
    design_matrices=design_matrices,
    tr=2.0,  # Repetition time in seconds
    preprocess=True,  # Apply mean-centering
)

data, design = loader.load()

print(f"Loaded {data.n_sessions} sessions with {data.n_voxels} voxels")


# ============================================================================
# 2. Define Regions of Interest
# ============================================================================

# Option A: Load ROIs from NIfTI files
region_config = RegionConfig(
    regions=[
        Path("/path/to/V1_left.nii.gz"),
        Path("/path/to/V1_right.nii.gz"),
        Path("/path/to/FFA_left.nii.gz"),
        Path("/path/to/FFA_right.nii.gz"),
    ],
    region_names=["V1_L", "V1_R", "FFA_L", "FFA_R"],
    min_voxels=10,  # Minimum voxels per region
    allow_overlap=False,  # Ensure regions don't overlap
)

# Option B: Use pre-loaded masks (numpy arrays)
# Assuming you have 3D boolean masks:
# v1_left = np.load('v1_left_mask.npy')
# v1_right = np.load('v1_right_mask.npy')
#
# region_config = RegionConfig(
#     regions=[v1_left, v1_right],
#     region_names=['V1_L', 'V1_R'],
#     min_voxels=10,
# )


# ============================================================================
# 3. Define Contrasts
# ============================================================================

# Auto-generate contrasts for 2×2 factorial design
contrasts = ContrastSpec(
    factors=["Condition_AB", "Condition_CD"],
    levels=[2, 2],
    effects="main",  # Only main effects, not interactions
)

# Or specify manually:
# contrasts = [
#     np.array([[0, 1, 0]]),  # Condition A vs B
#     np.array([[0, 0, 1]]),  # Condition C vs D
# ]
# contrast_names = ["A_vs_B", "C_vs_D"]


# ============================================================================
# 4. Run Region Analysis
# ============================================================================

analysis_config = AnalysisConfig(
    regularization=0.01,  # Small ridge regularization
    permute=True,
    max_permutations=5000,
    verbose=1,
)

# Create estimator
estimator = RegionCvManova(
    region_config=region_config,
    contrasts=contrasts,
    analysis_config=analysis_config,
)

# Fit and score
result = estimator.fit_score(data, design)

print(f"\nRegion analysis complete!")
print(f"Analyzed {region_config.n_regions} regions")


# ============================================================================
# 5. Export and Analyze Results
# ============================================================================

# Export to pandas DataFrame
df = result.to_dataframe()
print("\nResults DataFrame:")
print(df.head(10))

# Save to CSV
output_path = Path("./region_results.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")

# Access results for specific region and contrast
for region_name in region_config.region_names:
    for contrast_name in result.contrast_names:
        D = result.get_region_contrast(region_name, contrast_name)
        print(
            f"  {region_name} - {contrast_name}: "
            f"mean D = {np.mean(D):.4f}, std = {np.std(D):.4f}"
        )


# ============================================================================
# 6. Visualize Results
# ============================================================================

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot discriminability across regions for each contrast
    fig, axes = plt.subplots(1, len(result.contrast_names), figsize=(12, 4))

    if len(result.contrast_names) == 1:
        axes = [axes]

    for ax, contrast_name in zip(axes, result.contrast_names):
        # Get mean discriminability per region
        region_means = []
        region_labels = []

        for region_name in region_config.region_names:
            D = result.get_region_contrast(region_name, contrast_name)
            region_means.append(np.mean(D))
            region_labels.append(region_name)

        # Bar plot
        ax.bar(range(len(region_means)), region_means)
        ax.set_xticks(range(len(region_labels)))
        ax.set_xticklabels(region_labels, rotation=45, ha="right")
        ax.set_ylabel("Discriminability")
        ax.set_title(contrast_name)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("region_discriminability.png", dpi=300)
    print("\nSaved bar plot to region_discriminability.png")

    # Heatmap of all regions × contrasts
    if len(result.contrast_names) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create matrix of mean discriminability
        D_matrix = np.zeros((region_config.n_regions, len(result.contrast_names)))

        for i, region_name in enumerate(region_config.region_names):
            for j, contrast_name in enumerate(result.contrast_names):
                D = result.get_region_contrast(region_name, contrast_name)
                D_matrix[i, j] = np.mean(D)

        # Heatmap
        sns.heatmap(
            D_matrix,
            xticklabels=result.contrast_names,
            yticklabels=region_config.region_names,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
        )
        ax.set_title("Discriminability by Region and Contrast")
        plt.tight_layout()
        plt.savefig("region_heatmap.png", dpi=300)
        print("Saved heatmap to region_heatmap.png")

except ImportError:
    print("Install matplotlib and seaborn for plots: pip install matplotlib seaborn")


# ============================================================================
# 7. Statistical Testing (if permutations used)
# ============================================================================

if result.discriminability.shape[2] > 1:  # Has permutations
    print("\n=== Permutation-based significance ===")

    for region_name in region_config.region_names:
        for contrast_name in result.contrast_names:
            D = result.get_region_contrast(region_name, contrast_name)

            # D shape: (n_permutations,)
            observed = D[0]  # First permutation is observed
            null_dist = D[1:]  # Rest are null distribution

            # One-tailed p-value
            p_value = np.mean(null_dist >= observed)

            print(
                f"  {region_name} - {contrast_name}: "
                f"D = {observed:.4f}, p = {p_value:.4f}"
            )
