"""
Basic searchlight analysis with the modern cvManova API.

This example demonstrates:
1. Loading data using SPMLoader
2. Configuring a searchlight analysis
3. Defining contrasts
4. Running the analysis with SearchlightCvManova
5. Visualizing and saving results
"""

from pathlib import Path
from cvmanova import (
    SearchlightCvManova,
    SPMLoader,
    SearchlightConfig,
    ContrastSpec,
    AnalysisConfig,
)

# ============================================================================
# 1. Load Data
# ============================================================================

# Load data from SPM.mat file
spm_dir = Path("/path/to/your/spm/directory")
loader = SPMLoader(
    spm_dir,
    whiten=True,  # Apply whitening
    high_pass_filter=True,  # Apply high-pass filtering
)

data, design = loader.load()

print(f"Loaded {data.n_sessions} sessions")
print(f"Each session has {data.n_voxels} voxels")
print(f"Design matrix: {design.matrices[0].shape}")


# ============================================================================
# 2. Configure Searchlight
# ============================================================================

searchlight_config = SearchlightConfig(
    radius=3.0,  # Searchlight radius in mm
    n_jobs=-1,  # Use all available CPU cores
    show_progress=True,  # Show progress bar
    checkpoint_dir=Path("./checkpoints"),  # Save checkpoints for resuming
    checkpoint_name="my_searchlight",
)


# ============================================================================
# 3. Define Contrasts
# ============================================================================

# Option A: Auto-generate from factorial design
contrasts = ContrastSpec(
    factors=["Face", "House"],  # Factor names
    levels=[2, 2],  # Levels per factor
    effects="all",  # Include main effects and interactions
)

# Option B: Manually specify contrast matrices
# import numpy as np
# contrasts = [
#     np.array([[0, 1, 0, 0]]),  # Face effect
#     np.array([[0, 0, 1, 0]]),  # House effect
#     np.array([[0, 0, 0, 1]]),  # Interaction
# ]


# ============================================================================
# 4. Run Analysis
# ============================================================================

# Configure analysis parameters
analysis_config = AnalysisConfig(
    regularization=0.0,  # Ridge regularization (lambda)
    permute=True,  # Use sign permutation for significance testing
    max_permutations=5000,  # Maximum number of permutations
    verbose=1,  # Print progress messages
)

# Create estimator
estimator = SearchlightCvManova(
    searchlight_config=searchlight_config,
    contrasts=contrasts,
    analysis_config=analysis_config,
)

# Fit and score (combined operation)
result = estimator.fit_score(data, design)

# Or do it in two steps:
# estimator.fit(data, design)
# result = estimator.score()

print(f"Analysis complete!")
print(f"Result shape: {result.discriminability.shape}")


# ============================================================================
# 5. Visualize and Save Results
# ============================================================================

# Save all results to directory
output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)

# Save each contrast as a NIfTI file
for contrast_name in result.contrast_names:
    nifti_path = output_dir / f"{contrast_name}_discriminability.nii.gz"
    result.to_nifti(contrast_name, nifti_path)
    print(f"Saved {contrast_name} to {nifti_path}")

# Get peak coordinates for a specific contrast
peaks = result.get_peaks("Face", n=10, min_distance=8.0)
print("\nTop 10 peaks for Face contrast:")
for i, peak in enumerate(peaks, 1):
    print(f"  {i}. MNI coords: {peak['mni_coords']}, D = {peak['value']:.4f}")

# Export results to DataFrame (useful for ROI analysis or further processing)
df = result.to_dataframe()
df.to_csv(output_dir / "searchlight_results.csv", index=False)
print(f"\nSaved summary to CSV")

# Plot results (requires matplotlib and nilearn)
try:
    from nilearn import plotting

    # Glass brain plot
    fig = result.plot_glass_brain(
        "Face", threshold=0.1, colorbar=True, title="Face Discriminability"
    )
    fig.savefig(output_dir / "face_glass_brain.png", dpi=300)

    # Statistical map overlay
    fig = result.plot_stat_map(
        "Face", threshold=0.1, cut_coords=(-40, -20, 0, 20, 40), title="Face vs House"
    )
    fig.savefig(output_dir / "face_stat_map.png", dpi=300)

    print("Saved visualization plots")
except ImportError:
    print("Install nilearn for visualization: pip install nilearn matplotlib")


# ============================================================================
# 6. Access Raw Results
# ============================================================================

# Get discriminability for specific contrast
face_D = result.get_contrast("Face")
print(f"\nFace discriminability shape: {face_D.shape}")

# Get number of voxels in each searchlight
n_voxels = result.n_voxels
print(f"Searchlight sizes shape: {n_voxels.shape}")

# Access permutation results (if permute=True)
if result.discriminability.shape[1] > 1:
    print(
        f"Permutation results available: "
        f"{result.discriminability.shape[1]} permutations"
    )
