"""
Integration with nilearn for advanced preprocessing.

This example demonstrates:
1. Using NilearnMaskerLoader with custom preprocessing
2. Leveraging nilearn's advanced features (smoothing, filtering, standardization)
3. Combining cvManova with nilearn's visualization tools
"""

from pathlib import Path
import numpy as np

# ============================================================================
# 1. Setup Nilearn Masker with Preprocessing
# ============================================================================

try:
    from nilearn.maskers import NiftiMasker
    from cvmanova import (
        SearchlightCvManova,
        NilearnMaskerLoader,
        SearchlightConfig,
        ContrastSpec,
        AnalysisConfig,
    )

    # Configure nilearn masker with sophisticated preprocessing
    masker = NiftiMasker(
        mask_img="/path/to/mask.nii.gz",
        # Spatial smoothing
        smoothing_fwhm=6.0,  # 6mm FWHM Gaussian smoothing
        # Temporal filtering
        high_pass=0.01,  # High-pass filter cutoff (Hz)
        low_pass=0.1,  # Low-pass filter cutoff (Hz)
        t_r=2.0,  # Repetition time in seconds
        # Standardization
        standardize=True,  # Z-score each voxel time series
        detrend=True,  # Remove linear trends
        # Memory and computation
        memory="/tmp/nilearn_cache",  # Cache computations
        memory_level=1,
        verbose=1,
    )

    # Define input files
    bold_files = [
        Path("/path/to/sub-01_task-faces_run-1_bold.nii.gz"),
        Path("/path/to/sub-01_task-faces_run-2_bold.nii.gz"),
    ]

    # Create design matrices
    # In a real analysis, you might generate these from events.tsv files
    n_scans = 200

    # Simple block design: Face, House, Face, House, ...
    design_block = np.zeros((n_scans, 3))
    design_block[:, 0] = 1  # Intercept
    block_length = 20
    for i in range(0, n_scans, 2 * block_length):
        design_block[i : i + block_length, 1] = 1  # Face blocks
        design_block[
            i + block_length : i + 2 * block_length, 2
        ] = 1  # House blocks

    design_matrices = [design_block, design_block]

    # ============================================================================
    # 2. Load Data with Nilearn Preprocessing
    # ============================================================================

    loader = NilearnMaskerLoader(
        bold_files=bold_files, masker=masker, design_matrices=design_matrices
    )

    data, design = loader.load()

    print(f"Loaded and preprocessed {data.n_sessions} sessions")
    print(f"Data shape: {data.sessions[0].shape}")
    print("Applied preprocessing:")
    print("  - 6mm smoothing")
    print("  - Temporal filtering (0.01-0.1 Hz)")
    print("  - Standardization and detrending")

    # ============================================================================
    # 3. Run Searchlight Analysis
    # ============================================================================

    # Define contrasts
    contrasts = ContrastSpec(
        factors=["Face", "House"], levels=[2, 2], effects="main"
    )

    # Configure and run
    estimator = SearchlightCvManova(
        searchlight_config=SearchlightConfig(radius=3.0, n_jobs=-1),
        contrasts=contrasts,
        analysis_config=AnalysisConfig(permute=True, verbose=1),
    )

    result = estimator.fit_score(data, design)

    # ============================================================================
    # 4. Visualize with Nilearn
    # ============================================================================

    from nilearn import plotting

    # Save results as NIfTI
    face_nifti = Path("./face_discriminability.nii.gz")
    result.to_nifti("Face", face_nifti)

    # Glass brain plot
    plotting.plot_glass_brain(
        str(face_nifti),
        threshold=0.1,
        colorbar=True,
        title="Face Discriminability",
        output_file="glass_brain.png",
    )

    # Statistical map on anatomical background
    plotting.plot_stat_map(
        str(face_nifti),
        bg_img="/path/to/T1.nii.gz",  # Anatomical reference
        threshold=0.1,
        display_mode="z",
        cut_coords=10,
        title="Face vs House",
        output_file="stat_map.png",
    )

    # Interactive view (opens in browser)
    view = plotting.view_img(
        str(face_nifti), threshold=0.1, cmap="cold_hot", symmetric_cmap=False
    )
    view.save_as_html("interactive_view.html")

    print("\nSaved nilearn visualizations:")
    print("  - glass_brain.png")
    print("  - stat_map.png")
    print("  - interactive_view.html")

    # ============================================================================
    # 5. Compare with Different Preprocessing
    # ============================================================================

    # Run again with minimal preprocessing for comparison
    masker_minimal = NiftiMasker(
        mask_img="/path/to/mask.nii.gz",
        smoothing_fwhm=None,  # No smoothing
        standardize=False,  # No standardization
        detrend=False,  # No detrending
        high_pass=None,  # No filtering
        low_pass=None,
        t_r=2.0,
    )

    loader_minimal = NilearnMaskerLoader(
        bold_files=bold_files,
        masker=masker_minimal,
        design_matrices=design_matrices,
    )

    data_minimal, _ = loader_minimal.load()

    estimator_minimal = SearchlightCvManova(
        searchlight_config=SearchlightConfig(radius=3.0, n_jobs=-1),
        contrasts=contrasts,
        analysis_config=AnalysisConfig(permute=False, verbose=0),
    )

    result_minimal = estimator_minimal.fit_score(data_minimal, design)

    # Compare results
    print("\n=== Preprocessing Comparison ===")
    print(
        f"With preprocessing - Mean D: "
        f"{np.mean(result.get_contrast('Face')):.4f}"
    )
    print(
        f"Without preprocessing - Mean D: "
        f"{np.mean(result_minimal.get_contrast('Face')):.4f}"
    )

    # Correlation between maps
    D_preprocessed = result.get_contrast("Face").flatten()
    D_minimal = result_minimal.get_contrast("Face").flatten()

    # Remove NaN values
    valid = ~(np.isnan(D_preprocessed) | np.isnan(D_minimal))
    correlation = np.corrcoef(D_preprocessed[valid], D_minimal[valid])[0, 1]

    print(f"Correlation between maps: {correlation:.4f}")

except ImportError as e:
    print(f"This example requires nilearn: pip install nilearn")
    print(f"Error: {e}")


# ============================================================================
# 6. Advanced: Atlas-based Region Analysis
# ============================================================================

try:
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    from cvmanova import RegionCvManova, RegionConfig

    # Fetch Harvard-Oxford atlas
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = atlas.maps
    labels = atlas.labels

    # Use atlas masker to extract regional time series
    atlas_masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        memory="/tmp/nilearn_cache",
        verbose=1,
    )

    # Extract time series for each region
    # Note: This is different from searchlight - we're getting one value per region
    region_timeseries = []
    for bold_file in bold_files:
        ts = atlas_masker.fit_transform(str(bold_file))
        region_timeseries.append(ts)

    print(f"\nExtracted time series for {len(labels)} atlas regions")
    print(f"Time series shape: {region_timeseries[0].shape}")

    # For a true region-based cvManova, you'd still load voxel-level data
    # and use the atlas to define regions, but this shows integration possibilities

except ImportError:
    print("\nAtlas analysis requires nilearn: pip install nilearn")
