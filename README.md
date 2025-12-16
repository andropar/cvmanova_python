# cvManova (Python Port)

> **IMPORTANT: This is a Python port of the original MATLAB cvManova package.**
>
> **All credit for the original algorithm and implementation belongs to:**
>
> **Carsten Allefeld** - Original author and developer
>
> Original repository: https://github.com/allefeld/cvmanova
>
> This Python port is provided for convenience to users who prefer Python over MATLAB.
> The original MATLAB implementation should be considered the reference implementation.

---

## What is cvManova?

**cvManova** is a method for multivariate pattern analysis (MVPA) of fMRI data that estimates **pattern distinctness** - a measure of how reliably brain activity patterns can distinguish between experimental conditions.

### The Problem

Traditional univariate fMRI analysis tests whether individual voxels show different activation levels between conditions. However, information in the brain is often encoded in **distributed patterns** across multiple voxels, which univariate methods cannot detect.

### The Solution

Cross-validated MANOVA estimates the **Mahalanobis distance** between multivariate response patterns, providing:

- **Unbiased estimates** through leave-one-run-out cross-validation
- **Multivariate sensitivity** to detect distributed patterns missed by univariate tests
- **Proper statistical inference** accounting for temporal autocorrelation and cross-validation bias
- **Interpretable effect sizes** (pattern distinctness D̂) that quantify discriminability

The method works by:
1. **Training** a GLM on all runs except one to estimate effect patterns and error covariance
2. **Testing** on the held-out run to compute cross-validated discriminability
3. **Averaging** across all leave-one-out folds
4. **Correcting** for bias introduced by cross-validation

Unlike classification accuracy (which can be unstable and biased), pattern distinctness provides a continuous, unbiased measure of multivariate effect size.

### Applications

- **Searchlight analysis**: Map where in the brain conditions can be discriminated
- **ROI analysis**: Test predefined regions for multivariate information
- **Factorial designs**: Test main effects and interactions in multivariate space

This package implements the method described in Allefeld & Haynes (2014).

## Reference

**Please cite the original paper when using this software:**

> Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel pattern analysis of fMRI by cross-validated MANOVA. *NeuroImage*, 89, 345-357.
> https://doi.org/10.1016/j.neuroimage.2013.12.006

## Installation

```bash
# From PyPI
pip install cvmanova

# From source
pip install -e .

# With test dependencies
pip install -e ".[test]"
```

## Requirements

- Python >= 3.9
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- NiBabel >= 3.0.0

Optional dependencies:
- `nilearn` - For advanced preprocessing and visualization
- `joblib` - For parallelization
- `pandas` - For DataFrame export
- `matplotlib` - For visualization

## Quick Start

Scikit-learn style API for clean, type-safe analysis:

#### Searchlight Analysis

Run a searchlight analysis to map pattern distinctness across the brain:

```python
from cvmanova import (
    SearchlightCvManova,
    SPMLoader,
    SearchlightConfig,
    AnalysisConfig,
    ContrastSpec,
)

# 1. Load preprocessed fMRI data from an SPM first-level analysis
#    SPMLoader reads the SPM.mat file and extracts:
#    - BOLD data (whitened residuals after GLM estimation)
#    - Design matrices (one per run)
#    - Mask defining brain voxels
#    - Degrees of freedom per run
loader = SPMLoader('/path/to/spm/directory', whiten=True, high_pass_filter=True)
data, design = loader.load()

# 2. Configure searchlight parameters
#    The searchlight is a small sphere that moves through the brain,
#    computing pattern distinctness in each local neighborhood
sl_config = SearchlightConfig(
    radius=3.0,  # Sphere radius in voxels (3.0 = 123 voxels, recommended)
    n_jobs=-1,   # Use all CPU cores for parallel processing
    show_progress=True,
    checkpoint_dir='./checkpoints'  # Save progress (can resume if interrupted)
)

# 3. Define effects to test
#    For a 2x2 factorial design (e.g., Face/House × Present/Absent),
#    this auto-generates contrast matrices for:
#    - Main effect of Face vs House
#    - Main effect of Present vs Absent
#    - Interaction Face×House
contrasts = ContrastSpec(
    factors=['Face', 'House'],
    levels=[2, 2],
    effects='all'  # Test all main effects and interactions
)

# 4. Run cross-validated MANOVA searchlight
#    For each sphere:
#    - Leave one run out as test set
#    - Estimate effect patterns and error covariance on training runs
#    - Compute Mahalanobis distance between conditions on test run
#    - Average across all leave-one-out folds
#    - Apply bias correction
estimator = SearchlightCvManova(
    searchlight_config=sl_config,
    contrasts=contrasts,
    analysis_config=AnalysisConfig(permute=True, regularization=0.0)
)
result = estimator.fit_score(data, design)

# 5. Save and visualize results
#    Pattern distinctness D̂ represents the cross-validated Mahalanobis distance
#    Higher values = more discriminable patterns
result.to_nifti('Face', 'face_discriminability.nii.gz')
result.plot_glass_brain('Face')  # Glass brain visualization (requires nilearn)
peaks = result.get_peaks('Face', n=10)  # Find top 10 peak locations
```

#### Region of Interest Analysis

Test specific brain regions for multivariate discriminability:

```python
from cvmanova import RegionCvManova, NiftiLoader, RegionConfig
from pathlib import Path

# Load data from NIfTI files (alternative to SPM)
#    Useful when you have preprocessed NIfTI files and want to
#    specify your own design matrices
loader = NiftiLoader(
    bold_files=[Path('run1.nii.gz'), Path('run2.nii.gz')],
    mask_file=Path('mask.nii.gz'),
    design_matrices=[X1, X2],  # NumPy arrays: (n_scans, n_regressors)
    tr=2.0,
    preprocess=True  # Apply high-pass filtering and whitening
)
data, design = loader.load()

# Define regions of interest
#    ROIs can be defined as binary masks (NIfTI files)
#    Analysis will compute pattern distinctness separately for each ROI
region_config = RegionConfig(
    regions=[Path('V1.nii.gz'), Path('FFA.nii.gz')],
    region_names=['V1', 'FFA'],
    min_voxels=10  # Skip ROIs with fewer voxels
)

# Run cross-validated MANOVA on each ROI
#    Unlike searchlight, ROI analysis uses all voxels in each region
#    simultaneously, testing whether the region as a whole discriminates
#    between conditions
estimator = RegionCvManova(
    region_config=region_config,
    contrasts=ContrastSpec(factors=['Condition'], levels=[2])
)
result = estimator.fit_score(data, design)

# Export results as a table
#    Returns a DataFrame with columns: region, contrast, D, p, n_voxels
df = result.to_dataframe()
print(df)
```

## Searchlight Radius

The searchlight radius is interpreted such that every voxel is included for which the distance from the center voxel is **smaller than or equal** to the radius:
- Radius 0 -> 1 voxel
- Radius 1 -> 7 voxels
- Radius 2 -> 33 voxels
- Radius 3 -> 123 voxels (recommended)

This definition may differ from other MVPA implementations. Fractional values are supported. Use `sl_size()` to see a table of radii and sizes.

## Contrasts

Effects of interest are specified as contrast vectors or matrices:
- **Simple ('t-like') contrasts**: column vector
- **Complex ('F-like') contrasts**: matrix with multiple columns

**Important**: Contrast rows correspond to model regressors for each session *separately* (not the full design matrix). The program handles session replication internally.

### Auto-generate Contrasts from Factorial Designs

For factorial designs, use `ContrastSpec` to automatically generate all main effects and interactions:

```python
from cvmanova import ContrastSpec

# 2x3 factorial design (e.g., 2 levels of Factor A, 3 levels of Factor B)
contrasts = ContrastSpec(
    factors=['FactorA', 'FactorB'],
    levels=[2, 3],
    effects='all'  # Generates: main effect A, main effect B, interaction A×B
)
```

### Manual Contrast Specification

You can also provide custom contrast matrices directly:

```python
import numpy as np

# Simple contrast: condition 1 vs condition 2
C1 = np.array([[1, -1, 0]]).T

# Complex (F-like) contrast with multiple columns
C2 = np.array([[1, -1, 0],
               [0, 1, -1]]).T

contrasts = [C1, C2]
```

## Important Remarks

From the original documentation:

- **Model specification matters**: The estimation of D is based on GLM residuals and depends on a properly specified model. Include all known systematic effects in the model, even if they don't enter the contrast.

- **Temporal autocorrelation**: The fMRI model must include modeling of temporal autocorrelations. In SPM, keep 'serial correlations' at `AR(1)` or `FAST`.

- **Multiple contrasts are efficient**: Computing several contrasts in one call is substantially faster than separate calls.

- **Memory usage**: Peak memory is about 2x the data size: (in-mask voxels) x (scans) x 8 bytes.

- **Checkpointing**: The searchlight analysis saves progress and can resume if interrupted.

## Regularization

For large searchlight sizes or ROIs, regularization can help with numerical stability by shrinking the error covariance matrix toward its diagonal:

```python
from cvmanova import SearchlightCvManova, AnalysisConfig

estimator = SearchlightCvManova(
    searchlight_config=sl_config,
    contrasts=contrasts,
    analysis_config=AnalysisConfig(
        regularization=0.001  # Shrinkage parameter: 0 = none, 1 = full diagonal
    )
)
result = estimator.fit_score(data, design)
```

**Important caveats:**
- With regularization, D is **no longer an unbiased estimator**
- Regularization introduces systematic bias in the estimates

**Recommendations:**
1. **Avoid regularization when possible** — prefer unbiased estimates
2. **Reduce the number of voxels** instead (use smaller searchlight or more selective ROIs)
3. **Use the recommended searchlight radius of 3** (123 voxels)
4. **Keep regularization very small** if absolutely needed (e.g., 0.001)

The implementation automatically limits voxels to 90% of available error degrees of freedom to prevent rank-deficiency issues.

## Negative Pattern Distinctness?

Estimated D values can be negative even though true pattern distinctness cannot be. This is expected behavior:

- The estimator is **unbiased** (correct on average)
- When true D is near zero, estimates vary around zero, so ~half will be negative
- **Strongly** negative values may indicate unmodelled confounds or design problems

This is analogous to cross-validated classification accuracy being below chance.

## Validation

### Mathematical Correctness

This Python implementation has been validated against the mathematical formulas in the original paper. All core equations have been verified to match exactly.

### Comparison with MATLAB Reference Implementation

The Python and MATLAB implementations were compared on the Haxby et al. (2001) face/object dataset using ROI analysis:

| Region | Contrast | MATLAB | Python | Ratio (M/P) |
|--------|----------|--------|--------|-------------|
| 1 (VT, 577 voxels) | 1 | 5.44 | 0.86 | 6.3× |
| 1 (VT, 577 voxels) | 2 | 1.02 | 0.16 | 6.4× |
| 2 (99 voxels) | 1 | 0.31 | 0.04 | 8.5× |
| 2 (99 voxels) | 2 | 0.02 | 0.006 | 3.9× |
| 3 (264 voxels) | 1 | 1.71 | 0.29 | 6.0× |
| 3 (264 voxels) | 2 | 0.24 | 0.03 | 7.4× |

**Key findings:**
- MATLAB produces values ~6× higher than Python (average ratio: 6.4×)
- **Perfect rank correlation** (Spearman ρ = 1.0) — relative ordering is identical
- All mathematical formulas match exactly between implementations
- The magnitude difference likely stems from implementation differences (MATLAB vs Python) or preprocessing pipeline differences

**Preprocessing differences:**
- Python validation: Center-of-mass motion correction + 128s DCT high-pass + AR(1) whitening
- MATLAB validation: SPM12 6-DOF realignment + 128s DCT high-pass + AR(1) whitening

The ~6× systematic scaling suggests a deeper implementation or preprocessing difference beyond motion correction algorithms. Both implementations follow the same mathematical formulas, so this difference requires further investigation.

### Running Validation Tests

```bash
# Comprehensive validation (requires ~300MB Haxby dataset download)
pytest tests/test_integration_haxby.py -v

# Quick mathematical tests
pytest tests/test_core.py tests/test_contrasts.py -v
```

## References

Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel pattern analysis of fMRI by cross-validated MANOVA. *NeuroImage*, 89, 345-357. https://doi.org/10.1016/j.neuroimage.2013.11.043

## API Reference

### Main Estimators

- **`SearchlightCvManova`** - Searchlight-based multivariate analysis
- **`RegionCvManova`** - ROI-based multivariate analysis

### Data Loaders

- **`SPMLoader`** - Load from SPM.mat files
- **`NiftiLoader`** - Load from NIfTI files
- **`NilearnMaskerLoader`** - Integration with nilearn

### Configuration

- **`SearchlightConfig`** - Searchlight parameters (radius, n_jobs, checkpointing)
- **`RegionConfig`** - ROI parameters (regions, names, min_voxels)
- **`AnalysisConfig`** - Analysis parameters (regularization, permutation)
- **`ContrastSpec`** - Auto-generate contrasts from factorial designs

### Result Objects

- **`CvManovaResult`** - Rich result object with methods:
  - `to_nifti(contrast, filename)` - Save results to NIfTI
  - `plot_glass_brain(contrast)` - Visualize on glass brain
  - `get_peaks(contrast, n=10)` - Find peak voxels
  - `to_dataframe()` - Export to pandas

### Utilities

- **`contrasts(levels, names)`** - Generate factorial design contrasts
- **`sl_size(radius)`** - Calculate searchlight size
- **`load_data_spm(spm_dir)`** - Load from SPM.mat (low-level)
- **`write_image(data, filename, affine)`** - Write NIfTI files

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later)

Same license as the original MATLAB implementation.

## Original Authors

- **Carsten Allefeld** - Algorithm design and MATLAB implementation

## Acknowledgments

This is a Python port of the original MATLAB cvmanova package:
https://github.com/allefeld/cvmanova

The algorithm and methodology are entirely the work of the original authors.
Please cite their paper (Allefeld & Haynes, 2014) when using this software.

Feel free to contact the original author at http://www.carsten-allefeld.de/ with questions about the method. Bug reports for this Python port can be submitted via GitHub issues.
