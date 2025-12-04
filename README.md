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

A Python implementation of cross-validated MANOVA for fMRI data analysis.

This package implements multivariate pattern analysis (MVPA) using cross-validated MANOVA as introduced by Allefeld & Haynes (2014).

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

## Quick Start

### Searchlight Analysis

```python
import numpy as np
from cvmanova import cv_manova_searchlight, contrasts

# Load your data (Ys: list of session data, Xs: list of design matrices)
# mask: 3D boolean array
# fE: degrees of freedom per session

# Generate contrasts for a 2x2 factorial design
Cs, names = contrasts([2, 2], ['Factor1', 'Factor2'])

# Run searchlight analysis
D, p, n_contrasts, n_perms = cv_manova_searchlight(
    Ys, Xs, mask,
    sl_radius=3.0,  # searchlight radius in voxels
    Cs=Cs,
    fE=fE,
    permute=False,   # set True for permutation testing
    lambda_=0.0      # regularization parameter (0-1)
)
```

### Region of Interest Analysis

```python
from cvmanova import cv_manova_region

# region_indices: list of arrays with mask voxel indices per region
D, p = cv_manova_region(
    Ys, Xs, Cs, fE,
    region_indices,
    permute=False,
    lambda_=0.0
)

# Print results
for ri in range(D.shape[2]):
    for ci in range(D.shape[0]):
        print(f"Region {ri+1}, Contrast {ci+1}: D = {D[ci, 0, ri]:.6f}")
```

### Loading Data from SPM.mat

If you have an existing SPM analysis, you can load data directly:

```python
from cvmanova import load_data_spm
from cvmanova.api import searchlight_analysis, region_analysis

# Load data from SPM.mat
Ys, Xs, mask, misc = load_data_spm('/path/to/spm/directory')

# Or use the high-level API
D, p, n_contrasts, n_perms = searchlight_analysis(
    '/path/to/spm/directory',
    sl_radius=3.0,
    Cs=Cs,
    permute=False
)
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

Example for a 2x3 factorial design:
```python
from cvmanova import contrasts

Cs, names = contrasts([2, 3])
# Returns: main effect A, main effect B, interaction AxB
```

## Important Remarks

From the original documentation:

- **Model specification matters**: The estimation of D is based on GLM residuals and depends on a properly specified model. Include all known systematic effects in the model, even if they don't enter the contrast.

- **Temporal autocorrelation**: The fMRI model must include modeling of temporal autocorrelations. In SPM, keep 'serial correlations' at `AR(1)` or `FAST`.

- **Multiple contrasts are efficient**: Computing several contrasts in one call is substantially faster than separate calls.

- **Memory usage**: Peak memory is about 2x the data size: (in-mask voxels) x (scans) x 8 bytes.

- **Checkpointing**: The searchlight analysis saves progress and can resume if interrupted.

## Regularization

For large searchlight sizes or ROIs, regularization can help with numerical stability:

```python
D, p, _, _ = cv_manova_searchlight(..., lambda_=0.001)
```

**However**, with regularization, D is no longer an unbiased estimator. It's recommended to:
1. Avoid regularization when possible
2. Reduce the number of voxels instead
3. Use the recommended searchlight radius of 3 (123 voxels)
4. Keep `lambda_` very small if needed (e.g., 0.001)

The implementation limits voxels to 90% of available error degrees of freedom.

## Negative Pattern Distinctness?

Estimated D values can be negative even though true pattern distinctness cannot be. This is expected behavior:

- The estimator is **unbiased** (correct on average)
- When true D is near zero, estimates vary around zero, so ~half will be negative
- **Strongly** negative values may indicate unmodelled confounds or design problems

This is analogous to cross-validated classification accuracy being below chance.

## Validation Against MATLAB Implementation

The Python port has been tested against the original MATLAB implementation using the Haxby et al. (2001) dataset.

**MATLAB expected values (SPM12):**
```
Region 1, Contrast 1: D = 5.443427
Region 1, Contrast 2: D = 1.021870
Region 2, Contrast 1: D = 0.314915
Region 2, Contrast 2: D = 0.021717
Region 3, Contrast 1: D = 1.711423
Region 3, Contrast 2: D = 0.241187
```

**Python values (simplified preprocessing):**
```
Region 1, Contrast 1: D = 1.168399
Region 1, Contrast 2: D = 0.251478
Region 2, Contrast 1: D = 0.044688
Region 2, Contrast 2: D = -0.002491
Region 3, Contrast 1: D = 0.431727
Region 3, Contrast 2: D = 0.044129
```

Note: Values differ due to preprocessing differences (Python uses simplified preprocessing without motion correction). The **relative pattern is preserved** (Region 1 > Region 3 > Region 2) with **Spearman rho = 1.0** (perfect rank correlation).

To run integration tests:
```bash
# Tests will automatically download Haxby data (~300MB) if not present
pytest tests/test_integration_haxby.py -v
```

## API Reference

### Core Functions

#### `CvManovaCore`
Core computation engine for cross-validated MANOVA.

```python
from cvmanova import CvManovaCore

cmc = CvManovaCore(Ys, Xs, Cs, fE, permute=False, lambda_=0.0)
D = cmc.compute(voxel_indices)
```

#### `cv_manova_searchlight`
Run cross-validated MANOVA on searchlight.

```python
D, p, n_contrasts, n_perms = cv_manova_searchlight(
    Ys, Xs, mask, sl_radius, Cs, fE,
    permute=False, lambda_=0.0, checkpoint=None
)
```

#### `cv_manova_region`
Run cross-validated MANOVA on regions of interest.

```python
D, p = cv_manova_region(
    Ys, Xs, Cs, fE, region_indices,
    permute=False, lambda_=0.0
)
```

### Utility Functions

#### `contrasts`
Generate contrast matrices for factorial designs.

```python
from cvmanova import contrasts

c_matrix, c_name = contrasts([2, 3], ['Factor1', 'Factor2'])
```

#### `sl_size`
Calculate searchlight size for a given radius.

```python
from cvmanova import sl_size

n_voxels = sl_size(3.0)  # Returns 123
```

#### `sign_permutations`
Generate sign permutations for permutation testing.

```python
from cvmanova import sign_permutations

perms, n_perms = sign_permutations(n_sessions, max_perms=5000)
```

#### `inestimability`
Check if a contrast is estimable.

```python
from cvmanova import inestimability

ie = inestimability(C, X)  # Should be ~0 for estimable contrasts
```

### I/O Functions

#### `load_data_spm`
Load fMRI data from SPM.mat file.

```python
from cvmanova import load_data_spm

Ys, Xs, mask, misc = load_data_spm('/path/to/spm_dir', regions=None)
```

#### `write_image`
Write data to NIfTI file.

```python
from cvmanova import write_image

write_image(data, 'output.nii', affine, descrip='description')
```

#### `read_vols_masked`
Read masked voxels from NIfTI files.

```python
from cvmanova import read_vols_masked

Y, mask = read_vols_masked(volume_files, mask)
```

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
