# cvManova (Python Port)

> **⚠️ IMPORTANT: This is a Python port of the original MATLAB cvManova package.**
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
# From source
cd python
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

## Validation Against MATLAB Implementation

The original MATLAB test suite uses the Haxby et al. (2001) dataset and produces the following expected results:

**Region Analysis (with SPM12):**
```
Region 1, Contrast 1: D = 5.443427
Region 1, Contrast 2: D = 1.021870
Region 2, Contrast 1: D = 0.314915
Region 2, Contrast 2: D = 0.021717
Region 3, Contrast 1: D = 1.711423
Region 3, Contrast 2: D = 0.241187
```

**Searchlight Analysis MD5 checksums:**
```
03adb4e589c9e1da8f08829c839b26d9  spmD_C0001_P0001.nii
7a8f0d5918363c213e0d749a1bfdd665  spmD_C0002_P0001.nii
8bfe2b4261920127b2fcf5fe5358a340  spmDs_C0001_P0001.nii
e7d2c583c5159feb671dea7ff2b72570  spmDs_C0002_P0001.nii
```

To validate this Python port against the MATLAB implementation, run the integration test with the Haxby dataset (requires downloading ~300MB of data):

```bash
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

n_voxels = sl_size(3.0)  # Number of voxels in searchlight
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

## Parameters

### Pattern Discriminability (D)
The main output is the pattern discriminability D, which measures how well multivariate patterns distinguish between conditions. Positive D indicates above-chance discrimination.

### Regularization (lambda_)
The `lambda_` parameter (0-1) controls shrinkage regularization of the error covariance matrix towards its diagonal. This can improve numerical stability when the number of voxels approaches the degrees of freedom. Default is 0 (no regularization).

### Permutation Testing
Set `permute=True` to compute permutation values for statistical inference. The number of permutations depends on the number of sessions (2^n / 2 for full enumeration, or max_perms for Monte Carlo).

## Testing

```bash
cd python
pip install -e ".[test]"
pytest tests/
```

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later)

Same license as the original MATLAB implementation.

## Original Authors

- **Carsten Allefeld** - Algorithm design and MATLAB implementation

## Python Port Contributors

- Python port contributors

## Acknowledgments

This is a Python port of the original MATLAB cvmanova package:
https://github.com/allefeld/cvmanova

The algorithm and methodology are entirely the work of the original authors.
Please cite their paper (Allefeld & Haynes, 2014) when using this software.
