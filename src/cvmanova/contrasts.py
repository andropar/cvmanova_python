"""
Contrast generation for factorial designs.

This module provides functions to generate contrast matrices for
main effects and interactions in factorial experimental designs.
"""

import numpy as np
from itertools import product


def contrasts(
    f_level: list[int] | np.ndarray, f_name: list[str] | None = None
) -> tuple[list[np.ndarray], list[str]]:
    """
    Generate contrasts for main effects and interactions of a factorial design.

    Parameters
    ----------
    f_level : array-like
        Number of levels for each factor.
    f_name : list of str, optional
        Names of factors. If not provided, generic names ('A', 'B', ...) are used.

    Returns
    -------
    c_matrix : list of ndarray
        Contrast matrices.
    c_name : list of str
        Contrast names.

    Notes
    -----
    The first factor is the one being enumerated slowest.
    Contrasts are not orthonormalized.

    Examples
    --------
    >>> c_matrix, c_name = contrasts([2, 3])
    >>> c_name
    ['A', 'B', 'A×B']

    >>> c_matrix, c_name = contrasts([2, 2], ['Face', 'House'])
    >>> c_name
    ['Face', 'House', 'Face×House']
    """
    f_level = np.asarray(f_level).flatten()
    nf = len(f_level)

    if f_name is None:
        # Generate generic names for factors
        f_name = [chr(ord("A") + i) for i in range(nf)]

    # Generate sorted list of contrast signatures
    # Each signature is a binary pattern indicating which factors are involved
    nc = 2**nf - 1
    cs = np.zeros((nc, nf), dtype=bool)
    for i in range(nc):
        for j in range(nf):
            cs[i, j] = bool((i + 1) >> j & 1)

    # Sort by number of factors involved (main effects first, then interactions)
    order = np.argsort(np.sum(cs, axis=1))
    cs = cs[order]

    # Compute contrast elements for each factor
    # e[0, fi] = ones (averaging)
    # e[1, fi] = difference coding
    e = [[None, None] for _ in range(nf)]
    for fi in range(nf):
        e[fi][0] = np.ones((f_level[fi], 1))
        # Difference coding: -diff(eye(n)).T
        eye_n = np.eye(f_level[fi])
        e[fi][1] = -np.diff(eye_n, axis=0).T

    # Compute contrasts
    c_matrix = []
    c_name = []
    for ci in range(nc):
        # Contrast matrix via Kronecker product
        contrast = np.array([[1.0]])
        for fi in range(nf):
            idx = 1 if cs[ci, fi] else 0
            contrast = np.kron(contrast, e[fi][idx])
        c_matrix.append(contrast)

        # Contrast name
        involved_factors = [f_name[fi] for fi in range(nf) if cs[ci, fi]]
        c_name.append("×".join(involved_factors))

    return c_matrix, c_name


def print_contrasts(
    f_level: list[int] | np.ndarray, f_name: list[str] | None = None
) -> None:
    """
    Print contrasts for a factorial design.

    Parameters
    ----------
    f_level : array-like
        Number of levels for each factor.
    f_name : list of str, optional
        Names of factors.
    """
    c_matrix, c_name = contrasts(f_level, f_name)
    for name, matrix in zip(c_name, c_matrix):
        print(f"{name}:")
        print(matrix)
        print()
