"""
Validation functions for linear algebra.

This module contains functions that are used to validate various properties
and conditions related to linear algebra operations and structures. These
functions help ensure that input arrays and matrices meet the necessary
criteria for performing linear algebra computations correctly.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar("_T", bound=np.generic)


def is_square(matrix: npt.NDArray[_T]) -> bool:
    """
    Check if a given matrix is square.

    A matrix is considered square if it has the same number of rows and
    columns.

    Parameters
    ----------
    matrix : npt.NDArray[_T]
        The input matrix to be checked.

    Returns
    -------
    bool
        True if the matrix is square, False otherwise.

    Examples
    --------
    >>> is_square(np.array([[1, 2], [3, 4]]))
    True

    >>> is_square(np.array([[1, 2, 3], [4, 5, 6]]))
    False
    """
    if matrix.ndim != 2:
        return False
    rows, cols = matrix.shape
    # Circumvent a regression in numpy 2.1.
    # Note that this regression is already fixed in numpy 2.2.
    # reveal_type(rows) -> Any
    # reveal_type(cols) -> Any
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    return rows == cols


def is_qubitop(matrix: npt.NDArray[_T]) -> bool:
    """
    Check if the input matrix is a square matrix with a dimension that is a power of 2.

    Parameters
    ----------
    matrix : npt.NDArray[_T]
        The input matrix to be checked.

    Returns
    -------
    bool
        True if the matrix is a square matrix and its dimension is a power of 2,
        False otherwise.

    Notes
    -----
    A matrix is considered a square matrix if it has the same number of rows
    and columns. The dimension is a power of 2 if it can be expressed as
    2^n for some non-negative integer n.
    """
    if not is_square(matrix):
        return False
    size, _ = matrix.shape
    # Circumvent a regression in numpy 2.1.
    # Note that this regression is already fixed in numpy 2.2.
    # reveal_type(size) -> Any
    assert isinstance(size, int)
    return size > 0 and size & (size - 1) == 0


def is_hermitian(matrix: npt.NDArray[_T]) -> bool:
    """
    Check if the provided matrix is Hermitian.

    A matrix is considered Hermitian if it is equal to its own conjugate transpose.
    This means that for a matrix \( A \), it is Hermitian if \( A = A^H \).

    Parameters
    ----------
    matrix : ndarray
        A square matrix to be checked for Hermitian property.

    Returns
    -------
    bool
        True if the matrix is Hermitian, False otherwise.

    Raises
    ------
    ValueError
        If the input matrix is not square.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2 + 1j], [2 - 1j, 3]])
    >>> is_hermitian(A)
    True

    >>> B = np.array([[1, 2], [3, 4]])
    >>> is_hermitian(B)
    False
    """
    if not is_square(matrix):
        return False
    return np.allclose(matrix, matrix.transpose().conjugate())


def is_psd(matrix: npt.NDArray[_T], tol: float = 1e-15) -> bool:
    """
    Check if a density matrix is positive semidefinite by diagonalizing.

    Parameters
    ----------
    matrix : npt.NDArray
        The matrix to check for positive semidefiniteness.
    tol : float, optional
        The tolerance for considering small negative eigenvalues as zero. Default is 1e-15.

    Returns
    -------
    bool
        True if the matrix is positive semidefinite, False otherwise.
    """
    if not is_square(matrix):
        return False
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if not is_hermitian(matrix):
        return False
    evals = np.linalg.eigvalsh(matrix.astype(np.complex128))
    return all(evals >= -tol)


def is_unit_trace(matrix: npt.NDArray[_T]) -> bool:
    """
    Check if the given square matrix has a trace equal to 1.

    Parameters
    ----------
    matrix : array_like
        A square matrix represented as a NumPy array.

    Returns
    -------
    bool
        True if the trace of the matrix is 1, False otherwise.

    Notes
    -----
    The trace of a matrix is defined as the sum of the elements
    on its main diagonal. This function assumes that the input
    matrix is square; no checks for this condition are performed.
    """
    if not is_square(matrix):
        return False
    return np.allclose(matrix.trace(), 1.0)
