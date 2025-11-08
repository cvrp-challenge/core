# utils/symmetric_matrix_read.py

from typing import Dict, Tuple


def get_symmetric_value(matrix: Dict[Tuple[int, int], float], i: int, j: int) -> float:
    """
    Returns the symmetric matrix value M_ij, checking both (i, j) and (j, i).

    Used for all half-matrix structures in the CVRP DRI framework,
    e.g. spatial, demand, and combined dissimilarities.

    Args:
        matrix: dict storing values for pairs (i, j) where i < j
        i, j: node indices

    Returns:
        The matrix value if found, or 0.0 if i == j.
    """
    if i == j:
        return 0.0
    return matrix.get((i, j)) or matrix.get((j, i))
