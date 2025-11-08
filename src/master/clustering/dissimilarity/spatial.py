# clustering/dissimilarity/spatial.py

import math
from typing import Dict, Tuple, Optional
from utils.loader import load_instance
from clustering.dissimilarity.polar_coordinates import compute_polar_angle


def compute_lambda(coords: Dict[int, Tuple[float, float]]) -> float:
    """λ = (1 / (2n)) * Σ_i (x_i + y_i)."""
    n = len(coords)
    if n == 0:
        raise ValueError("Coordinate dictionary is empty.")
    return sum(x + y for x, y in coords.values()) / (2 * n)


def spatial_dissimilarity(instance_name: str, instance: Optional[dict] = None) -> Dict[Tuple[int, int], float]:
    """
    Computes spatial dissimilarity S^s_ij:
        S^s_ij = sqrt((x_j - x_i)^2 + (y_j - y_i)^2 + λ * (θ_j - θ_i)^2)
    Only stores (i, j) for i < j for efficiency.
    """
    if instance is None:
        instance = load_instance(instance_name)

    coords_arr = instance["node_coord"]
    coords_full = {i + 1: tuple(coords_arr[i]) for i in range(len(coords_arr))}
    DEPOT_ID = 1

    # Exclude depot
    coords = {i: coords_full[i] for i in coords_full if i != DEPOT_ID}

    angles = compute_polar_angle(instance_name, instance)
    lam = compute_lambda(coords)

    nodes = list(coords.keys())
    S: Dict[Tuple[int, int], float] = {}

    for idx_i, i in enumerate(nodes):
        x_i, y_i = coords[i]
        theta_i = angles[i]
        for j in nodes[idx_i + 1:]:
            x_j, y_j = coords[j]
            theta_j = angles[j]
            S[(i, j)] = math.sqrt(
                (x_j - x_i) ** 2 + (y_j - y_i) ** 2 + lam * (theta_j - theta_i) ** 2
            )

    return S


if __name__ == "__main__":
    instance = load_instance("X-n101-k25.vrp")
    S = spatial_dissimilarity("X-n101-k25.vrp", instance)
    print("First 3 spatial dissimilarities:", list(S.items())[:3])
