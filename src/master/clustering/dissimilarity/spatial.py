# cvrp_dri/dissimilarity/spatial.py

import os
import math
import vrplib
from typing import Dict, Tuple
from polar_coordinates import compute_polar_angle


def compute_lambda(coords: Dict[int, Tuple[float, float]]) -> float:
    """
    λ = (1 / (2n)) * Σ_i (x_i + y_i)
    """
    n = len(coords)
    if n == 0:
        raise ValueError("Coordinate dictionary is empty.")
    return sum(x + y for x, y in coords.values()) / (2 * n)


def spatial_dissimilarity(instance_name: str) -> Dict[Tuple[int, int], float]:
    """
    Computes spatial dissimilarity S^s_ij for all customer pairs in a VRP instance:
        S^s_ij = sqrt((x_j - x_i)^2 + (y_j - y_i)^2 + λ * (θ_j - θ_i)^2)

    Optimized for large instances:
        - Only stores (i, j) for i < j  (half matrix)
        - Avoids redundant computations

    Automatically searches for the instance in:
        core/instances/test-instances/x
        core/instances/test-instances/xl
    """
    # Build base path relative to this file
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../../"))
    instances_root = os.path.join(core_root, "instances", "test-instances")

    # Try both folders (x and xl)
    possible_paths = [
        os.path.join(instances_root, "x", instance_name),
        os.path.join(instances_root, "xl", instance_name),
    ]
    instance_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if instance_path is None:
        raise FileNotFoundError(
            f"Instance '{instance_name}' not found in: {possible_paths}"
        )

    print(f"→ Loading instance from: {instance_path}")

    # Load instance
    instance = vrplib.read_instance(instance_path)
    coords_full = instance["node_coord"]
    coords = {i: coords_full[i] for i in range(1, len(coords_full))}  # exclude depot

    # Compute polar angles
    angles = compute_polar_angle(instance_name)

    # Compute λ and initialize result dict
    lam = compute_lambda(coords)
    nodes = list(coords.keys())
    S = {}

    # Efficient half-matrix computation
    for idx_i, i in enumerate(nodes):
        x_i, y_i = coords[i]
        theta_i = angles[i]
        for j in nodes[idx_i + 1:]:
            x_j, y_j = coords[j]
            theta_j = angles[j]

            # Spatial dissimilarity (Eq. 34)
            dist = math.sqrt(
                (x_j - x_i) ** 2 + (y_j - y_i) ** 2 + lam * (theta_j - theta_i) ** 2
            )

            S[(i, j)] = dist  # store only one direction

    return S


def get_spatial_dissim(S: Dict[Tuple[int, int], float], i: int, j: int) -> float:
    """Helper for symmetric lookup of S^s_ij regardless of order."""
    if i == j:
        return 0.0
    return S.get((i, j)) or S.get((j, i))


if __name__ == "__main__":
    S = spatial_dissimilarity("X-n101-k25.vrp")
    print("First 3 spatial dissimilarities:", list(S.items())[:3])
