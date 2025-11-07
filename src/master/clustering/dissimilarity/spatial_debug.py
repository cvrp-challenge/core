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

    Only stores (i, j) for i < j for efficiency.
    """
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../../"))
    instances_root = os.path.join(core_root, "instances", "test-instances")

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

    instance = vrplib.read_instance(instance_path)
    coords_full = instance["node_coord"]
    coords = {i: coords_full[i] for i in range(1, len(coords_full))}  # exclude depot

    angles = compute_polar_angle(instance_name)
    lam = compute_lambda(coords)
    nodes = list(coords.keys())
    S = {}

    print(f"λ (lambda) = {lam:.4f}")

    # Efficient half-matrix computation
    debug_count = 0
    for idx_i, i in enumerate(nodes):
        x_i, y_i = coords[i]
        theta_i = angles[i]
        for j in nodes[idx_i + 1:]:
            x_j, y_j = coords[j]
            theta_j = angles[j]

            dx2 = (x_j - x_i) ** 2
            dy2 = (y_j - y_i) ** 2
            dtheta2 = lam * (theta_j - theta_i) ** 2
            dist = math.sqrt(dx2 + dy2 + dtheta2)

            S[(i, j)] = dist

            # Print debug info for first 3 pairs only
            if debug_count < 3:
                print(f"\nPair ({i}, {j}):")
                print(f"  x_i={x_i:.2f}, y_i={y_i:.2f}, θ_i={theta_i:.4f}")
                print(f"  x_j={x_j:.2f}, y_j={y_j:.2f}, θ_j={theta_j:.4f}")
                print(f"  (x_j - x_i)^2 = {dx2:.4f}")
                print(f"  (y_j - y_i)^2 = {dy2:.4f}")
                print(f"  λ*(θ_j - θ_i)^2 = {dtheta2:.4f}")
                print(f"  → sqrt sum = {dist:.4f}")
                debug_count += 1

            if debug_count >= 3:
                break
        if debug_count >= 3:
            break

    return S


if __name__ == "__main__":
    S = spatial_dissimilarity("X-n101-k25.vrp")
    print("\nFirst 3 spatial dissimilarities:", list(S.items())[:3])

    # Also show λ explicitly for cross-check
    instance_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../instances/test-instances/x/X-n101-k25.vrp"
    )
    instance = vrplib.read_instance(instance_path)
    coords = {
        i: coord
        for i, coord in enumerate(instance["node_coord"][1:], start=1)
    }
    print("λ (computed separately for check):", compute_lambda(coords))
