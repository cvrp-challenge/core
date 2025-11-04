# cvrp_dri/dissimilarity/combined.py

import os
import vrplib
from typing import Dict, Tuple
from .spatial import spatial_dissimilarity, get_spatial_dissim  # adjust name if your file is 'spatial.py'

def combined_dissimilarity(instance_name: str) -> Dict[Tuple[int, int], float]:
    """
    Computes the combined spatial-demand dissimilarity S^sd_ij:
        S^sd_ij = S^s_ij * (1 + (d_i + d_j) / Q)

    - Uses spatial dissimilarities from spatial_dissimilarity()
    - Only stores (i, j) for i < j for efficiency
    - The higher the demand of i and j, the higher the overall dissimilarity
    """
    # Base directory for locating instances
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
    Q = instance["capacity"]
    demands_full = instance["demand"]
    demands = {i: demands_full[i] for i in range(1, len(demands_full))}

    # Compute spatial dissimilarity once
    S_s = spatial_dissimilarity(instance_name)

    nodes = list(demands.keys())
    S_sd = {}

    print(f"Vehicle capacity Q = {Q}")

    # Combine spatial and demand components
    for idx_i, i in enumerate(nodes):
        d_i = demands[i]
        for j in nodes[idx_i + 1:]:
            d_j = demands[j]

            # Retrieve S^s_ij
            S_s_ij = get_spatial_dissim(S_s, i, j)

            # Combined dissimilarity (Eq.)
            S_sd_ij = S_s_ij * (1 + (d_i + d_j) / Q)
            S_sd[(i, j)] = S_sd_ij

            # Debug print for first 3 pairs
            if idx_i == 0 and j <= nodes[3]:
                print(f"\nPair ({i}, {j}):")
                print(f"  S^s_ij = {S_s_ij:.4f}")
                print(f"  (d_i + d_j)/Q = {(d_i + d_j)/Q:.4f}")
                print(f"  S^sd_ij = {S_sd_ij:.4f}")

    return S_sd


if __name__ == "__main__":
    S_sd = combined_dissimilarity("X-n101-k25.vrp")
    print("\nFirst 3 combined dissimilarities:", list(S_sd.items())[:3])
