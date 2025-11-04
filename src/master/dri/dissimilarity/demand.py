# cvrp_dri/dissimilarity/demand.py

import os
import vrplib
from typing import Dict, Tuple

def demand_dissimilarity(instance_name: str) -> Dict[Tuple[int, int], float]:
    """
    Computes the pairwise demand dissimilarity S^d_ij:
        S^d_ij = (d_i + d_j) / Q
    Only stores (i, j) with i < j for efficiency.
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
    Q = instance["capacity"]
    demands_full = instance["demand"]
    demands = {i: demands_full[i] for i in range(1, len(demands_full))}

    nodes = list(demands.keys())
    S_d = {}

    for idx_i, i in enumerate(nodes):
        d_i = demands[i]
        for j in nodes[idx_i + 1:]:
            d_j = demands[j]
            S_d[(i, j)] = (d_i + d_j) / Q  # store only one direction

    return S_d


def get_demand_dissim(S_d: Dict[Tuple[int, int], float], i: int, j: int) -> float:
    """Helper to retrieve S^d_ij regardless of ordering."""
    if i == j:
        return 0.0
    return S_d.get((i, j)) or S_d.get((j, i))


if __name__ == "__main__":
    S_d = demand_dissimilarity("X-n101-k25.vrp")
    # Example symmetric lookup
    print("S_d(1,2) =", get_demand_dissim(S_d, 1, 2))
    print("S_d(2,1) =", get_demand_dissim(S_d, 2, 1))
