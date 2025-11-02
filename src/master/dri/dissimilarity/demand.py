# cvrp_dri/dissimilarity/demand.py

import os
import vrplib
from typing import Dict, Tuple

def demand_dissimilarity(instance_name: str) -> Dict[Tuple[int, int], float]:
    """
    Computes the pairwise demand dissimilarity S^d_ij:
        S^d_ij = (d_i + d_j) / Q

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

    # Load VRP instance
    instance = vrplib.read_instance(instance_path)
    Q = instance["capacity"]
    demands_full = instance["demand"]
    demands = {i: demands_full[i] for i in range(1, len(demands_full))}  # exclude depot

    # Compute dissimilarities
    nodes = list(demands.keys())
    S_d = {}

    for idx_i, i in enumerate(nodes):
        d_i = demands[i]
        for j in nodes[idx_i + 1:]:
            d_j = demands[j]
            S_ij = (d_i + d_j) / Q
            S_d[(i, j)] = S_d[(j, i)] = S_ij  # symmetric

    return S_d


if __name__ == "__main__":
    S_d = demand_dissimilarity("X-n101-k25.vrp")
    print("First 3 demand dissimilarities:", list(S_d.items())[:3])



# # cvrp_dri/dissimilarity/demand.py

# from typing import Dict, Tuple

# def demand_dissimilarity(
#     demands: Dict[int, int],
#     capacity: int
# ) -> Dict[Tuple[int, int], float]:
#     """
#     Computes the pairwise demand dissimilarity S^d_ij as defined in the DRI framework:
#         S^d_ij = (d_i + d_j) / Q

#     Args:
#         demands: {node_id: demand_i}
#         capacity: vehicle capacity Q

#     Returns:
#         dissimilarity: {(i, j): S^d_ij} for all unordered pairs i < j
#     """
#     if capacity <= 0:
#         raise ValueError("Vehicle capacity Q must be positive.")

#     nodes = list(demands.keys())
#     S_d = {}

#     for idx_i, i in enumerate(nodes):
#         d_i = demands[i]

#         for j in nodes[idx_i + 1:]:
#             d_j = demands[j]
#             S_ij = (d_i + d_j) / capacity
#             S_d[(i, j)] = S_ij
#             S_d[(j, i)] = S_ij  # symmetric

#     return S_d


# if __name__ == "__main__":
#     # Example usage
#     demands = {2: 38, 3: 51, 4: 73}
#     Q = 206

#     S_d = demand_dissimilarity(demands, Q)
#     print("First 3 demand dissimilarities:", list(S_d.items())[:3])
