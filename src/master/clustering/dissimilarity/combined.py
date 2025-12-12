# clustering/dissimilarity/combined.py

from typing import Dict, Tuple
from master.utils.loader import load_instance
from master.utils.symmetric_matrix_read import get_symmetric_value
from master.clustering.dissimilarity.spatial import spatial_dissimilarity
from master.clustering.dissimilarity.demand import demand_dissimilarity


def combined_dissimilarity(instance_name: str) -> Dict[Tuple[int, int], float]:
    """
    Computes combined spatial-demand dissimilarity:
        S^sd_ij = S^s_ij * (1 + (d_i + d_j) / Q)
    Only stores (i, j) for i < j for efficiency.
    """
    instance = load_instance(instance_name)
    Q = int(instance["capacity"])
    demands_arr = instance["demand"]
    demands = {i + 1: int(demands_arr[i]) for i in range(len(demands_arr))}
    DEPOT_ID = 1

    # Compute spatial and demand dissimilarities with shared instance
    S_s = spatial_dissimilarity(instance_name, instance)
    S_d = demand_dissimilarity(instance_name, instance)

    nodes = list({n for pair in S_s.keys() for n in pair})
    S_sd: Dict[Tuple[int, int], float] = {}

    for idx_i, i in enumerate(nodes):
        for j in nodes[idx_i + 1:]:
            S_s_ij = get_symmetric_value(S_s, i, j)
            S_d_ij = get_symmetric_value(S_d, i, j)
            S_sd[(i, j)] = S_s_ij * (1 + S_d_ij)

    return S_sd


if __name__ == "__main__":
    S_sd = combined_dissimilarity("X-n101-k25.vrp")
    print("First 3 combined dissimilarities:", list(S_sd.items())[:3])
