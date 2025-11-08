# clustering/dissimilarity/demand.py

from typing import Dict, Tuple, Optional
from utils.loader import load_instance


def demand_dissimilarity(instance_name: str, instance: Optional[dict] = None) -> Dict[Tuple[int, int], float]:
    """
    Computes the pairwise demand dissimilarity S^d_ij:
        S^d_ij = (d_i + d_j) / Q
    Only stores (i, j) for i < j for efficiency.
    """
    if instance is None:
        instance = load_instance(instance_name)

    Q = int(instance["capacity"])
    demands_arr = instance["demand"]
    demands = {i + 1: int(demands_arr[i]) for i in range(len(demands_arr))}
    DEPOT_ID = 1

    # Exclude depot
    demands = {i: d for i, d in demands.items() if i != DEPOT_ID}

    nodes = list(demands.keys())
    S_d: Dict[Tuple[int, int], float] = {}

    for idx_i, i in enumerate(nodes):
        d_i = demands[i]
        for j in nodes[idx_i + 1:]:
            d_j = demands[j]
            S_d[(i, j)] = (d_i + d_j) / Q

    return S_d


if __name__ == "__main__":
    instance = load_instance("X-n101-k25.vrp")
    S_d = demand_dissimilarity("X-n101-k25.vrp", instance)
    print("First 3 demand dissimilarities:", list(S_d.items())[:3])
