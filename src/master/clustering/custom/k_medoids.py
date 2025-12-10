# clustering/k_medoids.py
"""
Unified, normalized K-Medoids clustering for the DRI framework.
Returns:
    clusters : {cluster_id → [customer_ids]}
    medoids  : {cluster_id → medoid_id}
"""

import random
from typing import Dict, List, Tuple, Optional
from utils.symmetric_matrix_read import get_symmetric_value
from utils.loader import load_instance
from clustering.dissimilarity.combined import combined_dissimilarity
from clustering.dissimilarity.spatial import spatial_dissimilarity


# -----------------------------------------------------------
# Initialization (Eq. 16)
# -----------------------------------------------------------

def initialize_medoids(nodes: List[int], S: Dict[Tuple[int, int], float], k: int) -> List[int]:
    """Select k far-apart vertices as initial medoids."""
    scores = {}
    denom = sum(S.get((i, j), S.get((j, i), 0)) for i in nodes for j in nodes if i != j)

    for i in nodes:
        scores[i] = sum(get_symmetric_value(S, i, j) for j in nodes) / denom

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


# -----------------------------------------------------------
# Assignment
# -----------------------------------------------------------

def assign_to_medoids(nodes: List[int], medoids: List[int], S) -> Dict[int, List[int]]:
    """Assign each non-medoid node to its closest medoid."""
    clusters_raw = {m: [] for m in medoids}

    for i in nodes:
        if i in medoids:
            continue
        closest = min(medoids, key=lambda m: get_symmetric_value(S, i, m))
        clusters_raw[closest].append(i)

    return clusters_raw


# -----------------------------------------------------------
# Update
# -----------------------------------------------------------

def update_medoids(clusters_raw: Dict[int, List[int]], S) -> List[int]:
    """Assign new medoid per cluster (Eq. 17)."""
    new_medoids = []

    for old_m, members in clusters_raw.items():
        nodes = [old_m] + members
        best = min(nodes, key=lambda i:
                   sum(get_symmetric_value(S, i, j) for j in nodes if j != i))
        new_medoids.append(best)

    return new_medoids


# -----------------------------------------------------------
# Main Algorithm
# -----------------------------------------------------------

def k_medoids(
    instance_name: str,
    k: int,
    instance: Optional[dict] = None,
    use_combined: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Normalized K-Medoids:
      returns (clusters, medoids) with cluster IDs 1..k.
    """
    if instance is None:
        instance = load_instance(instance_name)

    S = combined_dissimilarity(instance_name) if use_combined else spatial_dissimilarity(instance_name)

    nodes = sorted({n for a, b in S.keys() for n in (a, b)})

    # --- initialize ---
    medoids = initialize_medoids(nodes, S, k)

    # --- iterative refinement ---
    while True:
        clusters_raw = assign_to_medoids(nodes, medoids, S)
        new_medoids = update_medoids(clusters_raw, S)

        if set(new_medoids) == set(medoids):
            break

        medoids = new_medoids

    # -------------------------------------------------------
    # Normalize output   (THIS WAS THE BUG YOU NEEDED FIXED)
    # -------------------------------------------------------
    clusters: Dict[int, List[int]] = {}
    medoids_dict: Dict[int, int] = {}

    for cid, (m, members) in enumerate(clusters_raw.items(), start=1):
        clusters[cid] = members
        medoids_dict[cid] = m

    return clusters, medoids_dict


if __name__ == "__main__":
    clusters, medoids = k_medoids("X-n101-k25.vrp", k=5)
    print(clusters)
    print(medoids)
