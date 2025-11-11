# clustering/k-medoids.py
"""
K-Medoids clustering for the DRI framework
------------------------------------------
Implements Algorithm 1 from:
Kerscher & Minner (2025), 'Decompose-route-improve framework for solving large-scale VRPTWs'.
Customers are clustered based on a precomputed dissimilarity matrix, e.g., S^s_ij or S^sd_ij.
"""

import random
from typing import Dict, List, Tuple, Optional
from utils.symmetric_matrix_read import get_symmetric_value
from utils.loader import load_instance
from clustering.dissimilarity.combined import combined_dissimilarity
from clustering.dissimilarity.spatial import spatial_dissimilarity


def initialize_medoids(nodes: List[int], S: Dict[Tuple[int, int], float], k: int) -> List[int]:
    """
    Initialization as described in Eq. (16): select |P_I| most dissimilar vertices.
    """
    scores = {}
    for i in nodes:
        denom = sum(S.get((j, l), S.get((l, j), 0)) for j in nodes for l in nodes if j != l)
        scores[i] = sum(get_symmetric_value(S, i, j) for j in nodes) / denom
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def assign_to_clusters(nodes: List[int], medoids: List[int], S: Dict[Tuple[int, int], float]) -> Dict[int, List[int]]:
    """Assign each node to the closest medoid."""
    clusters = {m: [] for m in medoids}
    for i in nodes:
        if i in medoids:
            continue
        closest = min(medoids, key=lambda m: get_symmetric_value(S, i, m))
        clusters[closest].append(i)
    return clusters


def update_medoids(clusters: Dict[int, List[int]], S: Dict[Tuple[int, int], float]) -> List[int]:
    """
    Update each cluster’s medoid following Eq. (17):
    m_P = argmin_i∈V_P sum_j∈V_P S_ij.
    """
    new_medoids = []
    for old_medoid, members in clusters.items():
        cluster_nodes = [old_medoid] + members
        best = min(cluster_nodes,
                   key=lambda i: sum(get_symmetric_value(S, i, j) for j in cluster_nodes if j != i))
        new_medoids.append(best)
    return new_medoids


def k_medoids(instance_name: str, k: int, instance: Optional[dict] = None, use_combined: bool = False,
) -> Dict[int, List[int]]:
    """
    Performs k-medoids clustering using the combined dissimilarity matrix.
    """
    if instance is None:
        instance = load_instance(instance_name)

    if use_combined:
        S = combined_dissimilarity(instance_name)
    else:
        S = spatial_dissimilarity(instance_name)    
    
    nodes = sorted({n for pair in S.keys() for n in pair})

    # --- Initialization (Eq. 16) ---
    medoids = initialize_medoids(nodes, S, k)

    # --- Repeat until convergence (Algorithm 1) ---
    while True:
        clusters = assign_to_clusters(nodes, medoids, S)
        new_medoids = update_medoids(clusters, S)
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids

    # Final cluster mapping: medoid → member list
    final_clusters = assign_to_clusters(nodes, medoids, S)
    return final_clusters


if __name__ == "__main__":
    clusters = k_medoids("X-n101-k25.vrp", k=5)
    for i, (m, members) in enumerate(clusters.items(), 1):
        print(f"Cluster {i}: medoid {m}, size {len(members) + 1}")
