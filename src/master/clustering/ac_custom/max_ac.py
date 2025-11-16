# max agglomerative clustering

# clustering/max_ac.py
"""
Agglomerative Clustering (Complete Linkage) for the DRI framework.
Implements Algorithm 3 from Kerscher & Minner (2025),
adapted for CVRP without time windows.

Each customer starts as its own cluster, and clusters are iteratively merged
based on the minimum maximum pairwise dissimilarity (Eq. 23 using complete linkage).
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from utils.loader import load_instance
from utils.symmetric_matrix_read import get_symmetric_value
from clustering.dissimilarity.combined import combined_dissimilarity
from clustering.dissimilarity.spatial import spatial_dissimilarity


def complete_linkage_distance(
    cluster_a: List[int],
    cluster_b: List[int],
    S: Dict[Tuple[int, int], float],
) -> float:
    """
    Computes complete-linkage dissimilarity between two clusters:
        S̄_p,θ = max_{i∈V_p, j∈V_θ} S_ij
    (Eq. 25)
    """
    max_dist = -math.inf
    for i in cluster_a:
        for j in cluster_b:
            val = get_symmetric_value(S, i, j)
            if val > max_dist:
                max_dist = val
    return max_dist


def find_closest_clusters(
    clusters: Dict[int, List[int]],
    S: Dict[Tuple[int, int], float],
) -> Tuple[int, int]:
    """
    Selects the cluster pair (P, Θ)' with minimum complete-linkage dissimilarity.
    Randomly breaks ties (equal minimum).
    """
    best_pair = None
    best_dist = math.inf
    cluster_ids = list(clusters.keys())

    for idx_a, a in enumerate(cluster_ids):
        for b in cluster_ids[idx_a + 1:]:
            dist = complete_linkage_distance(clusters[a], clusters[b], S)
            if dist < best_dist:
                best_dist = dist
                best_pair = (a, b)
            elif dist == best_dist and random.random() < 0.5:
                best_pair = (a, b)

    return best_pair


def compute_cluster_medoids(
    clusters: Dict[int, List[int]],
    S: Dict[Tuple[int, int], float],
) -> Dict[int, int]:
    """
    Computes the medoid (most central node) for each cluster:
        m_p = argmin_i Σ_j S_ij,  j ∈ V_p
    """
    medoids = {}
    for cid, members in clusters.items():
        best_node, best_val = None, float("inf")
        for i in members:
            val = sum(get_symmetric_value(S, i, j) for j in members if i != j)
            if val < best_val:
                best_val, best_node = val, i
        medoids[cid] = best_node
    return medoids


def agglomerative_clustering_complete(
    instance_name: str,
    k: int,
    instance: Optional[dict] = None,
    use_combined: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Executes Agglomerative Clustering (complete linkage) for a CVRP instance.

    Returns:
        clusters: {cluster_id: [node_ids]}
        medoids: {cluster_id: medoid_node}
    """
    if instance is None:
        instance = load_instance(instance_name)
    if use_combined:
        S = combined_dissimilarity(instance_name)
    else:
        S = spatial_dissimilarity(instance_name)

    # Step 1: start with each customer as its own cluster
    nodes = sorted({n for pair in S.keys() for n in pair})
    clusters: Dict[int, List[int]] = {i: [i] for i in nodes}

    # Step 2: merge until number of clusters = k
    while len(clusters) > k:
        pair = find_closest_clusters(clusters, S)
        if pair is None:
            break
        a, b = pair

        merged = clusters[a] + clusters[b]
        new_id = min(a, b)
        clusters[new_id] = merged
        for old in [a, b]:
            if old in clusters and old != new_id:
                del clusters[old]

    # Step 3: compute medoids
    medoids = compute_cluster_medoids(clusters, S)
    return clusters, medoids


if __name__ == "__main__":
    instance_name = "X-n101-k25.vrp"
    k = 5
    clusters, medoids = agglomerative_clustering_complete(instance_name, k)

    print(f"✅ Agglomerative Clustering (Complete Linkage) complete for {instance_name}")
    print(f"Number of clusters: {len(clusters)}\n")

    for idx, (cid, members) in enumerate(clusters.items(), start=1):
        m = medoids[cid]
        print(f"Cluster {idx}:")
        print(f"  Medoid: {m}")
        print(f"  Members: {len(members)} customers")
        print(f"  Example members: {members[:10]}")
