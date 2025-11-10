# clustering/avg_ac.py
"""
Agglomerative Clustering (Average Linkage) for the DRI framework.
Implements Algorithm 3 from Kerscher & Minner (2025),
adapted for CVRP without time windows.

Each customer starts as its own cluster, and clusters are iteratively merged
based on minimum average pairwise dissimilarity (Eq. 23 using average linkage).
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from utils.loader import load_instance
from utils.symmetric_matrix_read import get_symmetric_value
from clustering.dissimilarity.combined import combined_dissimilarity


def average_linkage_distance(
    cluster_a: List[int],
    cluster_b: List[int],
    S: Dict[Tuple[int, int], float],
) -> float:
    """
    Computes average pairwise dissimilarity between two clusters.
    Eq. (23) using average linkage:
        S̄_p,θ = (1 / (|V_p| * |V_θ|)) * Σ_{i∈V_p} Σ_{j∈V_θ} S_ij
    """
    total = 0.0
    count = 0
    for i in cluster_a:
        for j in cluster_b:
            total += get_symmetric_value(S, i, j)
            count += 1
    return total / count if count > 0 else math.inf


def find_closest_clusters(
    clusters: Dict[int, List[int]],
    S: Dict[Tuple[int, int], float],
) -> Tuple[int, int]:
    """
    Selects the cluster pair (P, Θ)' with minimum average dissimilarity.
    Randomly breaks ties (equal minimum).
    """
    best_pair = None
    best_dist = math.inf
    cluster_ids = list(clusters.keys())

    for idx_a, a in enumerate(cluster_ids):
        for b in cluster_ids[idx_a + 1:]:
            dist = average_linkage_distance(clusters[a], clusters[b], S)
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


def agglomerative_clustering_average(
    instance_name: str,
    k: int,
    instance: Optional[dict] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Executes Agglomerative Clustering (average linkage) for a CVRP instance.

    Returns:
        clusters: {cluster_id: [node_ids]}
        medoids: {cluster_id: medoid_node}
    """
    # --- load instance only once ---
    if instance is None:
        instance = load_instance(instance_name)

    # --- compute dissimilarity (without modifying combined_dissimilarity) ---
    # only pass instance_name to keep compatibility with all other code
    S = combined_dissimilarity(instance_name)

    # Step 1: start with each customer as its own cluster
    nodes = sorted({n for pair in S.keys() for n in pair})
    clusters: Dict[int, List[int]] = {i: [i] for i in nodes}

    # Step 2: merge until number of clusters = k
    while len(clusters) > k:
        pair = find_closest_clusters(clusters, S)
        if pair is None:
            break
        a, b = pair

        # Merge clusters
        merged = clusters[a] + clusters[b]
        new_id = min(a, b)  # assign smallest index as new id
        clusters[new_id] = merged
        # remove old cluster entries safely
        for old in [a, b]:
            if old in clusters and old != new_id:
                del clusters[old]

    # Step 3: compute medoids for each final cluster
    medoids = compute_cluster_medoids(clusters, S)
    return clusters, medoids


if __name__ == "__main__":
    instance_name = "X-n101-k25.vrp"
    k = 5
    clusters, medoids = agglomerative_clustering_average(instance_name, k)

    print(f"✅ Agglomerative Clustering (Average Linkage) complete for {instance_name}")
    print(f"Number of clusters: {len(clusters)}\n")

    for idx, (cid, members) in enumerate(clusters.items(), start=1):
        m = medoids[cid]
        print(f"Cluster {idx}:")
        print(f"  Medoid: {m}")
        print(f"  Members: {len(members)} customers")
        print(f"  Example members: {members[:10]}")
