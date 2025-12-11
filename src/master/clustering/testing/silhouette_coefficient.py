# evaluation/silhouette_score.py
"""
Computes the Silhouette Coefficient ζ for clustering quality evaluation,
as defined in Eq. (27) of Kerscher & Minner (2025).

This implementation works for any clustering result from the DRI framework:
K-Medoids, Fuzzy C-Medoids, Agglomerative (Average/Complete).

Assumes that time dissimilarity is disregarded (CVRP only).
"""

from typing import Dict, List, Tuple
from utils.symmetric_matrix_read import get_symmetric_value
from clustering.dissimilarity.combined import combined_dissimilarity
import math


def silhouette_coefficient(
    instance_name: str,
    clusters: Dict[int, List[int]],
    medoids: Dict[int, int],
) -> float:
    """
    Computes the overall silhouette coefficient ζ ∈ [-1, 1].

    Args:
        instance_name: name of the instance (e.g., "X-n101-k25.vrp")
        clusters: {cluster_id: [list of nodes]}
        medoids: {cluster_id: medoid_node}

    Returns:
        ζ (float): average silhouette coefficient for all customers
    """
    # Load dissimilarity matrix (spatial + demand)
    S = combined_dissimilarity(instance_name)

    # Build reverse map node -> cluster
    node_to_cluster = {}
    for cid, members in clusters.items():
        for n in members:
            node_to_cluster[n] = cid

    ζ_values = []

    # Loop over all customers
    for i in node_to_cluster.keys():
        P = node_to_cluster[i]         # current cluster
        m_P = medoids[P]               # its medoid
        S_i_mP = get_symmetric_value(S, i, m_P)  # dissimilarity to own medoid

        # Find most similar (closest) other cluster
        best_cluster = None
        best_dist = math.inf
        for O, m_O in medoids.items():
            if O == P:
                continue
            S_i_mO = get_symmetric_value(S, i, m_O)
            if S_i_mO < best_dist:
                best_dist = S_i_mO
                best_cluster = O

        if best_cluster is None:
            continue  # should not happen unless only 1 cluster

        S_i_mO = best_dist
        denom = max(S_i_mP, S_i_mO)
        if denom == 0:
            ζ_i = 0
        else:
            ζ_i = (S_i_mO - S_i_mP) / denom

        ζ_values.append(ζ_i)

    # Global ζ
    if len(ζ_values) == 0:
        return 0.0
    return sum(ζ_values) / len(ζ_values)


if __name__ == "__main__":
    # Example test
    from clustering.k_medoids import k_medoids

    instance_name = "X-n101-k25.vrp"
    k = 5
    clusters = k_medoids(instance_name, k)

    # Convert from {medoid: members} to {id: members}, {id: medoid}
    clusters_dict = {}
    medoids_dict = {}
    for idx, (m, members) in enumerate(clusters.items(), start=1):
        clusters_dict[idx] = members
        medoids_dict[idx] = m

    ζ = silhouette_coefficient(instance_name, clusters_dict, medoids_dict)
    print(f"✅ Silhouette Coefficient for {instance_name}: ζ = {ζ:.4f}")
