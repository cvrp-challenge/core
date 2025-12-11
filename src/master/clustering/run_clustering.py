"""
Unified clustering runner with consistent naming and guaranteed full coverage.

This file ensures:
    - A single interface for all clustering methods.
    - Each clustering method returns:
        clusters : Dict[int, List[int]]
        medoids  : Dict[int, int] | None
    - ALL customers are always assigned to exactly one cluster.
      => Prevents SCP errors: "customers not covered by any route"
      => Prevents routing failures due to missing customers

Clustering methods supported:
    custom_ac_avg
    custom_ac_complete
    custom_ac_min
    custom_k_medoids
    pyclust_k_medoids
    sk_ac_avg
    sk_ac_complete
    sk_ac_min
    sk_kmeans
    fcm
"""

from typing import Dict, List, Tuple, Optional

# --- Instance loader (needed for enforcing full coverage)
from utils.loader import load_instance

# --- Custom AC implementations ---
from master.clustering.custom.avg_ac import agglomerative_clustering_average
from master.clustering.custom.max_ac import agglomerative_clustering_complete
from master.clustering.custom.min_ac import agglomerative_clustering_min

# --- Custom K-Medoids ---
from master.clustering.custom.k_medoids import k_medoids
from master.clustering.k_medoids_pyclustering import k_medoids_pyclustering

# --- sklearn-based clustering ---
from master.clustering.scikit_clustering import (
    run_sklearn_ac,
    run_sklearn_kmeans,
)

# --- Fuzzy C-means ---
from master.clustering.fcm_scikit_fuzzy import run_sklearn_fcm


# =====================================================================
# Coverage Enforcement
# =====================================================================

def enforce_full_coverage(
    clusters: Dict[int, List[int]],
    all_customers: List[int],
) -> Dict[int, List[int]]:
    """
    Ensures that ALL customers are assigned to exactly one cluster.

    Missing customers are appended to the smallest cluster (by size).
    This prevents missing-customer failures in routing and SCP.

    Parameters
    ----------
    clusters : dict[int -> list[int]]
        Raw clusters from the chosen clustering algorithm.
    all_customers : list[int]
        List of all customer IDs (1..N), excluding depot.

    Returns
    -------
    dict
        Corrected cluster dictionary with full coverage.
    """
    assigned = set()
    for members in clusters.values():
        assigned.update(members)

    missing = [c for c in all_customers if c not in assigned]

    if not missing:
        return clusters

    # Find cluster with minimal size
    smallest_cluster_id = min(clusters, key=lambda cid: len(clusters[cid]))

    print(f"[Clustering WARNING] Missing customers detected: {missing}")
    print(f"[Clustering FIX] Assigning them to cluster {smallest_cluster_id}")

    clusters[smallest_cluster_id].extend(missing)
    return clusters


# =====================================================================
# Unified Clustering Dispatch
# =====================================================================

def run_clustering(
    method: str,
    instance_name: str,
    k: int,
    use_combined: bool = False,
    **kwargs,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]:
    """
    Unified clustering interface.

    Parameters
    ----------
    method : str
        One of:
            custom_ac_avg
            custom_ac_complete
            custom_ac_min
            custom_k_medoids
            pyclust_k_medoids
            sk_ac_avg
            sk_ac_complete
            sk_ac_min
            sk_kmeans
            fcm

    Returns
    -------
    clusters : dict[int -> list[int]]
    medoids  : dict[int -> int] | None
    """
    method = method.lower()

    # -------------------------------
    # Dispatch to implementation
    # -------------------------------

    if method == "custom_ac_avg":
        return agglomerative_clustering_average(instance_name, k, **kwargs)
    if method == "custom_ac_complete":
        return agglomerative_clustering_complete(instance_name, k, **kwargs)
    if method == "custom_ac_min":
        return agglomerative_clustering_min(instance_name, k, **kwargs)

    # ============================================================
    # CUSTOM K-MEDOIDS
    # ============================================================
    if method == "custom_k_medoids":
        return k_medoids(instance_name, k, **kwargs)

    if method == "pyclust_k_medoids":
        # pyclustering format is {medoid -> members}
        clust_by_medoid = k_medoids_pyclustering(instance_name, k, **kwargs)
        clusters = {}
        medoids = {}
        for cid, (med, members) in enumerate(clust_by_medoid.items(), start=1):
            clusters[cid] = members
            medoids[cid] = med
        return clusters, medoids

    # ============================================================
    # SKLEARN AC
    # ============================================================
    if method == "sk_ac_avg":
        return run_sklearn_ac(instance_name, k, linkage="average", **kwargs)
    if method == "sk_ac_complete":
        return run_sklearn_ac(instance_name, k, linkage="complete", **kwargs)
    if method == "sk_ac_min":
        return run_sklearn_ac(instance_name, k, linkage="single", **kwargs)

    # ============================================================
    # SKLEARN K-MEANS
    # ============================================================
    if method == "sk_kmeans":
        clusters, medoids, _ = run_sklearn_kmeans(instance_name, k, **kwargs)
        return clusters, medoids

    # ============================================================
    # FCM (scikit-fuzzy)
    # ============================================================
    if method == "fcm":
        clusters, medoids, _, _ = run_sklearn_fcm(instance_name, k, **kwargs)
        return clusters, medoids

    else:
        raise ValueError(f"Unknown clustering method '{method}'.")

    # =====================================================================
    # COVERAGE ENFORCEMENT
    # =====================================================================
    inst = load_instance(instance_name)

    # inst["demand"] is a list: index = node id, index 0 = depot
    num_nodes = len(inst["demand"])

    # Depot is explicitly node 1 in VRPLIB CVRP (DEPOT_SECTION = 1)
    depot_id = 1

    # Customers are nodes 2..N
    all_customers = list(range(2, num_nodes))


    clusters = enforce_full_coverage(clusters, all_customers)


    return clusters, medoids
