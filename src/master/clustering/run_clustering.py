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
from master.clustering.ac_custom.avg_ac import agglomerative_clustering_average
from master.clustering.ac_custom.max_ac import agglomerative_clustering_complete
from master.clustering.ac_custom.min_ac import agglomerative_clustering_min

# --- Custom K-Medoids ---
from master.clustering.k_medoids import k_medoids
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
    **kwargs,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]:
    """
    Unified clustering interface.

    Parameters
    ----------
    method : str
        Name of clustering algorithm.
    instance_name : str
        VRPLIB instance to load.
    k : int
        Number of clusters.

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
        clusters, medoids = agglomerative_clustering_average(instance_name, k, **kwargs)

    elif method == "custom_ac_complete":
        clusters, medoids = agglomerative_clustering_complete(instance_name, k, **kwargs)

    elif method == "custom_ac_min":
        clusters, medoids = agglomerative_clustering_min(instance_name, k, **kwargs)

    elif method == "custom_k_medoids":
        clusters, medoids = k_medoids(instance_name, k, **kwargs)

    elif method == "pyclust_k_medoids":
        # Format returned by pyclustering: {medoid: members}
        cl_raw = k_medoids_pyclustering(instance_name, k, **kwargs)
        clusters, medoids = {}, {}
        for cid, (med, members) in enumerate(cl_raw.items(), start=1):
            clusters[cid] = members
            medoids[cid] = med

    elif method in ("sk_ac_avg", "sk_ac_complete", "sk_ac_min"):
        linkage = {
            "sk_ac_avg": "average",
            "sk_ac_complete": "complete",
            "sk_ac_min": "single",
        }[method]
        clusters, medoids = run_sklearn_ac(instance_name, k, linkage=linkage, **kwargs)

    elif method == "sk_kmeans":
        clusters, medoids, _ = run_sklearn_kmeans(instance_name, k, **kwargs)

    elif method == "fcm":
        clusters, medoids, _, _ = run_sklearn_fcm(instance_name, k, **kwargs)

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
