# run_clustering.py
"""
Unified clustering runner.
Standardizes all outputs to:

    clusters : Dict[int, List[int]]
    medoids  : Dict[int, int] | None

Every clustering function from this project is wrapped so that downstream
modules (routing, DRSCI iterations, SCP) can use a single interface.
"""

from typing import Dict, List, Tuple, Optional

# --- Custom AC ---
from master.clustering.ac_custom.avg_ac import agglomerative_clustering_average
from master.clustering.ac_custom.max_ac import agglomerative_clustering_complete
from master.clustering.ac_custom.min_ac import agglomerative_clustering_min

# --- K-Medoids ---
from master.clustering.k_medoids import k_medoids
from master.clustering.k_medoids_pyclustering import k_medoids_pyclustering

# --- sklearn-based clustering ---
from master.clustering.scikit_clustering import (
    run_sklearn_ac,
    run_sklearn_kmeans,
)

# --- FCM (skit-fuzzy based) ---
from master.clustering.fcm_scikit_fuzzy import run_sklearn_fcm


# ============================================================
# Unified Clustering Interface
# ============================================================

def run_clustering(
    method: str,
    instance_name: str,
    k: int,
    **kwargs,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]:
    """
    Run any clustering algorithm in the project and normalize output.

    Returns:
        clusters : Dict[int, List[int]]
        medoids  : Dict[int, int] | None
    """
    method = method.lower()

    # ============================================
    # CUSTOM AGGLOMERATIVE CLUSTERING
    # ============================================
    if method == "ac_avg":
        clusters, medoids = agglomerative_clustering_average(instance_name, k, **kwargs)
        return clusters, medoids

    if method == "ac_max":
        clusters, medoids = agglomerative_clustering_complete(instance_name, k, **kwargs)
        return clusters, medoids

    if method == "ac_min":
        clusters, medoids = agglomerative_clustering_min(instance_name, k, **kwargs)
        return clusters, medoids

    # ============================================
    # CUSTOM K-MEDOIDS
    # ============================================
    if method == "k_medoids":
        clusters, medoids = k_medoids(instance_name, k, **kwargs)
        return clusters, medoids

    if method == "k_medoids_pyclustering":
        # returns {medoid_node_id: [members...]}
        clust_by_medoid = k_medoids_pyclustering(instance_name, k, **kwargs)
        clusters = {}
        medoids = {}
        for cid, (med, members) in enumerate(clust_by_medoid.items(), start=1):
            clusters[cid] = members
            medoids[cid] = med
        return clusters, medoids

    # ============================================
    # SKLEARN-BASED AC
    # ============================================
    if method == "sk_ac_avg":
        clusters, medoids = run_sklearn_ac(instance_name, k, linkage="average", **kwargs)
        return clusters, medoids

    if method == "sk_ac_complete":
        clusters, medoids = run_sklearn_ac(instance_name, k, linkage="complete", **kwargs)
        return clusters, medoids
    
    if method == "sk_ac_min":
        clusters, medoids = run_sklearn_ac(instance_name, k, linkage="single", **kwargs)
        return clusters, medoids

    # ============================================
    # SKLEARN K-MEANS
    # ============================================
    if method == "sk_kmeans":
        clusters, medoids, _centers = run_sklearn_kmeans(instance_name, k, **kwargs)
        return clusters, medoids

    # ============================================
    # FCM (scikit-fuzzy)
    # ============================================
    if method == "fcm":
        # returns: clusters, medoids, memberships, cntr
        clusters, medoids, memberships, cntr = run_sklearn_fcm(instance_name, k, **kwargs)
        return clusters, medoids

    # ============================================
    # ERROR: unknown method
    # ============================================
    raise ValueError(f"Unknown clustering method '{method}'.")
