"""
Unified clustering runner with consistent naming.

Standard output:
    clusters : Dict[int, List[int]]
    medoids  : Dict[int, int] | None

Naming conventions:
    custom_*     → custom implementations (AC, K-Medoids)
    sk_*         → sklearn-based (AC, KMeans)
    fcm          → scikit-fuzzy
    pyclust_*    → pyclustering-based
"""

from typing import Dict, List, Tuple, Optional

# --- Custom AC (slow, high-quality) ---
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

# --- scikit-fuzzy ---
from master.clustering.fcm_scikit_fuzzy import run_sklearn_fcm


# =====================================================================
# Unified Interface
# =====================================================================

def run_clustering(
    method: str,
    instance_name: str,
    k: int,
    **kwargs,
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]:
    """
    Run any clustering algorithm with standardized naming.

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
    clusters : Dict[int, List[int]]
    medoids  : Dict[int, int] | None
    """
    method = method.lower()

    # ============================================================
    # CUSTOM AC
    # ============================================================
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

    # ============================================================
    # ERROR
    # ============================================================
    raise ValueError(f"Unknown clustering method '{method}'.")
