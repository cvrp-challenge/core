"""
Route-Based Decomposition (RBD) for DRSCI.

Uses consistent naming across the whole project:
    sk_ac_avg
    sk_ac_complete
    sk_ac_min
    sk_kmeans
    fcm
    pyclust_k_medoids

Custom AC and custom K-medoids are intentionally NOT supported here
because RBD must remain fast for large-scale VRP instances.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Literal

from utils.loader import load_instance

# sklearn-based clustering (customer-based functions, but reused here)
from master.clustering.scikit_clustering import (
    run_sklearn_ac,
    run_sklearn_kmeans,
)

# fuzzy c-means
from master.clustering.fcm_scikit_fuzzy import run_sklearn_fcm

# pyclustering k-medoids
from master.clustering.k_medoids_pyclustering import k_medoids_pyclustering


# ======================================================================
# Step 1: Extract route feature vectors
# ======================================================================

def compute_route_features(
    instance_name: str,
    routes: List[List[int]],
    *,
    use_angle: bool = True,
    use_load: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build feature vectors τ_r per route.

    Features:
        - centroid_x, centroid_y
        - optional centroid_angle
        - optional route_load
    """
    inst = load_instance(instance_name)
    coords = inst["node_coord"]
    demands = inst["demand"]

    X = []
    route_ids = list(range(len(routes)))

    for r_idx, route in enumerate(routes):
        customers = [nid for nid in route if nid != 1]  # remove depot

        if not customers:
            X.append([0, 0] + ([0] if use_angle else []) + ([0] if use_load else []))
            continue

        cx = np.mean([coords[nid - 1][0] for nid in customers])
        cy = np.mean([coords[nid - 1][1] for nid in customers])

        feat = [cx, cy]

        if use_angle:
            theta = np.arctan2(cy, cx)
            feat.append(theta)

        if use_load:
            q = sum(int(demands[nid - 1]) for nid in customers)
            feat.append(q)

        X.append(feat)

    return np.asarray(X, float), route_ids


# ======================================================================
# Step 2: Route clustering using unified naming
# ======================================================================

def cluster_routes(
    X: np.ndarray,
    k: int,
    method: str
) -> Dict[int, List[int]]:
    """
    Clusters route-feature matrix X using the standardized method names.

    Supported:
        sk_ac_avg
        sk_ac_complete
        sk_ac_min
        sk_kmeans
        fcm
        pyclust_k_medoids
    """
    method = method.lower()

    # ---------------------------------------------------------
    # sklearn Agglomerative Clustering
    # ---------------------------------------------------------
    if method in ("sk_ac_avg", "sk_ac_complete", "sk_ac_min"):
        linkage_map = {
            "sk_ac_avg": "average",
            "sk_ac_complete": "complete",
            "sk_ac_min": "single",
        }
        linkage = linkage_map[method]

        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)

        clusters = {cid: [] for cid in range(k)}
        for ridx, lab in enumerate(labels):
            clusters[lab].append(ridx)
        return clusters

    # ---------------------------------------------------------
    # sklearn KMeans
    # ---------------------------------------------------------
    if method == "sk_kmeans":
        # Slight override: we want to supply X directly.
        # Trick: call run_sklearn_kmeans with X_override.
        clusters, medoids, _ = run_sklearn_kmeans(
            instance_name=None,
            k=k,
            instance={"dummy": True},
            use_polar=False,
            use_demand=False,
            X_override=X,
        )
        return clusters

    # ---------------------------------------------------------
    # scikit-fuzzy FCM
    # ---------------------------------------------------------
    if method == "fcm":
        import skfuzzy as fuzz
        cntr, u, _, _, _, _, _ = fuzz.cmeans(
            X.T,
            k,
            2.0,
            error=1e-4,
            maxiter=200,
        )
        labels = np.argmax(u, axis=0)

        clusters = {cid: [] for cid in range(k)}
        for ridx, lab in enumerate(labels):
            clusters[lab].append(ridx)
        return clusters

    # ---------------------------------------------------------
    # pyclustering K-Medoids
    # ---------------------------------------------------------
    if method == "pyclust_k_medoids":
        from pyclustering.cluster.kmedoids import kmedoids
        import random
        initial = random.sample(range(len(X)), k)

        km = kmedoids(X.tolist(), initial)
        km.process()
        clusters_raw = km.get_clusters()

        clusters = {}
        for cid, group in enumerate(clusters_raw):
            clusters[cid] = group
        return clusters

    # ---------------------------------------------------------
    # ERROR
    # ---------------------------------------------------------
    raise ValueError(
        f"Unsupported RBD clustering method '{method}'. "
        f"Allowed: sk_ac_avg, sk_ac_complete, sk_ac_min, sk_kmeans, fcm, pyclust_k_medoids"
    )


# ======================================================================
# Step 3: Convert route clusters → customer clusters
# ======================================================================

def route_clusters_to_customer_clusters(
    routes: List[List[int]],
    route_clusters: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    """
    Converts route-level clusters into customer-level clusters.
    """
    customer_clusters = {}

    for cid, route_ids in route_clusters.items():
        custs = []
        for ridx in route_ids:
            for nid in routes[ridx]:
                if nid != 1:
                    custs.append(nid)
        customer_clusters[cid] = sorted(set(custs))

    return customer_clusters


# ======================================================================
# Public API
# ======================================================================

def route_based_decomposition(
    instance_name: str,
    global_routes: List[List[int]],
    k: int,
    method: str,
    *,
    use_angle: bool = True,
    use_load: bool = True,
) -> Dict[int, List[int]]:
    """
    Full RBD pipeline:

        1) Extract route features
        2) Cluster routes using unified naming
        3) Convert route clusters back to customer clusters
    """
    # 1) Features
    X, route_ids = compute_route_features(
        instance_name,
        global_routes,
        use_angle=use_angle,
        use_load=use_load,
    )

    # 2) Route clustering
    route_clusters = cluster_routes(X, k=k, method=method)

    # 3) Convert to customer clusters
    return route_clusters_to_customer_clusters(global_routes, route_clusters)
