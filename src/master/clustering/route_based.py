# decomposition/route_based.py
"""
Route-Based Decomposition (RBD) for DRSCI.

This module clusters *routes* instead of customers. It is used after
obtaining a global solution (after routing or after LS) to decompose
the problem again for the next DRSCI iteration.

Supported clustering backends (package-based only):
    - scikit-learn KMeans
    - scikit-learn AgglomerativeClustering (average / complete / single)
    - scikit-fuzzy FCM  (via run_sklearn_fcm)
    - k_medoids_pyclustering

These are fast enough for large n. Custom AC/KMedoids versions are excluded.

Output:
    clusters : Dict[int, List[int]]
        Just like vertex-based decomposition: a dict of cluster_id -> customer IDs.

Usage:
    route_clusters = route_based_decomposition(
        instance_name,
        global_routes,
        k=10,
        method="kmeans",
        use_angle=True,
        use_load=True,
    )
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Literal

from utils.loader import load_instance
from master.clustering.scikit_clustering import run_sklearn_ac, run_sklearn_kmeans
from master.clustering.fcm_scikit_fuzzy import run_sklearn_fcm
from master.clustering.k_medoids_pyclustering import k_medoids_pyclustering


# ============================================================
# Step 1: Extract route feature vectors
# ============================================================

def compute_route_features(
    instance_name: str,
    routes: List[List[int]],
    *,
    use_angle: bool = True,
    use_load: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute feature vectors τ_r for each route.

    Features:
        centroid_x, centroid_y,
        (optional) centroid_angle,
        (optional) route_load

    Returns:
        X : array shape (R, d)
        route_ids : list of indices 0..R-1
    """
    inst = load_instance(instance_name)
    coords = inst["node_coord"]
    demands = inst["demand"]

    X = []
    route_ids = list(range(len(routes)))

    for ridx, route in enumerate(routes):
        # Remove depot (node_id = 1)
        customers = [nid for nid in route if nid != 1]

        if not customers:
            # empty route (unlikely)
            X.append([0, 0] + ([0] if use_angle else []) + ([0] if use_load else []))
            continue

        # --- 1) Centroid ---
        cx = np.mean([coords[nid - 1][0] for nid in customers])
        cy = np.mean([coords[nid - 1][1] for nid in customers])

        feat = [cx, cy]

        # --- 2) Angle ---
        if use_angle:
            theta = np.arctan2(cy, cx)
            feat.append(theta)

        # --- 3) Load ---
        if use_load:
            q = sum(int(demands[nid - 1]) for nid in customers)
            feat.append(q)

        X.append(feat)

    return np.asarray(X, float), route_ids


# ============================================================
# Step 2: Cluster routes using chosen backend
# ============================================================

def cluster_routes(
    X: np.ndarray,
    k: int,
    method: Literal[
        "kmeans",
        "ac_avg",
        "ac_complete",
        "ac_min",
        "fcm",
        "kmedoids_pyclust",
    ] = "kmeans",
) -> Dict[int, List[int]]:
    """
    Clusters route feature vectors.

    Returns:
        Dict[cluster_id -> [route_indices]]
    """
    n = X.shape[0]

    # --------------------------------------------------------
    # 1) scikit-learn KMeans
    # --------------------------------------------------------
    if method == "kmeans":
        model_result = run_sklearn_kmeans(
            instance_name=None,   # we won't use instance inside
            k=k,
            instance={"dummy": True},  # bypass loader expectation
            use_polar=False,
            use_demand=False,
            X_override=X          # <-- we will extend run_sklearn_kmeans to accept this
        )
        clusters, medoids, _ = model_result
        # clusters: cluster_id -> route_IDs
        return clusters

    # --------------------------------------------------------
    # 2) scikit-learn Agglomerative Clustering
    # --------------------------------------------------------
    if method.startswith("ac_"):
        linkage = {
            "ac_avg": "average",
            "ac_complete": "complete",
            "ac_min": "single",
        }[method]

        # Use sklearn AC directly on X
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)

        clusters: Dict[int, List[int]] = {cid: [] for cid in range(k)}
        for ridx, lab in enumerate(labels):
            clusters[lab].append(ridx)
        return clusters

    # --------------------------------------------------------
    # 3) scikit-fuzzy FCM
    # --------------------------------------------------------
    if method == "fcm":
        # run_sklearn_fcm expects VRPLIB customers → adapt:
        import skfuzzy as fuzz
        cntr, u, _, _, _, _, _ = fuzz.cmeans(
            X.T, k, 2.0, error=1e-4, maxiter=200
        )
        labels = np.argmax(u, axis=0)

        clusters: Dict[int, List[int]] = {cid: [] for cid in range(k)}
        for ridx, lab in enumerate(labels):
            clusters[lab].append(ridx)
        return clusters

    # --------------------------------------------------------
    # 4) pyclustering k-medoids
    # --------------------------------------------------------
    if method == "kmedoids_pyclust":
        from pyclustering.cluster.kmedoids import kmedoids
        import random

        # pick initial medoids
        initial = random.sample(range(len(X)), k)

        # use Euclidean between feature vectors
        km = kmedoids(X.tolist(), initial)
        km.process()
        clusters_raw = km.get_clusters()   # list[list[route_ids]]

        clusters: Dict[int, List[int]] = {}
        for cid, group in enumerate(clusters_raw):
            clusters[cid] = group
        return clusters

    raise ValueError(f"Unknown method '{method}' for route clustering.")


# ============================================================
# Step 3: Convert route clusters → customer clusters
# ============================================================

def route_clusters_to_customer_clusters(
    routes: List[List[int]],
    route_clusters: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """
    For each route-cluster, gather all customers appearing in the routes.
    """
    customer_clusters: Dict[int, List[int]] = {}

    for cid, route_ids in route_clusters.items():
        customers = []
        for ridx in route_ids:
            route = routes[ridx]
            for nid in route:
                if nid != 1:      # ignore depot
                    customers.append(nid)
        # deduplicate to avoid repeating customers
        customer_clusters[cid] = sorted(set(customers))

    return customer_clusters


# ============================================================
# Public API
# ============================================================

def route_based_decomposition(
    instance_name: str,
    global_routes: List[List[int]],
    k: int,
    method: Literal[
        "kmeans",
        "ac_avg",
        "ac_complete",
        "ac_min",
        "fcm",
        "kmedoids_pyclust",
    ] = "kmeans",
    *,
    use_angle: bool = True,
    use_load: bool = True,
) -> Dict[int, List[int]]:
    """
    The full Route-Based Decomposition pipeline.

    1) Build route features τ_r
    2) Cluster routes
    3) Convert route clusters back to customer clusters
    """
    # 1) route feature extraction
    X, route_ids = compute_route_features(
        instance_name,
        global_routes,
        use_angle=use_angle,
        use_load=use_load,
    )

    # 2) route clustering
    route_clusters = cluster_routes(
        X,
        k=k,
        method=method,
    )

    # 3) convert route clusters → customer clusters
    return route_clusters_to_customer_clusters(global_routes, route_clusters)
