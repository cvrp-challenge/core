# sklearn_clustering.py
"""
Unified Scikit-Learn Clustering Interface:
- Agglomerative Clustering (average / complete / single)
- K-Means Clustering

Supports:
  • spatial dissimilarity (feature-based)
  • spatial-polar-demand feature vectors
  • combined dissimilarity (S_ij)
  • precomputed dissimilarity mode (AC only)
  • proper scaling for K-Means

Output format is compatible with:
  • silhouette_coefficient(...)
  • plot_viz_clustering(...)
  • custom k-medoids / FCM / AC implementations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

from utils.loader import load_instance
from clustering.dissimilarity.spatial import spatial_dissimilarity
from clustering.dissimilarity.combined import combined_dissimilarity
from clustering.dissimilarity.polar_coordinates import compute_polar_angle
from utils.symmetric_matrix_read import get_symmetric_value


# ---------------------------------------------------------
# Feature Vector Builder
# ---------------------------------------------------------

def build_feature_matrix(instance_name: str,
                         instance: dict,
                         use_polar: bool = True,
                         use_demand: bool = False) -> np.ndarray:
    """
    Build feature matrix X for sklearn algorithms using:
      x, y, (optional) θ, (optional) demand

    Always excludes depot.
    """
    coords = instance["node_coord"][1:]      # remove depot
    demands = instance["demand"][1:]
    angles = compute_polar_angle(instance_name, instance)

    X = []
    node_ids = []

    for i in range(2, len(coords) + 2):  # customers 2..n+1
        x, y = coords[i - 2]
        theta = angles[i]
        q = int(demands[i - 2])

        fv = [x, y]
        if use_polar:
            fv.append(theta)
        if use_demand:
            fv.append(q)

        X.append(fv)
        node_ids.append(i)

    return np.array(X, dtype=float), node_ids


# ---------------------------------------------------------
# Medoid Extraction
# ---------------------------------------------------------

def compute_medoids(clusters: Dict[int, List[int]],
                    S: Dict[Tuple[int, int], float]) -> Dict[int, int]:
    """
    Compute medoid (central node) per cluster:
        m_p = argmin_i sum_j S_ij
    """
    medoids = {}
    for cid, members in clusters.items():
        best_node = None
        best_val = float("inf")
        for i in members:
            val = sum(get_symmetric_value(S, i, j) for j in members if j != i)
            if val < best_val:
                best_val = val
                best_node = i
        medoids[cid] = best_node
    return medoids


# ---------------------------------------------------------
# Agglomerative Clustering (scikit-learn)
# ---------------------------------------------------------

def run_sklearn_ac(instance_name: str,
                   k: int,
                   linkage: str = "average",       # "average" | "complete" | "single"
                   use_dissimilarity: bool = False,
                   use_combined: bool = False,
                   use_polar: bool = True,
                   use_demand: bool = False,
                   instance: Optional[dict] = None
                   ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Unified interface for sklearn AgglomerativeClustering.

    Modes:
      • Feature-based:
            use_dissimilarity = False
            → X = [x,y,(θ),(q)]
            → model.fit(X_scaled)

      • Precomputed dissimilarity:
            use_dissimilarity = True
            → S matrix used directly (AC only)

    Supported linkages:
        "average"      ← average linkage
        "complete"     ← max linkage
        "single"       ← min linkage
    """
    assert linkage in ("average", "complete", "single"), \
        f"Invalid linkage '{linkage}', must be one of: 'average', 'complete', 'single'"

    if instance is None:
        instance = load_instance(instance_name)

    # Build feature matrix
    X, node_ids = build_feature_matrix(
        instance_name, instance,
        use_polar=use_polar,
        use_demand=use_demand
    )

    # ---------------------------------------------------------
    # PRECOMPUTED DISSIMILARITY MODE
    # ---------------------------------------------------------
    if use_dissimilarity:
        S = combined_dissimilarity(instance_name) if use_combined else spatial_dissimilarity(instance_name)

        # convert S dict to dense NxN matrix
        n = len(node_ids)
        D = np.zeros((n, n))
        for a_idx, a in enumerate(node_ids):
            for b_idx, b in enumerate(node_ids):
                if a == b:
                    D[a_idx, b_idx] = 0.0
                else:
                    D[a_idx, b_idx] = get_symmetric_value(S, a, b)

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric="precomputed"
        )
        labels = model.fit_predict(D)

        # construct clusters
        clusters = {cid: [] for cid in range(k)}
        for idx, lab in enumerate(labels):
            clusters[lab].append(node_ids[idx])

        medoids = compute_medoids(clusters, S)
        return clusters, medoids

    # ---------------------------------------------------------
    # FEATURE-BASED MODE
    # ---------------------------------------------------------
    X_scaled = StandardScaler().fit_transform(X)

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage
    )
    labels = model.fit_predict(X_scaled)

    clusters = {cid: [] for cid in range(k)}
    for idx, lab in enumerate(labels):
        clusters[lab].append(node_ids[idx])

    # compute medoids with spatial S
    S = spatial_dissimilarity(instance_name)
    medoids = compute_medoids(clusters, S)

    return clusters, medoids


# ---------------------------------------------------------
# K-Means (scikit-learn)
# ---------------------------------------------------------

def run_sklearn_kmeans(instance_name: str,
                       k: int,
                       use_combined: bool = False,
                       use_polar: bool = True,
                       use_demand: bool = False,
                       instance: Optional[dict] = None
                       ) -> Tuple[Dict[int, List[int]], Dict[int, int], np.ndarray]:
    """
    K-Means clustering (feature-based only).

    Returns:
        clusters  : {cluster_id → [node_ids]}
        medoids   : medoid per cluster
        centroids : cluster centroid positions (scaled feature space)
    """

    if instance is None:
        instance = load_instance(instance_name)

    # If use_combined is True, add both polar and demand features
    # If use_combined is False, use only polar features
    if use_combined:
        use_polar = True
        use_demand = True

    X, node_ids = build_feature_matrix(
        instance_name, instance,
        use_polar=use_polar,
        use_demand=use_demand
    )

    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=k, n_init="auto")
    labels = model.fit_predict(X_scaled)
    centroids = model.cluster_centers_

    clusters = {cid: [] for cid in range(k)}
    for idx, lab in enumerate(labels):
        clusters[lab].append(node_ids[idx])

    S = spatial_dissimilarity(instance_name)
    medoids = compute_medoids(clusters, S)

    return clusters, medoids, centroids
