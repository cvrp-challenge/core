# fcm_scikit_fuzzy.py
"""
Fuzzy C-Means using scikit-fuzzy (fuzz.cmeans)
------------------------------------------------

This implements feature-based FCM for VRP instances using optional:
  - Cartesian coordinates (always used)
  - Polar angle θ (optional)
  - Demand q (optional)

Returns:
  • clusters (crisp): {cluster_id -> [node_ids]}
  • medoids: {cluster_id -> medoid_node}
  • memberships: {node_id -> {cluster: μ_i,p}}
  • centroids: FCM centroids in scaled feature space

This is fully compatible with:
  - silhouette scoring
  - VRP solvers
  - your visualization tools
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler    # ← REQUIRED

from utils.loader import load_instance
from clustering.dissimilarity.spatial import spatial_dissimilarity
from clustering.dissimilarity.polar_coordinates import compute_polar_angle
from utils.symmetric_matrix_read import get_symmetric_value


# -------------------------------------------------------------
# Feature Builder
# -------------------------------------------------------------

def build_fcm_feature_matrix(
    instance_name: str,
    instance: dict,
    use_polar: bool = True,
    use_demand: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """
    Builds feature matrix for FCM:
        X[i] = [x, y, (θ optional), (q optional)]
    Excludes depot. Returns:
        X : np.ndarray (n_customers × dim)
        node_ids : list of node IDs
    """

    coords = instance["node_coord"][1:]         # remove depot
    demands = instance["demand"][1:]
    angles = compute_polar_angle(instance_name, instance)

    X = []
    node_ids = []

    for node_id in range(2, len(coords) + 2):   # customers are 2..n+1
        x, y = coords[node_id - 2]
        theta = angles[node_id]
        q = int(demands[node_id - 2])

        fv = [x, y]
        if use_polar:
            fv.append(theta)
        if use_demand:
            fv.append(q)

        X.append(fv)
        node_ids.append(node_id)

    return np.array(X, dtype=float), node_ids


# -------------------------------------------------------------
# Medoid Extraction (convert centroid -> closest node)
# -------------------------------------------------------------

def compute_medoids_from_centroids(
    clusters: Dict[int, List[int]],
    S: Dict[Tuple[int, int], float]
) -> Dict[int, int]:
    """
    Compute a medoid for each cluster using spatial S_ij.
    The medoid is the node minimizing total dissimilarity.
    """
    medoids = {}
    for cid, members in clusters.items():
        best_node, best_val = None, float("inf")
        for i in members:
            val = sum(get_symmetric_value(S, i, j) for j in members if j != i)
            if val < best_val:
                best_val = val
                best_node = i
        medoids[cid] = best_node
    return medoids


# -------------------------------------------------------------
# FCM using scikit-fuzzy
# -------------------------------------------------------------

def run_sklearn_fcm(
    instance_name: str,
    k: int,
    m: float = 2.0,
    max_iter: int = 150,
    error: float = 1e-5,
    use_combined: bool = False,
    use_polar: bool = True,
    use_demand: bool = False,
    instance: Optional[dict] = None
) -> Tuple[
    Dict[int, List[int]],             # clusters (crisp)
    Dict[int, int],                   # medoids
    Dict[int, Dict[int, float]],      # memberships μ[i][p]
    np.ndarray                        # centroids (scaled space)
]:
    """
    Fuzzy C-Means clustering using scikit-fuzzy.

    Steps:
      1. Build feature matrix X
      2. Scale X (FCM is sensitive to magnitude differences)
      3. Run fuzz.cmeans
      4. Convert membership matrix to crisp clusters
      5. Convert centroids -> medoids using spatial distance
      6. Construct membership dict
      7. Return everything
    """

    if instance is None:
        instance = load_instance(instance_name)

    # If use_combined is True, add both polar and demand features
    # If use_combined is False, use only polar features
    if use_combined:
        use_polar = True
        use_demand = True
        
    # Build feature matrix
    X, node_ids = build_fcm_feature_matrix(
        instance_name,
        instance,
        use_polar=use_polar,
        use_demand=use_demand
    )

    # Scale features (critical for FCM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # FCM expects data shape (features, samples)
    data = X_scaled.T  # shape: (dim, n_customers)

    # ---------------------------------------------------------
    # Run scikit-fuzzy FCM
    # ---------------------------------------------------------
    cntr, u, _, _, _, _, _ = fuzz.cmeans(
        data=data,
        c=k,
        m=m,
        error=error,
        maxiter=max_iter,
        init=None
    )

    # u: membership matrix shape (k, n_customers)
    # cntr: (k × dim) centroids

    # ---------------------------------------------------------
    # Crisp clusters using argmax
    # ---------------------------------------------------------
    labels = np.argmax(u, axis=0)

    clusters: Dict[int, List[int]] = {cid: [] for cid in range(k)}
    for idx, cid in enumerate(labels):
        clusters[cid].append(node_ids[idx])

    # ---------------------------------------------------------
    # Convert centroids → medoids
    # ---------------------------------------------------------
    S_spatial = spatial_dissimilarity(instance_name)
    medoids = compute_medoids_from_centroids(clusters, S_spatial)

    # ---------------------------------------------------------
    # Build membership dictionary
    # ---------------------------------------------------------
    memberships: Dict[int, Dict[int, float]] = {}
    for node_idx, node_id in enumerate(node_ids):
        memberships[node_id] = {p: float(u[p, node_idx]) for p in range(k)}

    return clusters, medoids, memberships, cntr
