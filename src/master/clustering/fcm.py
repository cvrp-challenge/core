# clustering/fuzzy_c_medoids.py
"""
Normalized Fuzzy C-Medoids (FCM)
Returns:
    clusters : {cluster_id → [node_ids]}
    medoids  : {cluster_id → medoid_id}
    τ_P      : cluster feature vectors
"""

import random
from typing import Dict, List, Tuple, Optional
from utils.loader import load_instance
from utils.symmetric_matrix_read import get_symmetric_value
from clustering.dissimilarity.polar_coordinates import compute_polar_angle
from clustering.dissimilarity.combined import combined_dissimilarity
from clustering.dissimilarity.spatial import spatial_dissimilarity
from clustering.k_medoids import initialize_medoids


# ------------------------------------------------------------------
# Feature vectors
# ------------------------------------------------------------------

def compute_customer_features(instance_name: str, instance) -> Dict[int, Tuple[float, float, float, float]]:
    coords = instance["node_coord"]
    demands = instance["demand"]
    angles = compute_polar_angle(instance_name, instance)

    features = {}
    for i in range(2, len(coords)+1):
        x, y = coords[i-1]
        q = int(demands[i-1])
        theta = angles[i]
        features[i] = (x, y, theta, q)

    return features


def compute_cluster_features(U, features):
    k = len(next(iter(U.values())))
    τ = {p: [0.0]*4 for p in range(k)}
    weights = {p: 0.0 for p in range(k)}

    for i, μi in U.items():
        fi = features[i]
        for p in range(k):
            for d in range(4):
                τ[p][d] += μi[p] * fi[d]
            weights[p] += μi[p]

    for p in range(k):
        if weights[p] > 0:
            τ[p] = [v/weights[p] for v in τ[p]]

    return τ


# ------------------------------------------------------------------
# Membership update (Eq. 18)
# ------------------------------------------------------------------

def update_membership(nodes, medoids, S, kappa):
    U = {}
    for i in nodes:
        U[i] = {}
        for p, m_p in enumerate(medoids):
            denom = 0.0
            Sip = max(get_symmetric_value(S, i, m_p), 1e-10)
            for m_o in medoids:
                Sio = max(get_symmetric_value(S, i, m_o), 1e-10)
                denom += (Sip / Sio) ** (2/(kappa-1))
            U[i][p] = 1/denom
    return U


# ------------------------------------------------------------------
# FCM main
# ------------------------------------------------------------------

def fuzzy_c_medoids(
    instance_name: str,
    k: int,
    kappa: float = 2.0,
    epsilon: float = 1e-4,
    max_iter: int = 100,
    instance=None,
    use_combined: bool = False,
):
    if instance is None:
        instance = load_instance(instance_name)

    S = combined_dissimilarity(instance_name) if use_combined else spatial_dissimilarity(instance_name)

    nodes = sorted({n for a,b in S.keys() for n in (a,b)})
    features = compute_customer_features(instance_name, instance)

    # --- init ---
    medoids = initialize_medoids(nodes, S, k)
    U_prev = update_membership(nodes, medoids, S, kappa)
    prev_obj = 1e15

    # --- iterate ---
    for _ in range(max_iter):

        # update medoids (Eq. 19)
        new_medoids = []
        used = set()
        for p in range(k):
            best_node, best_val = None, float("inf")
            for i in nodes:
                if i in used: continue
                val = sum((U_prev[j][p]**2)*get_symmetric_value(S,i,j) for j in nodes if j!=i)
                if val < best_val:
                    best_val, best_node = val, i
            new_medoids.append(best_node)
            used.add(best_node)

        medoids = new_medoids

        # update memberships
        U_new = update_membership(nodes, medoids, S, kappa)

        # convergence test
        diff = sum(abs(U_new[i][p]-U_prev[i][p]) for i in nodes for p in range(k))
        obj = sum((U_new[i][p]**2) * get_symmetric_value(S, i, medoids[p])
                  for i in nodes for p in range(k))

        if diff < epsilon and abs(obj-prev_obj) < 1e-3:
            U_prev = U_new
            break

        U_prev = U_new
        prev_obj = obj

    # ------------------------------------------------------------
    # NORMALIZE OUTPUT FORMAT  (the fix!)
    # ------------------------------------------------------------
    clusters = {cid: [] for cid in range(1, k+1)}
    medoids_dict = {cid: medoids[cid-1] for cid in range(1, k+1)}

    for i in nodes:
        assigned = max(U_prev[i], key=U_prev[i].get)
        clusters[assigned+1].append(i)

    τ_P = compute_cluster_features(U_prev, features)

    return clusters, medoids_dict, τ_P
