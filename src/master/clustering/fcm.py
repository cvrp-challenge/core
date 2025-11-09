# clustering/fuzzy_c_medoids.py
"""
Fuzzy C-Medoids (FCM) clustering for the DRI framework.
Implements Algorithm 2 and Eq. (18) from:
Kerscher & Minner (2025) 'Decompose-route-improve framework for solving large-scale VRPTWs'.

Adapted for CVRP without time windows:
Each customer i has a feature vector τ_i = (x_i, y_i, θ_i, q_i),
and a membership degree μ[i,p] ∈ [0,1] for each cluster p.
Clusters are represented by medoid nodes.
"""

import random
import math
from typing import Dict, Tuple, List, Optional
from utils.loader import load_instance
from utils.symmetric_matrix_read import get_symmetric_value
from clustering.dissimilarity.polar_coordinates import compute_polar_angle
from clustering.dissimilarity.combined import combined_dissimilarity


# ============================================================
# --- Feature Computation ------------------------------------
# ============================================================

def compute_customer_features(instance_name: str, instance: dict) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Computes τ_i = (x_i, y_i, θ_i, q_i) for all customers (excluding depot).
    Returns dict: {i: (x, y, θ, q)}.
    """
    coords_arr = instance["node_coord"]
    demands_arr = instance["demand"]
    angles = compute_polar_angle(instance_name, instance)

    DEPOT_ID = 1
    features = {}

    for i in range(1, len(coords_arr) + 1):
        if i == DEPOT_ID:
            continue
        x_i, y_i = coords_arr[i - 1]
        q_i = int(demands_arr[i - 1])
        theta_i = angles[i]
        features[i] = (x_i, y_i, theta_i, q_i)

    return features


def compute_cluster_features(U, features):
    """
    Computes cluster-level feature vectors:
        τ_P = (Σ_i μ_i,P * τ_i) / Σ_i μ_i,P
    Returns dict {p: [x̄, ȳ, θ̄, q̄]}.
    """
    k = len(next(iter(U.values())))
    τ = {p: [0.0, 0.0, 0.0, 0.0] for p in range(k)}
    weight_sum = {p: 0.0 for p in range(k)}

    for i, μ_i in U.items():
        τ_i = features[i]
        for p in range(k):
            for dim in range(4):
                τ[p][dim] += μ_i[p] * τ_i[dim]
            weight_sum[p] += μ_i[p]

    for p in range(k):
        if weight_sum[p] > 0:
            τ[p] = [v / weight_sum[p] for v in τ[p]]
    return τ



# ============================================================
# --- FCM Core Steps -----------------------------------------
# ============================================================

def initialize_membership_matrix(nodes: List[int], k: int) -> Dict[int, Dict[int, float]]:
    """
    Initialize membership matrix U⁰ with random values that sum to 1 for each i.
    Returns μ[i][p] for i in nodes, p in range(k).
    """
    U = {}
    for i in nodes:
        rand_vals = [random.random() for _ in range(k)]
        s = sum(rand_vals)
        U[i] = {p: rand_vals[p] / s for p in range(k)}
    return U


def update_membership_matrix(
    nodes: List[int],
    medoids: List[int],
    S: Dict[Tuple[int, int], float],
    kappa: float,
) -> Dict[int, Dict[int, float]]:
    """
    Update μᵢₚ following Eq. (18):
        μᵢₚ = (Σₒ ((Sᵢₚ / Sᵢₒ)^(2 / (κ−1))))⁻¹
    """
    U = {}
    for i in nodes:
        U[i] = {}
        for p_idx, m_p in enumerate(medoids):
            denom = 0.0
            S_ip = max(get_symmetric_value(S, i, m_p), 1e-10)  # avoid 0-division
            for o_idx, m_o in enumerate(medoids):
                S_io = max(get_symmetric_value(S, i, m_o), 1e-10)
                denom += (S_ip / S_io) ** (2 / (kappa - 1))
            U[i][p_idx] = 1 / denom
    return U


def update_medoids(
    U: Dict[int, Dict[int, float]],
    nodes: List[int],
    S: Dict[Tuple[int, int], float],
    k: int,
) -> List[int]:
    """
    Update cluster medoids mₚ as argmin Σ μᵢₚ^κ · Sᵢⱼ (Eq. 19).
    """
    medoids = []
    for p in range(k):
        best_node, best_val = None, float("inf")
        for i in nodes:
            val = sum((U[j][p] ** 2) * get_symmetric_value(S, i, j) for j in nodes if i != j)
            if val < best_val:
                best_val, best_node = val, i
        medoids.append(best_node)
    return medoids


# ============================================================
# --- Main FCM Algorithm -------------------------------------
# ============================================================

def fuzzy_c_medoids(
    instance_name: str,
    k: int,
    kappa: float = 2.0,
    epsilon: float = 1e-4,
    max_iter: int = 100,
    instance: Optional[dict] = None,
) -> Tuple[Dict[int, Dict[int, float]], List[int], Dict[int, List[float]]]:
    """
    Executes the Fuzzy C-Medoids algorithm (FCM) adapted for CVRP.
    Stable version including:
      – K-Medoids-based initialization
      – unique medoid enforcement
      – smooth convergence criterion (ΔU + objective stability)
    Returns:
      (U, medoids, τ_P)
        U        : membership matrix μ[i][p]
        medoids  : final list of medoid node IDs
        τ_P      : cluster-level feature vectors τ_P = (x̄, ȳ, θ̄, q̄)
    """
    import numpy as np
    from clustering.k_medoids import initialize_medoids

    if instance is None:
        instance = load_instance(instance_name)

    # --- Precompute all necessary data ---
    S = combined_dissimilarity(instance_name)
    features = compute_customer_features(instance_name, instance)
    nodes = sorted({n for pair in S.keys() for n in pair})

    # --- Step 1: Initialization ---
    medoids = initialize_medoids(nodes, S, k)  # far-apart start
    U_prev = initialize_membership_matrix(nodes, k)
    prev_obj = float("inf")

    # --- Step 2: Iterative optimization ---
    for iteration in range(max_iter):
        # --- Step 3: Update medoids (Eq. 19) with uniqueness enforcement ---
        new_medoids = []
        used_nodes = set()
        for p in range(k):
            best_node, best_val = None, float("inf")
            for i in nodes:
                if i in used_nodes:
                    continue
                val = sum((U_prev[j][p] ** 2) * get_symmetric_value(S, i, j)
                          for j in nodes if i != j)
                if val < best_val:
                    best_val, best_node = val, i
            if best_node is None:  # fallback
                remaining = [n for n in nodes if n not in used_nodes]
                best_node = random.choice(remaining)
            new_medoids.append(best_node)
            used_nodes.add(best_node)
        medoids = new_medoids

        # --- Step 4: Update membership matrix (Eq. 18) ---
        U_new = update_membership_matrix(nodes, medoids, S, kappa)

        # --- Step 5: Convergence check ---
        # 1) ΔU change
        diff = sum(abs(U_new[i][p] - U_prev[i][p]) for i in nodes for p in range(k))

        # 2) Objective function change
        total_obj = sum(
            (U_new[i][p] ** 2) * get_symmetric_value(S, i, medoids[p])
            for i in nodes for p in range(k)
        )
        obj_diff = abs(total_obj - prev_obj)
        prev_obj = total_obj

        if diff < epsilon and obj_diff < 1e-3:
            break

        U_prev = U_new

    # --- Step 6: Compute cluster-level feature vectors τ_P ---
    τ_P = compute_cluster_features(U_prev, features)

    return U_prev, medoids, τ_P

def fuzzy_c_medoids_debug(
    instance_name: str,
    k: int,
    kappa: float = 2.0,
    epsilon: float = 1e-4,
    max_iter: int = 100,
    instance: Optional[dict] = None,
) -> Tuple[Dict[int, Dict[int, float]], List[int], Dict[int, List[float]]]:
    """
    Debug version of Fuzzy C-Medoids.
    Prints detailed diagnostics per iteration:
      – medoid choices and objective values
      – ΔU changes
      – inter-cluster correlation of memberships
    """
    import numpy as np
    from clustering.k_medoids import initialize_medoids

    if instance is None:
        instance = load_instance(instance_name)

    # Precompute dissimilarities and features
    S = combined_dissimilarity(instance_name)
    features = compute_customer_features(instance_name, instance)
    nodes = sorted({n for pair in S.keys() for n in pair})

    # --- Initialization ---
    medoids = initialize_medoids(nodes, S, k)  # far-apart medoids
    U_prev = initialize_membership_matrix(nodes, k)

    print(f"\n[DEBUG] Initial medoids: {medoids}")
    print(f"[DEBUG] First 3 membership rows (node → memberships):")
    for i in list(U_prev.keys())[:3]:
        print(f"  Node {i}: {U_prev[i]}")

    prev_diff = None
    for iteration in range(max_iter):
        print(f"\n========== Iteration {iteration + 1} ==========")

        # --- Step 3–4: update medoids with uniqueness enforcement ---
        new_medoids = []
        used_nodes = set()
        for p in range(k):
            best_node, best_val = None, float("inf")
            for i in nodes:
                if i in used_nodes:
                    continue
                val = sum((U_prev[j][p] ** 2) * get_symmetric_value(S, i, j)
                          for j in nodes if i != j)
                if val < best_val:
                    best_val, best_node = val, i
            if best_node is None:
                remaining = [n for n in nodes if n not in used_nodes]
                best_node = random.choice(remaining)
            new_medoids.append(best_node)
            used_nodes.add(best_node)

            print(f"[DEBUG] Cluster {p+1}: chose medoid {best_node} "
                  f"(weighted dissimilarity sum = {best_val:.2f})")

        medoids = new_medoids
        print(f"[DEBUG] Medoids after update: {medoids}")

        # --- Step 5: update memberships ---
        U_new = update_membership_matrix(nodes, medoids, S, kappa)

        # --- Convergence and diagnostic info ---
        diff = sum(abs(U_new[i][p] - U_prev[i][p]) for i in nodes for p in range(k))
        if prev_diff is not None:
            print(f"[DEBUG] ΔU change: {diff:.6f} (previous {prev_diff:.6f})")
        else:
            print(f"[DEBUG] ΔU change: {diff:.6f}")
        prev_diff = diff

        # --- Compute correlation between membership columns ---
        U_mat = np.array([[U_new[i][p] for p in range(k)] for i in nodes])
        corr = np.corrcoef(U_mat.T)
        mean_corr = np.mean(corr[np.triu_indices_from(corr, 1)])
        print(f"[DEBUG] Mean inter-cluster corr: {mean_corr:.4f}")

        # --- Compute total objective value ---
        total_obj = sum(
            (U_new[i][p] ** 2) * get_symmetric_value(S, i, medoids[p])
            for i in nodes for p in range(k)
        )
        print(f"[DEBUG] Objective value: {total_obj:.2f}")

        # --- Duplicate medoid check ---
        medoid_counts = {m: medoids.count(m) for m in medoids}
        duplicate_meds = [m for m, c in medoid_counts.items() if c > 1]
        if duplicate_meds:
            print(f"[DEBUG] WARNING: duplicate medoids {duplicate_meds}")

        # --- Convergence criterion ---
        if diff < epsilon:
            print(f"[DEBUG] Converged after {iteration + 1} iterations.")
            break

        U_prev = U_new

    # --- Final cluster feature vectors τ_P ---
    τ_P = compute_cluster_features(U_prev, features)

    print(f"\n[DEBUG] Final medoids: {medoids}")
    return U_prev, medoids, τ_P




# ============================================================
# --- Standalone Test ----------------------------------------
# ============================================================

if __name__ == "__main__":
    instance_name = "X-n101-k25.vrp"
    k = 5
    U, medoids, τ_P = fuzzy_c_medoids(instance_name, k)

    print(f"\n✅ Fuzzy C-Medoids finished for {instance_name}")
    print(f"Medoids: {medoids}\n")

    # Print cluster summary
    for p, m in enumerate(medoids):
        memberships = [(n, round(U[n][p], 3)) for n in list(U.keys())[:5]]
        τ = τ_P[p]
        print(f"Cluster {p+1}: medoid {m}")
        print(f"  τ_P = (x̄={τ[0]:.2f}, ȳ={τ[1]:.2f}, θ̄={τ[2]:.3f}, q̄={τ[3]:.2f})")
        print(f"  Example memberships: {memberships}\n")
