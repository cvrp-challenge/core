# benchmark_clustering_speed.py
"""
Benchmark the runtime of all clustering algorithms on a large instance.
Instance: XLTEST-n1421-k9.vrp

Produces:
 - Runtime per method
 - Cluster count
 - Optional silhouette score (DISABLED by default for speed)

Author: Robin & ChatGPT
"""

import os
import sys
import time

# Ensure correct import paths
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# --- Imports for all clustering methods ---
from clustering.k_medoids import k_medoids
from clustering.fcm import fuzzy_c_medoids
from clustering.ac_custom.avg_ac import agglomerative_clustering_average
from clustering.ac_custom.max_ac import agglomerative_clustering_complete
from clustering.ac_custom.min_ac import agglomerative_clustering_min
from clustering.scikit_clustering import run_sklearn_ac, run_sklearn_kmeans
from clustering.k_medoids_pyclustering import k_medoids_pyclustering
from clustering.silhouette_coefficient import silhouette_coefficient


# ============================================================
# Helpers
# ============================================================

def time_it(name, func, *args, **kwargs):
    """Measure wall-clock time of a method call."""
    print(f"\n>>> Running {name} ...")
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    runtime = t1 - t0
    print(f"{name} finished in {runtime:.3f} seconds.")
    return result, runtime


def print_summary(name, clusters, medoids):
    """Print quick summary of clustering results."""
    print(f"=== {name} ===")
    print(f"Clusters: {len(clusters)}")
    for cid, members in clusters.items():
        print(f"  Cluster {cid}: size={len(members)}, medoid={medoids[cid]}")
    print()


# ============================================================
# Benchmark Config
# ============================================================

INSTANCE = "XLTEST-n1421-k9.vrp"
K = 9

# For speed, silhouette is optional
COMPUTE_SILHOUETTE = False  # Set True if needed, but will be slow


# ============================================================
# Main Benchmark Script
# ============================================================

if __name__ == "__main__":

    results = []  # (name, runtime)

    # --------------------------------------------------------
    # 1) Custom K-Medoids
    # --------------------------------------------------------
    (clusters_km, _), t_km = time_it("Custom K-Medoids",
                                    k_medoids,
                                    INSTANCE, K, None, False)

    # Convert medoid → index format
    medoids_km = {i+1: m for i, (m, _) in enumerate(clusters_km.items())}

    # Convert medoid-keyed clusters to index-keyed clusters
    clusters_km_idx = {i+1: members for i, (_, members) in enumerate(clusters_km.items())}

    print_summary("Custom K-Medoids", clusters_km_idx, medoids_km)
    results.append(("Custom K-Medoids", t_km))

    # --------------------------------------------------------
    # 2) Custom Fuzzy C-Medoids
    # --------------------------------------------------------
    (clusters_fcm, medoids_fcm, τP_fcm), t_fcm = time_it(
        "Custom Fuzzy C-Medoids",
        fuzzy_c_medoids,
        INSTANCE, K, 2.0, 1e-4, 100, None, False
    )

    # clusters_fcm: {cluster_id → [node_ids]}
    # medoids_fcm : {cluster_id → medoid_node}
    print_summary("Custom Fuzzy C-Medoids", clusters_fcm, medoids_fcm)
    results.append(("Custom Fuzzy C-Medoids", t_fcm))



    # --------------------------------------------------------
    # 3) Custom Agglomerative Average
    # --------------------------------------------------------
    (clusters_avg, medoids_avg), t = time_it("Custom AC - Average",
                                             agglomerative_clustering_average,
                                             INSTANCE, K, None)
    print_summary("Custom AC - Average", clusters_avg, medoids_avg)
    results.append(("Custom AC - Average", t))

    # --------------------------------------------------------
    # 4) Custom Agglomerative Complete
    # --------------------------------------------------------
    (clusters_comp, medoids_comp), t = time_it("Custom AC - Complete",
                                               agglomerative_clustering_complete,
                                               INSTANCE, K, None)
    print_summary("Custom AC - Complete", clusters_comp, medoids_comp)
    results.append(("Custom AC - Complete", t))

    # --------------------------------------------------------
    # 5) Custom Agglomerative Min (Single Linkage)
    # --------------------------------------------------------
    (clusters_min, medoids_min), t = time_it("Custom AC - Min",
                                             agglomerative_clustering_min,
                                             INSTANCE, K, None)
    print_summary("Custom AC - Min", clusters_min, medoids_min)
    results.append(("Custom AC - Min", t))

    # --------------------------------------------------------
    # 6) sklearn AC (Average)
    # --------------------------------------------------------
    (clusters_savg, medoids_savg), t = time_it(
        "sklearn AC - Average",
        run_sklearn_ac,
        INSTANCE, K, "average", False, False, True, False)
    print_summary("sklearn AC - Average", clusters_savg, medoids_savg)
    results.append(("sklearn AC - Average", t))

    # --------------------------------------------------------
    # 7) sklearn AC (Complete)
    # --------------------------------------------------------
    (clusters_scomp, medoids_scomp), t = time_it(
        "sklearn AC - Complete",
        run_sklearn_ac,
        INSTANCE, K, "complete", False, False, True, False)
    print_summary("sklearn AC - Complete", clusters_scomp, medoids_scomp)
    results.append(("sklearn AC - Complete", t))

    # --------------------------------------------------------
    # 8) sklearn K-Means
    # --------------------------------------------------------
    (clusters_skm, medoids_skm, _), t = time_it(
        "sklearn K-Means",
        run_sklearn_kmeans,
        INSTANCE, K, True, False)
    print_summary("sklearn K-Means", clusters_skm, medoids_skm)
    results.append(("sklearn K-Means", t))

    # --------------------------------------------------------
    # 9) pyclustering K-Medoids (⚠ heavy: distance matrix)
    # --------------------------------------------------------
    # Safe for ~1000 nodes; for XL you can skip
    if len(clusters) < 2000:  # simple guard
        clusters_py = None
        try:
            clusters_py, t = time_it("pyclustering K-Medoids",
                                     k_medoids_pyclustering,
                                     INSTANCE, K, None, False)
            medoids_py = {i + 1: m for i, (m, _) in enumerate(clusters_py.items())}
            clusters_py_num = {i + 1: members for i, (_, members) in enumerate(clusters_py.items())}
            print_summary("pyclustering K-Medoids", clusters_py_num, medoids_py)
            results.append(("pyclustering K-Medoids", t))
        except MemoryError:
            print("pyclustering K-Medoids FAILED (distance matrix too large)")
    else:
        print("\nSkipping pyclustering K-Medoids (instance too large).")

    # ============================================================
    # Print ranking
    # ============================================================
    print("\n=========== RUNTIME SUMMARY ===========")
    for name, rt in sorted(results, key=lambda x: x[1]):
        print(f"{name:28s} : {rt:8.3f} s")

    print("\nDone.")
