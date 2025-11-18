import os, sys

# Ensure module paths
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# ---- Custom clustering imports ----
from clustering.k_medoids import k_medoids
from clustering.fcm import fuzzy_c_medoids
from clustering.ac_custom.avg_ac import agglomerative_clustering_average
from clustering.ac_custom.max_ac import agglomerative_clustering_complete
from clustering.ac_custom.min_ac import agglomerative_clustering_min

# ---- Sklearn clustering imports ----
from clustering.scikit_clustering import (
    run_sklearn_ac,
    run_sklearn_kmeans
)

from clustering.k_medoids_pyclustering import k_medoids_pyclustering


# ---- Scikit-fuzzy FCM ----
from clustering.fcm_scikit_fuzzy import run_sklearn_fcm

# ---- Silhouette ----
from clustering.silhouette_coefficient import silhouette_coefficient


# ===================================================================
#                         TEST CONFIGURATION
# ===================================================================

instance_name = "X-n101-k25.vrp"     # small instance
k = 5                                 # number of clusters


# ===================================================================
#                       HELPER PRINT FUNCTION
# ===================================================================

def print_clusters(title, clusters, medoids):
    print(f"\n=== {title} ===")
    print(f"Clusters: {len(clusters)}")
    for cid, members in clusters.items():
        print(f"  Cluster {cid}: size={len(members)}, medoid={medoids[cid]}")


# ===================================================================
#                         RUN ALL METHODS
# ===================================================================

if __name__ == "__main__":

    # =========================
    # 1. K-MEDOIDS (CUSTOM)
    # =========================
    print("Running custom K-Medoids...")

    clusters_km, medoids_km = k_medoids(instance_name, k)

    print("\n=== K-Medoids ===")
    print(f"Clusters: {len(clusters_km)}")
    for cid, members in clusters_km.items():
        print(f"  Cluster {cid}: size={len(members)}, medoid={medoids_km[cid]}")

    ζ_km = silhouette_coefficient(instance_name, clusters_km, medoids_km)
    print(f"Silhouette: {ζ_km:.4f}")



    # =========================
    # 2. FCM (CUSTOM)
    # =========================
    print("Running custom Fuzzy C-Medoids...")

    clusters_fcm, medoids_fcm, tau_fcm = fuzzy_c_medoids(instance_name, k)

    print("\n=== Fuzzy C-Medoids ===")
    print(f"Clusters: {len(clusters_fcm)}")

    for cid, members in clusters_fcm.items():
        print(f"  Cluster {cid}: size={len(members)}, medoid={medoids_fcm[cid]}")

    ζ_fcm = silhouette_coefficient(instance_name, clusters_fcm, medoids_fcm)
    print(f"Silhouette: {ζ_fcm:.4f}")


    # =========================
    # 3. Agglomerative (CUSTOM)
    # =========================
    print("\nRunning custom Average AC...")
    clusters_avg, medoids_avg = agglomerative_clustering_average(instance_name, k)
    print_clusters("Custom AC - Average", clusters_avg, medoids_avg)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_avg, medoids_avg):.4f}")

    print("\nRunning custom Complete AC...")
    clusters_comp, medoids_comp = agglomerative_clustering_complete(instance_name, k)
    print_clusters("Custom AC - Complete", clusters_comp, medoids_comp)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_comp, medoids_comp):.4f}")

    print("\nRunning custom Min AC...")
    clusters_min, medoids_min = agglomerative_clustering_min(instance_name, k)
    print_clusters("Custom AC - Min", clusters_min, medoids_min)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_min, medoids_min):.4f}")


    # =========================
    # 4. Agglomerative (SKLEARN)
    # =========================
    print("\nRunning sklearn AC - Average...")
    clusters_savg, medoids_savg = run_sklearn_ac(instance_name, k, linkage="average")
    print_clusters("Sklearn AC - Average", clusters_savg, medoids_savg)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_savg, medoids_savg):.4f}")

    print("\nRunning sklearn AC - Complete...")
    clusters_scomp, medoids_scomp = run_sklearn_ac(instance_name, k, linkage="complete")
    print_clusters("Sklearn AC - Complete", clusters_scomp, medoids_scomp)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_scomp, medoids_scomp):.4f}")

    print("\nRunning sklearn AC - Single...")
    clusters_ssing, medoids_ssing = run_sklearn_ac(instance_name, k, linkage="single")
    print_clusters("Sklearn AC - Single", clusters_ssing, medoids_ssing)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_ssing, medoids_ssing):.4f}")


    # =========================
    # 5. K-MEANS (SKLEARN)
    # =========================
    print("\nRunning sklearn K-Means...")
    clusters_kmeans, medoids_kmeans, centroids = run_sklearn_kmeans(instance_name, k)
    print_clusters("Sklearn K-Means", clusters_kmeans, medoids_kmeans)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_kmeans, medoids_kmeans):.4f}")


    # =========================
    # 6. FCM (SCIKIT FUZZY)
    # =========================
    print("\nRunning scikit-fuzzy FCM...")
    clusters_fcm_sf, medoids_fcm_sf, memberships_sf, centroids_sf = run_sklearn_fcm(instance_name, k)
    print_clusters("Scikit-Fuzzy FCM", clusters_fcm_sf, medoids_fcm_sf)
    print(f"Silhouette: {silhouette_coefficient(instance_name, clusters_fcm_sf, medoids_fcm_sf):.4f}")


    print("\n=== All clustering methods tested successfully ===")

    # ---------------------------------------------------------
    # 7. pyclustering K-Medoids (distance-matrix based)
    # ---------------------------------------------------------
    print("\nRunning pyclustering K-Medoids...")

    clusters_km_py = k_medoids_pyclustering(instance_name, k, use_combined=False)

    # convert {medoid: [members]} → {cluster_id: [members]}, {cluster_id: medoid}
    clusters_py = {}
    medoids_py = {}
    for cid, (m, members) in enumerate(clusters_km_py.items(), start=1):
        clusters_py[cid] = members
        medoids_py[cid] = m

    print_clusters("pyclustering K-Medoids", clusters_py, medoids_py)
    sil_py = silhouette_coefficient(instance_name, clusters_py, medoids_py)
    print(f"Silhouette: {sil_py:.4f}")
