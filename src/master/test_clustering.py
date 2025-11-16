# D:\PS_CVRP\core\src\master\test_clustering.py

import os, sys

# --- Make sure Python finds all subpackages like utils, clustering, etc. ---
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from clustering.k_medoids import k_medoids
from clustering.fcm import fuzzy_c_medoids
# from clustering.avg_ac import agglomerative_clustering_average
# from clustering.max_ac import agglomerative_clustering_complete
from clustering.silhouette_coefficient import silhouette_coefficient


if __name__ == "__main__":
    # # Choose a small benchmark instance for testing
    # instance_name = "X-n101-k25.vrp"   # or e.g. "XLTEST-n1094-k6.vrp"
    # k = 5  # number of clusters

    # # --------------------------------------------------
    # # 1️⃣ K-MEDOIDS
    # # --------------------------------------------------
    # print(f"→ Running K-Medoids clustering on instance '{instance_name}' with k={k}")
    # clusters_kmedoids = k_medoids(instance_name, k)

    # print("\n✅ K-Medoids complete!")
    # print(f"Number of clusters: {len(clusters_kmedoids)}\n")

    # for idx, (medoid, members) in enumerate(clusters_kmedoids.items(), start=1):
    #     print(f"Cluster {idx}:")
    #     print(f"  Medoid: {medoid}")
    #     print(f"  Members: {len(members)} customers")
    #     print(f"  All members: {members}\n")

    # # --------------------------------------------------
    # # 2️⃣ FUZZY C-MEDOIDS
    # # --------------------------------------------------
    # print(f"→ Running Fuzzy C-Medoids clustering on instance '{instance_name}' with k={k}")
    # U, medoids_fcm, τ_P = fuzzy_c_medoids(instance_name, k)

    # print("\n✅ Fuzzy C-Medoids complete!")
    # print(f"Number of clusters: {len(medoids_fcm)}\n")

    # for p, m in enumerate(medoids_fcm):
    #     assigned = [i for i in U.keys() if max(U[i], key=U[i].get) == p]
    #     avg_membership = sum(U[i][p] for i in U.keys()) / len(U)
    #     top5 = sorted(U.items(), key=lambda x: x[1][p], reverse=True)[:5]

    #     τ = τ_P[p]
    #     print(f"Cluster {p+1}:")
    #     print(f"  Medoid: {m}")
    #     print(f"  Size (argmax-membership customers): {len(assigned)}")
    #     print(f"  Avg membership μ̄: {avg_membership:.3f}")
    #     print(f"  τ_P = (x̄={τ[0]:.2f}, ȳ={τ[1]:.2f}, θ̄={τ[2]:.3f}, q̄={τ[3]:.2f})")
    #     print(f"  Top 5 strongest members: {[(i, round(U[i][p],3)) for i,_ in top5]}\n")

    # # --------------------------------------------------
    # # 3️⃣ AGGLOMERATIVE CLUSTERING (AVERAGE LINKAGE)
    # # --------------------------------------------------
    # print(f"→ Running Agglomerative Clustering (Average Linkage) on instance '{instance_name}' with k={k}\n")
    # clusters_avgac, medoids_avgac = agglomerative_clustering_average(instance_name, k)

    # print("✅ Agglomerative Clustering (Average Linkage) complete!")
    # print(f"Number of clusters: {len(clusters_avgac)}\n")

    # for idx, (cid, members) in enumerate(clusters_avgac.items(), start=1):
    #     m = medoids_avgac[cid]
    #     print(f"Cluster {idx}:")
    #     print(f"  Medoid: {m}")
    #     print(f"  Size: {len(members)} customers")
    #     if len(members) > 10:
    #         preview = members[:10]
    #         print(f"  Example members: {preview} ...\n")
    #     else:
    #         print(f"  Members: {members}\n")

    # # --------------------------------------------------
    # # 4️⃣ AGGLOMERATIVE CLUSTERING (COMPLETE LINKAGE)
    # # --------------------------------------------------
    # print(f"→ Running Agglomerative Clustering (Complete Linkage) on instance '{instance_name}' with k={k}\n")
    # clusters_maxac, medoids_maxac = agglomerative_clustering_complete(instance_name, k)

    # print("✅ Agglomerative Clustering (Complete Linkage) complete!")
    # print(f"Number of clusters: {len(clusters_maxac)}\n")

    # for idx, (cid, members) in enumerate(clusters_maxac.items(), start=1):
    #     m = medoids_maxac[cid]
    #     print(f"Cluster {idx}:")
    #     print(f"  Medoid: {m}")
    #     print(f"  Size: {len(members)} customers")
    #     if len(members) > 10:
    #         print(f"  Example members: {members[:10]} ...\n")
    #     else:
    #         print(f"  Members: {members}\n")

    # # --------------------------------------------------
    # # 📊 SILHOUETTE COEFFICIENT COMPARISON
    # # --------------------------------------------------
    # print("\n=== Silhouette Coefficient Comparison ===")

    # # --- K-Medoids ---
    # clusters_dict_k = {}
    # medoids_dict_k = {}
    # for idx, (m, members) in enumerate(clusters_kmedoids.items(), start=1):
    #     clusters_dict_k[idx] = members
    #     medoids_dict_k[idx] = m
    # ζ_kmedoids = silhouette_coefficient(instance_name, clusters_dict_k, medoids_dict_k)
    # print(f"K-Medoids: ζ = {ζ_kmedoids:.4f}")

    # # --- Fuzzy C-Medoids ---
    # clusters_dict_fcm = {p + 1: [] for p in range(len(medoids_fcm))}
    # for i in U.keys():
    #     assigned_cluster = max(U[i], key=U[i].get)
    #     clusters_dict_fcm[assigned_cluster + 1].append(i)
    # medoids_dict_fcm = {p + 1: medoids_fcm[p] for p in range(len(medoids_fcm))}
    # ζ_fcm = silhouette_coefficient(instance_name, clusters_dict_fcm, medoids_dict_fcm)
    # print(f"Fuzzy C-Medoids: ζ = {ζ_fcm:.4f}")

    # # --- Agglomerative (Average Linkage) ---
    # ζ_avgac = silhouette_coefficient(instance_name, clusters_avgac, medoids_avgac)
    # print(f"Agglomerative (Average): ζ = {ζ_avgac:.4f}")

    # # --- Agglomerative (Complete Linkage) ---
    # ζ_maxac = silhouette_coefficient(instance_name, clusters_maxac, medoids_maxac)
    # print(f"Agglomerative (Complete): ζ = {ζ_maxac:.4f}")

    # print("\n✅ Silhouette Coefficient comparison complete.")
    from clustering.fcm_scikit_fuzzy import run_sklearn_fcm

    instance_name = "X-n101-k25.vrp"
    k = 5

    clusters, medoids, memberships, centroids = run_sklearn_fcm(
        instance_name,
        k,
        use_polar=True,
        use_demand=True
    )

    print("Clusters:", clusters)
    print("Medoids:", medoids)

    # Example: print membership of node 10
    print("Membership of node 10:", memberships[10])

