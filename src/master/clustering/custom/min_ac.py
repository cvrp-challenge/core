# clustering/min_ac.py
"""
Correct Single-Linkage (Minimum Linkage) Agglomerative Clustering
implemented via Minimum Spanning Tree (MST).

This follows the mathematically correct definition:
  1. Build MST from the dissimilarity graph
  2. Remove the (k-1) largest edges
  3. Connected components → clusters

Supports both spatial and combined dissimilarity.
"""

import math
from typing import Dict, List, Tuple, Optional
from master.utils.loader import load_instance
from master.utils.symmetric_matrix_read import get_symmetric_value
from master.clustering.dissimilarity.combined import combined_dissimilarity
from master.clustering.dissimilarity.spatial import spatial_dissimilarity


# ---------------------------------------------------------------------
# 1. Build MST using Prim's algorithm
# ---------------------------------------------------------------------

def build_mst(nodes: List[int], S: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float]]:
    """
    Build a Minimum Spanning Tree (MST) using Prim's algorithm.
    Returns a list of edges: (i, j, weight).
    """

    import heapq

    start = nodes[0]
    visited = set([start])

    edges = []          # MST edges
    heap = []           # (weight, i, j)

    # init heap
    for j in nodes:
        if j != start:
            heapq.heappush(heap, (get_symmetric_value(S, start, j), start, j))

    while heap and len(visited) < len(nodes):
        w, i, j = heapq.heappop(heap)
        if j in visited:
            continue
        visited.add(j)
        edges.append((i, j, w))

        for k in nodes:
            if k not in visited:
                heapq.heappush(heap, (get_symmetric_value(S, j, k), j, k))

    return edges


# ---------------------------------------------------------------------
# 2. Convert MST to clusters by cutting the k-1 largest edges
# ---------------------------------------------------------------------

def mst_to_clusters(nodes: List[int], mst_edges: List[Tuple[int, int, float]], k: int) -> Dict[int, List[int]]:
    """
    Convert MST edges to clusters by cutting the (k-1) largest edges.
    """

    # Sort edges by weight descending → candidate edges to cut
    edges_sorted = sorted(mst_edges, key=lambda e: e[2], reverse=True)

    # cut k-1 largest edges
    edges_to_keep = edges_sorted[k-1:]

    # Build adjacency list
    adj = {n: [] for n in nodes}
    for i, j, _ in edges_to_keep:
        adj[i].append(j)
        adj[j].append(i)

    # BFS/DFS to extract connected components
    visited = set()
    clusters = {}
    cid = 1

    for start in nodes:
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        cluster = []

        while stack:
            u = stack.pop()
            cluster.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        clusters[cid] = sorted(cluster)
        cid += 1

    return clusters


# ---------------------------------------------------------------------
# 3. Compute medoids
# ---------------------------------------------------------------------

def compute_cluster_medoids(clusters: Dict[int, List[int]], S: Dict[Tuple[int, int], float]) -> Dict[int, int]:
    """Medoid = argmin_i Σ_j S_ij inside cluster."""
    medoids = {}
    for cid, members in clusters.items():
        best_node = None
        best_val = float("inf")
        for i in members:
            v = sum(get_symmetric_value(S, i, j) for j in members if j != i)
            if v < best_val:
                best_val = v
                best_node = i
        medoids[cid] = best_node
    return medoids


# ---------------------------------------------------------------------
# 4. Main interface
# ---------------------------------------------------------------------

def agglomerative_clustering_min(
    instance_name: str,
    k: int,
    instance: Optional[dict] = None,
    use_combined: bool = False
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Correct Single-Linkage AC via MST-based clustering.
    """

    if instance is None:
        instance = load_instance(instance_name)

    # choose dissimilarity
    S = combined_dissimilarity(instance_name) if use_combined else spatial_dissimilarity(instance_name)

    nodes = sorted({n for pair in S.keys() for n in pair})

    # build MST
    mst_edges = build_mst(nodes, S)

    # convert MST to clusters
    clusters = mst_to_clusters(nodes, mst_edges, k)

    # compute medoids
    medoids = compute_cluster_medoids(clusters, S)

    return clusters, medoids


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    inst = "X-n101-k25.vrp"
    clusters, medoids = agglomerative_clustering_min(inst, k=5)

    print("✅ Correct Single-Linkage AC complete\n")
    for cid, members in clusters.items():
        print(f"Cluster {cid}: size={len(members)}, medoid={medoids[cid]}")
