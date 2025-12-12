# clustering/k_medoids_pyclustering.py
"""
K-Medoids via pyclustering
--------------------------
Wrapper around pyclustering.cluster.kmedoids using your dissimilarity matrices.

Supports:
  - spatial or combined dissimilarity
  - same output format as custom k_medoids:
        {medoid_node_id: [member_node_ids]}
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

from master.utils.loader import load_instance
from master.utils.symmetric_matrix_read import get_symmetric_value
from master.clustering.dissimilarity.spatial import spatial_dissimilarity
from master.clustering.dissimilarity.combined import combined_dissimilarity
from master.clustering.custom.k_medoids import initialize_medoids, k_medoids


def _build_distance_matrix(
    instance_name: str,
    use_combined: bool = False,
) -> Tuple[np.ndarray, List[int], Dict[Tuple[int, int], float]]:
    """
    Build dense NxN distance matrix D from your dissimilarity dict S.

    Returns:
        D        : NxN numpy array (distance matrix)
        node_ids : list of node ids in the same order as D's indices
        S        : original dissimilarity dict
    """
    if use_combined:
        S = combined_dissimilarity(instance_name)
    else:
        S = spatial_dissimilarity(instance_name)

    # Use exactly the same node ordering as your other clustering methods
    node_ids = sorted({n for pair in S.keys() for n in pair})
    n = len(node_ids)

    D = np.zeros((n, n), dtype=float)
    for i_idx, i in enumerate(node_ids):
        for j_idx, j in enumerate(node_ids):
            if i == j:
                D[i_idx, j_idx] = 0.0
            else:
                D[i_idx, j_idx] = get_symmetric_value(S, i, j)

    return D, node_ids, S


def k_medoids_pyclustering(
    instance_name: str,
    k: int,
    instance: Optional[dict] = None,
    use_combined: bool = False,
    use_custom_init: bool = True,
) -> Dict[int, List[int]]:
    """
    Run K-Medoids using pyclustering on your dissimilarity matrix.

    Args:
        instance_name : name of the VRP instance (e.g. 'X-n101-k25.vrp')
        k             : number of clusters
        instance      : optional preloaded instance (not used, for symmetry)
        use_combined  : if True, use combined dissimilarity, else spatial
        use_custom_init:
            - True: use your far-apart initialize_medoids(...)
            - False: use pyclustering's medoid indices as given

    Returns:
        clusters_by_medoid: {medoid_node_id: [member_node_ids]}
                            (medoid NOT included in the member list)
    """
    # Build distance matrix and node ordering
    D, node_ids, S = _build_distance_matrix(instance_name, use_combined=use_combined)
    n = len(node_ids)

    # --- Initial medoids (indices into D) ---
    if use_custom_init:
        # your initializer returns medoid *node ids*
        init_medoids_nodes = initialize_medoids(node_ids, S, k)
        init_medoids = [node_ids.index(m) for m in init_medoids_nodes]
    else:
        # simple fallback: first k nodes
        init_medoids = list(range(k))

    # --- Run pyclustering K-Medoids ---
    # data_type='distance_matrix' tells it D[i][j] is a distance
    kmed = k_medoids(D.tolist(), init_medoids, data_type='distance_matrix')
    kmed.process()

    clusters_indices = kmed.get_clusters()   # list of lists of indices
    medoids_indices = kmed.get_medoids()     # list of medoid indices

    # --- Convert to {medoid_node_id: [member_node_ids]} ---
    clusters_by_medoid: Dict[int, List[int]] = {}

    # Map each cluster (by medoid index) to node IDs
    for m_idx, cluster_indices in zip(medoids_indices, clusters_indices):
        medoid_node = node_ids[m_idx]
        members = [node_ids[i] for i in cluster_indices if i != m_idx]
        clusters_by_medoid[medoid_node] = members

    return clusters_by_medoid


if __name__ == "__main__":
    # Simple manual test on a small instance
    inst = "X-n101-k25.vrp"
    k = 5
    clusters = k_medoids_pyclustering(inst, k, use_combined=False)

    print(f"✅ pyclustering K-Medoids finished for {inst}")
    print(f"Number of clusters: {len(clusters)}")
    for i, (m, members) in enumerate(clusters.items(), start=1):
        print(f"  Cluster {i}: medoid={m}, size={len(members) + 1}")
