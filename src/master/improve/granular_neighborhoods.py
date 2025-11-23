# granular_neighborhoods.py
"""
Granular neighborhoods Φ_i for large-scale VRP local search.

This module builds per-customer neighbor lists based on your existing
dissimilarity matrices:

    - spatial_dissimilarity(instance_name, instance)
    - combined_dissimilarity(instance_name)

We *do not* change the dissimilarity computation itself. Instead, we:
    1) Load S_ij (spatial or combined).
    2) Normalize S_ij to [0, 1] via min–max.
    3) For each customer i, pick the φ most similar customers j
       (smallest normalized S_ij).

This can be used as a "granular neighborhood" structure for local search,
similar to the data-based LS in Kerscher's DRI / DRSCI frameworks.

Design notes:
-------------
- Depot (node 1) is automatically ignored, because your dissimilarities
  are already defined only on customers (2..n).
- FCM-specific behavior (borderline vertices, etc.) is handled *outside*.
  Here you can pass:
      focus_nodes: nodes for which to build neighborhoods (i's)
      candidate_nodes: nodes allowed as neighbors (j's)
  So FCM code can pass its "borderline" set as focus_nodes, for example.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Optional, Literal
import math
import heapq

from master.clustering.dissimilarity.spatial import spatial_dissimilarity
from master.clustering.dissimilarity.combined import combined_dissimilarity
from utils.symmetric_matrix_read import get_symmetric_value


DissimilarityDict = Dict[Tuple[int, int], float]
NeighborsDict = Dict[int, List[int]]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_dissimilarity(S: DissimilarityDict) -> DissimilarityDict:
    """
    Min–max normalization of S_ij to [0, 1].

    Input S only stores i < j; output has the same structure.

    If all values in S are equal, all normalized values are set to 0.0.
    """
    if not S:
        return {}

    vals = list(S.values())
    smin = min(vals)
    smax = max(vals)

    if math.isclose(smin, smax):
        # All distances identical → normalized to 0.
        return {key: 0.0 for key in S.keys()}

    scale = 1.0 / (smax - smin)
    S_norm: DissimilarityDict = {}
    for (i, j), val in S.items():
        S_norm[(i, j)] = (val - smin) * scale

    return S_norm


# ---------------------------------------------------------------------------
# Granular neighborhoods builder
# ---------------------------------------------------------------------------

def _extract_nodes(S: DissimilarityDict) -> List[int]:
    """
    Extract sorted list of all node IDs that appear in S.
    (Depot is not present if S is built as in your current code.)
    """
    nodes = set()
    for i, j in S.keys():
        nodes.add(i)
        nodes.add(j)
    return sorted(nodes)


def build_granular_neighborhoods(
    instance_name: str,
    phi: int,
    mode: Literal["spatial", "combined"] = "spatial",
    S: Optional[DissimilarityDict] = None,
    focus_nodes: Optional[Iterable[int]] = None,
    candidate_nodes: Optional[Iterable[int]] = None,
) -> NeighborsDict:
    """
    Build granular neighborhoods Φ_i for all or some customers.

    Parameters
    ----------
    instance_name : str
        VRPLIB instance name (e.g. "X-n101-k25.vrp").
        Only used if S is not provided; then we compute it via your
        spatial/combined dissimilarity functions.
    phi : int
        Neighborhood size: for each i, we keep up to φ nearest neighbors.
    mode : {"spatial", "combined"}
        Which dissimilarity to use if S is None:
            "spatial"  → spatial_dissimilarity(instance_name)
            "combined" → combined_dissimilarity(instance_name)
    S : dict[(int,int), float], optional
        Optional precomputed dissimilarity dictionary. If given, this
        overrides `mode` and `instance_name` for the dissimilarity.
    focus_nodes : iterable[int], optional
        Nodes i for which we want neighborhoods Φ_i. If None, build Φ_i
        for all nodes in S.
    candidate_nodes : iterable[int], optional
        Nodes allowed as neighbors j. If None, all nodes in S can be
        candidates (except i itself).

    Returns
    -------
    neighbors : dict[int, list[int]]
        neighbors[i] = list of up to φ nearest neighbor node IDs, sorted
        from closest to farthest (in terms of normalized dissimilarity).

    Usage examples
    --------------
    # 1) Simple spatial neighborhoods, all nodes
    neighbors = build_granular_neighborhoods("X-n101-k25.vrp", phi=20)

    # 2) Combined spatial + demand, all nodes
    neighbors = build_granular_neighborhoods("X-n101-k25.vrp", phi=20, mode="combined")

    # 3) FCM-guided: only build Φ_i for borderline nodes
    borderline_nodes = [...]  # computed outside using memberships
    neighbors = build_granular_neighborhoods(
        "X-n101-k25.vrp",
        phi=20,
        mode="spatial",
        focus_nodes=borderline_nodes,
    )
    """
    # --------------------------------------------
    # 1) Obtain dissimilarity S_ij
    # --------------------------------------------
    if S is None:
        if mode == "spatial":
            S_raw = spatial_dissimilarity(instance_name)
        elif mode == "combined":
            S_raw = combined_dissimilarity(instance_name)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'spatial' or 'combined'.")
    else:
        S_raw = S

    if not S_raw:
        return {}

    # --------------------------------------------
    # 2) Normalize S_ij to [0, 1]
    # --------------------------------------------
    S_norm = normalize_dissimilarity(S_raw)

    # --------------------------------------------
    # 3) Determine node sets
    # --------------------------------------------
    all_nodes = _extract_nodes(S_norm)  # customers only, sorted

    if focus_nodes is None:
        focus = all_nodes
    else:
        focus = [i for i in focus_nodes if i in all_nodes]

    if candidate_nodes is None:
        cand = set(all_nodes)
    else:
        cand = {j for j in candidate_nodes if j in all_nodes}

    # --------------------------------------------
    # 4) Build Φ_i for each i in focus
    # --------------------------------------------
    neighbors: NeighborsDict = {}
    phi = max(0, int(phi))

    for i in focus:
        # Candidate list: all j != i that are in candidate set
        candidates: List[Tuple[float, int]] = []

        for j in cand:
            if j == i:
                continue

            # Use get_symmetric_value to read S_norm(i,j) from (i<j or j<i)
            dij = get_symmetric_value(S_norm, i, j)
            candidates.append((dij, j))

        if not candidates or phi == 0:
            neighbors[i] = []
            continue

        # We want the φ smallest dissimilarities (closest neighbors).
        # Use nsmallest for efficiency: O(n log φ) per node.
        smallest = heapq.nsmallest(phi, candidates, key=lambda t: t[0])
        # store only the node IDs, sorted by distance
        neighbors[i] = [j for (_d, j) in smallest]

    return neighbors


if __name__ == "__main__":
    # Simple manual test
    inst_name = "X-n101-k25.vrp"
    phi_test = 5

    print(f"[granular] Building spatial neighborhoods for {inst_name}, φ={phi_test} ...")
    neigh = build_granular_neighborhoods(inst_name, phi=phi_test, mode="spatial")
    first_items = list(neigh.items())[:5]
    for i, nbrs in first_items:
        print(f"  Node {i}: neighbors = {nbrs}")
