# master/improve/ls_controller.py
"""
Local search controller using PyVRP's LocalSearch.

Responsibilities:
  - Build a full PyVRP ProblemData for the original instance.
  - Convert VRPLIB-style routes (node IDs, depot = 1) to a PyVRP Solution.
  - Build a granular neighbourhood from our own dissimilarity metric
    (spatial or combined).
  - Run PyVRP LocalSearch and convert the improved solution back to
    VRPLIB-style routes.

Assumptions:
  - Single-depot CVRP in standard VRPLIB format.
  - Depot node ID is 1, customers are 2..n+1.
  - PyVRP's VRPLIB reader is used to keep distances consistent.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Literal, Optional

from pyvrp import (
    read,                      # read(instance_path, "vrplib")
    CostEvaluator,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.search import (
    LocalSearch,
    NODE_OPERATORS,
    ROUTE_OPERATORS,
)

from master.utils.loader import load_instance
from master.clustering.dissimilarity.spatial import spatial_dissimilarity
from master.clustering.dissimilarity.combined import combined_dissimilarity
from master.utils.symmetric_matrix_read import get_symmetric_value


# ======================================================================
# Helpers: resolve instance path for PyVRP.read
# ======================================================================

def _resolve_instance_path(instance_name: str) -> str:
    """
    Reconstructs the path to the instance file, consistent with loader.py:
      core/instances/test-instances/x
      core/instances/test-instances/xl
    """
    base_dir = os.path.dirname(__file__)                 # .../core/src/master/improve
    core_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    instances_root = os.path.join(core_root, "instances", "test-instances")

    for sub in ("x", "xl"):
        p = os.path.join(instances_root, sub, instance_name)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Instance '{instance_name}' not found in: "
        f"[{os.path.join(instances_root, 'x')}, {os.path.join(instances_root, 'xl')}]"
    )


# ======================================================================
# Helpers: neighbourhood from DRI dissimilarities
# ======================================================================

def _build_dri_neighbours(
    instance_name: str,
    num_locations: int,
    use_demand: bool = False,
    max_neighbours: int = 40,
) -> List[List[int]]:
    """
    Build a granular neighbourhood structure for PyVRP LocalSearch based on
    our own dissimilarity metric.

    PyVRP expects:
        neighbours: list[list[int]] of length = num_locations
    where:
        - lower indices correspond to depots and are empty;
        - remaining indices correspond to client locations.

    Our VRPLIB convention:
        node 1  -> depot  -> location 0
        node i  -> client -> location (i-1), for i >= 2

    We compute:
        - S^s_ij  (spatial)        if use_demand = False
        - S^sd_ij (combined)       if use_demand = True
    and, for each customer i, keep the max_neighbours smallest S_ij.

    The neighbour lists are expressed in terms of *location indices*
    (0..num_locations-1) as required by PyVRP.
    """
    # Choose which S to use
    if use_demand:
        S = combined_dissimilarity(instance_name)
    else:
        S = spatial_dissimilarity(instance_name)

    # Collect all customer node IDs appearing in S (should be 2..n+1)
    nodes = sorted({n for pair in S.keys() for n in pair})

    neighbours: List[List[int]] = [[] for _ in range(num_locations)]

    # For each customer node i, pick the closest neighbours in S
    for i in nodes:
        # build list of (dissimilarity, j)
        cand: List[Tuple[float, int]] = []
        for j in nodes:
            if j == i:
                continue
            key = (i, j) if i < j else (j, i)
            dij = get_symmetric_value(S, i, j)
            cand.append((dij, j))

        cand.sort(key=lambda x: x[0])
        selected = [j for _, j in cand[:max_neighbours]]

        loc_i = i - 1           # VRPLIB node 2 -> loc 1, etc.
        neighbours[loc_i] = [j - 1 for j in selected]   # map to loc indices

    # depot (location 0) remains neighbours[0] = []
    return neighbours


# ======================================================================
# Conversion between VRPLIB routes and PyVRP Solution
# ======================================================================

def _vrplib_routes_to_solution(data, routes_vrplib):
    routes_clients = []

    for r in routes_vrplib:
        # convert VRPLIB nodes to location indices
        clients = [(nid - 1) for nid in r if nid != 1]   # FIXED
        if clients:
            routes_clients.append(clients)

    return Solution(data, routes_clients)


def _solution_to_vrplib_routes(sol):
    vrp_routes = []

    for route in sol.routes():
        clients = list(route.visits())
        if not clients:
            continue
        seq = [1] + [(idx + 1) for idx in clients] + [1]   # FIXED
        vrp_routes.append(seq)

    return vrp_routes


# ======================================================================
# Public API: improve with LocalSearch
# ======================================================================

def improve_with_local_search(
    instance_name: str,
    routes_vrplib: List[List[int]],
    *,
    neighbourhood: Literal["dri_spatial", "dri_combined"] = "dri_spatial",
    max_neighbours: int = 40,
    seed: int = 0,
    load_penalty: int = 20,
    dist_penalty: int = 1,
) -> Dict[str, object]:
    """
    Run PyVRP's LocalSearch on top of an existing decomposition solution.

    Args:
        instance_name:
            Name of the VRPLIB instance (e.g. "X-n101-k25.vrp").
        routes_vrplib:
            List of routes in VRPLIB format, e.g. [[1, 2, 6, 1], [1, 3, 4, 1], ...].
        neighbourhood:
            "dri_spatial"  -> spatial dissimilarity S^s_ij
            "dri_combined" -> combined dissimilarity S^sd_ij (spatial + demand)
        max_neighbours:
            Maximum number of neighbours per client in the granular neighbourhood.
        seed:
            Seed for PyVRP's RandomNumberGenerator.
        load_penalty:
            Penalty coefficient for capacity violations (PyVRP CostEvaluator).
        dist_penalty:
            Distance scaling in the CostEvaluator.

    Returns:
        dict with:
            - "initial_cost"
            - "improved_cost"
            - "routes_initial"
            - "routes_improved"
            - "ls_moves"
            - "ls_improving_moves"
    """
    # --------------------------------------------------------------
    # 1) Read instance with PyVRP to get ProblemData
    # --------------------------------------------------------------
    instance_path = _resolve_instance_path(instance_name)
    data = read(instance_path)  # single-depot CVRP

    # Dimension check using vrplib loader (optional sanity)
    inst = load_instance(instance_name)
    dim = int(inst["dimension"])
    assert data.num_locations == dim, (
        f"PyVRP data.num_locations={data.num_locations} "
        f"!= VRPLIB DIMENSION={dim}"
    )

    # --------------------------------------------------------------
    # 2) Build CostEvaluator & RNG
    # --------------------------------------------------------------
    rng = RandomNumberGenerator(seed)
    cost_eval = CostEvaluator(
        load_penalties=[load_penalty],   # single load dimension
        tw_penalty=0,                    # no time windows in pure CVRP
        dist_penalty=dist_penalty,
    )

    # --------------------------------------------------------------
    # 3) Build granular neighbourhood based on DRI dissimilarity
    # --------------------------------------------------------------
    use_demand = (neighbourhood == "dri_combined")
    neighbours = _build_dri_neighbours(
        instance_name=instance_name,
        num_locations=data.num_locations,
        use_demand=use_demand,
        max_neighbours=max_neighbours,
    )

    # --------------------------------------------------------------
    # 4) Setup LocalSearch with default NODE/ROUTE operators
    # --------------------------------------------------------------
    ls = LocalSearch(data, rng, neighbours)

    for node_op in NODE_OPERATORS:
        ls.add_node_operator(node_op(data))

    for route_op in ROUTE_OPERATORS:
        ls.add_route_operator(route_op(data))

    # --------------------------------------------------------------
    # 5) Convert VRPLIB routes -> Solution, run LS, convert back
    # --------------------------------------------------------------
    sol_initial = _vrplib_routes_to_solution(data, routes_vrplib)
    initial_cost = cost_eval.penalised_cost(sol_initial)

    sol_improved = ls(sol_initial, cost_eval)
    improved_cost = cost_eval.penalised_cost(sol_improved)

    routes_improved = _solution_to_vrplib_routes(sol_improved)

    stats = ls.statistics  # LocalSearchStatistics

    return {
        "initial_cost": float(initial_cost),
        "improved_cost": float(improved_cost),
        "routes_initial": routes_vrplib,
        "routes_improved": routes_improved,
        "ls_moves": int(stats.num_moves),
        "ls_improving_moves": int(stats.num_improving),
        "ls_updates": int(stats.num_updates),
    }
