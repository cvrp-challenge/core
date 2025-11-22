# src/master/dri/routing_controller.py
"""
Routing controller: glue between clustering and PyVRP.

Minimal prototype:
    1) Load instance via utils.loader.load_instance (vrplib dict).
    2) For each cluster (set of customers, VRPLIB indices, depot = 1):
        - Build a PyVRP model restricted to that cluster (+ depot).
        - Solve with PyVRP.
        - Extract routes in original node indices.
    3) Aggregate routes & cost.

Notes
-----
- Clusters are expected to contain **customer IDs in VRPLIB notation**:
    depot is node 1, customers are 2..DIMENSION.
  The depot (1) is **automatically** added to each cluster subproblem.
- Internally, vrplib uses 0-based indices:
    - node_coord[0] = depot (VRPLIB node 1),
    - node_coord[1] = customer 2, etc.
  This file takes care of that conversion.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np

from utils.loader import load_instance

# ---------------------------------------------------------------------------
# PyVRP import (works both if pip-installed or only as git submodule in solver/pyvrp)
# ---------------------------------------------------------------------------

try:
    from pyvrp import Model, solve
    from pyvrp.stop import MaxRuntime
except ImportError:
    # Try to import from project submodule: core/solver/pyvrp
    CURRENT_DIR = os.path.dirname(__file__)
    CORE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
    PYVRP_DIR = os.path.join(CORE_ROOT, "solver", "pyvrp")
    if PYVRP_DIR not in sys.path:
        sys.path.append(PYVRP_DIR)

    from pyvrp import Model, solve  # type: ignore
    from pyvrp.stop import MaxRuntime  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_cluster_model(
    instance: Dict[str, Any],
    cluster_nodes_vrplib: List[int],
) -> Tuple[Model, List[int]]:
    """
    Build a PyVRP Model for a single cluster.

    Parameters
    ----------
    instance
        vrplib instance dict (from load_instance).
    cluster_nodes_vrplib
        List of *VRPLIB node IDs* in this cluster (e.g. [4, 7, 29, ...]).
        These are customer indices; depot (1) is *not* required and is ignored
        if present.

    Returns
    -------
    model : pyvrp.Model
        PyVRP model restricted to depot + this cluster.
    location_to_node_id : list[int]
        Mapping from PyVRP location index -> original VRPLIB node ID.
        - location_to_node_id[0] = 1 (depot)
        - location_to_node_id[i] = customer node ID in this cluster, for i>=1.
    """
    coords = instance["node_coord"]            # shape (n,2), 0-based: 0 = depot
    demands = instance["demand"]              # shape (n,), 0-based
    capacity = int(instance["capacity"])
    edge_mat = instance.get("edge_weight")    # may be None if distances not computed
    depot_idx0 = int(instance["depot"][0])    # vrplib already made this 0-based (normally 0)

    # --------------------------------------------------------
    # Clean & sort cluster nodes (VRPLIB indices)
    # --------------------------------------------------------
    # Drop depot (1) if someone passed it by accident, and unique+sort.
    cluster_customers = sorted(
        {nid for nid in cluster_nodes_vrplib if nid != depot_idx0 + 1}
    )

    # VRPLIB node ID -> 0-based index in vrplib arrays
    #   VRPLIB node 1 -> idx0 = 0 (depot)
    #   VRPLIB node k -> idx0 = k-1
    cluster_idx0 = [nid - 1 for nid in cluster_customers]

    # Order of *original* nodes in this subproblem:
    #   first depot, then cluster customers in cluster_customers order
    idx0_nodes = [depot_idx0] + cluster_idx0
    location_to_node_id = [depot_idx0 + 1] + cluster_customers  # 1-based VRPLIB IDs

    # --------------------------------------------------------
    # Build PyVRP model
    # --------------------------------------------------------
    m = Model()

    # Add depot (location index 0)
    depot_coord = coords[depot_idx0]
    m.add_depot(x=float(depot_coord[0]), y=float(depot_coord[1]), name="depot")

    # Add clients (location indices 1..len(cluster_customers))
    for nid in cluster_customers:
        idx0 = nid - 1
        xy = coords[idx0]
        demand = int(demands[idx0])
        m.add_client(
            x=float(xy[0]),
            y=float(xy[1]),
            delivery=demand,
            name=f"cust_{nid}",
        )

    # Add vehicle type: enough vehicles so capacity is never the binding limit here.
    # You might tighten this later (e.g. global fleet split).
    # For now: upper bound = number of customers in this cluster.
    m.add_vehicle_type(num_available=max(1, len(cluster_customers)), capacity=capacity)

    # Add edges using original distance matrix if available;
    # otherwise compute Euclidean distances on the fly for this cluster.
    locations = m.locations  # [Depot, Client(cluster_customers[0]), ...]
    num_loc = len(locations)

    for i in range(num_loc):
        for j in range(num_loc):
            if i == j:
                continue

            orig_i = idx0_nodes[i]
            orig_j = idx0_nodes[j]

            if edge_mat is not None:
                dist = edge_mat[orig_i, orig_j]
                # PyVRP expects integer distances; vrplib gives float -> round.
                dist_int = int(round(float(dist)))
            else:
                # Fallback: compute Euclidean distance
                dx = float(coords[orig_i, 0] - coords[orig_j, 0])
                dy = float(coords[orig_i, 1] - coords[orig_j, 1])
                dist_int = int(round((dx * dx + dy * dy) ** 0.5))

            m.add_edge(locations[i], locations[j], distance=dist_int)

    return m, location_to_node_id


def _solve_cluster_with_pyvrp(
    instance: Dict[str, Any],
    cluster_nodes_vrplib: List[int],
    time_limit: float = 10.0,
    seed: int = 0,
) -> Tuple[List[List[int]], float]:
    """
    Solve one cluster subproblem with PyVRP.

    Parameters
    ----------
    instance
        vrplib instance dict (from load_instance).
    cluster_nodes_vrplib
        List of VRPLIB node IDs in this cluster (customers only).
    time_limit
        Per-cluster time limit in seconds for MaxRuntime.
    seed
        Random seed for PyVRP.

    Returns
    -------
    routes_cluster : list[list[int]]
        Routes for this cluster in original VRPLIB node IDs, including depot
        at start and end, e.g. [1, 4, 7, 29, 1].
    cost_cluster : float
        Objective value of the cluster solution (PyVRP's cost()).
        Will be +inf if infeasible or something went very wrong.
    """
    if not cluster_nodes_vrplib:
        return [], 0.0

    model, loc_to_node = _build_cluster_model(instance, cluster_nodes_vrplib)
    data = model.data()

    result = solve(
        data,
        stop=MaxRuntime(time_limit),
        seed=seed,
        collect_stats=False,
        display=False,
    )

    cost = float(result.cost())
    best = result.best

    if not result.is_feasible() or best is None or not np.isfinite(cost):
        # Infeasible cluster (should not really happen in CVRP),
        # but we fail gracefully.
        return [], float("inf")

    routes_cluster: List[List[int]] = []

    # Solution.routes() -> iterable of Route objects; Route.visits() gives
    # location indices (0 = depot here), see PyVRP docs.
    for route in best.routes():
        loc_indices = list(route.visits())  # e.g. [0, 3, 1, 0]

        # Map location indices back to VRPLIB node IDs, using loc_to_node.
        # loc_to_node[0] = depot (1), loc_to_node[i>=1] = cluster customer ID.
        node_ids = [loc_to_node[idx] for idx in loc_indices]

        # Just in case, enforce depot at start and end (VRPLIB node 1).
        depot_id = loc_to_node[0]
        if node_ids[0] != depot_id:
            node_ids.insert(0, depot_id)
        if node_ids[-1] != depot_id:
            node_ids.append(depot_id)

        routes_cluster.append(node_ids)

    return routes_cluster, cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_clusters_with_pyvrp(
    instance_name: str,
    clusters: Dict[int, List[int]],
    time_limit_per_cluster: float = 10.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    High-level entry point: solve all clusters of an instance with PyVRP.

    Parameters
    ----------
    instance_name
        File name of the instance, e.g. "X-n101-k25.vrp".
        Must be found by utils.loader.load_instance (x/xl folders).
    clusters
        Mapping cluster_id -> list of VRPLIB node IDs (customers) in that cluster.
        Example:
            {
                1: [4, 7, 29, 53],
                2: [3, 8, 9],
                ...
            }
        Depot (1) is automatically added internally.
    time_limit_per_cluster
        MaxRuntime in seconds for each cluster subproblem.
    seed
        Base random seed; per-cluster, we use seed + cluster_id.

    Returns
    -------
    result : dict
        {
            "routes": List[List[int]],   # all routes (global), VRPLIB node IDs
            "total_cost": float,        # sum of cluster costs
            "cluster_costs": Dict[int, float],  # per-cluster objective
        }
    """
    print(f"[routing] Loading instance '{instance_name}' ...")
    instance = load_instance(instance_name)

    all_routes: List[List[int]] = []
    total_cost: float = 0.0
    cluster_costs: Dict[int, float] = {}

    for cid, nodes in clusters.items():
        # Ensure we do not accidentally pass depot as a customer.
        customers = [nid for nid in nodes if nid != 1]

        print(
            f"[routing] Solving cluster {cid}: "
            f"{len(customers)} customers (VRPLIB IDs) ..."
        )

        routes_c, cost_c = _solve_cluster_with_pyvrp(
            instance,
            customers,
            time_limit=time_limit_per_cluster,
            seed=seed + cid,
        )

        print(
            f"[routing]   -> Cluster {cid}: "
            f"{len(routes_c)} routes, cost = {cost_c:.2f}"
        )

        all_routes.extend(routes_c)
        cluster_costs[cid] = cost_c
        total_cost += cost_c

    print(
        f"[routing] Done. Total routes: {len(all_routes)}, "
        f"aggregated cost = {total_cost:.2f}"
    )

    return {
        "routes": all_routes,
        "total_cost": total_cost,
        "cluster_costs": cluster_costs,
    }


if __name__ == "__main__":
    # Small manual smoke-test example, adjust as needed:
    #
    # Imagine you already ran clustering on X-n101-k25 and got something like:
    #   clusters = {1: [2,3,4], 2: [5,6,7], ...}
    #
    # You can quickly test integration by hardcoding a tiny example here.
    demo_clusters = {
        1: [2, 3, 4, 5],
        2: [6, 7, 8, 9],
    }
    res = solve_clusters_with_pyvrp(
        "X-n101-k25.vrp",
        demo_clusters,
        time_limit_per_cluster=3.0,
        seed=0,
    )
    print("[routing] Example result:", res)
