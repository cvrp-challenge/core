# src/master/routing/routing_controller.py
"""
Routing controller: glue between clustering and a routing backend.

Current behaviour:
    - Default backend is PyVRP.
    - Solves each cluster as a subproblem (depot + cluster customers).
    - Aggregates the cluster solutions into one Result-like object for the rest
      of the pipeline.

Goal of this controller:
    - Provide ONE stable API (solve_clusters) that can dispatch to different
      routing engines (pyvrp, hexaly, future FILO1/2, ...).

Important:
    - We keep the existing PyVRP cluster subproblem implementation exactly as-is
      (to preserve behaviour and performance).
    - For non-PyVRP solvers, we dispatch via master.routing.solver.solve.
      Those solver adapters are expected to return routes in VRPLIB format
      via SolveOutput.metadata["routes_vrplib"].
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Mapping, Optional

import numpy as np

from master.utils.loader import load_instance

# ---------------------------------------------------------------------------
# PyVRP import (works both if pip-installed or only as git submodule in solver/pyvrp)
# ---------------------------------------------------------------------------

try:
    from pyvrp import Model, solve, Solution, Result, Statistics  # type: ignore
    from pyvrp.stop import MaxRuntime  # type: ignore
except ImportError:
    CURRENT_DIR = os.path.dirname(__file__)
    CORE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
    PYVRP_DIR = os.path.join(CORE_ROOT, "solver", "pyvrp")
    if PYVRP_DIR not in sys.path:
        sys.path.append(PYVRP_DIR)

    from pyvrp import Model, solve, Solution, Result, Statistics  # type: ignore
    from pyvrp.stop import MaxRuntime  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers (PyVRP cluster model)
# ---------------------------------------------------------------------------

def _build_cluster_model(
    instance: Dict[str, Any],
    cluster_nodes_vrplib: List[int],
) -> Tuple[Model, List[int]]:
    """
    Build a PyVRP Model for a single cluster subproblem.

    cluster_nodes_vrplib are VRPLIB node IDs (customers only), depot=1.
    Depot is added automatically.
    """
    coords = instance["node_coord"]
    demands = instance["demand"]
    capacity = int(instance["capacity"])
    edge_mat = instance.get("edge_weight")
    depot_idx0 = int(instance["depot"][0])  # usually 0

    # Clean & sort cluster nodes, drop depot if present
    cluster_customers = sorted({nid for nid in cluster_nodes_vrplib if nid != depot_idx0 + 1})
    cluster_idx0 = [nid - 1 for nid in cluster_customers]

    # Node order in this subproblem: depot first, then customers
    idx0_nodes = [depot_idx0] + cluster_idx0
    location_to_node_id = [depot_idx0 + 1] + cluster_customers  # VRPLIB IDs

    m = Model()

    depot_coord = coords[depot_idx0]
    m.add_depot(x=float(depot_coord[0]), y=float(depot_coord[1]), name="depot")

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

    m.add_vehicle_type(num_available=max(1, len(cluster_customers)), capacity=capacity)

    locations = m.locations
    num_loc = len(locations)

    for i in range(num_loc):
        for j in range(num_loc):
            if i == j:
                continue

            orig_i = idx0_nodes[i]
            orig_j = idx0_nodes[j]

            if edge_mat is not None:
                dist = edge_mat[orig_i, orig_j]
                dist_int = int(round(float(dist)))
            else:
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
    Solve one cluster subproblem with PyVRP and return VRPLIB routes.
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
        return [], float("inf")

    routes_cluster: List[List[int]] = []
    for route in best.routes():
        loc_indices = list(route.visits())
        node_ids = [loc_to_node[idx] for idx in loc_indices]

        depot_id = loc_to_node[0]
        if not node_ids or node_ids[0] != depot_id:
            node_ids.insert(0, depot_id)
        if node_ids[-1] != depot_id:
            node_ids.append(depot_id)

        routes_cluster.append(node_ids)

    return routes_cluster, cost


# ---------------------------------------------------------------------------
# Unified model (for returning a PyVRP Result that downstream code already expects)
# ---------------------------------------------------------------------------

def _build_unified_model(instance: Dict[str, Any]) -> Tuple[Model, Dict[int, int]]:
    """
    Build a PyVRP Model for the entire instance (all customers).
    Returns (model, vrplib_to_pyvrp_location_index).
    """
    coords = instance["node_coord"]
    demands = instance["demand"]
    capacity = int(instance["capacity"])
    edge_mat = instance.get("edge_weight")
    depot_idx0 = int(instance["depot"][0])

    all_customers = list(range(2, len(demands) + 1))  # VRPLIB node IDs

    vrplib_to_pyvrp = {1: 0}
    for idx, customer_id in enumerate(all_customers, start=1):
        vrplib_to_pyvrp[customer_id] = idx

    m = Model()

    depot_coord = coords[depot_idx0]
    m.add_depot(x=float(depot_coord[0]), y=float(depot_coord[1]), name="depot")

    for customer_id in all_customers:
        idx0 = customer_id - 1
        xy = coords[idx0]
        demand = int(demands[idx0])
        m.add_client(
            x=float(xy[0]),
            y=float(xy[1]),
            delivery=demand,
            name=f"cust_{customer_id}",
        )

    m.add_vehicle_type(num_available=len(all_customers), capacity=capacity)

    locations = m.locations
    num_loc = len(locations)

    for i in range(num_loc):
        for j in range(num_loc):
            if i == j:
                continue

            orig_i = depot_idx0 if i == 0 else (all_customers[i - 1] - 1)
            orig_j = depot_idx0 if j == 0 else (all_customers[j - 1] - 1)

            if edge_mat is not None:
                dist = edge_mat[orig_i, orig_j]
                dist_int = int(round(float(dist)))
            else:
                dx = float(coords[orig_i, 0] - coords[orig_j, 0])
                dy = float(coords[orig_i, 1] - coords[orig_j, 1])
                dist_int = int(round((dx * dx + dy * dy) ** 0.5))

            m.add_edge(locations[i], locations[j], distance=dist_int)

    return m, vrplib_to_pyvrp


# ---------------------------------------------------------------------------
# Public API (solver-agnostic)
# ---------------------------------------------------------------------------

def solve_clusters(
    instance_name: str,
    clusters: Dict[int, List[int]],
    *,
    solver: str = "pyvrp",
    solver_options: Optional[Mapping[str, Any]] = None,
    time_limit_per_cluster: float = 10.0,
    seed: int = 0,
) -> Result:
    """
    Solve all clusters for an instance using the selected routing solver.

    - If solver="pyvrp": uses the existing in-file PyVRP cluster solve (unchanged).
    - Otherwise: dispatches to master.routing.solver.solve, expecting the adapter to
      return routes in VRPLIB format in SolveOutput.metadata["routes_vrplib"].

    Returns a PyVRP Result object containing the aggregated routes, so the rest of
    the pipeline remains unchanged.
    """
    solver_key = solver.lower()
    solver_options = dict(solver_options or {})

    instance = load_instance(instance_name)

    all_routes: List[List[int]] = []
    total_cost: float = 0.0
    total_runtime: float = 0.0
    cluster_costs: Dict[int, float] = {}

    # Lazy import to avoid circular deps / heavy imports in worker processes
    routing_solve = None
    if solver_key != "pyvrp":
        from master.routing.solver import solve as routing_solve  # type: ignore

    for cid, nodes in clusters.items():
        customers = [nid for nid in nodes if nid != 1]

        cluster_start = time.time()

        if solver_key == "pyvrp":
            routes_c, cost_c = _solve_cluster_with_pyvrp(
                instance,
                customers,
                time_limit=time_limit_per_cluster,
                seed=seed + cid,
            )
        else:
            # Contract for non-pyvrp adapters:
            # - They receive cluster_nodes in solver_options.
            # - They must return VRPLIB routes in SolveOutput.metadata["routes_vrplib"].
            per_cluster_opts = {
                **solver_options,
                "cluster_nodes": customers,
                "seed": seed + cid,
                "max_runtime": time_limit_per_cluster,
            }
            out = routing_solve(  # type: ignore[misc]
                instance=instance_name,
                solver=solver_key,
                solver_options=per_cluster_opts,
            )

            routes_c = out.metadata.get("routes_vrplib")
            if routes_c is None:
                raise RuntimeError(
                    f"Routing solver '{solver_key}' did not provide metadata['routes_vrplib']."
                )
            cost_c = float(out.cost)

        cluster_runtime = time.time() - cluster_start
        total_runtime += cluster_runtime

        all_routes.extend(routes_c)
        cluster_costs[cid] = float(cost_c)
        total_cost += float(cost_c)

    # Build unified model for downstream compatibility (Result/Solution)
    unified_model, _ = _build_unified_model(instance)
    unified_data = unified_model.data()

    # Convert VRPLIB routes to PyVRP "location indices" (same as your previous logic)
    routes_pyvrp: List[List[int]] = []
    for route_vrplib in all_routes:
        locations = [(nid - 1) for nid in route_vrplib if nid != 1]
        if locations:
            routes_pyvrp.append(locations)

    # Basic guard
    max_location = unified_data.num_locations - 1
    for idx, route in enumerate(routes_pyvrp):
        for visit in route:
            if visit < 1 or visit > max_location:
                raise ValueError(
                    f"Invalid location index {visit} in route {idx} (expected 1..{max_location})."
                )

    solution = Solution(unified_data, routes_pyvrp)
    stats = Statistics()

    result = Result(
        best=solution,
        stats=stats,
        num_iterations=len(clusters),
        runtime=total_runtime,
    )

    # Attach extras (same as before)
    result.cluster_costs = cluster_costs
    result.data = unified_data

    return result


# Backward compatibility: old name used by pipeline
solve_clusters_with_pyvrp = solve_clusters


if __name__ == "__main__":
    demo_clusters = {
        1: [2, 3, 4, 5],
        2: [6, 7, 8, 9],
    }
    res = solve_clusters(
        "X-n101-k25.vrp",
        demo_clusters,
        solver="pyvrp",
        time_limit_per_cluster=3.0,
        seed=0,
    )
    print("[routing] Example result:", res)
