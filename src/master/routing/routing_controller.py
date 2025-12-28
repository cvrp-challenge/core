# src/master/routing/routing_controller.py
"""
Routing controller: glue between clustering and a routing backend.

Behaviour:
    - Adaptive cluster runtime based on cluster size
    - PyVRP: pure time-limit stopping ONLY (no stagnation support)
    - Non-PyVRP solvers (e.g. Hexaly): receive both max_runtime and stall_time
    - Full transparency via debug prints for tuning
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Any, Mapping, Optional

import numpy as np

from master.utils.loader import load_instance

# ---------------------------------------------------------------------------
# PyVRP imports (NO stagnation stopping – confirmed unsupported)
# ---------------------------------------------------------------------------

from pyvrp import Model, solve, Solution, Result, Statistics  # type: ignore
from pyvrp.stop import MaxRuntime  # type: ignore


# ---------------------------------------------------------------------------
# Adaptive timing logic (THIS is what you tune)
# ---------------------------------------------------------------------------

def _adaptive_cluster_time(
    n: int,
    *,
    base: float = 1.0,
    alpha: float = 0.08,
    exponent: float = 0.85,
    min_time: float = 2.0,
    max_time: float = 120.0,
) -> float:
    if n <= 0:
        return min_time
    t = base + alpha * (n ** exponent)
    return max(min_time, min(max_time, t))


def _adaptive_stall_time(
    n_customers: int,
    *,
    ratio: float = 0.35,
    min_stall: float = 1.0,
) -> float:
    """
    Stall budget for solvers that support stagnation stopping (Hexaly).
    """
    return max(min_stall, ratio * _adaptive_cluster_time(n_customers))


# ---------------------------------------------------------------------------
# PyVRP cluster model
# ---------------------------------------------------------------------------

def _build_cluster_model(
    instance: Dict[str, Any],
    cluster_nodes_vrplib: List[int],
) -> Tuple[Model, List[int]]:

    coords = instance["node_coord"]
    demands = instance["demand"]
    capacity = int(instance["capacity"])
    edge_mat = instance.get("edge_weight")
    depot_idx0 = int(instance["depot"][0])

    cluster_customers = sorted({nid for nid in cluster_nodes_vrplib if nid != depot_idx0 + 1})
    cluster_idx0 = [nid - 1 for nid in cluster_customers]

    idx0_nodes = [depot_idx0] + cluster_idx0
    location_to_node_id = [depot_idx0 + 1] + cluster_customers

    m = Model()
    depot_coord = coords[depot_idx0]
    m.add_depot(x=float(depot_coord[0]), y=float(depot_coord[1]), name="depot")

    for nid in cluster_customers:
        idx0 = nid - 1
        xy = coords[idx0]
        m.add_client(
            x=float(xy[0]),
            y=float(xy[1]),
            delivery=int(demands[idx0]),
            name=f"cust_{nid}",
        )

    m.add_vehicle_type(num_available=max(1, len(cluster_customers)), capacity=capacity)

    locations = m.locations
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j:
                continue
            oi, oj = idx0_nodes[i], idx0_nodes[j]
            if edge_mat is not None:
                dist = int(round(float(edge_mat[oi, oj])))
            else:
                dx = coords[oi][0] - coords[oj][0]
                dy = coords[oi][1] - coords[oj][1]
                dist = int(round((dx * dx + dy * dy) ** 0.5))
            m.add_edge(locations[i], locations[j], distance=dist)

    return m, location_to_node_id


def _solve_cluster_with_pyvrp(
    instance: Dict[str, Any],
    cluster_nodes_vrplib: List[int],
    *,
    time_limit: float,
    seed: int,
) -> Tuple[List[List[int]], float]:

    if not cluster_nodes_vrplib:
        return [], 0.0

    model, loc_to_node = _build_cluster_model(instance, cluster_nodes_vrplib)
    data = model.data()

    # IMPORTANT: PyVRP supports ONLY pure time limits
    stop = MaxRuntime(time_limit)

    result = solve(
        data,
        stop=stop,
        seed=seed,
        collect_stats=False,
        display=False,
    )

    cost = float(result.cost())
    best = result.best

    if not result.is_feasible() or best is None or not np.isfinite(cost):
        return [], float("inf")

    routes: List[List[int]] = []
    for r in best.routes():
        visits = [loc_to_node[i] for i in r.visits()]
        depot = loc_to_node[0]
        if visits[0] != depot:
            visits.insert(0, depot)
        if visits[-1] != depot:
            visits.append(depot)
        routes.append(visits)

    return routes, cost


# ---------------------------------------------------------------------------
# Unified model for downstream compatibility
# ---------------------------------------------------------------------------

def _build_unified_model(instance: Dict[str, Any]) -> Tuple[Model, Dict[int, int]]:
    coords = instance["node_coord"]
    demands = instance["demand"]
    capacity = int(instance["capacity"])
    edge_mat = instance.get("edge_weight")
    depot_idx0 = int(instance["depot"][0])

    all_customers = list(range(2, len(demands) + 1))

    m = Model()
    depot_coord = coords[depot_idx0]
    m.add_depot(x=float(depot_coord[0]), y=float(depot_coord[1]), name="depot")

    for cid in all_customers:
        idx0 = cid - 1
        xy = coords[idx0]
        m.add_client(
            x=float(xy[0]),
            y=float(xy[1]),
            delivery=int(demands[idx0]),
            name=f"cust_{cid}",
        )

    m.add_vehicle_type(num_available=len(all_customers), capacity=capacity)

    locations = m.locations
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j:
                continue
            oi = depot_idx0 if i == 0 else all_customers[i - 1] - 1
            oj = depot_idx0 if j == 0 else all_customers[j - 1] - 1
            if edge_mat is not None:
                dist = int(round(float(edge_mat[oi, oj])))
            else:
                dx = coords[oi][0] - coords[oj][0]
                dy = coords[oi][1] - coords[oj][1]
                dist = int(round((dx * dx + dy * dy) ** 0.5))
            m.add_edge(locations[i], locations[j], distance=dist)

    return m, {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_clusters(
    instance_name: str,
    clusters: Dict[int, List[int]],
    *,
    solver: str = "pyvrp",
    solver_options: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> Result:

    solver_key = solver.lower()
    solver_options = dict(solver_options or {})
    instance = load_instance(instance_name)

    all_routes: List[List[int]] = []
    total_runtime = 0.0
    cluster_costs: Dict[int, float] = {}

    routing_solve = None
    if solver_key != "pyvrp":
        from master.routing.solver import solve as routing_solve  # type: ignore

    for cid, nodes in clusters.items():
        customers = [nid for nid in nodes if nid != 1]
        n = len(customers)

        cluster_time = _adaptive_cluster_time(n)
        stall_time = _adaptive_stall_time(n)

        cluster_time = _adaptive_cluster_time(n)
        stall_time   = _adaptive_stall_time(n)

        print(
            f"[ROUTING] solver={solver_key} | cluster={cid} | "
            f"n={n} | cluster_time={cluster_time:.2f}s | stall_time={stall_time:.2f}s",
            flush=True,
        )


        t0 = time.time()

        if solver_key == "pyvrp":
            routes, cost = _solve_cluster_with_pyvrp(
                instance,
                customers,
                time_limit=cluster_time,
                seed=seed + cid,
            )
        else:
            out = routing_solve(
                instance=instance_name,
                solver=solver_key,
                solver_options={
                    **solver_options,
                    "cluster_nodes": customers,
                    "seed": seed + cid,
                    "max_runtime": cluster_time,
                    "stall_time": stall_time,
                },
            )
            routes = out.metadata["routes_vrplib"]
            cost = float(out.cost)

        total_runtime += time.time() - t0
        all_routes.extend(routes)
        cluster_costs[cid] = cost

    unified_model, _ = _build_unified_model(instance)
    unified_data = unified_model.data()

    routes_pyvrp = [
        [nid - 1 for nid in r if nid != 1]
        for r in all_routes
        if len(r) > 2
    ]

    solution = Solution(unified_data, routes_pyvrp)
    result = Result(
        best=solution,
        stats=Statistics(),
        num_iterations=len(clusters),
        runtime=total_runtime,
    )

    result.cluster_costs = cluster_costs
    result.data = unified_data
    return result


solve_clusters_with_pyvrp = solve_clusters
