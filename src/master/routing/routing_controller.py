# src/master/routing/routing_controller.py
"""
Routing controller: glue between clustering and a routing backend.

Behaviour:
    - Adaptive cluster runtime based on cluster size (for PyVRP and AILS2 time limit)
    - Adaptive no-improvement iterations based on cluster size:
      * n <= 100: 500 iterations
      * n >= 1500: 20000 iterations
      * Linear scaling in between
    - PyVRP: MultipleCriteria stopping (time limit AND no-improvement, stop when either met)
    - AILS2: Adaptive time limit (prioritized) and no-improvement iterations (fallback)
    - FILO1/FILO2: Only no-improvement criterion (no time limit)
    - Non-PyVRP solvers (e.g. Hexaly): receive stall_time if enabled
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
from pyvrp.stop import MaxRuntime, NoImprovement, MultipleCriteria  # type: ignore


# ---------------------------------------------------------------------------
# Adaptive timing logic (THIS is what you tune)
# ---------------------------------------------------------------------------

def _adaptive_cluster_time(
    n: int,
    *,
    base: float = 1.0,
    alpha: float = 0.25,            #was 0.16
    exponent: float = 0.9,
    min_time: float = 2.0,
    max_time: float = 180.0,
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


def _adaptive_no_improvement(n: int) -> int:
    if n <= 100 and n >= 10:
        return 500
    elif n <= 10: # edge case for ac_min
        return 10
    elif n >= 1500:
        return 20000
    else:
        # Linear interpolation: 10000 + (100000 - 10000) * ((n - 100) / (10000 - 100))
        # = 10000 + 90000 * ((n - 100) / 9900)
        ratio = (n - 100) / (1500 - 100)
        return int(500 + 19500 * ratio)


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
    no_improvement: Optional[int] = None,
) -> Tuple[List[List[int]], float]:

    if not cluster_nodes_vrplib:
        return [], 0.0

    model, loc_to_node = _build_cluster_model(instance, cluster_nodes_vrplib)
    data = model.data()

    # PyVRP: Always use multiple stopping criteria (time limit AND no-improvement)
    # Stop when either criterion is met
    from pyvrp.stop import NoImprovement, MultipleCriteria
    
    if no_improvement is None:
        raise ValueError("no_improvement must be provided for PyVRP")
    
    stop = MultipleCriteria([
        NoImprovement(no_improvement),
        MaxRuntime(time_limit),  # Adaptive time limit by cluster size
    ])

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
    no_improvement: Optional[int] = None,
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
        
        # Calculate adaptive no-improvement iterations based on cluster size
        adaptive_no_improvement = _adaptive_no_improvement(n)
        
        # Override with provided no_improvement if given (for testing/debugging)
        effective_no_improvement = adaptive_no_improvement
        if no_improvement is not None:
            effective_no_improvement = no_improvement

        # if solver_key == "pyvrp":
        #     # print(
        #     #     f"[ROUTING] solver={solver_key} | cluster={cid} | "
        #     #     f"n={n} | no_improvement={effective_no_improvement} | "
        #     #     f"time_limit={cluster_time:.2f}s",
        #     #     flush=True,
        #     # )
        # else:
        #     print(
        #         f"[ROUTING] solver={solver_key} | cluster={cid} | "
        #         f"n={n} | no_improvement={effective_no_improvement}",
        #         flush=True,
        #     )

        t0 = time.time()

        if solver_key == "pyvrp":
            routes, cost = _solve_cluster_with_pyvrp(
                instance,
                customers,
                time_limit=cluster_time,
                seed=seed + cid,
                no_improvement=effective_no_improvement*10,
            )
        else:
            opts = {
                **solver_options,
                "cluster_nodes": customers,
                "seed": seed + cid,
            }
            
            # AILS2: Use adaptive time limit (prioritized) and no-improvement (fallback)
            # AILS2 supports either Time OR Iteration stopping criterion, not both.
            # We provide both, but AILS2 will use max_runtime (adaptive time limit) when provided.
            if solver_key == "ails2":
                opts["max_runtime"] = cluster_time  # Adaptive time limit based on cluster size
                opts["no_improvement"] = effective_no_improvement  # Fallback if max_runtime not used
            else:
                # FILO: Only use no_improvement, no time limit
                opts["no_improvement"] = effective_no_improvement

            # Only add stall_time if explicitly enabled (for Hexaly)
            if solver_options.get("use_stall", False):
                opts["stall_time"] = stall_time

            out = routing_solve(
                instance=instance_name,
                solver=solver_key,
                solver_options=opts,
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
