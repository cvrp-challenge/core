# core/src/master/routing/solver_hexaly.py

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, List, Optional

import hexaly.optimizer

from master.routing.solver import SolveOutput, register_solver
from master.utils.loader import load_instance


@register_solver("hexaly")
def _solve_with_hexaly(
    instance_path: Path,
    options: Mapping[str, Any],
) -> SolveOutput:
    """
    Hexaly routing backend (CVRP) for a *cluster subproblem*.

    Contract (as used by routing_controller.solve_clusters):
      - options["cluster_nodes"] : list of VRPLIB customer node IDs (no depot)
      - options["seed"]         : int
      - options["max_runtime"]  : float seconds
    Must return:
      - SolveOutput.metadata["routes_vrplib"] : List[List[int]] in VRPLIB format
    """

    start = time.time()

    # ---------------------------------------------------------
    # Read options
    # ---------------------------------------------------------
    cluster_nodes: Optional[List[int]] = None
    if "cluster_nodes" in options and options["cluster_nodes"] is not None:
        cluster_nodes = list(options["cluster_nodes"])

    seed = int(options.get("seed", 0))
    max_runtime = options.get("max_runtime", None)
    if max_runtime is not None:
        max_runtime = float(max_runtime)

    # ---------------------------------------------------------
    # Load instance (your existing loader expects the filename)
    # ---------------------------------------------------------
    inst = load_instance(instance_path.name)

    coords = inst["node_coord"]      # coords[0] is VRPLIB node 1 (depot)
    demands = inst["demand"]         # demands[0] depot, demands[i] node i+1
    capacity = int(inst["capacity"])

    # ---------------------------------------------------------
    # Determine which customers are in this subproblem
    # ---------------------------------------------------------
    if cluster_nodes is None:
        # full instance customers
        cluster_customers = list(range(2, len(demands) + 1))
    else:
        # cluster customers (VRPLIB IDs), ensure no depot
        cluster_customers = sorted({nid for nid in cluster_nodes if nid != 1})

    k = len(cluster_customers)
    if k == 0:
        runtime = time.time() - start
        return SolveOutput(
            solver="hexaly",
            instance=instance_path,
            cost=0.0,
            runtime=runtime,
            num_iterations=0,
            feasible=True,
            data=None,  # type: ignore[assignment]
            raw_result=[],
            metadata={"routes_vrplib": []},
        )

    # Map local index (0..k-1) -> VRPLIB node id
    local_to_vrplib = cluster_customers

    # Local demand array aligned with local indices
    local_demands = [int(demands[nid - 1]) for nid in local_to_vrplib]

    # ---------------------------------------------------------
    # Build integer-rounded distance matrix for {depot + cluster customers}
    # Indices:
    #   0 = depot
    #   1..k = cluster customers in local_to_vrplib order
    # ---------------------------------------------------------
    def dist_int_vrplib(u: int, v: int) -> int:
        x1, y1 = coords[u - 1]
        x2, y2 = coords[v - 1]
        return int(round(math.hypot(x2 - x1, y2 - y1)))

    nodes = [1] + local_to_vrplib  # VRPLIB node IDs
    dist_matrix: List[List[int]] = []
    for a in nodes:
        row = []
        for b in nodes:
            row.append(dist_int_vrplib(a, b))
        dist_matrix.append(row)

    # ---------------------------------------------------------
    # Hexaly model
    # ---------------------------------------------------------
    with hexaly.optimizer.HexalyOptimizer() as opt:
        m = opt.model

        # Optional params (best effort; Hexaly versions vary)
        try:
            opt.param.seed = seed
        except Exception:
            pass

        if max_runtime is not None:
            try:
                opt.param.time_limit = float(max_runtime)
            except Exception:
                opt.param.time_limit = int(max_runtime)

        # One route list per vehicle (upper bound = k vehicles)
        routes = [m.list(k) for _ in range(k)]

        # Each customer served exactly once across all routes
        m.constraint(m.partition(routes))

        # Arrays for indexing inside expressions
        demand_arr = m.array(local_demands)     # index: local customer (0..k-1)
        dist_arr = m.array(dist_matrix)         # index: 0..k (0=depot, 1..k customers)

        route_costs = []

        for r in routes:
            size = m.count(r)  # number of customers on this route

            # Capacity constraint:
            # sum_{customer in r} demand(customer) <= capacity
            load = m.sum(
                r,
                m.lambda_function(lambda cust: m.at(demand_arr, cust))
            )
            m.constraint(load <= capacity)

            # Travel cost inside the route:
            # sum_{p=1..size-1} dist( r[p-1], r[p] )
            # Note: dist_arr uses +1 offset because 0 is depot.
            internal = m.sum(
                m.range(1, size),
                m.lambda_function(
                    lambda p: m.at(dist_arr, m.at(r, p - 1) + 1, m.at(r, p) + 1)
                )
            )

            # Add depot legs if route used
            start_leg = m.at(dist_arr, 0, m.at(r, 0) + 1)
            end_leg = m.at(dist_arr, m.at(r, size - 1) + 1, 0)

            cost_r = m.iif(size > 0, internal + start_leg + end_leg, 0)
            route_costs.append(cost_r)

        m.minimize(m.sum(route_costs))
        m.close()

        opt.solve()

        runtime = time.time() - start

        # ---------------------------------------------------------
        # Extract solution -> VRPLIB routes
        # ---------------------------------------------------------
        routes_vrplib: List[List[int]] = []

        for r in routes:
            # r.value is a Python list of local customer indices if used; [] if unused
            vals = getattr(r, "value", None)
            if not vals:
                continue

            # Convert local indices -> VRPLIB node IDs
            cust_nodes = [local_to_vrplib[int(i)] for i in vals]
            routes_vrplib.append([1] + cust_nodes + [1])

        # Compute cost consistently with dist_matrix
        # (dist_matrix indices: 0=depot, 1..k customers in local_to_vrplib order)
        vrplib_to_matrix_index = {1: 0}
        for idx, nid in enumerate(local_to_vrplib, start=1):
            vrplib_to_matrix_index[nid] = idx

        total_cost = 0
        for route in routes_vrplib:
            for a, b in zip(route, route[1:]):
                ia = vrplib_to_matrix_index[a]
                ib = vrplib_to_matrix_index[b]
                total_cost += dist_matrix[ia][ib]

        # Status (best effort)
        status_str = "unknown"
        try:
            status_str = str(getattr(opt, "status", "unknown"))
        except Exception:
            pass

        feasible = True  # if we got values, we assume feasible

        return SolveOutput(
            solver="hexaly",
            instance=instance_path,
            cost=float(total_cost),
            runtime=runtime,
            num_iterations=0,
            feasible=feasible,
            data=None,  # type: ignore[assignment]
            raw_result=routes_vrplib,  # fine for debugging
            metadata={
                "backend": "hexaly",
                "status": status_str,
                "routes_vrplib": routes_vrplib,  # IMPORTANT for routing_controller
                "num_routes": len(routes_vrplib),
            },
        )
