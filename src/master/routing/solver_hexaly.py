# core/src/master/routing/solver_hexaly.py

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, List, Optional, Tuple

import hexaly.optimizer

from master.routing.solver import SolveOutput, register_solver
from master.utils.loader import load_instance


def _dist_int_vrplib(coords, u: int, v: int) -> int:
    x1, y1 = coords[u - 1]
    x2, y2 = coords[v - 1]
    return int(round(math.hypot(x2 - x1, y2 - y1)))


def _extract_routes_vrplib(routes, local_to_vrplib: List[int]) -> List[List[int]]:
    """
    Extract VRPLIB routes from Hexaly list variables.
    routes: list of Hexaly list decision variables
    local_to_vrplib: mapping local idx -> VRPLIB customer id
    """
    routes_vrplib: List[List[int]] = []
    for r in routes:
        vals = getattr(r, "value", None)
        if not vals:
            continue
        cust_nodes = [local_to_vrplib[int(i)] for i in vals]
        routes_vrplib.append([1] + cust_nodes + [1])
    return routes_vrplib


def _compute_cost_from_dist_matrix(
    routes_vrplib: List[List[int]],
    dist_matrix: List[List[int]],
    local_to_vrplib: List[int],
) -> int:
    """
    Compute total integer cost from the precomputed dist_matrix.
    dist_matrix indexing: 0=depot, 1..k customers in local_to_vrplib order.
    """
    vrplib_to_matrix_index = {1: 0}
    for idx, nid in enumerate(local_to_vrplib, start=1):
        vrplib_to_matrix_index[nid] = idx

    total_cost = 0
    for route in routes_vrplib:
        for a, b in zip(route, route[1:]):
            ia = vrplib_to_matrix_index[a]
            ib = vrplib_to_matrix_index[b]
            total_cost += dist_matrix[ia][ib]
    return total_cost


def _try_get_hexaly_objective(opt) -> Optional[float]:
    """
    Best-effort retrieval of current best objective from Hexaly, if available.
    Hexaly APIs vary. If we can't get it reliably, return None.
    """
    # Common attribute names (best effort; harmless if missing)
    for name in ("best_objective", "objective_value", "best_value", "value"):
        try:
            val = getattr(opt, name)
            if callable(val):
                val = val()
            if isinstance(val, (int, float)):
                return float(val)
        except Exception:
            pass

    # Sometimes opt has a "solution" object
    for name in ("solution", "best_solution"):
        try:
            sol = getattr(opt, name, None)
            if sol is None:
                continue
            for subname in ("objective_value", "value", "cost", "best_objective"):
                v = getattr(sol, subname, None)
                if v is None:
                    continue
                if callable(v):
                    v = v()
                if isinstance(v, (int, float)):
                    return float(v)
        except Exception:
            pass

    return None


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
      - options["stall_time"]   : float seconds (stop if no improvement)
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
    max_runtime = float(max_runtime) if max_runtime is not None else None

    stall_time = options.get("stall_time", None)
    stall_time = float(stall_time) if stall_time is not None else None

    # Chunk size for stall-based stopping (tunable). 1.0s is a good default.
    chunk_time = float(options.get("chunk_time", 1.0))

    # ---------------------------------------------------------
    # Load instance (loader expects filename)
    # ---------------------------------------------------------
    inst = load_instance(instance_path.name)

    coords = inst["node_coord"]      # coords[0] is VRPLIB node 1 (depot)
    demands = inst["demand"]         # demands[0] depot, demands[i] node i+1
    capacity = int(inst["capacity"])

    # ---------------------------------------------------------
    # Determine customers in this subproblem
    # ---------------------------------------------------------
    if cluster_nodes is None:
        cluster_customers = list(range(2, len(demands) + 1))
    else:
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

    local_to_vrplib = cluster_customers
    local_demands = [int(demands[nid - 1]) for nid in local_to_vrplib]

    # ---------------------------------------------------------
    # Build integer-rounded distance matrix for {depot + customers}
    # Indices:
    #   0 = depot
    #   1..k = customers in local_to_vrplib order
    # ---------------------------------------------------------
    nodes = [1] + local_to_vrplib
    dist_matrix: List[List[int]] = []
    for a in nodes:
        row = []
        for b in nodes:
            row.append(_dist_int_vrplib(coords, a, b))
        dist_matrix.append(row)

    # ---------------------------------------------------------
    # Hexaly model
    # ---------------------------------------------------------
    with hexaly.optimizer.HexalyOptimizer() as opt:
        m = opt.model

        # Best-effort params
        try:
            opt.param.seed = seed
        except Exception:
            pass

        # One route list per vehicle (upper bound = k vehicles)
        routes = [m.list(k) for _ in range(k)]

        # Each customer served exactly once across all routes
        m.constraint(m.partition(routes))

        demand_arr = m.array(local_demands)     # 0..k-1 customers
        dist_arr = m.array(dist_matrix)         # 0..k with 0=depot

        route_costs = []
        for r in routes:
            size = m.count(r)

            load = m.sum(
                r,
                m.lambda_function(lambda cust: m.at(demand_arr, cust))
            )
            m.constraint(load <= capacity)

            internal = m.sum(
                m.range(1, size),
                m.lambda_function(
                    lambda p: m.at(dist_arr, m.at(r, p - 1) + 1, m.at(r, p) + 1)
                )
            )

            start_leg = m.at(dist_arr, 0, m.at(r, 0) + 1)
            end_leg = m.at(dist_arr, m.at(r, size - 1) + 1, 0)

            cost_r = m.iif(size > 0, internal + start_leg + end_leg, 0)
            route_costs.append(cost_r)

        m.minimize(m.sum(route_costs))
        m.close()

        # ---------------------------------------------------------
        # Solve with optional "stall_time" stopping via chunking
        # ---------------------------------------------------------
        if max_runtime is None:
            # Be conservative: if no max_runtime passed, give a reasonable bound
            max_runtime = 60.0

        # If no stall_time, keep old behavior (single solve call)
        use_stall = stall_time is not None and stall_time > 0.0

        best_cost_seen: Optional[float] = None
        last_improvement = time.time()

        elapsed_total = 0.0
        solve_calls = 0

        if not use_stall:
            # Old behavior: one solve call with full time limit
            try:
                opt.param.time_limit = float(max_runtime)
            except Exception:
                opt.param.time_limit = int(max_runtime)

            opt.solve()
            solve_calls = 1

        else:
            print(
                f"[HEXALY] stall_time enabled | stall={stall_time:.2f}s | "
                f"chunk={chunk_time:.2f}s | max_runtime={max_runtime:.2f}s",
                flush=True,
            )

            while elapsed_total < max_runtime:
                remaining = max_runtime - elapsed_total
                this_chunk = min(chunk_time, remaining)

                try:
                    opt.param.time_limit = float(this_chunk)
                except Exception:
                    opt.param.time_limit = int(this_chunk)

                opt.solve()
                solve_calls += 1

                # Try cheap objective read first
                obj = _try_get_hexaly_objective(opt)

                if obj is None:
                    # Fall back to extracting routes & computing integer cost
                    routes_tmp = _extract_routes_vrplib(routes, local_to_vrplib)
                    if routes_tmp:
                        obj = float(_compute_cost_from_dist_matrix(routes_tmp, dist_matrix, local_to_vrplib))

                now = time.time()
                elapsed_total = now - start

                if obj is not None:
                    if best_cost_seen is None or obj < best_cost_seen - 1e-9:
                        best_cost_seen = obj
                        last_improvement = now
                        print(
                            f"[HEXALY] improved | best={best_cost_seen:.2f} | "
                            f"t={elapsed_total:.2f}s | calls={solve_calls}",
                            flush=True,
                        )
                    else:
                        no_impr = now - last_improvement
                        if no_impr >= stall_time:
                            print(
                                f"[HEXALY] no improvement for {no_impr:.2f}s "
                                f"(>= {stall_time:.2f}s) -> early stop | "
                                f"t={elapsed_total:.2f}s | calls={solve_calls}",
                                flush=True,
                            )
                            break
                else:
                    # If we cannot measure objective at all, we cannot enforce stall reliably.
                    # In that case, keep running until max_runtime.
                    pass

        runtime = time.time() - start

        # ---------------------------------------------------------
        # Extract final solution -> VRPLIB routes
        # ---------------------------------------------------------
        routes_vrplib = _extract_routes_vrplib(routes, local_to_vrplib)

        total_cost = float(_compute_cost_from_dist_matrix(routes_vrplib, dist_matrix, local_to_vrplib)) \
            if routes_vrplib else float("inf")

        # Status (best effort)
        status_str = "unknown"
        try:
            status_str = str(getattr(opt, "status", "unknown"))
        except Exception:
            pass

        feasible = bool(routes_vrplib)

        return SolveOutput(
            solver="hexaly",
            instance=instance_path,
            cost=float(total_cost),
            runtime=runtime,
            num_iterations=solve_calls,
            feasible=feasible,
            data=None,  # type: ignore[assignment]
            raw_result=routes_vrplib,
            metadata={
                "backend": "hexaly",
                "status": status_str,
                "routes_vrplib": routes_vrplib,
                "num_routes": len(routes_vrplib),
                "stall_time": stall_time,
                "chunk_time": chunk_time if use_stall else None,
                "solve_calls": solve_calls,
            },
        )
