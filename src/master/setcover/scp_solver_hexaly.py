# master/setcover/scp_solver_hexaly.py
"""
Set Covering Problem (SCP) solver for route pools using Hexaly.

Given:
    - instance_name (VRPLIB CVRP)
    - a pool of routes in VRPLIB format: [ [1, i1, i2, ..., 1], ... ]

We build and solve the SCP:

    min   sum_r c_r * x_r
    s.t.  sum_{r: i in route_r} x_r >= 1   for all customers i
          x_r ∈ {0,1}

where c_r is the route length (integer-rounded Euclidean distance over node coordinates).

Lives next to:
    master/setcover/scp_solver.py          (Gurobi)
    master/setcover/scp_solver_hexaly.py   (Hexaly)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

try:
    import hexaly.optimizer
except ImportError as e:
    raise ImportError(
        "hexaly is required for Hexaly-based SCP solving.\n"
        "Install with: pip install hexaly -i https://pip.hexaly.com\n"
        "You also need a working Hexaly Optimizer installation + license."
    ) from e

from master.utils.loader import load_instance


def _compute_route_costs_int(instance: dict, routes: List[List[int]]) -> List[int]:
    """
    Compute integer-rounded Euclidean route costs based on node coordinates.

    VRPLIB convention:
        - instance["node_coord"][0] = coordinates of node 1 (depot)
        - routes are node IDs, e.g. [1, 5, 23, 1]
    """
    coords = instance["node_coord"]  # index 0 -> node 1

    def dist_int(u: int, v: int) -> int:
        x1, y1 = coords[u - 1]
        x2, y2 = coords[v - 1]
        return int(round(math.hypot(x2 - x1, y2 - y1)))

    costs: List[int] = []
    for route in routes:
        if len(route) < 2:
            costs.append(0)
            continue
        c = 0
        for u, v in zip(route, route[1:]):
            c += dist_int(u, v)
        costs.append(c)

    return costs


def solve_scp(
    instance_name: str,
    route_pool: List[List[int]],
    time_limit: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Solve the route-based Set Covering Problem (SCP) with Hexaly.

    Returns a dict compatible with the Gurobi version:
        "solver"           : "hexaly"
        "optimal"          : bool
        "status"           : str (best-effort)
        "obj_value"        : float or None
        "selected_indices" : list[int]
        "selected_routes"  : list[list[int]]
    """
    if not route_pool:
        raise ValueError("Route pool is empty: cannot solve SCP.")

    # --------------------------------------------------------------
    # 1) Load instance and basic info
    # --------------------------------------------------------------
    inst = load_instance(instance_name)
    dim = int(inst["dimension"])
    customers = list(range(2, dim + 1))  # customer node IDs (depot = 1)

    if verbose:
        print(f"[SCP-HX] Instance: {instance_name}, DIMENSION={dim}")
        print(f"[SCP-HX] Route pool size: {len(route_pool)}")

    # --------------------------------------------------------------
    # 2) Build coverage structure
    # --------------------------------------------------------------
    route_cust: List[Set[int]] = []
    for route in route_pool:
        visited = {nid for nid in route if nid != 1}
        route_cust.append(visited)

    cust_routes: Dict[int, List[int]] = {i: [] for i in customers}
    for r_idx, visited in enumerate(route_cust):
        for i in visited:
            if i in cust_routes:
                cust_routes[i].append(r_idx)

    uncovered = [i for i, rlist in cust_routes.items() if len(rlist) == 0]
    if uncovered:
        raise ValueError(
            f"[SCP-HX] These customers are NOT covered by any route in the pool: {uncovered}"
        )

    # --------------------------------------------------------------
    # 3) Route costs (integer-rounded like your pipeline)
    # --------------------------------------------------------------
    costs = _compute_route_costs_int(inst, route_pool)
    num_routes = len(route_pool)

    # --------------------------------------------------------------
    # 4) Build + solve Hexaly model
    # --------------------------------------------------------------
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        m = optimizer.model

        # Decision vars: x[r] in {0,1}
        # (Hexaly typically supports model.bool() for a boolean decision var.)
        x = [m.bool() for _ in range(num_routes)]

        # Coverage constraints: for each customer i, sum_{r covers i} x[r] >= 1
        for i in customers:
            rlist = cust_routes[i]
            # IMPORTANT: pass a list into m.sum (not a generator)
            m.constraint(m.sum([x[r] for r in rlist]) >= 1)

        # Objective: minimize total cost
        # IMPORTANT: build terms as a list (not a generator)
        obj_terms = [costs[r] * x[r] for r in range(num_routes)]
        total_cost_expr = m.sum(obj_terms)
        m.minimize(total_cost_expr)

        m.close()

        # Params
        if time_limit is not None:
            # Hexaly expects seconds; many versions accept float.
            try:
                optimizer.param.time_limit = float(time_limit)
            except Exception:
                optimizer.param.time_limit = int(time_limit)

        if not verbose:
            try:
                optimizer.param.verbosity = 0
            except Exception:
                pass

        optimizer.solve()

        # ----------------------------------------------------------
        # 5) Extract solution
        # ----------------------------------------------------------
        selected_indices: List[int] = []
        for r in range(num_routes):
            try:
                if int(x[r].value) == 1:
                    selected_indices.append(r)
            except Exception:
                # If no value available, treat as not selected
                pass

        selected_routes = [route_pool[r] for r in selected_indices]

        # Objective value (best-effort across Hexaly versions)
        obj_value: Optional[float] = None
        try:
            v = getattr(total_cost_expr, "value", None)
            if v is not None:
                obj_value = float(v)
        except Exception:
            obj_value = None

        if obj_value is None:
            # fallback candidates some versions expose
            for attr in ("objective_value", "obj_value", "best_objective"):
                try:
                    v = getattr(optimizer, attr)
                    if v is not None:
                        obj_value = float(v)
                        break
                except Exception:
                    pass

        # Status / optimality (best-effort)
        status_str = "unknown"
        optimal = False
        try:
            st = getattr(optimizer, "status", None)
            if st is not None:
                status_str = str(st)
        except Exception:
            pass

        if "optimal" in status_str.lower():
            optimal = True

        if verbose:
            print(f"[SCP-HX] Status      : {status_str}")
            print(f"[SCP-HX] Objective   : {obj_value}")
            print(f"[SCP-HX] Routes used : {len(selected_indices)} / {num_routes}")

        return {
            "solver": "hexaly",
            "optimal": optimal,
            "status": status_str,
            "obj_value": obj_value,
            "selected_indices": selected_indices,
            "selected_routes": selected_routes,
        }


if __name__ == "__main__":
    print("[SCP-HX] This module is meant to be imported and used by DRSCI pipelines.")
