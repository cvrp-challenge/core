# master/setcover/scp_solver.py
"""
Set Covering Problem (SCP) solver for route pools using Gurobi.

Given:
    - an instance_name (VRPLIB CVRP)
    - a pool of routes in VRPLIB format: [ [1, i1, i2, ..., 1], ... ]

We build and solve the SCP:

    min   sum_r c_r * x_r
    s.t.  sum_{r: i in route_r} x_r >= 1   for all customers i
          x_r ∈ {0,1}

where c_r is the route length (Euclidean distance over node coordinates).

This is the SCP layer that will later be used in the DRSCI framework.
"""

from __future__ import annotations

import math
from typing import List, Dict, Optional

try:
    from gurobipy import Model, GRB, quicksum
except ImportError as e:
    raise ImportError(
        "gurobipy is required for SCP solving. "
        "Install it with: pip install gurobipy\n"
        "Note: Gurobi requires a valid license (academic or commercial)."
        "See https://www.gurobi.com/ for more information."
    ) from e

from master.utils.loader import load_instance


def _compute_route_costs(instance: dict,
                         routes: List[List[int]]) -> List[float]:
    """
    Compute Euclidean route costs based on node coordinates in the instance.

    VRPLIB convention:
        - instance["node_coord"][0] = coordinates of node 1 (depot)
        - routes are given as node IDs, e.g. [1, 5, 23, 1]
    """
    coords = instance["node_coord"]  # shape (n, 2), index 0 -> node 1

    def dist(u: int, v: int) -> float:
        x1, y1 = coords[u - 1]
        x2, y2 = coords[v - 1]
        return math.hypot(x2 - x1, y2 - y1)

    costs: List[float] = []
    for route in routes:
        if len(route) < 2:
            costs.append(0.0)
            continue
        c = 0.0
        for u, v in zip(route, route[1:]):
            c += dist(u, v)
        costs.append(c)

    return costs


def solve_scp(
    instance_name: str,
    route_pool: List[List[int]],
    time_limit: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Solve the route-based Set Covering Problem (SCP) with Gurobi.

    Args
    ----
    instance_name : str
        Name of the VRPLIB instance (e.g. "X-n101-k25.vrp").
    route_pool : list[list[int]]
        VRPLIB routes: each route is a list of node IDs [1, ..., 1].
    time_limit : float, optional
        Time limit in seconds for Gurobi. If None, no explicit limit.
    verbose : bool
        If True, print a short log of the model and solution.

    Returns
    -------
    dict with keys:
        "status"           : Gurobi status code (int)
        "obj_value"        : objective value (float or None)
        "selected_indices" : list[int] indices of chosen routes in route_pool
        "selected_routes"  : list[list[int]] of chosen VRPLIB routes
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
        print(f"[SCP] Instance: {instance_name}, DIMENSION={dim}")
        print(f"[SCP] Route pool size: {len(route_pool)}")

    # --------------------------------------------------------------
    # 2) Build coverage structure: which routes cover each customer
    # --------------------------------------------------------------
    # route_cust[r] = set of customers contained in route r
    route_cust = []
    for r_idx, route in enumerate(route_pool):
        visited = {nid for nid in route if nid != 1}
        route_cust.append(visited)

    # cust_routes[i] = list of route indices that visit customer i
    cust_routes: Dict[int, List[int]] = {i: [] for i in customers}
    for r_idx, visited in enumerate(route_cust):
        for i in visited:
            if i in cust_routes:   # ignore if some node outside 2..dim appears
                cust_routes[i].append(r_idx)

    # Sanity check: make sure each customer is covered at least once
    uncovered = [i for i, rlist in cust_routes.items() if len(rlist) == 0]
    if uncovered:
        # This is a critical issue for DRSCI: some customers never appear in the route pool.
        # For now, raise an error so we catch it early.
        raise ValueError(
            f"[SCP] The following customers are NOT covered by any route in the pool: {uncovered}"
        )

    # --------------------------------------------------------------
    # 3) Route costs
    # --------------------------------------------------------------
    costs = _compute_route_costs(inst, route_pool)

    # --------------------------------------------------------------
    # 4) Build Gurobi model
    # --------------------------------------------------------------
    model = Model("route_scp")

    # Control Gurobi output level
    model.Params.OutputFlag = 1 if verbose else 0

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    # Decision vars: x_r ∈ {0,1} for each route r
    num_routes = len(route_pool)
    x = model.addVars(num_routes, vtype=GRB.BINARY, name="x")

    # Objective: minimize sum_r c_r * x_r
    model.setObjective(
        quicksum(costs[r] * x[r] for r in range(num_routes)),
        GRB.MINIMIZE,
    )

    # Coverage constraints: each customer must be visited by at least one selected route
    for i in customers:
        routes_covering_i = cust_routes[i]
        model.addConstr(
            quicksum(x[r] for r in routes_covering_i) >= 1,
            name=f"cover_{i}",
        )

    # --------------------------------------------------------------
    # 5) Solve
    # --------------------------------------------------------------
    model.optimize()

    status = model.Status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE:
        if verbose:
            print("[SCP] Model infeasible or unbounded.")
        return {
            "status": status,
            "obj_value": None,
            "selected_indices": [],
            "selected_routes": [],
        }

    if status != GRB.OPTIMAL and status != GRB.TIME_LIMIT:
        if verbose:
            print(f"[SCP] Model ended with status {status}.")
        # still try to extract incumbent if any
    obj_value = model.objVal if model.SolCount > 0 else None

    # --------------------------------------------------------------
    # 6) Extract chosen routes
    # --------------------------------------------------------------
    selected_indices: List[int] = []
    if model.SolCount > 0:
        model_sol = model.getAttr("X", x)
        for r_idx in range(num_routes):
            if model_sol[r_idx] > 0.5:
                selected_indices.append(r_idx)

    selected_routes = [route_pool[r] for r in selected_indices]

    if verbose:
        print(f"[SCP] Status      : {status}")
        print(f"[SCP] Objective   : {obj_value}")
        print(f"[SCP] Routes used : {len(selected_indices)} / {num_routes}")

    return {
        "status": status,
        "obj_value": obj_value,
        "selected_indices": selected_indices,
        "selected_routes": selected_routes,
    }


if __name__ == "__main__":
    # This is only a minimal internal check; you can ignore or adapt it.
    # For real testing, you'll call solve_scp() from a run_* script.
    print("[SCP] This module is meant to be imported and used by DRSCI pipelines.")
