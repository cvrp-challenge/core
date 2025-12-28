"""
Restricted Set Partitioning (RSP) solver using Gurobi (Binary MIP)
------------------------------------------------------------------

Solve SP on a restricted pool of routes:

    min   Σ c_r x_r
    s.t.  Σ_{r covers i} x_r == 1   for all customers i
          x_r ∈ {0,1}

Optionally:
- Warm start using incumbent routes (if present in pool)
- Optional constraint on number of routes (K) if desired later

This is the "exact recombination" step after dual-based filtering.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

try:
    from gurobipy import Model, GRB, quicksum
except ImportError as e:
    raise ImportError(
        "gurobipy is required for restricted SP solving.\n"
        "Install it with: pip install gurobipy\n"
        "Gurobi requires a valid license."
    ) from e

from master.utils.loader import load_instance

Route = List[int]
Routes = List[Route]


def _custset(route: Route, depot_id: int = 1) -> frozenset[int]:
    return frozenset(n for n in route if n != depot_id)


def _compute_route_costs_euclidean(instance: dict, routes: Routes) -> List[float]:
    coords = instance["node_coord"]

    def dist(u: int, v: int) -> float:
        x1, y1 = coords[u - 1]
        x2, y2 = coords[v - 1]
        return math.hypot(x2 - x1, y2 - y1)

    costs: List[float] = []
    for route in routes:
        c = 0.0
        for u, v in zip(route, route[1:]):
            c += dist(u, v)
        costs.append(c)
    return costs


def solve_restricted_sp(
    *,
    instance_name: str,
    route_pool: Routes,
    time_limit: Optional[float] = None,
    verbose: bool = False,
    depot_id: int = 1,
    costs: Optional[List[float]] = None,
    warm_start_routes: Optional[Routes] = None,
) -> Dict[str, object]:
    """
    Returns dict compatible with your SCP solver schema:
      solver, optimal, status, obj_value, selected_indices, selected_routes
    """
    if not route_pool:
        raise ValueError("[RSP] route_pool is empty.")

    inst = load_instance(instance_name)
    dim = int(inst["dimension"])
    customers = list(range(2, dim + 1))

    # Build coverage
    route_cust: List[Set[int]] = []
    for r in route_pool:
        route_cust.append({nid for nid in r if nid != depot_id})

    cust_routes: Dict[int, List[int]] = {i: [] for i in customers}
    for r_idx, custs in enumerate(route_cust):
        for i in custs:
            if i in cust_routes:
                cust_routes[i].append(r_idx)

    uncovered = [i for i, rlist in cust_routes.items() if not rlist]
    if uncovered:
        raise ValueError(f"[RSP] infeasible restricted pool: uncovered customers={uncovered[:20]}")

    if costs is None:
        costs = _compute_route_costs_euclidean(inst, route_pool)

    n_routes = len(route_pool)

    model = Model("restricted_sp")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    # Binary variables
    x = model.addVars(n_routes, vtype=GRB.BINARY, name="x")

    model.setObjective(quicksum(costs[r] * x[r] for r in range(n_routes)), GRB.MINIMIZE)

    # Partition constraints: exactly once
    for i in customers:
        model.addConstr(quicksum(x[r] for r in cust_routes[i]) == 1, name=f"par_{i}")

    # Warm start (by customer-set signature)
    if warm_start_routes:
        sig_to_idx = {_custset(r, depot_id): idx for idx, r in enumerate(route_pool)}
        warm = set()
        for r in warm_start_routes:
            sig = _custset(r, depot_id)
            if sig in sig_to_idx:
                warm.add(sig_to_idx[sig])

        # Set starts
        for r in range(n_routes):
            x[r].Start = 1.0 if r in warm else 0.0

    model.optimize()

    status = model.Status
    optimal = status == GRB.OPTIMAL

    if model.SolCount == 0:
        return {
            "solver": "gurobi_rsp",
            "optimal": False,
            "status": status,
            "obj_value": None,
            "selected_indices": [],
            "selected_routes": [],
        }

    obj_value = float(model.objVal)

    sol = model.getAttr("X", x)
    selected_indices = [r for r in range(n_routes) if sol[r] > 0.5]
    selected_routes = [route_pool[r] for r in selected_indices]

    return {
        "solver": "gurobi_rsp",
        "optimal": optimal,
        "status": status,
        "obj_value": obj_value,
        "selected_indices": selected_indices,
        "selected_routes": selected_routes,
    }
