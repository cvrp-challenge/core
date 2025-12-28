# master/setcover/scp_solver_gurobi_LP.py
"""
LP-based Set Covering Problem (SCP) solver for route pools using Gurobi.

This is a license-friendly alternative to the binary MIP SCP for large instances.

Given:
    - an instance_name (VRPLIB CVRP)
    - a pool of routes in VRPLIB format: [ [1, i1, i2, ..., 1], ... ]

We build and solve the LP relaxation of the SCP:

    min   sum_r c_r * x_r
    s.t.  sum_{r: i in route_r} x_r >= 1   for all customers i
          0 <= x_r <= 1

Then we construct an integer SCP solution via LP-guided greedy rounding:
  1) take routes with x_r close to 1
  2) greedily add routes minimizing cost / newly_covered until all customers covered

Returns the same output schema as the MIP solver so the caller can swap modules easily.
"""

from __future__ import annotations

import math
from typing import List, Dict, Optional, Set

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


def _compute_route_costs(instance: dict, routes: List[List[int]]) -> List[float]:
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


def _greedy_rounding(
    customers: List[int],
    route_cust: List[Set[int]],
    costs: List[float],
    x_vals: Dict[int, float],
    verbose: bool = False,
    take_threshold: float = 0.999,
) -> List[int]:
    """
    LP-guided greedy rounding for SCP.

    Strategy:
      1) select all routes with x_r >= take_threshold
      2) while some customers uncovered:
           pick route minimizing cost / newly_covered

    Returns:
      list of selected route indices
    """
    all_customers: Set[int] = set(customers)

    selected: Set[int] = set()
    covered: Set[int] = set()

    # 1) Take almost-integer routes first
    for r, val in x_vals.items():
        if val >= take_threshold:
            selected.add(r)
            covered |= route_cust[r]

    uncovered = all_customers - covered
    if verbose:
        print(f"[SCP-LP] Rounding: preselected {len(selected)} routes, "
              f"covered {len(covered)}/{len(all_customers)} customers.")

    # 2) Greedy completion
    #    (Iterate over all routes; if you ever need speed, add candidate filtering here.)
    num_routes = len(route_cust)

    while uncovered:
        best_r: Optional[int] = None
        best_score = float("inf")
        best_new = 0

        for r in range(num_routes):
            if r in selected:
                continue
            newly = uncovered & route_cust[r]
            if not newly:
                continue

            # cost effectiveness for remaining uncovered
            score = costs[r] / len(newly)
            if score < best_score:
                best_score = score
                best_r = r
                best_new = len(newly)

        if best_r is None:
            # This should not happen if the pool covers all customers (we check earlier),
            # but keep it robust anyway.
            raise RuntimeError(
                "[SCP-LP] Greedy rounding failed: no route covers remaining uncovered customers. "
                f"Remaining uncovered count: {len(uncovered)}"
            )

        selected.add(best_r)
        covered |= route_cust[best_r]
        uncovered = all_customers - covered

        if verbose:
            print(f"[SCP-LP] Rounding: add route {best_r} "
                  f"(newly covered={best_new}, score={best_score:.4f}). "
                  f"Covered {len(covered)}/{len(all_customers)}.")

    return sorted(selected)

def _lns_scp_improve(
    *,
    customers: List[int],
    route_cust: List[Set[int]],
    costs: List[float],
    initial_selected: List[int],
    max_iters: int = 20,
    destroy_frac: float = 0.3,
    seed: int = 0,
    verbose: bool = False,
) -> List[int]:
    """
    Large Neighborhood Search (LNS) improvement for SCP.

    Starting from an initial SCP solution (set of routes),
    iteratively:
      - remove a fraction of routes
      - greedily repair uncovered customers
      - accept if improved

    Returns:
        improved list of selected route indices
    """
    import random

    rnd = random.Random(seed)

    all_customers = set(customers)
    best = set(initial_selected)
    best_cost = sum(costs[r] for r in best)

    current = set(best)

    num_routes = len(route_cust)

    for it in range(max_iters):
        if not current:
            break

        # -----------------------------
        # DESTROY
        # -----------------------------
        k = max(1, int(len(current) * destroy_frac))
        removed = set(rnd.sample(list(current), k))
        partial = current - removed

        covered = set()
        for r in partial:
            covered |= route_cust[r]

        uncovered = all_customers - covered

        # -----------------------------
        # REPAIR (greedy SCP)
        # -----------------------------
        while uncovered:
            best_r = None
            best_score = float("inf")

            for r in range(num_routes):
                if r in partial:
                    continue
                newly = uncovered & route_cust[r]
                if not newly:
                    continue

                score = costs[r] / len(newly)
                if score < best_score:
                    best_score = score
                    best_r = r

            if best_r is None:
                break

            partial.add(best_r)
            covered |= route_cust[best_r]
            uncovered = all_customers - covered

        new_cost = sum(costs[r] for r in partial)

        # -----------------------------
        # ACCEPTANCE
        # -----------------------------
        if new_cost < best_cost:
            best = set(partial)
            best_cost = new_cost
            current = set(partial)

            if verbose:
                print(f"[SCP-LNS] iter={it} improved cost={best_cost:.2f}")
        else:
            current = set(best)

    return sorted(best)


def solve_scp(
    instance_name: str,
    route_pool: List[List[int]],
    time_limit: Optional[float] = None,
    verbose: bool = True,
    seed: int = 0
) -> Dict[str, object]:
    """
    Solve the LP relaxation of the route-based Set Covering Problem (SCP) with Gurobi,
    then build an integer SCP solution via greedy rounding.

    Args
    ----
    instance_name : str
        Name of the VRPLIB instance (e.g. "X-n101-k25.vrp").
    route_pool : list[list[int]]
        VRPLIB routes: each route is a list of node IDs [1, ..., 1].
    time_limit : float, optional
        Time limit in seconds for Gurobi LP solve. If None, no explicit limit.
    verbose : bool
        If True, print a short log of the model and rounding progress.

    Returns
    -------
    dict with keys:
        "solver"           : str
        "optimal"          : bool (always False here: we do LP + heuristic rounding)
        "status"           : Gurobi status code (int)
        "obj_value"        : objective value of selected integer cover (float or None)
        "lp_obj_value"     : LP objective value (float or None)
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
        print(f"[SCP-LP] Instance: {instance_name}, DIMENSION={dim}")
        print(f"[SCP-LP] Route pool size: {len(route_pool)}")

    # --------------------------------------------------------------
    # 2) Build coverage structure: which routes cover each customer
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
            f"[SCP-LP] The following customers are NOT covered by any route in the pool: {uncovered}"
        )

    # --------------------------------------------------------------
    # 3) Route costs
    # --------------------------------------------------------------
    costs = _compute_route_costs(inst, route_pool)

    # --------------------------------------------------------------
    # 4) Build Gurobi LP model
    # --------------------------------------------------------------
    model = Model("route_scp_lp")

    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    num_routes = len(route_pool)

    # LP relaxation: 0 <= x_r <= 1, continuous
    x = model.addVars(
        num_routes,
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    model.setObjective(
        quicksum(costs[r] * x[r] for r in range(num_routes)),
        GRB.MINIMIZE,
    )

    for i in customers:
        model.addConstr(
            quicksum(x[r] for r in cust_routes[i]) >= 1.0,
            name=f"cover_{i}",
        )

    # --------------------------------------------------------------
    # 5) Solve LP
    # --------------------------------------------------------------
    model.optimize()

    status = model.Status

    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        if verbose:
            print("[SCP-LP] LP model infeasible or unbounded.")
        return {
            "solver": "gurobi_lp",
            "optimal": False,
            "status": status,
            "obj_value": None,
            "lp_obj_value": None,
            "selected_indices": [],
            "selected_routes": [],
        }

    if model.SolCount == 0:
        if verbose:
            print(f"[SCP-LP] LP model ended with status {status} and no solution.")
        return {
            "solver": "gurobi_lp",
            "optimal": False,
            "status": status,
            "obj_value": None,
            "lp_obj_value": None,
            "selected_indices": [],
            "selected_routes": [],
        }

    lp_obj_value = float(model.objVal)

    # Extract LP solution values
    x_attr = model.getAttr("X", x)  # dict-like: {idx: value}
    x_vals: Dict[int, float] = {r: float(x_attr[r]) for r in range(num_routes)}

    if verbose:
        # Basic LP diagnostics
        frac = sum(1 for v in x_vals.values() if 1e-6 < v < 1.0 - 1e-6)
        ones = sum(1 for v in x_vals.values() if v >= 1.0 - 1e-6)
        print(f"[SCP-LP] LP status    : {status}")
        print(f"[SCP-LP] LP objective : {lp_obj_value}")
        print(f"[SCP-LP] LP vars: ~1 => {ones}, fractional => {frac}, total => {num_routes}")

    # --------------------------------------------------------------
    # 6) LP-guided rounding to integer SCP solution
    # --------------------------------------------------------------
    selected_indices = _greedy_rounding(
        customers=customers,
        route_cust=route_cust,
        costs=costs,
        x_vals=x_vals,
        verbose=verbose,
        take_threshold=0.999,
    )

    # --------------------------------------------------------------
    # 7) Optional LNS-SCP improvement
    # --------------------------------------------------------------
    selected_indices = _lns_scp_improve(
        customers=customers,
        route_cust=route_cust,
        costs=costs,
        initial_selected=selected_indices,
        max_iters=20,
        destroy_frac=0.3,
        seed=seed,
        verbose=verbose,
    )

    selected_routes = [route_pool[r] for r in selected_indices]

    # Compute integer objective value for returned solution
    obj_value = float(sum(costs[r] for r in selected_indices))

    if verbose:
        print(f"[SCP-LP] Rounded objective : {obj_value}")
        print(f"[SCP-LP] Routes used        : {len(selected_indices)} / {num_routes}")

    return {
        "solver": "gurobi_lp",
        "optimal": False,  # by design: LP + heuristic rounding
        "status": status,
        "obj_value": obj_value,
        "lp_obj_value": lp_obj_value,
        "selected_indices": selected_indices,
        "selected_routes": selected_routes,
    }


if __name__ == "__main__":
    print("[SCP-LP] This module is meant to be imported and used by DRSCI pipelines.")
