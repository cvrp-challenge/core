# master/setcover/route_selection_dual/lp_relaxation.py
"""
LP Relaxation for Route-Based Set Covering / Set Partitioning (Restricted Master)
-------------------------------------------------------------------------------

This module solves the LP relaxation on a given route pool representation
and extracts:
- Primal values x_r (fractional route usage)
- Dual values pi_i for customer constraints
- LP objective value

Core information for paper-style filtering:
  Reduced cost of route r: rc_r = c_r - sum_{i in r} pi_i

Important:
- This module does NOT filter routes.
- This module does NOT solve an integer SCP/SP.
- This module does NOT do pricing/column generation.
It only solves the LP on the current pool and returns dual/primal info.

We support two constraint modes:
- "cover":      sum_{r covers i} x_r >= 1   (set covering relaxation)
- "partition":  sum_{r covers i} x_r == 1   (set partitioning relaxation)

LP:
  0 <= x_r <= 1
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Literal

try:
    from gurobipy import Model, GRB, quicksum, Constr
except ImportError as e:
    raise ImportError(
        "gurobipy is required for LP relaxation solving.\n"
        "Install it with: pip install gurobipy\n"
        "Gurobi requires a valid license."
    ) from e


Mode = Literal["cover", "partition"]


@dataclass(frozen=True)
class LpRelaxationResult:
    solver: str
    mode: Mode
    status: int
    runtime_sec: float

    lp_obj_value: Optional[float]

    # primal route usage
    x: Dict[int, float]              # route index -> x_r

    # duals on customer constraints
    pi: Dict[int, float]             # customer id -> pi_i

    # helpful diagnostics
    num_routes: int
    num_customers: int
    num_fractional: int
    num_near_one: int


def compute_reduced_costs(
    *,
    costs: List[float],
    route_cust: List[Set[int]],
    pi: Dict[int, float],
) -> List[float]:
    """
    Reduced cost:
      rc_r = c_r - sum_{i in route} pi_i
    """
    if len(costs) != len(route_cust):
        raise ValueError("[LP] compute_reduced_costs: costs and route_cust must align.")

    rc: List[float] = []
    for r, custs in enumerate(route_cust):
        dual_sum = 0.0
        for i in custs:
            dual_sum += pi.get(i, 0.0)
        rc.append(costs[r] - dual_sum)
    return rc


def solve_lp_relaxation(
    *,
    customers: List[int],
    route_cust: List[Set[int]],
    costs: List[float],
    mode: Mode = "cover",
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> LpRelaxationResult:
    """
    Solve the LP relaxation on a given route pool representation and extract duals.

    Parameters
    ----------
    customers:
        List of customer IDs (excluding depot)
    route_cust:
        route_cust[r] = set of customers covered by route r
    costs:
        costs[r] = cost of route r
    mode:
        "cover"      -> sum x_r >= 1  (set covering relaxation)
        "partition"  -> sum x_r == 1  (set partitioning relaxation)
    time_limit:
        Gurobi time limit (seconds)
    verbose:
        Print solver output

    Returns
    -------
    LpRelaxationResult
    """
    if not customers:
        raise ValueError("[LP] customers is empty.")
    if not route_cust:
        raise ValueError("[LP] route_cust is empty.")
    if len(route_cust) != len(costs):
        raise ValueError("[LP] costs and route_cust length mismatch.")

    num_routes = len(route_cust)
    num_customers = len(customers)

    # Build customer -> routes incidence
    cust_routes: Dict[int, List[int]] = {i: [] for i in customers}
    for r, custs in enumerate(route_cust):
        for i in custs:
            if i in cust_routes:
                cust_routes[i].append(r)

    # Sanity: full coverage
    uncovered = [i for i, rs in cust_routes.items() if not rs]
    if uncovered:
        # This usually means your pool is too small (or route_cust computed wrong)
        raise ValueError(f"[LP] Customers uncovered by route pool: {uncovered[:20]}")

    model = Model("route_pool_lp")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    # More stable LP behaviour
    model.Params.Method = 1  # dual simplex
    model.Params.Seed = 0

    # Variables: 0 <= x_r <= 1
    x = model.addVars(
        num_routes,
        lb=0.0,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    model.setObjective(
        quicksum(costs[r] * x[r] for r in range(num_routes)),
        GRB.MINIMIZE,
    )

    # Customer constraints
    constr: Dict[int, Constr] = {}

    if mode == "cover":
        for i in customers:
            constr[i] = model.addConstr(
                quicksum(x[r] for r in cust_routes[i]) >= 1.0,
                name=f"cov_{i}",
            )
    elif mode == "partition":
        for i in customers:
            constr[i] = model.addConstr(
                quicksum(x[r] for r in cust_routes[i]) == 1.0,
                name=f"par_{i}",
            )
    else:
        raise ValueError(f"[LP] Unknown mode: {mode}")

    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0
    status = model.Status

    if model.SolCount == 0 or status in (GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        return LpRelaxationResult(
            solver="gurobi",
            mode=mode,
            status=status,
            runtime_sec=runtime,
            lp_obj_value=None,
            x={},
            pi={},
            num_routes=num_routes,
            num_customers=num_customers,
            num_fractional=0,
            num_near_one=0,
        )

    lp_obj = float(model.objVal)

    # Primal x
    x_attr = model.getAttr("X", x)
    x_vals: Dict[int, float] = {r: float(x_attr[r]) for r in range(num_routes)}

    # Duals π_i
    pi: Dict[int, float] = {i: float(constr[i].Pi) for i in customers}

    frac = sum(1 for v in x_vals.values() if 1e-6 < v < 1.0 - 1e-6)
    near_one = sum(1 for v in x_vals.values() if v >= 1.0 - 1e-6)

    if verbose:
        print(
            f"[LP] mode={mode} | routes={num_routes} customers={num_customers} "
            f"| obj={lp_obj:.6f} | frac={frac} near1={near_one} | time={runtime:.2f}s"
        )

    return LpRelaxationResult(
        solver="gurobi",
        mode=mode,
        status=status,
        runtime_sec=runtime,
        lp_obj_value=lp_obj,
        x=x_vals,
        pi=pi,
        num_routes=num_routes,
        num_customers=num_customers,
        num_fractional=frac,
        num_near_one=near_one,
    )


if __name__ == "__main__":
    print("[lp_relaxation] This module is meant to be imported by the dual pipeline.")
