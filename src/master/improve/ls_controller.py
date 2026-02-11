"""
Local search controller (pluggable).

Responsibilities:
  - Provide ONE stable API: improve_with_local_search(...)
  - Support multiple LS backends:
      * "pyvrp"  : PyVRP LocalSearch with DRI neighbourhood (existing behaviour)
      * "hexaly" : time-limited Hexaly re-optimisation with a warm start
      * "none"   : no-op (returns input routes unchanged)
  - Keep VRPLIB route format throughout the pipeline:
      routes are lists of node IDs with depot = 1 at start/end:
          [1, i1, i2, ..., 1]

Assumptions:
  - Single-depot CVRP in standard VRPLIB format.
  - Depot node ID is 1, customers are 2..n+1.
  - For PyVRP LS, PyVRP's reader is used to keep distances consistent.
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple, Literal, Optional, Any

from master.utils.loader import load_instance
from master.clustering.dissimilarity.spatial import spatial_dissimilarity
from master.clustering.dissimilarity.combined import combined_dissimilarity
from master.utils.symmetric_matrix_read import get_symmetric_value

# PyVRP imports are optional at runtime (only needed if ls_solver="pyvrp")
try:
    from pyvrp import (
        read,  # read(instance_path)
        CostEvaluator,
        RandomNumberGenerator,
        Solution,
    )
    from pyvrp.search import (
        LocalSearch,
        NODE_OPERATORS,
        ROUTE_OPERATORS,
    )

    _HAS_PYVRP = True
except Exception:
    _HAS_PYVRP = False
        # Debug: print the actual import error to help diagnose
    import sys
    if "--debug-imports" in sys.argv:  # Only print if debug flag is set
        print(f"[DEBUG] PyVRP import failed")

# Hexaly imports are optional at runtime (only needed if ls_solver="hexaly")
try:
    import hexaly.optimizer  # pyright: ignore[reportMissingImports]

    _HAS_HEXALY = True
except Exception:
    _HAS_HEXALY = False


Route = List[int]
Routes = List[Route]


# ======================================================================
# Helpers: resolve instance path for PyVRP.read
# ======================================================================

def _resolve_instance_path(instance_name: str) -> str:
    """
    Resolves the path to an instance file.
    Searches in:
        core/instances/test-instances/x
        core/instances/test-instances/xl
        core/instances/challenge-instances
    
    Args:
        instance_name: Either a full path to the instance file, or just the filename.
                      If a full path is provided, only the basename will be used for searching.
    """
    base_dir = os.path.dirname(__file__)  # .../core/src/master/improve
    core_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    
    # Extract just the filename from instance_name (in case a full path is provided)
    instance_filename = os.path.basename(instance_name)
    
    # Define all search locations (order matters!)
    search_paths = [
        os.path.join(core_root, "instances", "test-instances", "x"),
        os.path.join(core_root, "instances", "test-instances", "xl"),
        os.path.join(core_root, "instances", "challenge-instances"),
    ]

    for path in search_paths:
        p = os.path.join(path, instance_filename)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Instance '{instance_filename}' not found in any of:\n  "
        + "\n  ".join(search_paths)
    )


# ======================================================================
# Helpers: neighbourhood from DRI dissimilarities (PyVRP-only)
# ======================================================================

def _build_dri_neighbours(
    instance_name: str,
    num_locations: int,
    use_demand: bool = False,
    max_neighbours: int = 100,
) -> List[List[int]]:
    """
    Build a granular neighbourhood structure for PyVRP LocalSearch based on
    our own dissimilarity metric.

    PyVRP expects:
        neighbours: list[list[int]] of length = num_locations
    where:
        - lower indices correspond to depots and are empty;
        - remaining indices correspond to client locations.

    Our VRPLIB convention:
        node 1  -> depot  -> location 0
        node i  -> client -> location (i-1), for i >= 2

    We compute:
        - S^s_ij  (spatial)        if use_demand = False
        - S^sd_ij (combined)       if use_demand = True
    and, for each customer i, keep the max_neighbours smallest S_ij.

    The neighbour lists are expressed in terms of *location indices*
    (0..num_locations-1) as required by PyVRP.
    """
    S = combined_dissimilarity(instance_name) if use_demand else spatial_dissimilarity(instance_name)

    nodes = sorted({n for pair in S.keys() for n in pair})
    neighbours: List[List[int]] = [[] for _ in range(num_locations)]

    for i in nodes:
        cand: List[Tuple[float, int]] = []
        for j in nodes:
            if j == i:
                continue
            dij = get_symmetric_value(S, i, j)
            cand.append((dij, j))

        cand.sort(key=lambda x: x[0])
        selected = [j for _, j in cand[:max_neighbours]]

        loc_i = i - 1  # VRPLIB node 2 -> loc 1, etc.
        neighbours[loc_i] = [j - 1 for j in selected]

    return neighbours


# ======================================================================
# Helpers: capacity feasibility check
# ======================================================================

def _check_capacity_feasibility(
    inst: Dict[str, object],
    routes_vrplib: Routes,
    label: str,
) -> None:
    cap = int(inst["capacity"])
    demands = inst["demand"]

    violating: List[Tuple[int, int]] = []
    for r_idx, route in enumerate(routes_vrplib):
        load = sum(int(demands[nid - 1]) for nid in route if nid != 1)
        if load > cap:
            violating.append((r_idx, load))

    if violating:
        preview = ", ".join(f"route {idx} load={load}" for idx, load in violating[:5])
        raise ValueError(
            f"[LS] {label} has capacity violations (cap={cap}). "
            f"Examples: {preview}. Total violating routes: {len(violating)}."
        )


# ======================================================================
# Helpers: route normalisation + simple integer-rounded distance cost
# ======================================================================

def _normalise_routes(routes_vrplib: Routes) -> Routes:
    """
    Ensure each route:
      - is non-empty
      - starts and ends with depot 1
      - has no depot in the middle (we drop any stray 1s)
    """
    out: Routes = []
    for r in routes_vrplib:
        if not r:
            continue
        core = [nid for nid in r if nid != 1]
        if not core:
            continue
        out.append([1] + core + [1])
    return out


def _integer_rounded_cost(inst: Dict[str, object], routes_vrplib: Routes) -> int:
    coords = inst["node_coord"]
    edge_mat = inst.get("edge_weight")

    def dist(u: int, v: int) -> int:
        u0, v0 = u - 1, v - 1
        if edge_mat is not None:
            return int(round(float(edge_mat[u0, v0])))
        x1, y1 = coords[u0]
        x2, y2 = coords[v0]
        return int(round(math.hypot(x2 - x1, y2 - y1)))

    total = 0
    for r in routes_vrplib:
        for a, b in zip(r, r[1:]):
            total += dist(a, b)
    return total


# ======================================================================
# PyVRP backend (existing behaviour, wrapped)
# ======================================================================

def _vrplib_routes_to_solution(data, routes_vrplib: Routes):
    routes_clients = []
    for r in routes_vrplib:
        clients = [(nid - 1) for nid in r if nid != 1]
        if clients:
            routes_clients.append(clients)
    return Solution(data, routes_clients)


def _solution_to_vrplib_routes(sol) -> Routes:
    vrp_routes: Routes = []
    for route in sol.routes():
        clients = list(route.visits())
        if not clients:
            continue
        seq = [1] + [(idx + 1) for idx in clients] + [1]
        vrp_routes.append(seq)
    return vrp_routes


def _improve_with_pyvrp_local_search(
    instance_name: str,
    routes_vrplib: Routes,
    *,
    neighbourhood: Literal["dri_spatial", "dri_combined"] = "dri_spatial",
    max_neighbours: int = 40,
    seed: int = 0,
    load_penalty: int = 1_000_000,
    dist_penalty: int = 1,
) -> Dict[str, object]:
    if not _HAS_PYVRP:
        raise ImportError("[LS] PyVRP is not available, cannot run ls_solver='pyvrp'.")

    instance_path = _resolve_instance_path(instance_name)
    data = read(instance_path)

    inst = load_instance(instance_name)
    dim = int(inst["dimension"])
    assert data.num_locations == dim, (
        f"PyVRP data.num_locations={data.num_locations} != VRPLIB DIMENSION={dim}"
    )

    routes_vrplib = _normalise_routes(routes_vrplib)
    _check_capacity_feasibility(inst, routes_vrplib, label="input routes to LS (pyvrp)")

    rng = RandomNumberGenerator(seed)
    cost_eval = CostEvaluator(
        load_penalties=[load_penalty],
        tw_penalty=0,
        dist_penalty=dist_penalty,
    )

    use_demand = (neighbourhood == "dri_combined")
    neighbours = _build_dri_neighbours(
        instance_name=instance_name,
        num_locations=data.num_locations,
        use_demand=use_demand,
        max_neighbours=max_neighbours,
    )

    ls = LocalSearch(data, rng, neighbours)
    for node_op in NODE_OPERATORS:
        ls.add_node_operator(node_op(data))
    for route_op in ROUTE_OPERATORS:
        ls.add_route_operator(route_op(data))

    sol_initial = _vrplib_routes_to_solution(data, routes_vrplib)
    initial_cost = cost_eval.penalised_cost(sol_initial)

    sol_improved = ls(sol_initial, cost_eval)
    improved_cost = cost_eval.penalised_cost(sol_improved)

    routes_improved = _solution_to_vrplib_routes(sol_improved)
    routes_improved = _normalise_routes(routes_improved)

    _check_capacity_feasibility(inst, routes_improved, label="routes after LS (pyvrp)")

    stats = ls.statistics
    return {
        "initial_cost": float(initial_cost),
        "improved_cost": float(improved_cost),
        "routes_initial": routes_vrplib,
        "routes_improved": routes_improved,
        "ls_moves": int(stats.num_moves),
        "ls_improving_moves": int(stats.num_improving),
        "ls_updates": int(stats.num_updates),
        "backend": "pyvrp",
    }


# ======================================================================
# Hexaly backend (time-limited re-optimisation warm-start)
# ======================================================================

def _improve_with_hexaly_local_search(
    instance_name: str,
    routes_vrplib: Routes,
    *,
    time_limit: float = 2.0,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Hexaly "LS" = short re-optimisation run with a warm start.

    Notes:
      - This does NOT use DRI neighbourhoods (those are PyVRP-specific).
      - It must preserve feasibility, so we keep capacity constraints in the model.
      - We treat current routes as an initial solution (warm-start), if possible.
    """
    if not _HAS_HEXALY:
        raise ImportError(
            "[LS] Hexaly is not available, cannot run ls_solver='hexaly'. "
            "Install with: pip install hexaly -i https://pip.hexaly.com"
        )

    inst = load_instance(instance_name)
    routes_vrplib = _normalise_routes(routes_vrplib)
    _check_capacity_feasibility(inst, routes_vrplib, label="input routes to LS (hexaly)")

    initial_cost_int = _integer_rounded_cost(inst, routes_vrplib)

    coords = inst["node_coord"]
    demands = inst["demand"]
    capacity = int(inst["capacity"])
    dim = int(inst["dimension"])
    depot = 1
    n_customers = dim - 1  # excluding depot

    def dist(u: int, v: int) -> int:
        x1, y1 = coords[u - 1]
        x2, y2 = coords[v - 1]
        return int(round(math.hypot(x2 - x1, y2 - y1)))

    dist_matrix = [
        [dist(i + 1, j + 1) for j in range(n_customers + 1)]
        for i in range(n_customers + 1)
    ]

    # Warm-start encoding: convert VRPLIB routes -> 0-based customer indices for Hexaly list vars
    warm_routes: List[List[int]] = []
    for r in routes_vrplib:
        core = [nid for nid in r if nid != 1]  # customers as VRPLIB ids
        if core:
            warm_routes.append([nid - 2 for nid in core])  # VRPLIB 2.. -> 0..

    with hexaly.optimizer.HexalyOptimizer() as opt:
        m = opt.model

        # Upper bound on number of vehicles: n_customers
        routes = [m.list(n_customers) for _ in range(n_customers)]
        m.constraint(m.partition(routes))  # each customer exactly once

        demand_arr = m.array([int(d) for d in demands[1:]])      # customers only, 0..n-1
        dist_arr = m.array(dist_matrix)

        route_costs = []

        for r in routes:
            size = m.count(r)

            # Capacity
            load = m.sum(r, m.lambda_function(lambda i: demand_arr[i]))
            m.constraint(load <= capacity)

            # Travel within route
            internal = m.sum(
                m.range(1, size),
                m.lambda_function(lambda i: m.at(dist_arr, r[i - 1] + 1, r[i] + 1)),
            )

            # Add depot legs
            depot_legs = m.iif(
                size > 0,
                m.at(dist_arr, 0, r[0] + 1) + m.at(dist_arr, r[size - 1] + 1, 0),
                0,
            )

            route_costs.append(internal + depot_legs)

        total_cost = m.sum(route_costs)
        m.minimize(total_cost)
        m.close()

        # Params
        try:
            opt.param.seed = int(seed)
        except Exception:
            pass

        if time_limit is not None:
            opt.param.time_limit = int(math.ceil(float(time_limit)))

        if not verbose:
            try:
                opt.param.verbosity = 0
            except Exception:
                pass

        # Warm-start (best-effort): set initial values if supported
        # Different Hexaly versions expose different APIs; we keep this robust.
        try:
            for idx, seq in enumerate(warm_routes):
                if idx >= len(routes):
                    break
                routes[idx].value = list(seq)  # some versions accept this
        except Exception:
            pass

        opt.solve()

        # Extract VRPLIB routes
        improved: Routes = []
        for r in routes:
            seq = getattr(r, "value", None)
            if seq:
                improved.append([1] + [int(i) + 2 for i in seq] + [1])

    improved = _normalise_routes(improved)
    _check_capacity_feasibility(inst, improved, label="routes after LS (hexaly)")

    improved_cost_int = _integer_rounded_cost(inst, improved)

    return {
        "initial_cost": float(initial_cost_int),
        "improved_cost": float(improved_cost_int),
        "routes_initial": routes_vrplib,
        "routes_improved": improved,
        "ls_moves": 0,
        "ls_improving_moves": 0,
        "ls_updates": 0,
        "backend": "hexaly",
    }


# ======================================================================
# AILS2 backend
# ======================================================================

def _improve_with_ails2_local_search(
    instance_name: str,
    routes_vrplib: Routes,
    *,
    time_limit: float = 10.0,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Improve routes using AILS2 re-optimisation.
    
    Note: AILS2 doesn't support warm-start solutions, so it will start from scratch
    and search for improvements. The input routes are used to compute initial cost
    for comparison.
    """
    from master.routing.solver import solve as routing_solve
    
    inst = load_instance(instance_name)
    routes_vrplib = _normalise_routes(routes_vrplib)
    _check_capacity_feasibility(inst, routes_vrplib, label="input routes to LS (ails2)")
    
    initial_cost_int = _integer_rounded_cost(inst, routes_vrplib)
    
    # Calculate adaptive time limit based on instance size
    # Use adaptive cluster time formula based on number of customers
    num_customers = len(inst.get("demand", [])) - 1  # Exclude depot
    if num_customers > 0:
        # Same formula as routing_controller._adaptive_cluster_time
        base = 1.0
        alpha = 0.25
        exponent = 0.9
        min_time = 2.0
        max_time = 180.0
        adaptive_time = base + alpha * (num_customers ** exponent)
        adaptive_time_limit = max(min_time, min(max_time, adaptive_time))
    else:
        adaptive_time_limit = time_limit  # Fallback to provided time_limit
    
    # Always use adaptive time limit (ignore input time_limit parameter)
    # This ensures AILS2 uses adaptive time based on instance size
    
    # Run AILS2 on the full instance with adaptive time limit
    result = routing_solve(
        instance=instance_name,
        solver="ails2",
        solver_options={
            "max_runtime": adaptive_time_limit,  # Use adaptive time based on instance size
            "rounded": True,
            "best": 0.0,  # We don't know the optimal, use 0
        },
    )
    
    improved = result.metadata.get("routes_vrplib", [])
    improved = _normalise_routes(improved)
    
    # Check feasibility
    if improved:
        _check_capacity_feasibility(inst, improved, label="routes after LS (ails2)")
    
    improved_cost_int = _integer_rounded_cost(inst, improved) if improved else float("inf")
    
    return {
        "initial_cost": float(initial_cost_int),
        "improved_cost": float(improved_cost_int),
        "routes_initial": routes_vrplib,
        "routes_improved": improved if improved else routes_vrplib,  # Fallback to original if no solution
        "ls_moves": 0,
        "ls_improving_moves": 0,
        "ls_updates": 0,
        "backend": "ails2",
    }


# ======================================================================
# Public API: improve with LocalSearch (dispatcher)
# ======================================================================

def improve_with_local_search(
    instance_name: str,
    routes_vrplib: Routes,
    *,
    neighbourhood: Literal["dri_spatial", "dri_combined"] = "dri_spatial",
    max_neighbours: int = 40,
    seed: int = 0,
    load_penalty: int = 1_000_000,
    dist_penalty: int = 1,
    ls_solver: Literal["pyvrp", "hexaly", "ails2", "none"] = "pyvrp",
    # Hexaly-specific knobs (ignored by pyvrp/ails2/none)
    hexaly_time_limit: float = 2.0,
    hexaly_verbose: bool = False,
    # AILS2-specific knobs (ignored by pyvrp/hexaly/none)
    ails2_time_limit: float = 10.0,
) -> Dict[str, object]:
    """
    Pluggable LS entry point.

    ls_solver:
      - "pyvrp": existing PyVRP LocalSearch (uses neighbourhood/max_neighbours)
      - "hexaly": short Hexaly re-optimisation (uses hexaly_time_limit)
      - "ails2": AILS2 re-optimisation (uses ails2_time_limit)
      - "none": no-op
    """
    solver = ls_solver.lower().strip()

    if solver == "none":
        routes_vrplib = _normalise_routes(routes_vrplib)
        return {
            "initial_cost": None,
            "improved_cost": None,
            "routes_initial": routes_vrplib,
            "routes_improved": routes_vrplib,
            "ls_moves": 0,
            "ls_improving_moves": 0,
            "ls_updates": 0,
            "backend": "none",
        }

    if solver == "pyvrp":
        return _improve_with_pyvrp_local_search(
            instance_name,
            routes_vrplib,
            neighbourhood=neighbourhood,
            max_neighbours=max_neighbours,
            seed=seed,
            load_penalty=load_penalty,
            dist_penalty=dist_penalty,
        )

    if solver == "hexaly":
        return _improve_with_hexaly_local_search(
            instance_name,
            routes_vrplib,
            time_limit=hexaly_time_limit,
            seed=seed,
            verbose=hexaly_verbose,
        )

    if solver == "ails2":
        return _improve_with_ails2_local_search(
            instance_name,
            routes_vrplib,
            time_limit=ails2_time_limit,
            seed=seed,
        )

    raise ValueError(f"[LS] Unknown ls_solver='{ls_solver}'. Use one of: pyvrp, hexaly, ails2, none.")
