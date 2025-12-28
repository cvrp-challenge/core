"""
DRSCI-DUAL (Decompose–Route–RestrictedSP–Improve)
------------------------------------------------

Dual-based extension of baseline DRSCI.

Key differences vs baseline:
- Explicit route pool management (RoutePoolManager)
- ONE LP relaxation on the GLOBAL pool to obtain dual prices
- Reduced-cost based route filtering (global)
- Restricted Set Partitioning instead of full SCP

Pipeline implemented here (as requested):
(cluster → routing → LS) × ALL methods/k
→ global pool
→ LP (partition)
→ dual filter
→ restricted SP
→ duplicate removal
→ LS

IMPORTANT:
- This file does NOT modify or replace run_drsci.py
- It is a side-by-side experimental pipeline
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent
SRC_ROOT = CURRENT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------
# Project imports (baseline components reused)
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.routing.routing_controller import solve_clusters
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.utils.loader import load_instance

# ---------------------------------------------------------
# Dual / restricted-SP components (NEW)
# ---------------------------------------------------------
from master.setcover.route_selection_dual.route_pool_manager import RoutePoolManager
from master.setcover.route_selection_dual.lp_relaxation import solve_lp_relaxation
from master.setcover.route_selection_dual.route_filtering import filter_by_reduced_cost
from master.setcover.route_selection_dual.sp_solver_gurobi_MIP import solve_restricted_sp


Route = List[int]   # VRPLIB route: [1, ..., 1]
Routes = List[Route]


# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------
VB_CLUSTER_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "k_medoids_pyclustering",
]

K_PER_METHOD_DEFAULT = {
    "sk_ac_avg": [2],
    "sk_ac_complete": [2],
    "sk_ac_min": [2],
    "sk_kmeans": [2],
    "fcm": [2],
    "k_medoids_pyclustering": [2],
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _result_to_vrplib_routes(result) -> Routes:
    best = result.best
    if best is None:
        return []

    routes: Routes = []
    for r in best.routes():
        visits = [v for v in r.visits() if v > 0]
        if visits:
            routes.append([1] + [v + 1 for v in visits] + [1])
    return routes


def compute_integer_rounded_cost(instance: dict, routes: Routes) -> int:
    coords = instance["node_coord"]
    edge_mat = instance.get("edge_weight")

    def dist(u: int, v: int) -> int:
        u_idx, v_idx = u - 1, v - 1
        if edge_mat is not None:
            return int(round(float(edge_mat[u_idx, v_idx])))
        dx = coords[u_idx][0] - coords[v_idx][0]
        dy = coords[u_idx][1] - coords[v_idx][1]
        return int(round(math.hypot(dx, dy)))

    total = 0
    for r in routes:
        for a, b in zip(r, r[1:]):
            total += dist(a, b)
    return total


def _format_cluster_sizes(clusters: Dict[Any, List[int]]) -> str:
    sizes = [len(v) for v in clusters.values()]
    if not sizes:
        return "sizes=[] | min=0 max=0 avg=0.0"
    avg = sum(sizes) / len(sizes)
    return f"sizes={sizes} | min={min(sizes)} max={max(sizes)} avg={avg:.1f}"


def _percentile_from_sorted(sorted_vals: List[float], p: float) -> float:
    """
    p in [0,1]. Simple, deterministic percentile from a pre-sorted list.
    """
    if not sorted_vals:
        return 0.0
    if p <= 0.0:
        return float(sorted_vals[0])
    if p >= 1.0:
        return float(sorted_vals[-1])
    idx = int(p * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


# =====================================================================
# MAIN DRIVER (DUAL)
# =====================================================================
def run_drsci_dual_for_instance(
    instance_name: str,
    *,
    seed: int = 0,
    time_limit_per_cluster: float = 30.0,
    sp_time_limit: float = 600.0,  # used for ONE LP + ONE restricted SP stage
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 100,
    ls_max_neighbours_restricted: int = 100,
    methods: Optional[List[str]] = None,
    k_per_method: Optional[Dict[str, List[int]]] = None,
    routing_solver: str = "pyvrp",
    # filtering knobs (keep simple for now)
    rc_threshold: float = 0.0,
    max_pool_size: Optional[int] = None,  # e.g. 20000 if you want a cap
    verbose_lp: bool = False,
) -> Dict[str, Any]:

    start_time = time.time()
    inst = load_instance(instance_name)

    if methods is None:
        methods = VB_CLUSTER_METHODS

    if k_per_method is None:
        k_per_method = {
            m: list(K_PER_METHOD_DEFAULT[m])
            for m in methods
            if m in K_PER_METHOD_DEFAULT
        }

    total_stages = sum(len(k_per_method[m]) for m in methods)
    stages = 0

    # --------------------------------------------------
    # Global route pool (filled across ALL methods/k)
    # --------------------------------------------------
    pool = RoutePoolManager()
    pool.attach_instance(inst)

    # --------------------------------------------------
    # Phase A: generate routes (cluster → routing → LS) for all methods/k
    # --------------------------------------------------
    for method in methods:
        for k in k_per_method[method]:
            stages += 1

            clusters, _ = run_clustering(
                method=method,
                instance_name=instance_name,
                k=k,
                use_combined=False,
            )

            print(
                f"[CLUSTER] method={method} k={k} | {_format_cluster_sizes(clusters)}",
                flush=True,
            )

            routing = solve_clusters(
                instance_name=instance_name,
                clusters=clusters,
                solver=routing_solver,
                time_limit_per_cluster=time_limit_per_cluster,
                seed=seed,
            )

            vb_routes = _result_to_vrplib_routes(routing)

            ls_vb = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=vb_routes,
                neighbourhood=ls_neighbourhood,
                max_neighbours=ls_after_routing_max_neighbours,
                seed=seed,
                ls_solver="pyvrp",  # keep fixed (as you wanted)
            )

            # Add improved routes to GLOBAL pool
            pool.add_routes(ls_vb["routes_improved"], source=f"VB:{method}:{k}")

            # Optional: cap pool size deterministically
            if max_pool_size is not None:
                pool.trim_to_max_size(int(max_pool_size))

            print(
                f"[POOL] stage={stages}/{total_stages} size={pool.size()}",
                flush=True,
            )

    # --------------------------------------------------
    # Phase B: ONE LP (partition) → ONE filter → ONE restricted SP
    # --------------------------------------------------
    print(f"[GLOBAL] pool_size={pool.size()} (before LP)", flush=True)

    # Edge case: empty pool
    if pool.size() == 0:
        runtime = time.time() - start_time
        return {
            "instance": instance_name,
            "best_cost": float("inf"),
            "routes": [],
            "runtime": runtime,
            "route_pool_size": 0,
            "restricted_pool_size": 0,
            "stages": stages,
        }

    lp_res = solve_lp_relaxation(
        customers=pool.customers,
        route_cust=pool.route_cust,
        costs=pool.costs,
        mode="partition",
        time_limit=sp_time_limit,
        verbose=verbose_lp,
    )

    # If LP failed / no solution: don't touch duals, don't crash
    if lp_res.lp_obj_value is None or not lp_res.pi:
        print(
            f"[LP] No usable LP solution (status={lp_res.status}). "
            f"Skipping dual-filter; using full pool as restricted pool.",
            flush=True,
        )
        duals: Dict[int, float] = {}
        filtered_routes = pool.routes
        print(
            f"[DUAL-FILTER] pool={pool.size()} → restricted={len(filtered_routes)} (thr={rc_threshold})",
            flush=True,
        )
    else:
        duals = lp_res.pi

        # --------------------------------------------------
        # Diagnostics: reduced cost distribution (safe scope)
        # --------------------------------------------------
        rc_vals: List[float] = []
        for r, custs in enumerate(pool.route_cust):
            rc_r = pool.costs[r] - sum(duals.get(i, 0.0) for i in custs)
            rc_vals.append(rc_r)

        rc_sorted = sorted(rc_vals)
        neg = sum(1 for v in rc_vals if v < 0.0)

        print(
            f"[RC-STATS] min={rc_sorted[0]:.2f} "
            f"p10={_percentile_from_sorted(rc_sorted, 0.10):.2f} "
            f"median={_percentile_from_sorted(rc_sorted, 0.50):.2f} "
            f"p90={_percentile_from_sorted(rc_sorted, 0.90):.2f} "
            f"neg={neg}/{len(rc_vals)}",
            flush=True,
        )

        # --------------------------------------------------
        # Dual-based filtering (global)
        # --------------------------------------------------
        filtered_routes = filter_by_reduced_cost(
            routes=pool.routes,   # property -> list of routes
            costs=pool.costs,
            duals=duals,
            threshold=rc_threshold,
        )

        print(
            f"[DUAL-FILTER] pool={pool.size()} → restricted={len(filtered_routes)} (thr={rc_threshold})",
            flush=True,
        )

        # Safety: never allow empty restricted pool (would make SP infeasible)
        if not filtered_routes:
            print(
                "[DUAL-FILTER] Restricted pool became empty; falling back to full pool.",
                flush=True,
            )
            filtered_routes = pool.routes

    # --------------------------------------------------
    # ONE Restricted SP
    # --------------------------------------------------

    print(
        f"[SP-INPUT] routes={len(filtered_routes)} "
        f"(reduction={(1 - len(filtered_routes)/pool.size()):.1%})",
        flush=True,
    )


    sp_res = solve_restricted_sp(
        instance_name=instance_name,
        route_pool=filtered_routes,   # IMPORTANT: keyword is route_pool
        time_limit=sp_time_limit,
        verbose=False,
    )

    selected = sp_res.get("selected_routes", [])
    if not selected:
        print("[SP] No selected routes returned; returning empty solution.", flush=True)
        runtime = time.time() - start_time
        return {
            "instance": instance_name,
            "best_cost": float("inf"),
            "routes": [],
            "runtime": runtime,
            "route_pool_size": pool.size(),
            "restricted_pool_size": len(filtered_routes),
            "stages": stages,
        }

    # --------------------------------------------------
    # Post: duplicate removal + LS on chosen routes
    # --------------------------------------------------
    dup_res = remove_duplicates(
        instance_name=instance_name,
        routes=selected,
        verbose=False,
        max_iters=50,
        ls_neighbourhood=ls_neighbourhood,
        ls_max_neighbours_restricted=ls_max_neighbours_restricted,
        seed=seed,
    )

    ls_res = improve_with_local_search(
        instance_name=instance_name,
        routes_vrplib=dup_res["routes"],
        neighbourhood=ls_neighbourhood,
        max_neighbours=ls_max_neighbours_restricted,
        seed=seed,
        ls_solver="pyvrp",
    )

    final_routes = ls_res["routes_improved"]
    final_cost = compute_integer_rounded_cost(inst, final_routes)

    # Mark incumbent if RoutePoolManager supports it (don’t crash if not)
    if hasattr(pool, "mark_incumbent"):
        try:
            pool.mark_incumbent(final_routes, source="INCUMBENT")
        except TypeError:
            # older signature without source kw
            pool.mark_incumbent(final_routes)
        except Exception:
            pass

    runtime = time.time() - start_time

    return {
        "instance": instance_name,
        "best_cost": final_cost,
        "routes": final_routes,
        "runtime": runtime,
        "route_pool_size": pool.size(),
        "restricted_pool_size": len(filtered_routes),
        "stages": stages,
    }


# ---------------------------------------------------------
# Debug run
# ---------------------------------------------------------
if __name__ == "__main__":
    res = run_drsci_dual_for_instance(
        instance_name="XLTEST-n1794-k408.vrp",
        seed=0,
        routing_solver="pyvrp",
        rc_threshold=-1.0,         # start here
        # max_pool_size=20000,      # optional safety cap
        verbose_lp=False,
    )

    print("\n[DEBUG] Best cost:", res["best_cost"])
    print("[DEBUG] #routes:", len(res["routes"]))
    print("[DEBUG] pool_size:", res["route_pool_size"])
    print("[DEBUG] restricted_pool_size:", res["restricted_pool_size"])
