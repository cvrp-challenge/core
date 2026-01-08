"""
DRSCI (Decompose–Route–SetCover–Improve) multi-k driver
-------------------------------------------------------

This script implements a DRSCI-style metaheuristic compatible with
benchmark_dri.py, with:

✔ PyVRP-compatible integer-rounded objective computation
✔ Full runtime tracking per instance
✔ Pluggable routing solvers (PyVRP / Hexaly / FILO)
✔ Pluggable SCP solvers (Gurobi / Hexaly)
✔ Additional diagnostics:
    - Full cluster size listing per stage
    - SCP timing + effectiveness (pool size -> selected size)
"""

from __future__ import annotations

import sys
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent
SRC_ROOT = CURRENT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.clustering.route_based import route_based_decomposition
from master.routing.routing_controller import solve_clusters
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.utils.loader import load_instance
from master.setcover.route_dominance_filter import filter_route_pool



# ---------------------------------------------------------
# SCP solver dispatcher
# ---------------------------------------------------------
def lazy_import_scp(scp_solver: str):
    if scp_solver == "gurobi_mip":
        from master.setcover.scp_solver_gurobi_MIP import solve_scp
    elif scp_solver == "gurobi_lp":
        from master.setcover.scp_solver_gurobi_LP import solve_scp
    elif scp_solver == "hexaly":
        from master.setcover.scp_solver_hexaly import solve_scp
    else:
        raise ValueError(f"Unknown SCP solver: {scp_solver}")
    return solve_scp


Route = List[int]   # VRPLIB format: [1, ..., 1]
Routes = List[Route]

# ---------------------------------------------------------
# Methods & defaults
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
    "sk_ac_avg": [4],
    "sk_ac_complete": [4],
    "sk_ac_min": [4],
    "sk_kmeans": [4],
    "fcm": [4],
    "k_medoids_pyclustering": [4],
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _result_to_vrplib_routes(result) -> Routes:
    best = result.best
    if best is None:
        return []

    routes = []
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
    """
    Returns a compact string listing ALL cluster sizes, plus min/max/avg.
    Example:
      sizes=[83, 102, 121, 110] | min=83 max=121 avg=104.0
    """
    sizes = [len(v) for v in clusters.values()]
    if not sizes:
        return "sizes=[] | min=0 max=0 avg=0.0"
    avg = sum(sizes) / len(sizes)
    return f"sizes={sizes} | min={min(sizes)} max={max(sizes)} avg={avg:.1f}"


def _scp_plus_ls(
    *,
    instance_name: str,
    route_pool: Routes,
    solve_scp,
    ls_neighbourhood: str,
    ls_max_neighbours_restricted: int,
    seed: int,
    scp_time_limit: float,
    best_cost_so_far: int | None,
) -> Tuple[Routes, int]:

    # --------------------------------------------------------------
    # SCP timing + effectiveness diagnostics (B + C)
    # --------------------------------------------------------------
    t0 = time.time()
    scp_res = solve_scp(
        instance_name=instance_name,
        route_pool=route_pool,
        time_limit=scp_time_limit,
        verbose=False,
    )
    scp_time = time.time() - t0

    selected = scp_res["selected_routes"]

    # Print SCP timing + effectiveness (pool -> selected)
    solver_tag = getattr(solve_scp, "__module__", "unknown").split(".")[-1]
    print(
        f"[SCP] solver={solver_tag} | pool={len(route_pool)} -> selected={len(selected)} | time={scp_time:.2f}s",
        flush=True,
    )

    dup_res = remove_duplicates(
        instance_name=instance_name,
        routes=selected,
        verbose=True,
        max_iters=50,
        ls_neighbourhood=ls_neighbourhood,
        ls_max_neighbours_restricted=ls_max_neighbours_restricted,
        seed=seed,
    )
    repaired = dup_res["routes"]

    ls_res = improve_with_local_search(
        instance_name=instance_name,
        routes_vrplib=repaired,
        neighbourhood=ls_neighbourhood,
        max_neighbours=ls_max_neighbours_restricted,
        seed=seed,
    )

    inst = load_instance(instance_name)
    final_routes = ls_res["routes_improved"]
    final_cost = compute_integer_rounded_cost(inst, final_routes)

    # --------------------------------------------------------------
    # Accept SCP solution ONLY if it improves the incumbent
    # --------------------------------------------------------------
    if best_cost_so_far is not None and final_cost >= best_cost_so_far:
        print(
            f"[SCP-REJECTED] best={best_cost_so_far} | "
            f"scp={final_cost} | Δ={final_cost - best_cost_so_far:+}",
            flush=True,
        )
        return [], best_cost_so_far


    return final_routes, final_cost


# =====================================================================
# MAIN DRIVER
# =====================================================================
def run_drsci_for_instance(
    instance_name: str,
    *,
    seed: int = 0,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 100,
    ls_max_neighbours_restricted: int = 100,
    scp_time_limit: float = 600.0,
    use_combined_dissimilarity: bool = False,
    methods: Optional[List[str]] = VB_CLUSTER_METHODS,
    k_per_method: Optional[Dict[str, List[int]]] = None,
    routing_solver: str = "pyvrp",
    scp_solver: str = "gurobi_mip",
    ls_solver: str = "pyvrp",
) -> Dict[str, Any]:

    start_time = time.time()
    solve_scp = lazy_import_scp(scp_solver)
    inst = load_instance(instance_name)

    if k_per_method is None:
        k_per_method = {m: list(K_PER_METHOD_DEFAULT[m]) for m in K_PER_METHOD_DEFAULT}

    total_stages = sum(len(k_per_method[m]) for m in methods) * 2

    global_pool: Routes = []
    best_routes: Optional[Routes] = None
    best_cost = float("inf")
    stages = 0

    for method in methods:
        for k in k_per_method[method]:
            stages += 1

            clusters, _ = run_clustering(
                method=method,
                instance_name=instance_name,
                k=k,
                use_combined=use_combined_dissimilarity,
            )

            # ----------------------------------------------------------
            # Cluster size diagnostics: list ALL cluster sizes
            # ----------------------------------------------------------
            print(
                f"[CLUSTER] method={method} k={k} | {_format_cluster_sizes(clusters)}",
                flush=True,
            )

            routing = solve_clusters(
                instance_name=instance_name,
                clusters=clusters,
                solver=routing_solver,
                seed=seed,
            )
            vb_routes = _result_to_vrplib_routes(routing)

            ls_vb = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=vb_routes,
                neighbourhood=ls_neighbourhood,
                max_neighbours=ls_after_routing_max_neighbours,
                seed=seed,
                ls_solver=ls_solver,
            )

            global_pool.extend(ls_vb["routes_improved"])
            global_pool = filter_route_pool(global_pool, depot_id=1, verbose=True)


            vb_final, vb_cost = _scp_plus_ls(
                instance_name=instance_name,
                route_pool=global_pool,
                solve_scp=solve_scp,
                ls_neighbourhood=ls_neighbourhood,
                ls_max_neighbours_restricted=ls_max_neighbours_restricted,
                seed=seed,
                scp_time_limit=scp_time_limit,
                best_cost_so_far=best_cost,

            )

            if vb_cost < best_cost:
                best_cost = vb_cost
                best_routes = vb_final

            if best_routes is None:
                print(f"{instance_name} arrived at stage {stages}/{total_stages}.", flush=True)
                continue

            stages += 1
            clusters_rb = route_based_decomposition(
                instance_name=instance_name,
                global_routes=best_routes,
                k=k,
                method="sk_kmeans",
                use_angle=True,
                use_load=True,
            )

            # ----------------------------------------------------------
            # Cluster size diagnostics for RB decomposition
            # (route_based_decomposition might return a different structure;
            #  if it's dict-like, this will work; otherwise, we fall back.)
            # ----------------------------------------------------------
            try:
                print(
                    f"[CLUSTER-RB] k={k} | {_format_cluster_sizes(clusters_rb)}",
                    flush=True,
                )
            except Exception:
                # Safe fallback: at least print how many clusters we got
                try:
                    print(f"[CLUSTER-RB] k={k} | num_clusters={len(clusters_rb)}", flush=True)
                except Exception:
                    print(f"[CLUSTER-RB] k={k} | (cluster stats unavailable)", flush=True)

            routing_rb = solve_clusters(
                instance_name=instance_name,
                clusters=clusters_rb,
                solver=routing_solver,
                seed=seed,
            )
            rb_routes = _result_to_vrplib_routes(routing_rb)

            ls_rb = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=rb_routes,
                neighbourhood=ls_neighbourhood,
                max_neighbours=ls_after_routing_max_neighbours,
                seed=seed,
                ls_solver=ls_solver,
            )

            global_pool.extend(ls_rb["routes_improved"])
            global_pool = filter_route_pool(global_pool, depot_id=1, verbose=True)

            rb_final, rb_cost = _scp_plus_ls(
                instance_name=instance_name,
                route_pool=global_pool,
                solve_scp=solve_scp,
                ls_neighbourhood=ls_neighbourhood,
                ls_max_neighbours_restricted=ls_max_neighbours_restricted,
                seed=seed,
                scp_time_limit=scp_time_limit,
                best_cost_so_far=best_cost,
            )

            if rb_cost < best_cost:
                best_cost = rb_cost
                best_routes = rb_final

            print(f"{instance_name} arrived at stage {stages}/{total_stages}.", flush=True)

    final_runtime = time.time() - start_time

    print("\n=== FINAL ROUTES (customer-only) ===")
    for i, r in enumerate(best_routes or [], 1):
        seq = [str(n - 1) for n in r if n != 1]
        print(f"Route #{i}: {' '.join(seq)}")

    print(f"\nBest cost : {best_cost}")
    print(f"Runtime   : {final_runtime:.2f}s")

    return {
        "instance": instance_name,
        "best_cost": best_cost,
        "routes": best_routes or [],
        "runtime": final_runtime,
        "route_pool_size": len(global_pool),
        "stages": stages,
    }


if __name__ == "__main__":
    res = run_drsci_for_instance(
        instance_name="X-n502-k39.vrp",
        seed=0,
        routing_solver="pyvrp",
        scp_solver="gurobi_mip",
    )

    print("\n[DEBUG] Best cost:", res["best_cost"])
    print("[DEBUG] #routes:", len(res["routes"]))
