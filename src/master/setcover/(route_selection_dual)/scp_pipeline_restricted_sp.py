"""
Dual-based Route Selection Pipeline
-----------------------------------

End-to-end "paper-style" master step:

  1) Solve LP relaxation on current pool -> duals pi
  2) Compute reduced costs for all routes
  3) Filter the pool using rules A/B/C + incumbent preservation
  4) Solve restricted set partitioning (binary MIP) on filtered pool

This module is meant to be called from run_drsci_dual.py (new pipeline),
NOT from the baseline run_drsci.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from master.setcover.route_selection_dual.lp_relaxation import (
    solve_lp_relaxation,
    compute_reduced_costs,
)
from master.setcover.route_selection_dual.route_filtering import (
    filter_routes_dual,
    FilterStats,
)
from master.setcover.route_selection_dual.sp_solver_gurobi_MIP import (
    solve_restricted_sp,
)

Route = List[int]
Routes = List[Route]


@dataclass(frozen=True)
class DualPipelineStats:
    lp_time: float
    lp_obj: float | None
    lp_frac: int
    lp_near_one: int

    filter_stats: FilterStats
    rsp_status: int
    rsp_obj: float | None
    rsp_selected: int
    rsp_pool: int


def solve_routes_via_dual_filtering(
    *,
    instance_name: str,
    route_pool: Routes,
    incumbent_routes: Optional[Routes],
    lp_mode: str = "cover",  # "cover" or "partition"
    lp_time_limit: float = 30.0,
    rsp_time_limit: float = 60.0,
    depot_id: int = 1,
    rc_threshold: float = -1e-6,
    k_per_customer: int = 10,
    n_cost_seed: int = 8000,
    n_final: int = 16000,
    verbose: bool = True,
) -> Tuple[Dict[str, object], DualPipelineStats]:
    """
    Returns:
      - rsp_result dict (schema: solver/optimal/status/obj/selected_routes...)
      - DualPipelineStats for logging/analysis
    """
    # 1) LP relaxation
    lp_res, costs, route_cust = solve_lp_relaxation(
        instance_name=instance_name,
        route_pool=route_pool,
        mode=lp_mode,  # type: ignore
        time_limit=lp_time_limit,
        verbose=False,
        depot_id=depot_id,
        costs=None,
    )

    if lp_res.lp_obj_value is None:
        # LP failed -> cannot compute duals reliably.
        # Fallback: just run restricted SP on the current pool (may be huge).
        rsp_res = solve_restricted_sp(
            instance_name=instance_name,
            route_pool=route_pool,
            time_limit=rsp_time_limit,
            verbose=False,
            depot_id=depot_id,
            costs=costs,
            warm_start_routes=incumbent_routes,
        )
        stats = DualPipelineStats(
            lp_time=lp_res.runtime_sec,
            lp_obj=None,
            lp_frac=0,
            lp_near_one=0,
            filter_stats=FilterStats(
                n_input=len(route_pool),
                n_output=len(route_pool),
                kept_negative_rc=0,
                kept_cost_seed=0,
                kept_topk_per_customer=0,
                kept_incumbent=0,
                uncovered_after_filter=0,
                repaired_added=0,
                rc_threshold=rc_threshold,
                k_per_customer=k_per_customer,
                n_cost_seed=n_cost_seed,
                n_final=n_final,
            ),
            rsp_status=int(rsp_res["status"]),
            rsp_obj=rsp_res["obj_value"],
            rsp_selected=len(rsp_res["selected_routes"]),
            rsp_pool=len(route_pool),
        )
        return rsp_res, stats

    # 2) Reduced costs
    rc = compute_reduced_costs(costs=costs, route_cust=route_cust, pi=lp_res.pi)

    # 3) Filtering
    customers = list(lp_res.pi.keys())
    filtered_routes, filtered_indices, filt_stats = filter_routes_dual(
        customers=customers,
        route_pool=route_pool,
        route_cust=route_cust,
        costs=costs,
        reduced_costs=rc,
        incumbent_routes=incumbent_routes,
        depot_id=depot_id,
        rc_threshold=rc_threshold,
        k_per_customer=k_per_customer,
        n_cost_seed=n_cost_seed,
        n_final=n_final,
        verbose=verbose,
    )

    # Align costs for filtered pool
    filtered_costs = [costs[i] for i in filtered_indices]

    # 4) Restricted SP on filtered pool
    rsp_res = solve_restricted_sp(
        instance_name=instance_name,
        route_pool=filtered_routes,
        time_limit=rsp_time_limit,
        verbose=False,
        depot_id=depot_id,
        costs=filtered_costs,
        warm_start_routes=incumbent_routes,
    )

    stats = DualPipelineStats(
        lp_time=lp_res.runtime_sec,
        lp_obj=lp_res.lp_obj_value,
        lp_frac=lp_res.num_fractional,
        lp_near_one=lp_res.num_near_one,
        filter_stats=filt_stats,
        rsp_status=int(rsp_res["status"]),
        rsp_obj=rsp_res["obj_value"],
        rsp_selected=len(rsp_res["selected_routes"]),
        rsp_pool=len(filtered_routes),
    )

    if verbose:
        print(
            f"[DUAL-PIPE] LP time={stats.lp_time:.2f}s obj={stats.lp_obj} "
            f"frac={stats.lp_frac} near1={stats.lp_near_one} | "
            f"RSP pool={stats.rsp_pool} selected={stats.rsp_selected} obj={stats.rsp_obj}",
            flush=True,
        )

    return rsp_res, stats
