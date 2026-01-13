from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
_HELPERS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _HELPERS_DIR.parents[2]

# ---------------------------------------------------------
# Types
# ---------------------------------------------------------
Route = List[int]
Routes = List[Route]

RouteKey = Tuple[int, ...]
Tag = Dict[str, Any]

# ============================================================
# HELPERS
# ============================================================

def _route_key(route: Route, *, depot_id: int = 1) -> RouteKey:
    """
    Canonical key for tagging:
    - remove depot visits
    - keep order (do NOT sort)
    """
    return tuple(n for n in route if n != depot_id)


def _tag_new_routes(
    route_tags: Dict[RouteKey, Tag],
    routes: Routes,
    *,
    tag: Tag,
    depot_id: int = 1,
) -> None:
    """
    Attach tag to each route if it doesn't already have one.
    We use setdefault so we preserve the first-known origin for that exact route.
    """
    for r in routes:
        key = _route_key(r, depot_id=depot_id)
        if not key:
            continue
        route_tags.setdefault(key, dict(tag))


def _result_to_vrplib_routes(result) -> Routes:
    best = result.best
    if best is None:
        return []

    routes: Routes = []
    for r in best.routes():
        # PyVRP returns location indices; keep non-negative visits.
        visits = [v for v in r.visits() if v >= 0]
        if visits:
            routes.append([1] + [v + 1 for v in visits] + [1])
    return routes


def _compute_integer_cost(instance: dict, routes: Routes) -> int:
    coords = instance["node_coord"]
    edge_mat = instance.get("edge_weight")

    def dist(u: int, v: int) -> int:
        u_idx, v_idx = u - 1, v - 1
        if edge_mat is not None:
            return int(round(float(edge_mat[u_idx, v_idx])))
        dx = coords[u_idx][0] - coords[v_idx][0]
        dy = coords[u_idx][1] - coords[v_idx][1]
        return int(round((dx * dx + dy * dy) ** 0.5))

    total = 0
    for r in routes:
        for a, b in zip(r, r[1:]):
            total += dist(a, b)
    return total


def _select_scp_solver_name(rng: random.Random, scp_solvers: List[str], scp_switch_prob: float) -> str:
    if not scp_solvers:
        raise ValueError("scp_solvers must contain at least one solver name.")
    if rng.random() < scp_switch_prob and len(scp_solvers) > 1:
        return rng.choice(scp_solvers)
    return scp_solvers[0]


def _load_bks_from_file(instance_name: str) -> Optional[int]:
    """
    Load BKS (Best Known Solution) cost for an instance from bks.json file.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n916-k207.vrp")
        
    Returns:
        BKS cost as int if found, None otherwise
    """
    # Get the path to bks.json file
    # CURRENT is src/master, so we go up to core, then instances/test-instances
    bks_file = PROJECT_ROOT / "instances" / "challenge-instances" / "challenge-bks.json"
    
    if not bks_file.exists():
        return None
    
    try:
        with open(bks_file, "r") as f:
            bks_data = json.load(f)
        
        instance_stem = Path(instance_name).stem
        bks_cost = bks_data.get(instance_stem)
        
        # Return None if BKS is null or not found
        if bks_cost is None:
            return None
        
        return int(bks_cost)
    except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
        return None


def _format_gap_to_bks(current_cost: float, bks_cost: Optional[int]) -> str:
    """
    Format the gap to BKS as a percentage string.
    
    Args:
        current_cost: Current solution cost
        bks_cost: Best Known Solution cost, or None if not available
        
    Returns:
        Formatted gap string like " | Gap: 10.0000%" or " | Gap: -10.0000%"
        Returns empty string if BKS is not available
        Negative gaps (better than BKS) are displayed in green
    """
    if bks_cost is None:
        return ""
    
    gap = ((current_cost - bks_cost) / bks_cost) * 100
    if gap < 0:
        # Green color for negative gaps (better than BKS)
        return f" | Gap: \033[38;2;41;209;47m{gap:.4f}%\033[0m"
    return f" | Gap: {gap:.4f}%"


def _convert_customer_ids_for_output(customers: List[int]) -> List[int]:
    """
    Convert customer IDs from VRPLIB format (2 to n) to official checker format (1 to n-1).
    This is a display-only conversion that doesn't affect calculations.
    
    Args:
        customers: List of customer IDs in VRPLIB format (2, 3, ..., n)
        
    Returns:
        List of customer IDs in official checker format (1, 2, ..., n-1)
    """
    return [c - 1 for c in customers]


def _write_sol_if_bks_beaten(
    *,
    instance_name: str,
    routes: Routes,
    cost: int,
    output_dir: str,
):
    """
    Write solution file if the current cost beats the BKS from bks.json.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n916-k207.vrp")
        routes: List of routes (VRPLIB format)
        cost: Current solution cost
        output_dir: Directory to write the solution file
    """
    bks_cost = _load_bks_from_file(instance_name)
    
    # If BKS is not found or current cost doesn't beat BKS, return early
    if bks_cost is None or cost >= bks_cost:
        return False

    os.makedirs(output_dir, exist_ok=True)
    base = Path(instance_name).stem
    sol_path = Path(output_dir) / f"BKS_{base}_{cost}.sol"

    # Official checker format: depot not mentioned, customers from 1 to n-1
    with open(sol_path, "w") as f:
        for idx, r in enumerate(routes, start=1):
            # Remove depot (assumes VRPLIB format [1, ..., 1])
            customers_vrplib = [v for v in r if v != 1]
            # Convert to official checker format (1 to n-1)
            customers = _convert_customer_ids_for_output(customers_vrplib)

            f.write(
                f"Route #{idx}: " + " ".join(map(str, customers)) + "\n"
            )

        f.write(f"Cost: {cost}\n")

    instance_base = Path(instance_name).stem

    print(
        # making the print appear in green for better visibility
        f"\033[38;2;41;209;47m[{instance_base} BKS] 🎉 BKS beaten! cost={cost} < BKS={bks_cost} "
        f"→ wrote {sol_path}\033[0m",
        flush=True,
    )
    return True

def _write_sol_unconditional(
    *,
    instance_name: str,
    routes: Routes,
    cost: int,
    output_dir: str,
    suffix: str = "INTERRUPTED",
):
    """
    Always write a .sol file, regardless of BKS.
    Used for interrupts / emergency checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = Path(instance_name).stem
    sol_path = Path(output_dir) / f"{base}_{suffix}_{cost}.sol"

    with open(sol_path, "w") as f:
        for idx, r in enumerate(routes, start=1):
            customers_vrplib = [v for v in r if v != 1]
            customers = _convert_customer_ids_for_output(customers_vrplib)
            f.write(f"Route #{idx}: " + " ".join(map(str, customers)) + "\n")
        f.write(f"Cost: {cost}\n")

    print(
        f"\033[93m[{base} INTERRUPT] wrote best-so-far solution → {sol_path}\033[0m",
        flush=True,
    )


def print_final_route_summary(
    *,
    best_routes: Routes,
    route_tags: Dict[RouteKey, Tag],
    depot_id: int = 1,
) -> None:
    """
    Summarize ONLY the routes that appear in the final solution.
    """
    final_tags = []

    for r in best_routes:
        key = _route_key(r, depot_id=depot_id)
        tag = route_tags.get(key)

        if tag is None:
            final_tags.append(("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"))
        else:
            final_tags.append(
                (
                    str(tag.get("mode")).upper(),
                    str(tag.get("method")),
                    str(tag.get("solver", "UNKNOWN")),
                    str(tag.get("stage")),
                )
            )

    counter = Counter(final_tags)

    print("\n[FINAL ROUTE SUMMARY]")
    for (mode, method, solver, stage), count in sorted(
        counter.items(), key=lambda x: (-x[1], x[0])
    ):
        print(f"  {count:4d} routes | {mode} | {method} | solver={solver} | stage={stage}")
