# core/src/master/routing/solver_filo.py
"""
FILO cluster solver (FILO1 / FILO2) for DRSCI subproblems.

Key points
----------
- FILO/FILO2 are C++ executables. We must call them via subprocess.
- For each cluster we build a temporary CVRP sub-instance .vrp (VRPLIB format),
  run FILO on it, parse the produced solution, and map routes back to global IDs.
- We also expose filo1/filo2 as routing backends via @register_solver so the
  existing routing_controller -> master.routing.solver.solve(..., solver=...) path
  can select them.

Configuration
-------------
solver_options can include:
- "cluster_nodes": list[int]   # VRPLIB customer node IDs (no depot) - for cluster subproblems
- "no_improvement": int        # Max iterations without improvement (required, adaptive by cluster size)
                               # Auto-converted to --coreopt-iterations <no_improvement * 10>
- "seed": int                  # Random seed (auto-added as --seed)
- "extra_args": list[str]      # Additional CLI arguments (e.g., --routemin-iterations)
- "keep_tmp": bool             # Keep temp dirs for debugging
- "filo1_exe": str|Path        # Override executable path
- "filo2_exe": str|Path        # Override executable path

Note: max_runtime is no longer used for FILO. Only no_improvement is used.
The no_improvement value is calculated adaptively by routing_controller based on
cluster size (10000 at n<=100 to 100000 at n>=10000, linear in between).

Stopping Criteria:
------------------
FILO1 and FILO2:
- Uses only no_improvement criterion (no time limit)
- no_improvement is required and adaptive by cluster size
- Converts to --coreopt-iterations <no_improvement * 10> (approximation)

Note: FILO doesn't have direct "no improvement" support, so we approximate by setting
--coreopt-iterations = no_improvement * 10. This gives FILO enough iterations to
have a chance to improve before stopping. The no_improvement value is calculated
adaptively by routing_controller based on cluster size (10000 at n<=100 to 100000 at n>=10000).

When used via routing_controller.solve_clusters():
- cluster_nodes is automatically provided for each cluster
- no_improvement is automatically calculated adaptively by routing_controller
- Routes are returned in VRPLIB format (global node IDs) via metadata["routes_vrplib"]

Example (cluster test)
----------------------
python -c "
from master.routing.solver_filo import solve_cluster_with_filo
r = solve_cluster_with_filo(
  instance_name='X-n502-k39.vrp',
  cluster_customers=[2,3,4,5,6,7,8],
  solver_variant='filo1',
  extra_args=[]
)
print(r.cost, r.runtime, len(r.routes_global))
"
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

from master.utils.loader import load_instance

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Route = List[int]    # VRPLIB-style [1, ..., 1] in GLOBAL node IDs
Routes = List[Route]


# ---------------------------------------------------------------------
# Instance helpers (vrplib-loaded dict)
# ---------------------------------------------------------------------
def _coord(inst: dict, node_id: int) -> Tuple[float, float]:
    coords = inst["node_coord"]
    x, y = coords[node_id - 1]
    return float(x), float(y)


def _demand(inst: dict, node_id: int) -> int:
    dem = inst.get("demand")
    if dem is None:
        return 0
    return int(dem[node_id - 1])


def _capacity(inst: dict) -> int:
    return int(inst["capacity"])


# ---------------------------------------------------------------------
# Write VRPLIB sub-instance
# ---------------------------------------------------------------------
def _write_subinstance_vrp(
    *,
    inst: dict,
    instance_name: str,
    cluster_customers_global: List[int],
    out_path: Path,
    solver_variant: str = "filo1",
) -> Tuple[Dict[int, int], Dict[int, int]]:
    customers = sorted({int(c) for c in cluster_customers_global if int(c) != 1})
    if not customers:
        raise ValueError("Cluster is empty (no customers).")

    global_to_local: Dict[int, int] = {1: 1}
    local_to_global: Dict[int, int] = {1: 1}
    for idx, g in enumerate(customers, start=2):
        global_to_local[g] = idx
        local_to_global[idx] = g

    cap = _capacity(inst)
    dim_local = 1 + len(customers)

    # For your X instances EUC_2D is fine.
    edge_type = "EUC_2D"

    name_stem = Path(instance_name).stem
    sub_name = f"{name_stem}_cluster_{len(customers)}"

    solver_variant = solver_variant.lower().strip()
    is_filo1 = (solver_variant == "filo1")
    
    lines: List[str] = []
    
    # Common header
    lines.append(f"NAME : {sub_name}")
    lines.append(f'COMMENT : "Cluster sub-instance extracted from {name_stem}"')
    lines.append("TYPE : CVRP")
    lines.append(f"DIMENSION : {dim_local}")
    lines.append(f"EDGE_WEIGHT_TYPE : {edge_type}")
    lines.append(f"CAPACITY : {cap}")
    
    if is_filo1:
        # FILO1 (XInstanceParser) expects:
        # - Skips 3 lines (NAME, COMMENT, TYPE)
        # - Reads DIMENSION
        # - Skips 1 line (EDGE_WEIGHT_TYPE)
        # - Reads CAPACITY
        # - Skips 1 blank line
        # - Reads coordinates directly (NO NODE_COORD_SECTION header!)
        lines.append("")  # Blank line before coordinates
    else:
        # FILO2 expects NODE_COORD_SECTION header
        lines.append("NODE_COORD_SECTION")
        lines.append("")  # Blank line after header

    # Write coordinates
    x1, y1 = _coord(inst, 1)
    lines.append(f"1 {x1:.0f} {y1:.0f}")

    for local_id in range(2, dim_local + 1):
        g = local_to_global[local_id]
        x, y = _coord(inst, g)
        lines.append(f"{local_id} {x:.0f} {y:.0f}")

    # DEMAND_SECTION header
    if is_filo1:
        # FILO1 reads DEMAND_SECTION header directly after coordinates
        lines.append("DEMAND_SECTION")
    else:
        # FILO2 expects blank line, then DEMAND_SECTION
        lines.append("")  # Blank line after coordinates
        lines.append("DEMAND_SECTION")
        lines.append("")  # Blank line after header

    # Write demands
    lines.append("1 0")
    for local_id in range(2, dim_local + 1):
        g = local_to_global[local_id]
        d = _demand(inst, g)
        lines.append(f"{local_id} {d}")

    # FILO parsers don't read DEPOT_SECTION, but include EOF for completeness
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return global_to_local, local_to_global


# ---------------------------------------------------------------------
# Parse FILO solution outputs
# ---------------------------------------------------------------------
_ROUTE_LINE = re.compile(r"(?:Route\s*#?\s*\d+|route\s*#?\s*\d+)\s*:\s*(.*)$")
_INT = re.compile(r"\d+")


def _parse_routes_from_text(text: str) -> List[List[int]]:
    """
    Parses routes from text containing lines like:
      Route #1: 2 7 5 1
    Returns list of routes as integer node IDs.
    """
    routes: List[List[int]] = []
    for line in text.splitlines():
        m = _ROUTE_LINE.search(line.strip())
        if not m:
            continue
        tail = m.group(1)
        nodes = [int(x) for x in _INT.findall(tail)]
        if nodes:
            routes.append(nodes)
    return routes


def _parse_cost_from_text(text: str) -> Optional[float]:
    # Flexible: "Cost: 12345", "cost = 12345.0", "Best cost : 12345"
    m = re.search(r"(?:Cost|cost|Best cost)\s*[:=]\s*([\d.]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _find_newest_file(root: Path, pattern: str) -> Optional[Path]:
    files = list(root.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


# ---------------------------------------------------------------------
# Locate executables
# ---------------------------------------------------------------------
def _repo_root() -> Path:
    # .../core/src/master/routing/solver_filo.py -> parents[3] == core
    return Path(__file__).resolve().parents[3]


def _candidate_exec_paths(variant: str) -> List[Path]:
    core = _repo_root()
    # Your repo layout: core/solver/filo1 and core/solver/filo2
    base = core / "solver" / variant

    names = []
    if variant == "filo1":
        names = ["filo", "filo.exe"]
    else:
        names = ["filo2", "filo2.exe"]

    candidates: List[Path] = []

    # Common CMake build locations
    for build_dir in [
        base / "build",
        base / "build" / "Release",
        base / "build" / "bin",
        base / "bin",
    ]:
        for n in names:
            candidates.append(build_dir / n)

    # Some people build directly in repo root
    for n in names:
        candidates.append(base / n)

    return candidates


def _resolve_executable(variant: str, override: Optional[str | Path] = None) -> Path:
    if override is not None:
        p = Path(override)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"{variant} executable override not found: {p}")

    for cand in _candidate_exec_paths(variant):
        if cand.exists():
            return cand.resolve()

    tried = "\n".join(str(p) for p in _candidate_exec_paths(variant))
    raise FileNotFoundError(
        f"Could not find {variant} executable. Tried:\n{tried}\n"
        f"Build it (cmake/make) or pass solver_options with "
        f"'filo1_exe'/'filo2_exe'."
    )


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class FiloClusterResult:
    routes_global: Routes
    cost: float
    runtime: float
    feasible: bool
    metadata: dict


# ---------------------------------------------------------------------
# Core runner (subprocess)
# ---------------------------------------------------------------------
def _run_filo_executable(
    *,
    variant: str,
    instance_vrp: Path,
    work_dir: Path,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Runs FILO/FILO2 and returns (stdout+stderr, runtime_seconds).
    """
    extra_args = extra_args or []

    if variant not in {"filo1", "filo2"}:
        raise ValueError("variant must be 'filo1' or 'filo2'.")

    exe = _resolve_executable(variant)

    cmd = [str(exe), str(instance_vrp)] + [str(a) for a in extra_args]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    t1 = time.time()

    out = proc.stdout or ""
    # If non-zero exit, still return output so caller can surface it.
    if proc.returncode != 0:
        raise RuntimeError(
            f"{variant} failed (exit={proc.returncode}). Output:\n{out}"
        )

    return out, (t1 - t0)


# ---------------------------------------------------------------------
# Public: solve ONE CLUSTER
# ---------------------------------------------------------------------
def solve_cluster_with_filo(
    *,
    instance_name: str,
    cluster_customers: Iterable[int],
    solver_variant: str = "filo1",
    keep_tmp: bool = False,
    extra_args: Optional[List[str]] = None,
    filo1_exe: Optional[str | Path] = None,
    filo2_exe: Optional[str | Path] = None,
) -> FiloClusterResult:
    solver_variant = str(solver_variant).lower().strip()
    if solver_variant not in {"filo1", "filo2"}:
        raise ValueError("solver_variant must be 'filo1' or 'filo2'.")

    inst = load_instance(instance_name)

    tmp_dir = Path(tempfile.mkdtemp(prefix="drsci_filo_cluster_"))
    try:
        sub_vrp_path = tmp_dir / f"{Path(instance_name).stem}_sub.vrp"

        _, local_to_global = _write_subinstance_vrp(
            inst=inst,
            instance_name=instance_name,
            cluster_customers_global=list(cluster_customers),
            out_path=sub_vrp_path,
            solver_variant=solver_variant,
        )

        # Allow overriding executable path per variant
        override = filo1_exe if solver_variant == "filo1" else filo2_exe
        # Resolve once so errors are clean
        _resolve_executable(solver_variant, override=override)

        # Run executable
        out, runtime = _run_filo_executable(
            variant=solver_variant,
            instance_vrp=sub_vrp_path,
            work_dir=tmp_dir,
            extra_args=extra_args,
        )

        # Try to find a produced solution file (native outputs vary; be flexible)
        sol_file = (
            _find_newest_file(tmp_dir, "*.vrp.sol")
            or _find_newest_file(tmp_dir, "*.sol")
            or None
        )

        text_for_parse = out
        if sol_file and sol_file.exists():
            try:
                text_for_parse = sol_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # fallback to stdout
                text_for_parse = out

        routes_local = _parse_routes_from_text(text_for_parse)

        # FILO outputs customer indices (1 to n-1), NOT node IDs!
        # In VRPLIB format: node 1 = depot, nodes 2,3,...,n = customers
        # FILO outputs: customer index 1, 2, ..., n-1 (where n-1 is the number of customers)
        # So we need to convert: customer_index -> node_ID = customer_index + 1
        #
        # Example: If sub-instance has 358 nodes (1 depot + 357 customers):
        #   - FILO outputs customer indices: 1, 2, ..., 357
        #   - These map to node IDs: 2, 3, ..., 358
        #   - Customer index 1 -> node ID 2 (first customer)
        #   - Customer index 79 -> node ID 80 (79th customer)
        
        dim_local = len(local_to_global)  # Number of nodes in sub-instance (1 depot + customers)
        num_customers = dim_local - 1  # Number of customers
        
        routes_local_norm: List[List[int]] = []
        for r in routes_local:
            rr = [int(x) for x in r]
            if not rr:
                continue
            
            # Check if values are in customer index range (1 to num_customers)
            # or node ID range (1 to dim_local)
            max_val = max(rr) if rr else 0
            min_val = min(rr) if rr else 0
            
            # If max value is <= num_customers and min >= 1, likely customer indices
            # If max value is <= dim_local and we see 1, might be node IDs (but 1 would be depot)
            # FILO typically outputs customer indices, so convert: customer_index -> node_ID = customer_index + 1
            if max_val <= num_customers and min_val >= 1:
                # Customer indices: convert to node IDs
                # Customer index 1 -> node ID 2, customer index 2 -> node ID 3, etc.
                rr = [n + 1 for n in rr]
            # If max_val > num_customers, assume they're already node IDs (shouldn't happen with FILO)
            
            # Filter out depot (1) from middle of route if present
            # FILO shouldn't output depot, but be safe
            rr = [n for n in rr if n != 1]
            
            if not rr:
                # Empty route after filtering, skip it
                continue
            
            # Enforce VRPLIB-style [1, ..., 1] in LOCAL ids (1-indexed)
            # Add depot at start/end
            rr = [1] + rr + [1]
            routes_local_norm.append(rr)

        routes_global: Routes = []
        for r in routes_local_norm:
            # Map local IDs (1-indexed) to global IDs
            # Filter out any invalid mappings (shouldn't happen, but be safe)
            mapped = []
            for n in r:
                global_id = local_to_global.get(int(n), None)
                if global_id is not None:
                    mapped.append(global_id)
                else:
                    # Fallback: if mapping fails, use depot (1)
                    mapped.append(1)
            routes_global.append(mapped)
        

        cost = _parse_cost_from_text(text_for_parse)
        if cost is None:
            # If we can't parse cost, set to NaN-like large and rely on downstream recomputation.
            cost = float("nan")

        feasible = len(routes_global) > 0

        meta = {
            "tmp_dir": str(tmp_dir),
            "sol_file": str(sol_file) if sol_file else None,
            "used_stdout": sol_file is None,
        }

        return FiloClusterResult(
            routes_global=routes_global,
            cost=float(cost),
            runtime=float(runtime),
            feasible=bool(feasible),
            metadata=meta,
        )

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def solve_cluster_with_filo1(
    *,
    instance_name: str,
    cluster_customers: Iterable[int],
    keep_tmp: bool = False,
    extra_args: Optional[List[str]] = None,
    filo1_exe: Optional[str | Path] = None,
) -> FiloClusterResult:
    return solve_cluster_with_filo(
        instance_name=instance_name,
        cluster_customers=cluster_customers,
        solver_variant="filo1",
        keep_tmp=keep_tmp,
        extra_args=extra_args,
        filo1_exe=filo1_exe,
    )


def solve_cluster_with_filo2(
    *,
    instance_name: str,
    cluster_customers: Iterable[int],
    keep_tmp: bool = False,
    extra_args: Optional[List[str]] = None,
    filo2_exe: Optional[str | Path] = None,
) -> FiloClusterResult:
    return solve_cluster_with_filo(
        instance_name=instance_name,
        cluster_customers=cluster_customers,
        solver_variant="filo2",
        keep_tmp=keep_tmp,
        extra_args=extra_args,
        filo2_exe=filo2_exe,
    )


# =====================================================================
# Routing backend adapters (for routing_controller / DRSCI integration)
# =====================================================================
# NOTE: This import is intentionally at the bottom to avoid import side effects
# until FILO is actually selected.
from master.routing.solver import SolveOutput, register_solver  # noqa: E402


def _solve_instance_with_filo_backend(
    instance_path: Path,
    *,
    solver_variant: str,
    options: Dict[str, Any],
) -> SolveOutput:
    """
    Adapter for master.routing.solver.solve(..., solver='filo1'|'filo2').

    Contract (as used by routing_controller.solve_clusters):
      - options["cluster_nodes"] : list of VRPLIB customer node IDs (no depot) [optional]
      - options["seed"]         : int [optional]
      - options["no_improvement"] : int [required, adaptive by cluster size]
      - options["extra_args"]   : list[str] [optional]
      - options["keep_tmp"]     : bool [optional]
      - options["filo1_exe"] / options["filo2_exe"] : Path [optional]
    
    Must return:
      - SolveOutput.metadata["routes_vrplib"] : List[List[int]] in VRPLIB format (global node IDs)

    Behavior:
      - If cluster_nodes is provided: uses solve_cluster_with_filo (handles subproblems)
      - Otherwise: runs FILO on the full instance
    """

    # Check if this is a cluster subproblem
    cluster_nodes: Optional[List[int]] = options.get("cluster_nodes", None)
    
    if cluster_nodes is not None:
        # Handle cluster subproblem using the dedicated cluster solver
        cluster_customers = sorted({nid for nid in cluster_nodes if nid != 1})
        
        if not cluster_customers:
            # Empty cluster
            return SolveOutput(
                solver=solver_variant,
                instance=instance_path,
                cost=0.0,
                runtime=0.0,
                num_iterations=0,
                feasible=True,
                data=None,
                raw_result=[],
                metadata={"routes_vrplib": []},
            )
        
        # Build extra_args from options
        extra_args = list(options.get("extra_args", []))
        
        # FILO: Only use no_improvement criterion (no time limit)
        # no_improvement should always be provided by routing_controller
        no_improvement = options.get("no_improvement", None)
        if no_improvement is None:
            raise ValueError("no_improvement must be provided for FILO solvers")
        
        # FILO doesn't have direct "no improvement" support, so we use --coreopt-iterations
        # as an approximation. Use no_improvement * 10 as coreopt-iterations.
        # This gives FILO enough iterations to have a chance to improve.
        coreopt_iterations = no_improvement * 10
        extra_args.extend(["--coreopt-iterations", str(int(coreopt_iterations))])
        
        # Add seed if provided
        seed = options.get("seed", None)
        if seed is not None:
            extra_args.extend(["--seed", str(int(seed))])
        
        # Get executable overrides
        filo1_exe = options.get("filo1_exe", None)
        filo2_exe = options.get("filo2_exe", None)
        
        keep_tmp = bool(options.get("keep_tmp", False))
        
        # Solve cluster subproblem
        result = solve_cluster_with_filo(
            instance_name=instance_path.name,
            cluster_customers=cluster_customers,
            solver_variant=solver_variant,
            keep_tmp=keep_tmp,
            extra_args=extra_args if extra_args else None,
            filo1_exe=filo1_exe,
            filo2_exe=filo2_exe,
        )
        
        # solve_cluster_with_filo returns routes in GLOBAL node IDs (VRPLIB format)
        return SolveOutput(
            solver=solver_variant,
            instance=instance_path,
            cost=result.cost,
            runtime=result.runtime,
            num_iterations=0,
            feasible=result.feasible,
            data=None,
            raw_result=result.routes_global,
            metadata={
                "routes_vrplib": result.routes_global,  # Already in VRPLIB format (global IDs)
                "tmp_dir": result.metadata.get("tmp_dir"),
                "sol_file": result.metadata.get("sol_file"),
            },
        )
    
    # Full instance mode (original behavior)
    keep_tmp = bool(options.get("keep_tmp", False))
    extra_args = options.get("extra_args", None)

    filo1_exe = options.get("filo1_exe", None)
    filo2_exe = options.get("filo2_exe", None)

    # Run in an isolated work dir (so FILO can write files)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"routing_{solver_variant}_"))
    try:
        # Copy instance into work dir to keep FILO outputs contained
        local_inst = tmp_dir / instance_path.name
        shutil.copy2(instance_path, local_inst)

        override = filo1_exe if solver_variant == "filo1" else filo2_exe
        _resolve_executable(solver_variant, override=override)

        out, runtime = _run_filo_executable(
            variant=solver_variant,
            instance_vrp=local_inst,
            work_dir=tmp_dir,
            extra_args=extra_args,
        )

        sol_file = (
            _find_newest_file(tmp_dir, "*.vrp.sol")
            or _find_newest_file(tmp_dir, "*.sol")
            or None
        )

        text_for_parse = out
        if sol_file and sol_file.exists():
            try:
                text_for_parse = sol_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text_for_parse = out

        routes = _parse_routes_from_text(text_for_parse)
        routes_norm: List[List[int]] = []
        for r in routes:
            rr = [int(x) for x in r]
            if not rr:
                continue
            if rr[0] != 1:
                rr = [1] + rr
            if rr[-1] != 1:
                rr = rr + [1]
            routes_norm.append(rr)

        cost = _parse_cost_from_text(text_for_parse)
        if cost is None:
            cost = float("nan")

        return SolveOutput(
            solver=solver_variant,
            instance=instance_path,
            cost=float(cost),
            runtime=float(runtime),
            num_iterations=0,
            feasible=len(routes_norm) > 0,
            data=None,
            raw_result=routes_norm,  # IMPORTANT: routes in THIS instance's node IDs
            metadata={
                "routes_vrplib": routes_norm,  # Add routes_vrplib for consistency
                "tmp_dir": str(tmp_dir),
                "sol_file": str(sol_file) if sol_file else None,
            },
        )

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@register_solver("filo1")
def _solve_with_filo1(instance_path: Path, options: Dict[str, Any]) -> SolveOutput:
    return _solve_instance_with_filo_backend(
        instance_path,
        solver_variant="filo1",
        options=options,
    )


@register_solver("filo2")
def _solve_with_filo2(instance_path: Path, options: Dict[str, Any]) -> SolveOutput:
    return _solve_instance_with_filo_backend(
        instance_path,
        solver_variant="filo2",
        options=options,
    )


# ---------------------------------------------------------------------
# Minimal manual test (cluster mode)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    res = solve_cluster_with_filo1(
        instance_name="X-n502-k39.vrp",
        cluster_customers=[2, 3, 4, 5, 6, 7, 8],
        keep_tmp=True,          # keep to inspect outputs
        extra_args=[],          # add FILO flags here if needed
    )
    print("Feasible:", res.feasible, "Cost:", res.cost, "Runtime:", res.runtime)
    for i, r in enumerate(res.routes_global, 1):
        print(f"Route {i}:", r)
