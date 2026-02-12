# core/src/master/routing/solver_ails2.py
"""
AILS2 solver for DRSCI subproblems.

Key points
----------
- AILS2 is a Java executable (JAR file). We must call it via subprocess with java -jar.
- For each cluster we build a temporary CVRP sub-instance .vrp (VRPLIB format),
  run AILS2 on it, parse the produced solution, and map routes back to global IDs.
- We expose ails2 as a routing backend via @register_solver so the
  existing routing_controller -> master.routing.solver.solve(..., solver=...) path
  can select it.

Configuration
-------------
solver_options can include:
- "cluster_nodes": list[int]   # VRPLIB customer node IDs (no depot) - for cluster subproblems
- "no_improvement": int        # Max iterations without improvement (used if max_runtime not provided)
- "max_runtime": float         # Time limit in seconds (converted to -limit with Time criterion)
- "seed": int                  # Random seed (if supported)
- "extra_args": list[str]      # Additional CLI arguments
- "keep_tmp": bool             # Keep temp dirs for debugging
- "ails2_jar": str|Path        # Override JAR path
- "rounded": bool              # Whether distances are rounded (default: true)
- "best": float                # Optimal solution value (default: 0)

Stopping Criteria:
------------------
AILS2 supports two stopping criteria:
- Time: stops after given time in seconds
- Iteration: stops after given number of iterations

We prioritize max_runtime if provided (adaptive time limit from routing_controller),
otherwise use no_improvement converted to iterations. Both are provided adaptively
based on cluster size by routing_controller.solve_clusters:
- max_runtime: adaptive time limit based on cluster size (t = base + alpha * n^exponent)
- no_improvement: adaptive iterations (500-20000 based on cluster size)
Since AILS2 only supports one stopping criterion at a time, max_runtime takes precedence.

When used via routing_controller.solve_clusters():
- cluster_nodes is automatically provided for each cluster
- Routes are returned in VRPLIB format (global node IDs) via metadata["routes_vrplib"]

Example (cluster test)
----------------------
python -c "
from master.routing.solver_ails2 import solve_cluster_with_ails2
r = solve_cluster_with_ails2(
  instance_name='X-n502-k39.vrp',
  cluster_customers=[2,3,4,5,6,7,8],
  max_runtime=30.0
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
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Write a VRPLIB-format sub-instance for AILS2."""
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

    edge_type = "EUC_2D"

    name_stem = Path(instance_name).stem
    sub_name = f"{name_stem}_cluster_{len(customers)}"

    lines: List[str] = []
    
    # VRPLIB format (standard) - AILS2 expects standard VRPLIB format
    lines.append(f"NAME : {sub_name}")
    lines.append(f'COMMENT : "Cluster sub-instance extracted from {name_stem}"')
    lines.append("TYPE : CVRP")
    lines.append(f"DIMENSION : {dim_local}")
    lines.append(f"EDGE_WEIGHT_TYPE : {edge_type}")
    lines.append(f"CAPACITY : {cap}")
    lines.append("NODE_COORD_SECTION")

    # Write coordinates (no blank line after header - AILS2 might be strict)
    x1, y1 = _coord(inst, 1)
    lines.append(f"1 {int(x1)} {int(y1)}")  # Use integers, not floats

    for local_id in range(2, dim_local + 1):
        g = local_to_global[local_id]
        x, y = _coord(inst, g)
        lines.append(f"{local_id} {int(x)} {int(y)}")  # Use integers

    # DEMAND_SECTION (no blank line before - AILS2 might be strict)
    lines.append("DEMAND_SECTION")

    # Write demands
    lines.append("1 0")
    for local_id in range(2, dim_local + 1):
        g = local_to_global[local_id]
        d = _demand(inst, g)
        lines.append(f"{local_id} {d}")

    # DEPOT_SECTION
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return global_to_local, local_to_global


# ---------------------------------------------------------------------
# Parse AILS2 solution outputs
# ---------------------------------------------------------------------
# AILS2 output format may vary - we'll try to parse common patterns
_ROUTE_LINE = re.compile(r"(?:Route\s*#?\s*\d+|route\s*#?\s*\d+)\s*[:=]\s*(.*)$", re.IGNORECASE)
_INT = re.compile(r"\d+")
_COST_PATTERNS = [
    re.compile(r"Cost\s+(\d+)", re.IGNORECASE),  # "Cost 682" format (AILS2 standard)
    re.compile(r"(?:Cost|cost|Best cost|Total cost|Objective)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"(\d+\.?\d*)\s*(?:is the|total)", re.IGNORECASE),
]


def _parse_routes_from_text(text: str) -> List[List[int]]:
    """
    Parses routes from text containing lines like:
      Route #1: 2 7 5 1
      Route 1: 2, 7, 5, 1
    Returns list of routes as integer node IDs.
    """
    routes: List[List[int]] = []
    for line in text.splitlines():
        m = _ROUTE_LINE.search(line.strip())
        if not m:
            continue
        tail = m.group(1)
        # Extract all integers from the route line
        nodes = [int(x) for x in _INT.findall(tail)]
        if nodes:
            routes.append(nodes)
    return routes


def _parse_cost_from_text(text: str) -> Optional[float]:
    """Try multiple patterns to extract cost from AILS2 output."""
    # First try explicit "Cost" line (from solution file or final output)
    for pattern in _COST_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    
    # If no explicit cost line, try to extract from progress output
    # AILS2 outputs: "solution quality: 27595.0 gap: ∞% K: 26 iteration: 1409..."
    # Extract the last/best solution quality (most recent improvement)
    quality_pattern = re.compile(r"solution quality:\s*([\d.]+)", re.IGNORECASE)
    matches = quality_pattern.findall(text)
    if matches:
        # Get the last (best) solution quality
        try:
            return float(matches[-1])
        except Exception:
            pass
    
    return None


def _find_newest_file(root: Path, pattern: str) -> Optional[Path]:
    files = list(root.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


# ---------------------------------------------------------------------
# Locate JAR executable
# ---------------------------------------------------------------------
def _repo_root() -> Path:
    # .../core/src/master/routing/solver_ails2.py -> parents[3] == core
    return Path(__file__).resolve().parents[3]


def _candidate_jar_paths() -> List[Path]:
    """Find AILS2 JAR file in common locations."""
    core = _repo_root()
    base = core / "solver" / "ails2"

    candidates: List[Path] = []

    # Common JAR locations
    jar_names = ["AILSII.jar", "AILS2.jar", "ails2.jar", "ailsii.jar"]
    for jar_name in jar_names:
        # Root of solver directory
        candidates.append(base / jar_name)
        # Common build/output directories
        for build_dir in ["build", "target", "out", "dist", "bin"]:
            candidates.append(base / build_dir / jar_name)
            candidates.append(base / build_dir / "libs" / jar_name)

    return candidates


def _resolve_jar(override: Optional[str | Path] = None) -> Path:
    """Resolve AILS2 JAR file path."""
    if override is not None:
        p = Path(override)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"AILS2 JAR override not found: {p}")

    for cand in _candidate_jar_paths():
        if cand.exists():
            return cand.resolve()

    tried = "\n".join(str(p) for p in _candidate_jar_paths())
    raise FileNotFoundError(
        f"Could not find AILS2 JAR file. Tried:\n{tried}\n"
        f"Build the JAR or pass solver_options with 'ails2_jar'."
    )


def _check_java() -> bool:
    """Check if Java is available."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class Ails2ClusterResult:
    routes_global: Routes
    cost: float
    runtime: float
    feasible: bool
    metadata: dict


# ---------------------------------------------------------------------
# Core runner (subprocess)
# ---------------------------------------------------------------------
def _run_ails2_executable(
    *,
    instance_vrp: Path,
    work_dir: Path,
    jar_path: Path,
    max_runtime: Optional[float] = None,
    no_improvement: Optional[int] = None,
    rounded: bool = True,
    best: float = 0.0,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Runs AILS2 and returns (stdout+stderr, runtime_seconds).
    
    AILS2 command format:
    java -jar AILSII.jar -file <instance> -rounded <true/false> -best <value> 
         -limit <value> -stoppingCriterion <Time|Iteration>
    """
    if not _check_java():
        raise RuntimeError("Java is not available. Please install Java to use AILS2.")

    extra_args = extra_args or []
    
    # Build command
    # AILS2 expects the file path - use relative path from work_dir if file is there,
    # otherwise use absolute path
    # Since we run with cwd=work_dir, use just the filename if the file is in work_dir
    if str(instance_vrp.parent) == str(work_dir) or instance_vrp.parent == work_dir:
        file_arg = instance_vrp.name  # Just filename since we're in work_dir
    else:
        file_arg = str(instance_vrp)  # Use absolute path
    
    cmd = ["java", "-jar", str(jar_path), "-file", file_arg]
    
    # Add rounded flag
    cmd.extend(["-rounded", "true" if rounded else "false"])
    
    # Add best value
    cmd.extend(["-best", str(int(best))])
    
    # Determine stopping criterion
    if max_runtime is not None:
        # Use time limit
        cmd.extend(["-stoppingCriterion", "Time"])
        cmd.extend(["-limit", str(int(max_runtime))])
    elif no_improvement is not None:
        # Use iteration limit (approximate no_improvement as iterations)
        cmd.extend(["-stoppingCriterion", "Iteration"])
        cmd.extend(["-limit", str(int(no_improvement))])
    else:
        # Default: use a reasonable time limit
        cmd.extend(["-stoppingCriterion", "Time"])
        cmd.extend(["-limit", "300"])  # 5 minutes default
    
    # Add any extra arguments
    cmd.extend([str(a) for a in extra_args])

    t0 = time.time()
    # Calculate timeout with buffer: use 30% extra or minimum 30 seconds, whichever is larger
    # AILS2 often runs past its -limit (cleanup, current iteration); avoid subprocess timeout
    timeout_value = None
    if max_runtime is not None:
        timeout_buffer = max(30.0, max_runtime * 1)  # At least 30s or 30% of runtime
        timeout_value = max_runtime + timeout_buffer
    
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_value,
    )
    t1 = time.time()

    out = proc.stdout or ""
    
    # If non-zero exit, still return output so caller can surface it.
    if proc.returncode != 0:
        # Add debug info about the file
        file_info = ""
        if instance_vrp.exists():
            try:
                lines = instance_vrp.read_text().splitlines()
                file_info = f"\nFile exists: True\nFile size: {instance_vrp.stat().st_size} bytes\nFirst 10 lines:\n"
                file_info += "\n".join(lines[:10])
            except Exception as e:
                file_info = f"\nCould not read file info: {e}"
        else:
            file_info = f"\nFile exists: False\nExpected path: {instance_vrp}"
        
        raise RuntimeError(
            f"AILS2 failed (exit={proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Working dir: {work_dir}\n"
            f"File path: {instance_vrp}{file_info}\n"
            f"Output:\n{out}"
        )

    return out, (t1 - t0)


# ---------------------------------------------------------------------
# Public: solve ONE CLUSTER
# ---------------------------------------------------------------------
def solve_cluster_with_ails2(
    *,
    instance_name: str,
    cluster_customers: Iterable[int],
    keep_tmp: bool = False,
    max_runtime: Optional[float] = None,
    no_improvement: Optional[int] = None,
    rounded: bool = True,
    best: float = 0.0,
    extra_args: Optional[List[str]] = None,
    ails2_jar: Optional[str | Path] = None,
) -> Ails2ClusterResult:
    """
    Solve a cluster subproblem using AILS2.
    
    Note: AILS2 may have issues with very small instances (< 10 customers).
    For such cases, consider using a different solver or handling gracefully.
    """
    """
    Solve a cluster subproblem using AILS2.
    
    Args:
        instance_name: Name of the instance file
        cluster_customers: Iterable of customer node IDs (global, VRPLIB format)
        keep_tmp: Keep temporary directories for debugging
        max_runtime: Time limit in seconds (takes precedence over no_improvement)
        no_improvement: Max iterations without improvement (used if max_runtime not provided)
        rounded: Whether distances are rounded (default: True)
        best: Optimal solution value (default: 0)
        extra_args: Additional CLI arguments
        ails2_jar: Override JAR file path
    
    Returns:
        Ails2ClusterResult with routes in global node IDs
    """
    inst = load_instance(instance_name)

    tmp_dir = Path(tempfile.mkdtemp(prefix="drsci_ails2_cluster_"))
    try:
        sub_vrp_path = tmp_dir / f"{Path(instance_name).stem}_sub.vrp"

        _, local_to_global = _write_subinstance_vrp(
            inst=inst,
            instance_name=instance_name,
            cluster_customers_global=list(cluster_customers),
            out_path=sub_vrp_path,
        )

        # Check cluster size - AILS2 may have issues with very small instances
        num_customers = len(list(cluster_customers))
        
        # Always calculate adaptive time limit based on cluster size
        # This ensures AILS2 uses adaptive time regardless of input max_runtime
        if num_customers > 0:
            # Same formula as routing_controller._adaptive_cluster_time
            base = 1.0
            alpha = 0.25
            exponent = 0.9
            min_time = 2.0
            max_time = 180.0
            adaptive_time = base + alpha * (num_customers ** exponent)
            max_runtime = max(min_time, min(max_time, adaptive_time))
        elif max_runtime is None:
            max_runtime = 10.0  # Fallback default
        
        if num_customers < 5:
            # Very small clusters - AILS2 may crash, return empty result
            return Ails2ClusterResult(
                routes_global=[],
                cost=float("inf"),
                runtime=0.0,
                feasible=False,
                metadata={"error": "Cluster too small for AILS2", "num_customers": num_customers},
            )

        # Resolve JAR path
        jar_path = _resolve_jar(override=ails2_jar)

        # Run executable
        try:
            out, runtime = _run_ails2_executable(
                instance_vrp=sub_vrp_path,
                work_dir=tmp_dir,
                jar_path=jar_path,
                max_runtime=max_runtime,
                no_improvement=no_improvement,
                rounded=rounded,
                best=best,
                extra_args=extra_args,
            )
        except RuntimeError as e:
            # AILS2 has known bugs with ArrayIndexOutOfBoundsException
            # This can happen with certain instance sizes or configurations
            error_msg = str(e)
            if "ArrayIndexOutOfBoundsException" in error_msg:
                # AILS2 has a bug with array indexing - return empty result gracefully
                return Ails2ClusterResult(
                    routes_global=[],
                    cost=float("inf"),
                    runtime=0.0,
                    feasible=False,
                    metadata={
                        "error": "AILS2 ArrayIndexOutOfBoundsException bug",
                        "num_customers": num_customers,
                        "dimension": num_customers + 1,
                        "original_error": error_msg[:300],
                    },
                )
            # Re-raise other errors
            raise

        # Try to find a produced solution file
        # AILS2 may write solution files in the same directory as the instance file
        instance_stem = sub_vrp_path.stem
        sol_file = None
        
        # Check common AILS2 solution file locations
        # AILS2 writes .sol files in the same directory as the .vrp file
        possible_sol_files = [
            sub_vrp_path.parent / f"{instance_stem}.sol",  # Same dir as instance
            tmp_dir / f"{instance_stem}.sol",  # Work dir
            tmp_dir / f"{sub_vrp_path.name}.sol",  # Work dir with full name
            _find_newest_file(tmp_dir, "*.vrp.sol"),
            _find_newest_file(tmp_dir, "*.sol"),
            _find_newest_file(sub_vrp_path.parent, "*.sol"),  # Instance dir
        ]
        
        for sf in possible_sol_files:
            if sf and sf.exists():
                sol_file = sf
                break

        # AILS2 may write solution to file or stdout
        # Try solution file first, then fallback to stdout
        text_for_parse = out
        if sol_file and sol_file.exists():
            try:
                sol_content = sol_file.read_text(encoding="utf-8", errors="ignore")
                # Prefer solution file content (more reliable)
                text_for_parse = sol_content
            except Exception:
                # Fallback to stdout if file read fails
                text_for_parse = out
        else:
            # No solution file found, use stdout
            # AILS2 outputs routes and cost to stdout
            text_for_parse = out

        routes_local = _parse_routes_from_text(text_for_parse)
        
        # Debug: log if we found routes (can be removed later)
        if not routes_local and keep_tmp:
            # Save output for debugging
            debug_file = tmp_dir / "ails2_debug_output.txt"
            debug_file.write_text(f"STDOUT:\n{out}\n\nSOL_FILE: {sol_file}\n\nPARSED_TEXT:\n{text_for_parse}", encoding="utf-8")

        # AILS2 outputs customer node IDs (local IDs for cluster subproblems)
        # Normalize routes to VRPLIB format [1, ..., 1] in LOCAL IDs
        routes_local_norm: List[List[int]] = []
        for r in routes_local:
            rr = [int(x) for x in r]
            if not rr:
                continue
            
            # AILS2 outputs customer IDs only (no depot)
            # Filter out depot (1) if it somehow appears
            rr = [n for n in rr if n != 1]
            
            if not rr:
                continue
            
            # Enforce VRPLIB-style [1, customer1, customer2, ..., 1]
            # AILS2 routes are customer IDs, so add depot at start and end
            rr = [1] + rr + [1]
            routes_local_norm.append(rr)

        # Map local IDs to global IDs
        routes_global: Routes = []
        for r in routes_local_norm:
            mapped = []
            for n in r:
                global_id = local_to_global.get(int(n), None)
                if global_id is not None:
                    mapped.append(global_id)
                else:
                    mapped.append(1)  # Fallback to depot
            routes_global.append(mapped)

        cost = _parse_cost_from_text(text_for_parse)
        if cost is None:
            # If no explicit cost, try to get from progress output
            cost = float("nan")
        
        # Check capacity feasibility and filter out infeasible routes
        cap = _capacity(inst)
        feasible_routes = []
        infeasible_count = 0
        for route in routes_global:
            route_load = sum(_demand(inst, node_id) for node_id in route if node_id != 1)
            if route_load <= cap:
                feasible_routes.append(route)
            else:
                infeasible_count += 1
        
        # Use only feasible routes
        routes_global = feasible_routes
        feasible = len(routes_global) > 0
        
        # Log warning if infeasible routes were found
        if infeasible_count > 0:
            import warnings
            warnings.warn(
                f"AILS2 produced {infeasible_count} infeasible route(s) "
                f"(capacity={cap}). Filtered out before returning.",
                UserWarning
            )
        
        # If no routes but we have cost, mark as partial result
        if not routes_global and not (cost is None or (isinstance(cost, float) and (cost != cost))):  # cost is not nan
            # We have cost but no routes - AILS2 might not output routes automatically
            # This is acceptable for cost tracking, but routes are needed for solve_clusters
            pass

        meta = {
            "tmp_dir": str(tmp_dir),
            "sol_file": str(sol_file) if sol_file else None,
            "used_stdout": sol_file is None,
            "jar_path": str(jar_path),
            "routes_found": len(routes_global),
            "cost_from_progress": cost is not None and sol_file is None,
        }

        return Ails2ClusterResult(
            routes_global=routes_global,
            cost=float(cost) if cost is not None else float("nan"),
            runtime=float(runtime),
            feasible=bool(feasible),
            metadata=meta,
        )

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =====================================================================
# Routing backend adapters (for routing_controller / DRSCI integration)
# =====================================================================
# NOTE: This import is intentionally at the bottom to avoid import side effects
# until AILS2 is actually selected.
from master.routing.solver import SolveOutput, register_solver  # noqa: E402


def _solve_instance_with_ails2_backend(
    instance_path: Path,
    options: Dict[str, Any],
) -> SolveOutput:
    """
    Adapter for master.routing.solver.solve(..., solver='ails2').

    Contract (as used by routing_controller.solve_clusters):
      - options["cluster_nodes"] : list of VRPLIB customer node IDs (no depot) [optional]
      - options["seed"]         : int [optional, may not be supported]
      - options["no_improvement"] : int [optional, converted to iterations]
      - options["max_runtime"]  : float [optional, time limit in seconds]
      - options["extra_args"]   : list[str] [optional]
      - options["keep_tmp"]     : bool [optional]
      - options["ails2_jar"]    : Path [optional]
      - options["rounded"]      : bool [optional, default: True]
      - options["best"]         : float [optional, default: 0]
    
    Must return:
      - SolveOutput.metadata["routes_vrplib"] : List[List[int]] in VRPLIB format (global node IDs)

    Behavior:
      - If cluster_nodes is provided: uses solve_cluster_with_ails2 (handles subproblems)
      - Otherwise: runs AILS2 on the full instance
    """

    # Check if this is a cluster subproblem
    cluster_nodes: Optional[List[int]] = options.get("cluster_nodes", None)
    
    if cluster_nodes is not None:
        # Handle cluster subproblem
        cluster_customers = sorted({nid for nid in cluster_nodes if nid != 1})
        
        if not cluster_customers:
            # Empty cluster
            return SolveOutput(
                solver="ails2",
                instance=instance_path,
                cost=0.0,
                runtime=0.0,
                num_iterations=0,
                feasible=True,
                data=None,
                raw_result=[],
                metadata={"routes_vrplib": []},
            )
        
        # Extract options
        max_runtime = options.get("max_runtime", None)
        no_improvement = options.get("no_improvement", None)
        rounded = options.get("rounded", True)
        best = options.get("best", 0.0)
        extra_args = list(options.get("extra_args", []))
        keep_tmp = bool(options.get("keep_tmp", False))
        ails2_jar = options.get("ails2_jar", None)
        
        # Solve cluster subproblem
        result = solve_cluster_with_ails2(
            instance_name=instance_path.name,
            cluster_customers=cluster_customers,
            keep_tmp=keep_tmp,
            max_runtime=max_runtime,
            no_improvement=no_improvement,
            rounded=rounded,
            best=best,
            extra_args=extra_args if extra_args else None,
            ails2_jar=ails2_jar,
        )
        
        # solve_cluster_with_ails2 returns routes in GLOBAL node IDs (VRPLIB format)
        return SolveOutput(
            solver="ails2",
            instance=instance_path,
            cost=result.cost,
            runtime=result.runtime,
            num_iterations=0,
            feasible=result.feasible,
            data=None,
            raw_result=result.routes_global,
            metadata={
                "routes_vrplib": result.routes_global,
                "tmp_dir": result.metadata.get("tmp_dir"),
                "sol_file": result.metadata.get("sol_file"),
            },
        )
    
    # Full instance mode
    keep_tmp = bool(options.get("keep_tmp", False))
    max_runtime = options.get("max_runtime", None)
    no_improvement = options.get("no_improvement", None)
    rounded = options.get("rounded", True)
    best = options.get("best", 0.0)
    extra_args = options.get("extra_args", None)
    ails2_jar = options.get("ails2_jar", None)

    # Load instance for capacity checking and adaptive time calculation
    from master.utils.loader import load_instance
    inst = load_instance(instance_path.name)
    
    # Always calculate adaptive time limit based on instance size
    # This ensures AILS2 uses adaptive time regardless of input max_runtime
    num_customers = len(inst.get("demand", [])) - 1  # Exclude depot
    if num_customers > 0:
        # Same formula as routing_controller._adaptive_cluster_time
        base = 1.0
        alpha = 0.25
        exponent = 0.9
        min_time = 2.0
        max_time = 180.0
        adaptive_time = base + alpha * (num_customers ** exponent)
        max_runtime = max(min_time, min(max_time, adaptive_time))
    else:
        max_runtime = max_runtime or 10.0  # Use provided or fallback default
    
    # Run in an isolated work dir
    tmp_dir = Path(tempfile.mkdtemp(prefix="routing_ails2_"))
    try:
        # Copy and preprocess instance file for AILS2
        # AILS2 is strict about VRP format - remove blank lines between headers
        local_inst = tmp_dir / instance_path.name
        
        # Read original file and clean it up for AILS2
        # AILS2 is very strict - it doesn't handle blank lines well
        content = instance_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        
        # Remove all blank lines (AILS2 expects no blank lines)
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Only keep non-blank lines
                cleaned_lines.append(line)
        
        # Write cleaned file (AILS2 format: no blank lines)
        local_inst.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
        local_inst.chmod(0o644)

        jar_path = _resolve_jar(override=ails2_jar)

        out, runtime = _run_ails2_executable(
            instance_vrp=local_inst,
            work_dir=tmp_dir,
            jar_path=jar_path,
            max_runtime=max_runtime,
            no_improvement=no_improvement,
            rounded=rounded,
            best=best,
            extra_args=extra_args,
        )

        # AILS2 writes solution files with .sol extension
        # AILS2 writes .sol files in the same directory as the .vrp file
        instance_stem = local_inst.stem
        sol_file = None
        
        # Check common AILS2 solution file locations
        possible_sol_files = [
            local_inst.parent / f"{instance_stem}.sol",  # Same dir as instance
            tmp_dir / f"{instance_stem}.sol",  # Work dir
            tmp_dir / f"{local_inst.name}.sol",  # Work dir with full name
            _find_newest_file(tmp_dir, "*.vrp.sol"),
            _find_newest_file(tmp_dir, "*.sol"),
            _find_newest_file(local_inst.parent, "*.sol"),  # Instance dir
        ]
        
        for sf in possible_sol_files:
            if sf and sf.exists():
                sol_file = sf
                break

        # Prefer solution file over stdout
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
            
            # AILS2 outputs customer node IDs (global IDs for full instances)
            # Filter out depot if present
            rr = [n for n in rr if n != 1]
            
            if not rr:
                continue
            
            # Enforce VRPLIB-style [1, customer1, customer2, ..., 1]
            rr = [1] + rr + [1]
            routes_norm.append(rr)

        cost = _parse_cost_from_text(text_for_parse)
        if cost is None:
            cost = float("nan")
        
        # Check capacity feasibility and filter out infeasible routes
        cap = _capacity(inst)
        feasible_routes = []
        infeasible_count = 0
        for route in routes_norm:
            route_load = sum(_demand(inst, node_id) for node_id in route if node_id != 1)
            if route_load <= cap:
                feasible_routes.append(route)
            else:
                infeasible_count += 1
        
        # Use only feasible routes
        routes_norm = feasible_routes
        
        # Log warning if infeasible routes were found
        if infeasible_count > 0:
            import warnings
            warnings.warn(
                f"AILS2 produced {infeasible_count} infeasible route(s) "
                f"(capacity={cap}). Filtered out before returning.",
                UserWarning
            )
        
        # Ensure routes_vrplib is always present (even if empty)
        # solve_clusters requires this field
        if not routes_norm:
            # No routes found - AILS2 might not output them automatically
            # Return empty routes but valid cost for cost tracking
            routes_norm = []

        return SolveOutput(
            solver="ails2",
            instance=instance_path,
            cost=float(cost),
            runtime=float(runtime),
            num_iterations=0,
            feasible=len(routes_norm) > 0,
            data=None,
            raw_result=routes_norm,
            metadata={
                "routes_vrplib": routes_norm,  # Required by solve_clusters
                "tmp_dir": str(tmp_dir),
                "sol_file": str(sol_file) if sol_file else None,
                "routes_found": len(routes_norm),
            },
        )

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@register_solver("ails2")
def _solve_with_ails2(instance_path: Path, options: Dict[str, Any]) -> SolveOutput:
    return _solve_instance_with_ails2_backend(
        instance_path,
        options=options,
    )


# ---------------------------------------------------------------------
# Minimal manual test (cluster mode)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    res = solve_cluster_with_ails2(
        instance_name="X-n502-k39.vrp",
        cluster_customers=[2, 3, 4, 5, 6, 7, 8],
        keep_tmp=True,
        max_runtime=30.0,
    )
    print("Feasible:", res.feasible, "Cost:", res.cost, "Runtime:", res.runtime)
    for i, r in enumerate(res.routes_global, 1):
        print(f"Route {i}:", r)
