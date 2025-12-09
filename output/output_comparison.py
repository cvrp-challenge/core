"""
Output comparison script for HGS, FILO, and FILO2 solvers.

Creates a comparison table with:
- Instance name
- BKS (Best Known Solution)
- HGS: gap and runtime
- FILO: gap and runtime
- FILO2: gap and runtime
"""

import sys
import re
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
CURRENT = Path(__file__).parent
PROJECT_ROOT = CURRENT.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from master.utils.solution_helpers import find_existing_solution

INSTANCES = [
    "X-n502-k39.vrp",
    "X-n524-k153.vrp",
    "X-n561-k42.vrp",
    "X-n641-k35.vrp",
    "X-n685-k75.vrp",
    "X-n716-k35.vrp",
    "X-n749-k98.vrp",
    "X-n801-k40.vrp",
    "X-n856-k95.vrp",
    "X-n916-k207.vrp",
    "XLTEST-n1048-k138.vrp",
    "XLTEST-n1794-k408.vrp",
    "XLTEST-n2541-k62.vrp",
    "XLTEST-n3147-k210.vrp",
    "XLTEST-n4153-k259.vrp",
    "XLTEST-n6034-k1685.vrp",
    "XLTEST-n6734-k1347.vrp",
    "XLTEST-n8028-k691.vrp",
    "XLTEST-n8766-k55.vrp",
    "XLTEST-n10001-k798.vrp",
]

# Output directories
HGS_DIR = CURRENT / "hgs_output"
FILO_DIR = CURRENT / "filo_output"
FILO2_DIR = CURRENT / "filo2_output"


def extract_cost_and_runtime(sol_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract cost and runtime from a solution file.
    
    Returns:
        Tuple of (cost, runtime_seconds) or (None, None) if not found
    """
    if not sol_path.exists():
        return None, None
    
    try:
        with open(sol_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract cost: "Cost: 69827" or "Cost: 69239.0"
        cost_match = re.search(r"Cost:\s*([\d.]+)", content)
        cost = float(cost_match.group(1)) if cost_match else None
        
        # Extract runtime: "Runtime: 30.13s"
        runtime_match = re.search(r"Runtime:\s*([\d.]+)s", content)
        runtime = float(runtime_match.group(1)) if runtime_match else None
        
        return cost, runtime
    except Exception as e:
        print(f"Warning: Could not parse {sol_path}: {e}")
        return None, None


def calculate_gap(cost: float, bks: float) -> float:
    """
    Calculate gap percentage: ((cost - bks) / bks) * 100
    
    Args:
        cost: The solution cost
        bks: The best known solution cost
        
    Returns:
        Gap percentage (positive means cost is worse)
    """
    if bks == 0:
        return float('inf') if cost > 0 else 0.0
    return ((cost - bks) / bks) * 100.0


def get_reference_bks(instance_name: str) -> Optional[float]:
    """Get reference Best Known Solution cost for an instance (if available)."""
    ref = find_existing_solution(instance_name)
    return ref[1] if ref else None


def create_comparison_table():
    """Create and print the comparison table."""
    rows = []
    
    for instance in INSTANCES:
        instance_stem = Path(instance).stem
        
        # Get reference BKS (if available)
        reference_bks = get_reference_bks(instance)
        
        # Get costs and runtimes from solution files
        hgs_sol = HGS_DIR / f"{instance_stem}.sol"
        hgs_cost, hgs_runtime = extract_cost_and_runtime(hgs_sol)
        
        filo_sol = FILO_DIR / f"{instance_stem}.sol"
        filo_cost, filo_runtime = extract_cost_and_runtime(filo_sol)
        
        filo2_sol = FILO2_DIR / f"{instance_stem}.sol"
        filo2_cost, filo2_runtime = extract_cost_and_runtime(filo2_sol)
        
        # Determine BKS: 
        # - For XL instances: always use best among solvers
        # - For X instances: use reference if available, otherwise best among solvers
        costs = [c for c in [hgs_cost, filo_cost, filo2_cost] if c is not None]
        is_xl_instance = instance_stem.startswith("XLTEST-")
        
        if costs:
            solver_bks = min(costs)
            if is_xl_instance:
                # For XL instances, always use best solver result as BKS
                bks = solver_bks
            else:
                # For X instances, prefer reference BKS if available
                bks = reference_bks if reference_bks is not None else solver_bks
        else:
            bks = reference_bks if not is_xl_instance else None
        
        # Calculate gaps relative to BKS
        hgs_gap = calculate_gap(hgs_cost, bks) if hgs_cost is not None and bks is not None else None
        filo_gap = calculate_gap(filo_cost, bks) if filo_cost is not None and bks is not None else None
        filo2_gap = calculate_gap(filo2_cost, bks) if filo2_cost is not None and bks is not None else None
        
        rows.append({
            "Instance": instance_stem,
            "BKS": bks,
            "HGS_Cost": hgs_cost,
            "HGS_Gap": hgs_gap,
            "HGS_Runtime": hgs_runtime,
            "FILO_Cost": filo_cost,
            "FILO_Gap": filo_gap,
            "FILO_Runtime": filo_runtime,
            "FILO2_Cost": filo2_cost,
            "FILO2_Gap": filo2_gap,
            "FILO2_Runtime": filo2_runtime,
        })
    
    # Print table
    print("\n" + "=" * 140)
    print("SOLVER COMPARISON TABLE")
    print("=" * 140)
    
    # Header
    header = (
        f"{'Instance':<25} | "
        f"{'BKS':>12} | "
        f"{'HGS Gap':>10} | {'HGS Time':>12} | "
        f"{'FILO Gap':>10} | {'FILO Time':>12} | "
        f"{'FILO2 Gap':>11} | {'FILO2 Time':>13}"
    )
    print(header)
    print("-" * 140)
    
    # Data rows
    for row in rows:
        instance = row["Instance"]
        bks = f"{row['BKS']:.2f}" if row["BKS"] is not None else "N/A"
        
        hgs_gap = f"{row['HGS_Gap']:+.2f}%" if row["HGS_Gap"] is not None else "N/A"
        hgs_time = f"{row['HGS_Runtime']:.2f}s" if row["HGS_Runtime"] is not None else "N/A"
        
        filo_gap = f"{row['FILO_Gap']:+.2f}%" if row["FILO_Gap"] is not None else "N/A"
        filo_time = f"{row['FILO_Runtime']:.2f}s" if row["FILO_Runtime"] is not None else "N/A"
        
        filo2_gap = f"{row['FILO2_Gap']:+.2f}%" if row["FILO2_Gap"] is not None else "N/A"
        filo2_time = f"{row['FILO2_Runtime']:.2f}s" if row["FILO2_Runtime"] is not None else "N/A"
        
        print(
            f"{instance:<25} | "
            f"{bks:>12} | "
            f"{hgs_gap:>10} | {hgs_time:>12} | "
            f"{filo_gap:>10} | {filo_time:>12} | "
            f"{filo2_gap:>11} | {filo2_time:>13}"
        )
    
    print("=" * 140)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 140)
    
    # Count instances with data
    hgs_count = sum(1 for r in rows if r["HGS_Gap"] is not None)
    filo_count = sum(1 for r in rows if r["FILO_Gap"] is not None)
    filo2_count = sum(1 for r in rows if r["FILO2_Gap"] is not None)
    
    print(f"Instances with HGS results: {hgs_count}/{len(rows)}")
    print(f"Instances with FILO results: {filo_count}/{len(rows)}")
    print(f"Instances with FILO2 results: {filo2_count}/{len(rows)}")
    
    # Average gaps (only for instances where we have BKS)
    hgs_gaps = [r["HGS_Gap"] for r in rows if r["HGS_Gap"] is not None and r["BKS"] is not None]
    filo_gaps = [r["FILO_Gap"] for r in rows if r["FILO_Gap"] is not None and r["BKS"] is not None]
    filo2_gaps = [r["FILO2_Gap"] for r in rows if r["FILO2_Gap"] is not None and r["BKS"] is not None]
    
    if hgs_gaps:
        avg_hgs_gap = sum(hgs_gaps) / len(hgs_gaps)
        print(f"Average HGS gap: {avg_hgs_gap:+.2f}%")
    
    if filo_gaps:
        avg_filo_gap = sum(filo_gaps) / len(filo_gaps)
        print(f"Average FILO gap: {avg_filo_gap:+.2f}%")
    
    if filo2_gaps:
        avg_filo2_gap = sum(filo2_gaps) / len(filo2_gaps)
        print(f"Average FILO2 gap: {avg_filo2_gap:+.2f}%")
    
    # Average runtimes
    if hgs_count > 0:
        avg_hgs_time = sum(r["HGS_Runtime"] for r in rows if r["HGS_Runtime"] is not None) / hgs_count
        print(f"Average HGS runtime: {avg_hgs_time:.2f}s")
    
    if filo_count > 0:
        avg_filo_time = sum(r["FILO_Runtime"] for r in rows if r["FILO_Runtime"] is not None) / filo_count
        print(f"Average FILO runtime: {avg_filo_time:.2f}s")
    
    if filo2_count > 0:
        avg_filo2_time = sum(r["FILO2_Runtime"] for r in rows if r["FILO2_Runtime"] is not None) / filo2_count
        print(f"Average FILO2 runtime: {avg_filo2_time:.2f}s")


if __name__ == "__main__":
    create_comparison_table()