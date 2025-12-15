"""
Solver comparison script including DRSCI.

Compares:
- HGS
- FILO
- FILO2
- DRSCI

All gaps are computed relative to BKS.

DRSCI files are expected as:
    <instance>_drsci.sol
"""

import sys
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CURRENT = Path(__file__).parent
PROJECT_ROOT = CURRENT.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from master.utils.solution_helpers import find_existing_solution

# ------------------------------------------------------------------
# Instances
# ------------------------------------------------------------------
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
]

# ------------------------------------------------------------------
# Output directories (as you specified)
# ------------------------------------------------------------------
HGS_DIR = Path(r"C:\Users\robin\Documents\PS_CVRP\core\output\hgs_output")
FILO_DIR = Path(r"C:\Users\robin\Documents\PS_CVRP\core\output\filo_output")
FILO2_DIR = Path(r"C:\Users\robin\Documents\PS_CVRP\core\output\filo2_output")

# DRSCI lives in the *general* output folder:
DRSCI_DIR = Path(r"C:\Users\robin\Documents\PS_CVRP\output")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def extract_cost_and_runtime(sol_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Extract cost and runtime from a .sol file."""
    if not sol_path.exists():
        return None, None

    try:
        content = sol_path.read_text(encoding="utf-8", errors="ignore")

        # e.g. "Cost: 69827" or "Cost: 69239.0"
        cost_match = re.search(r"Cost:\s*([\d.]+)", content)
        # e.g. "Runtime: 30.13s"
        runtime_match = re.search(r"Runtime:\s*([\d.]+)s", content)

        cost = float(cost_match.group(1)) if cost_match else None
        runtime = float(runtime_match.group(1)) if runtime_match else None

        return cost, runtime
    except Exception as e:
        print(f"[WARN] Could not parse {sol_path}: {e}")
        return None, None


def calculate_gap(cost: float, bks: float) -> float:
    """Gap in percent: ((cost - bks) / bks) * 100"""
    if bks == 0:
        return float("inf")
    return ((cost - bks) / bks) * 100.0


def get_reference_bks(instance: str) -> Optional[float]:
    """Reference BKS from solution_helpers (if available)."""
    ref = find_existing_solution(instance)
    return ref[1] if ref else None


def fmt_gap(x: Optional[float]) -> str:
    return f"{x:+.2f}%" if x is not None else "N/A"


def fmt_time(x: Optional[float]) -> str:
    return f"{x:.2f}s" if x is not None else "N/A"


def fmt_bks(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "N/A"


# ------------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------------
def create_comparison_table() -> None:
    rows: List[Dict[str, Any]] = []

    for instance in INSTANCES:
        stem = Path(instance).stem

        ref_bks = get_reference_bks(instance)

        hgs_cost, hgs_time = extract_cost_and_runtime(HGS_DIR / f"{stem}.sol")
        filo_cost, filo_time = extract_cost_and_runtime(FILO_DIR / f"{stem}.sol")
        filo2_cost, filo2_time = extract_cost_and_runtime(FILO2_DIR / f"{stem}.sol")

        # DRSCI special naming: <stem>_drsci.sol
        drsci_cost, drsci_time = extract_cost_and_runtime(DRSCI_DIR / f"{stem}_drsci.sol")

        # Choose BKS: reference if available, else best among solver results we have
        solver_costs = [c for c in [hgs_cost, filo_cost, filo2_cost, drsci_cost] if c is not None]
        if ref_bks is not None:
            bks = ref_bks
        else:
            bks = min(solver_costs) if solver_costs else None

        def gap(c: Optional[float]) -> Optional[float]:
            if c is None or bks is None:
                return None
            return calculate_gap(c, bks)

        rows.append({
            "Instance": stem,
            "BKS": bks,

            "HGS_Gap": gap(hgs_cost),
            "HGS_Time": hgs_time,

            "FILO_Gap": gap(filo_cost),
            "FILO_Time": filo_time,

            "FILO2_Gap": gap(filo2_cost),
            "FILO2_Time": filo2_time,

            "DRSCI_Gap": gap(drsci_cost),
            "DRSCI_Time": drsci_time,
        })

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 140)
    print("SOLVER COMPARISON TABLE (GAP vs BKS)")
    print("=" * 140)

    header = (
        f"{'Instance':<25} | "
        f"{'BKS':>12} | "
        f"{'HGS Gap':>10} |  "
        f"{'FILO Gap':>10} | "
        f"{'FILO2 Gap':>11} | "
        f"{'DRSCI Gap':>11} "
    )
    print(header)
    print("-" * 140)

    for r in rows:
        instance_str = r["Instance"]
        bks_str = fmt_bks(r["BKS"])

        print(
            f"{instance_str:<25} | "
            f"{bks_str:>12} | "
            f"{fmt_gap(r['HGS_Gap']):>10} | "
            f"{fmt_gap(r['FILO_Gap']):>10}  | "
            f"{fmt_gap(r['FILO2_Gap']):>11}  | "
            f"{fmt_gap(r['DRSCI_Gap']):>11} "
        )

    print("=" * 140)


if __name__ == "__main__":
    create_comparison_table()
