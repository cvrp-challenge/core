from pathlib import Path
from typing import Optional, Tuple
import re

# Ensure project root packages
CURRENT = Path(__file__).resolve()
# For src/master/utils/solution_helpers.py, parents[3] is the repo root (core)
REPO_ROOT = CURRENT.parents[3]
X_SOLUTIONS_DIR = REPO_ROOT / "instances" / "test-instances" / "x-solutions"
XL_SOLUTIONS_DIR = REPO_ROOT / "instances" / "test-instances" / "xl-solutions"

def extract_cost_from_sol(sol_path: Path) -> Optional[float]:
    """
    Extract cost value from a .sol file.
    
    Args:
        sol_path: Path to the .sol file
        
    Returns:
        Cost value if found, None otherwise
    """
    if not sol_path.exists():
        return None
    
    try:
        with open(sol_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Match "Cost: <number>" or "Cost <number>" at the end
            match = re.search(r"Cost[:\s]+(\d+\.?\d*)", content, re.IGNORECASE)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Warning: Could not read solution file {sol_path}: {e}")
    
    return None


def find_existing_solution(instance_name: str) -> Optional[Tuple[Path, float]]:
    """
    Check if a solution exists for the given instance in x-solutions or xl-solutions.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n101-k25.vrp")
        
    Returns:
        Tuple of (solution_path, cost) if found, None otherwise
    """
    instance_stem = Path(instance_name).stem  # e.g., "X-n101-k25"
    sol_name = f"{instance_stem}.sol"
    
    # Check x-solutions directory
    x_sol_path = X_SOLUTIONS_DIR / sol_name
    cost = extract_cost_from_sol(x_sol_path)
    if cost is not None:
        return (x_sol_path, cost)
    
    # Check xl-solutions directory
    xl_sol_path = XL_SOLUTIONS_DIR / sol_name
    cost = extract_cost_from_sol(xl_sol_path)
    if cost is not None:
        return (xl_sol_path, cost)
    
    return None


def calculate_gap(new_cost: float, reference_cost: float) -> float:
    """
    Calculate gap percentage: ((new_cost - reference_cost) / reference_cost) * 100
    
    Args:
        new_cost: The new solution cost
        reference_cost: The reference solution cost
        
    Returns:
        Gap percentage (positive means new_cost is worse)
    """
    if reference_cost == 0:
        return float('inf') if new_cost > 0 else 0.0
    return ((new_cost - reference_cost) / reference_cost) * 100.0


def _write_solution(
    where: Path,
    instance_name: str,
    data,
    result,
    solver: str,
    runtime: float,
    stopping_criteria: str,
    gap_percent: Optional[float] = None,
    clustering_method: Optional[str] = None,
    k_clusters: Optional[int] = None,
    dissimilarity: Optional[str] = None,
) -> None:
    """
    Write solution file in VRPLIB format with additional metadata.
    
    Format:
    1. Routes (standard VRPLIB format)
    2. Cost (standard VRPLIB format)
    3. Empty row
    4. Clustering Method, Dissimilarity, Solver
    5. Empty row
    6. Runtime, Stopping Criteria, Gap
    
    Args:
        where: Path where the .sol file should be written
        instance_name: Name of the instance (for filename: instance_name.sol)
        data: ProblemData object from pyvrp
        result: Result object from pyvrp
        solver: Name of the solver used
        runtime: Time used to get the solution (in seconds)
        stopping_criteria: Description of the stopping criteria
        gap_percent: Gap percentage if reference solution exists, None otherwise
        clustering_method: Clustering method used (e.g., "sk_ac_avg", "fcm")
        dissimilarity: Dissimilarity matrix used ("spatial" or "combined")
    """
    # Ensure the instance_name is just the stem (no extension)
    instance_stem = Path(instance_name).stem
    
    # Use instance_name.sol naming convention
    sol_path = where / f"{instance_stem}.sol"
    
    with open(sol_path, "w", encoding="utf-8") as handle:
        # Write standard VRPLIB format: routes
        if data.num_vehicle_types == 1:
            for idx, route in enumerate(result.best.routes(), 1):
                visits = [str(visit.location) for visit in route.schedule()]
                visits = visits[1:-1]  # drop depot markers
                handle.write(f"Route #{idx}: {' '.join(visits)}\n")
        else:
            # Multiple vehicle types
            type2vehicle = [
                (int(vehicle) for vehicle in vehicle_type.name.split(","))
                for vehicle_type in data.vehicle_types()
            ]
            
            routes = [f"Route #{idx + 1}:" for idx in range(data.num_vehicles)]
            for route in result.best.routes():
                visits = [str(visit.location) for visit in route.schedule()]
                visits = visits[1:-1]  # drop depot markers
                
                vehicle = next(type2vehicle[route.vehicle_type()])
                routes[vehicle] += " " + " ".join(visits)
            
            handle.writelines(route + "\n" for route in routes)
        
        # Write standard VRPLIB format: cost
        handle.write(f"Cost: {round(result.cost(), 2)}\n")

        # Empty row
        handle.write("\n")
        
        # Clustering Method, Dissimilarity, Solver
        clustering_str = clustering_method if clustering_method is not None else "n/a"
        k_clusters_str = k_clusters if k_clusters is not None else "n/a"
        dissimilarity_str = dissimilarity if dissimilarity is not None else "n/a"
        handle.write(f"Clustering: {clustering_str}\n")
        handle.write(f"# Clusters: {k_clusters_str}\n")
        handle.write(f"Dissimilarity: {dissimilarity_str}\n")
        handle.write(f"Solver: {solver}\n")
        
        # Empty row
        handle.write("\n")
        
        # Runtime, Stopping Criteria, Gap
        gap_str = f"{gap_percent:+.2f}%" if gap_percent is not None else "n/a"
        handle.write(f"Runtime: {runtime:.2f}s\n")
        handle.write(f"Stopping Criteria: {stopping_criteria}\n")
        handle.write(f"Gap: {gap_str}\n")