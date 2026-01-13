from pathlib import Path
from typing import Optional, Tuple, List
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
    cost: Optional[float] = None,
) -> None:
    """
    Write solution file in OFFICIAL CHECKER FORMAT.

    INTERNAL ASSUMPTION (CRITICAL):
    - All routes are in VRPLIB format:
        depot = 1
        customers = 2 .. n+1

    OUTPUT FORMAT:
    - depot is NOT written
    - customers are 1 .. n  (i.e. VRPLIB node - 1)
    """

    instance_stem = Path(instance_name).stem
    sol_path = where / f"{instance_stem}.sol"
    where.mkdir(parents=True, exist_ok=True)

    # number of customers for validation
    if isinstance(data, dict):
        n_customers = len(data["demand"]) - 1
    else:
        n_customers = data.num_locations - 1

    # detect route format
    is_list_of_routes = (
        isinstance(result, list)
        and len(result) > 0
        and isinstance(result[0], list)
    )

    def write_route(idx: int, vrplib_route: List[int]) -> None:
        """
        Convert VRPLIB route -> official checker route.
        """
        # VRPLIB invariant: depot == 1
        customers_vrplib = [v for v in vrplib_route if v != 1]

        # Convert to checker format: customer_id = vrplib_id - 1
        customers = [v - 1 for v in customers_vrplib]

        # Safety check (THIS would have caught your bug immediately)
        for c in customers:
            if not (1 <= c <= n_customers):
                raise ValueError(
                    f"Invalid customer id {c} in route #{idx} "
                    f"(instance has {n_customers} customers)"
                )

        handle.write(
            f"Route #{idx}: {' '.join(map(str, customers))}\n"
        )

    with open(sol_path, "w", encoding="utf-8") as handle:

        # --------------------------------------------------
        # CASE 1: list of VRPLIB routes
        # --------------------------------------------------
        if is_list_of_routes:
            for idx, route in enumerate(result, 1):
                if not route:
                    continue
                write_route(idx, route)

            if cost is None:
                raise ValueError("Cost must be provided when result is a list of routes")

            handle.write(f"Cost: {int(round(cost))}\n")

        # --------------------------------------------------
        # CASE 2: PyVRP Result object
        # --------------------------------------------------
        else:
            idx = 1
            for route in result.best.routes():
                visits = list(route.visits())  # locations, 0 = depot
                if not visits:
                    continue

                # convert PyVRP -> VRPLIB
                vrplib_route = [1] + [v + 1 for v in visits if v > 0] + [1]
                write_route(idx, vrplib_route)
                idx += 1

            handle.write(f"Cost: {int(round(result.cost()))}\n")

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------
        handle.write("\n")

        handle.write(f"Clustering: {clustering_method or 'n/a'}\n")
        handle.write(f"# Clusters: {k_clusters if k_clusters is not None else 'n/a'}\n")
        handle.write(f"Dissimilarity: {dissimilarity or 'n/a'}\n")
        handle.write(f"Solver: {solver}\n")

        handle.write("\n")

        gap_str = f"{gap_percent:+.2f}%" if gap_percent is not None else "n/a"
        handle.write(f"Runtime: {runtime:.2f}s\n")
        handle.write(f"Stopping Criteria: {stopping_criteria}\n")
        handle.write(f"Gap: {gap_str}\n")

def _write_solution_2(
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
    
    This is a variant of _write_solution specifically designed for PyVRP Result objects
    from routing_controller.py. It uses route.visits() instead of route.schedule() to
    avoid issues with schedule computation.
    
    Produces the exact same output format as _write_solution.
    
    Args:
        where: Path where the .sol file should be written
        instance_name: Name of the instance (for filename: instance_name.sol)
        data: ProblemData object from pyvrp (required, not a dict)
        result: PyVRP Result object with result.best (Solution) and result.cost()
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
    
    # Handle PyVRP Result object format using route.visits() instead of route.schedule()
    # route.visits() returns location indices (0=depot, 1+=clients)
    # Convert location indices to VRPLIB node IDs: location_index + 1 = VRPLIB_node_ID
    # (location 1 -> VRPLIB node 2, location 2 -> VRPLIB node 3, etc.)
    
    with open(sol_path, "w", encoding="utf-8") as handle:
        num_vehicle_types = 1
        if not isinstance(data, dict):
            num_vehicle_types = data.num_vehicle_types
        
        if num_vehicle_types == 1:
            for idx, route in enumerate(result.best.routes(), 1):
                # route.visits() returns location indices (1, 2, ..., n-1 for n total locations)
                # These are client location indices where location 1 = first client
                # Convert to customer node IDs: location_index = customer_node_ID
                # (location 1 -> customer 1, location 501 -> customer 501)
                # Note: This breaks VRPLIB compatibility (VRPLIB uses 2-n for customers)
                location_indices = list(route.visits())
                # Convert location indices directly to customer node IDs (1-indexed)
                # Filter out depot location 0
                customer_nodes = [loc_idx for loc_idx in location_indices if loc_idx > 0]
                if customer_nodes:  # Only write non-empty routes
                    visit_strs = [str(node_id) for node_id in customer_nodes]
                    handle.write(f"Route #{idx}: {' '.join(visit_strs)}\n")
        else:
            # Multiple vehicle types (only if data is ProblemData)
            type2vehicle = [
                (int(vehicle) for vehicle in vehicle_type.name.split(","))
                for vehicle_type in data.vehicle_types()
            ]
            
            routes = [f"Route #{idx + 1}:" for idx in range(data.num_vehicles)]
            for route in result.best.routes():
                # route.visits() returns location indices
                location_indices = list(route.visits())
                # Convert location indices directly to customer node IDs (1-indexed)
                # Filter out depot location 0
                # Note: This breaks VRPLIB compatibility (VRPLIB uses 2-n for customers)
                customer_nodes = [loc_idx for loc_idx in location_indices if loc_idx > 0]
                visit_strs = [str(node_id) for node_id in customer_nodes]
                
                vehicle = next(type2vehicle[route.vehicle_type()])
                routes[vehicle] += " " + " ".join(visit_strs)
            
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