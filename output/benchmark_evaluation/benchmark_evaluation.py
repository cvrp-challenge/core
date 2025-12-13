import re
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Paths
BENCHMARK_DIR = Path(__file__).parent
DRI_OUTPUT_DIR = BENCHMARK_DIR / "benchmark_dri_output"
DR_OUTPUT_DIR = BENCHMARK_DIR / "benchmark_dr_output"
HGS_OUTPUT_DIR = BENCHMARK_DIR.parent / "hgs_output"
FILO_OUTPUT_DIR = BENCHMARK_DIR.parent / "filo_output"
FILO2_OUTPUT_DIR = BENCHMARK_DIR.parent / "filo2_output"


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
            # Match "Cost: <number>" or "Cost <number>"
            match = re.search(r"Cost[:\s]+(\d+\.?\d*)", content, re.IGNORECASE)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Warning: Could not read solution file {sol_path}: {e}")
    
    return None


def extract_gap_from_sol(sol_path: Path) -> Optional[float]:
    """
    Extract gap percentage from a .sol file.
    
    Args:
        sol_path: Path to the .sol file
        
    Returns:
        Gap percentage value if found (e.g., 1.74 for "+1.74%"), None otherwise
    """
    if not sol_path.exists():
        return None
    
    try:
        with open(sol_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Match "Gap: +<number>%" or "Gap: -<number>%" or "Gap: <number>%"
            match = re.search(r"Gap[:\s]+([+-]?)(\d+\.?\d*)%", content, re.IGNORECASE)
            if match:
                sign = match.group(1)
                value = float(match.group(2))
                # Return the value with sign (positive gaps are worse, negative would be better)
                # But typically gaps are positive, so we just return the absolute value
                return abs(value) if sign == '-' else value
    except Exception as e:
        print(f"Warning: Could not read solution file {sol_path}: {e}")
    
    return None


def parse_filename(filename: str) -> Optional[Tuple[str, str, str, int]]:
    """
    Parse DRI filename to extract instance, method, dissimilarity, and k.
    
    Format: {instance}_{method}_{dissimilarity}_k={k}_dri.sol
    
    Returns:
        Tuple of (instance, method, dissimilarity, k) or None if parsing fails
    """
    # Remove .sol extension and _dri suffix
    name = filename.replace("_dri.sol", "").replace(".sol", "")
    
    # Match pattern: instance_method_dissimilarity_k=value
    match = re.match(r"(.+?)_(.+?)_(spatial|combined)_k=(\d+)", name)
    if match:
        instance = match.group(1)
        method = match.group(2)
        dissimilarity = match.group(3)
        k = int(match.group(4))
        return (instance, method, dissimilarity, k)
    
    return None


def parse_dr_filename(filename: str) -> Optional[Tuple[str, str, str, int]]:
    """
    Parse DR filename to extract instance, method, dissimilarity, and k.
    
    Format: {instance}_{method}_{dissimilarity}_k={k}.sol
    
    Returns:
        Tuple of (instance, method, dissimilarity, k) or None if parsing fails
    """
    # Remove .sol extension
    name = filename.replace(".sol", "")
    
    # Match pattern: instance_method_dissimilarity_k=value
    match = re.match(r"(.+?)_(.+?)_(spatial|combined)_k=(\d+)", name)
    if match:
        instance = match.group(1)
        method = match.group(2)
        dissimilarity = match.group(3)
        k = int(match.group(4))
        return (instance, method, dissimilarity, k)
    
    return None


def collect_dri_data(instance_name: str) -> Dict:
    """
    Collect all DRI data for a given instance.
    Uses BKS to recalculate gaps if gap is not found in file.
    
    Returns:
        Dictionary: {(method, k, dissimilarity): gap_percentage}
    """
    data = {}
    
    # Find local BKS for this instance
    bks = find_local_bks(instance_name)
    
    # Find all DRI files for this instance
    pattern = f"{instance_name}_*_dri.sol"
    # Also try without .vrp extension
    instance_stem = instance_name.replace(".vrp", "")
    pattern_stem = f"{instance_stem}_*_dri.sol"
    files = list(DRI_OUTPUT_DIR.glob(pattern)) + list(DRI_OUTPUT_DIR.glob(pattern_stem))
    
    for file_path in files:
        parsed = parse_filename(file_path.name)
        if parsed:
            instance, method, dissimilarity, k = parsed
            # Match instance name with or without .vrp extension
            if instance == instance_name or instance == instance_stem:
                # Try to get gap from file first
                gap = extract_gap_from_sol(file_path)
                
                # If gap not found, calculate from cost and BKS
                if gap is None and bks is not None:
                    cost = extract_cost_from_sol(file_path)
                    if cost is not None:
                        gap = calculate_gap_percent(cost, bks)
                
                if gap is not None:
                    data[(method, k, dissimilarity)] = gap
    
    return data


def find_local_bks(instance_name: str) -> Optional[float]:
    """
    Find the Best Known Solution (BKS) for an instance by searching through:
    1. All solutions in benchmark_dri_output
    2. Solutions in hgs_output, filo_output, and filo2_output
    
    Args:
        instance_name: Name of the instance (without .vrp extension)
        
    Returns:
        Best cost found (BKS), or None if no solutions found
    """
    best_cost = None
    instance_stem = instance_name.replace(".vrp", "")
    
    # 1. Search in benchmark_dri_output
    pattern = f"{instance_name}_*_dri.sol"
    if not instance_name.endswith("_dri.sol"):
        # Also try without .vrp extension
        pattern_stem = f"{instance_stem}_*_dri.sol"
        files = list(DRI_OUTPUT_DIR.glob(pattern)) + list(DRI_OUTPUT_DIR.glob(pattern_stem))
    else:
        files = list(DRI_OUTPUT_DIR.glob(pattern))
    
    for file_path in files:
        cost = extract_cost_from_sol(file_path)
        if cost is not None:
            if best_cost is None or cost < best_cost:
                best_cost = cost
    
    # 2. Search in hgs_output, filo_output, filo2_output
    output_dirs = [HGS_OUTPUT_DIR, FILO_OUTPUT_DIR, FILO2_OUTPUT_DIR]
    
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        
        # Try different filename patterns
        patterns = [
            f"{instance_name}.sol",
            f"{instance_stem}.sol",
            f"{instance_name.replace('.vrp', '')}.sol"
        ]
        
        for pattern in patterns:
            sol_file = output_dir / pattern
            if sol_file.exists():
                cost = extract_cost_from_sol(sol_file)
                if cost is not None:
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
    
    return best_cost


def calculate_gap_percent(cost: float, bks: float) -> float:
    """
    Calculate gap percentage: (cost - bks) / bks * 100
    
    Args:
        cost: Solution cost
        bks: Best known solution cost
        
    Returns:
        Gap percentage (positive value)
    """
    if bks is None or bks <= 0:
        return None
    return ((cost - bks) / bks) * 100.0


def collect_dr_data(instance_name: str) -> Dict:
    """
    Collect all DR data for a given instance.
    Uses BKS to recalculate gaps if gap is not found in file.
    
    Returns:
        Dictionary: {(method, k, dissimilarity): gap_percentage}
    """
    data = {}
    
    # Find local BKS for this instance
    bks = find_local_bks(instance_name)
    
    # Find all DR files for this instance
    pattern = f"{instance_name}_*.sol"
    # Also try without .vrp extension
    instance_stem = instance_name.replace(".vrp", "")
    pattern_stem = f"{instance_stem}_*.sol"
    files = list(DR_OUTPUT_DIR.glob(pattern)) + list(DR_OUTPUT_DIR.glob(pattern_stem))
    
    for file_path in files:
        parsed = parse_dr_filename(file_path.name)
        if parsed:
            instance, method, dissimilarity, k = parsed
            # Match instance name with or without .vrp extension
            if instance == instance_name or instance == instance_stem:
                # Try to get gap from file first
                gap = extract_gap_from_sol(file_path)
                
                # If gap not found, calculate from cost and BKS
                if gap is None and bks is not None:
                    cost = extract_cost_from_sol(file_path)
                    if cost is not None:
                        gap = calculate_gap_percent(cost, bks)
                
                if gap is not None:
                    data[(method, k, dissimilarity)] = gap
    
    return data


def discover_instances() -> list:
    """
    Discover all unique instances from DRI output files.
    
    Returns:
        List of unique instance names
    """
    instances = set()
    
    # Find all DRI files
    files = list(DRI_OUTPUT_DIR.glob("*_dri.sol"))
    
    for file_path in files:
        parsed = parse_filename(file_path.name)
        if parsed:
            instance, _, _, _ = parsed
            instances.add(instance)
    
    return sorted(list(instances))


def discover_dr_instances() -> list:
    """
    Discover all unique instances from DR output files.
    
    Returns:
        List of unique instance names
    """
    instances = set()
    
    # Find all DR files (exclude _dri files)
    files = list(DR_OUTPUT_DIR.glob("*.sol"))
    
    for file_path in files:
        if "_dri.sol" in file_path.name:
            continue  # Skip DRI files
        parsed = parse_dr_filename(file_path.name)
        if parsed:
            instance, _, _, _ = parsed
            instances.add(instance)
    
    return sorted(list(instances))


def create_heatmap(instance_name: str, data: Dict, dissimilarity_type: str):
    """
    Create a heatmap for the given instance and dissimilarity type.
    
    Heatmap structure:
    - Rows: k values (2, 4, 6, 9, 12)
    - Columns: methods (sk_ac_min, sk_ac_avg, sk_ac_complete, sk_kmeans, k_medoids_pyclustering, fcm)
    - Each cell shows the gap for the specified dissimilarity type
    
    Args:
        instance_name: Name of the instance
        data: Dictionary with {(method, k, dissimilarity): gap}
        dissimilarity_type: Either 'spatial' or 'combined'
    """
    # Define k values and methods (full names as they appear in filenames)
    k_values = [2, 4, 6, 9, 12]
    methods = ['sk_ac_min', 'sk_ac_avg', 'sk_ac_complete', 'sk_kmeans', 'k_medoids_pyclustering', 'fcm']
    
    # Mapping for display names (simplified for readability)
    method_display_names = {
        'sk_ac_min': 'ac_min',
        'sk_ac_avg': 'ac_avg',
        'sk_ac_complete': 'ac_max',  # Using ac_max as shown in reference
        'sk_kmeans': 'k_means',
        'k_medoids_pyclustering': 'k_medoids',
        'fcm': 'fcm'
    }
    
    # Collect gaps for the specified dissimilarity type
    gaps = []
    for method in methods:
        for k in k_values:
            if (method, k, dissimilarity_type) in data:
                gaps.append(data[(method, k, dissimilarity_type)])
    
    # Normalize gaps (lower is better)
    # Normalize to 0-1 where 0 = best (lowest gap), 1 = worst (highest gap)
    if gaps:
        gap_min = min(gaps)
        gap_max = max(gaps)
        gap_range = gap_max - gap_min if gap_max > gap_min else 1
    else:
        gap_min = gap_max = gap_range = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cell dimensions
    cell_width = 1.0
    cell_height = 1.0
    
    # Create custom colormaps based on dissimilarity type
    if dissimilarity_type == 'spatial':
        # Green gradient for spatial (very light green to saturated green)
        # Lower gap (better) = darker green, higher gap (worse) = lighter green
        colors = [(0.95, 1.0, 0.95), (0.2, 0.7, 0.2)]  # very light green to lighter saturated green
        cmap = LinearSegmentedColormap.from_list('spatial', colors, N=256)
    else:  # combined
        # Purple gradient for combined (very light purple to saturated purple)
        # Lower gap (better) = darker purple, higher gap (worse) = lighter purple
        colors = [(0.98, 0.95, 1.0), (0.7, 0.3, 0.7)]  # very light purple to lighter saturated purple
        cmap = LinearSegmentedColormap.from_list('combined', colors, N=256)
    
    # Draw cells
    # K values from top to bottom: 2, 4, 6, 9, 12
    for i, k in enumerate(k_values):
        for j, method in enumerate(methods):
            # Cell position (bottom-left corner)
            x = j * cell_width
            y = (len(k_values) - 1 - i) * cell_height  # k=2 at top (index 0), k=12 at bottom (index 4)
            
            # Get gap for this dissimilarity type
            gap = data.get((method, k, dissimilarity_type))
            
            # Draw cell as full rectangle
            if gap is not None:
                # Normalize: 0 = best (lowest gap), 1 = worst (highest gap)
                normalized = (gap - gap_min) / gap_range if gap_range > 0 else 0
                # Lower gap = more saturated (darker color) = use 1.0 - normalized
                # So best (normalized=0) -> 1.0 (darkest), worst (normalized=1) -> 0.0 (lightest)
                color = cmap(1.0 - normalized)
                
                cell_rect = patches.Rectangle(
                    (x, y), cell_width, cell_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(cell_rect)
            else:
                # Grey out if missing
                cell_rect = patches.Rectangle(
                    (x, y), cell_width, cell_height,
                    facecolor='lightgray',
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(cell_rect)
            
            # Add gap percentage text label (always black, centered)
            if gap is not None:
                gap_text = f"{gap:.2f}%"
                ax.text(x + cell_width / 2, y + cell_height / 2, gap_text, 
                       fontsize=10, ha='center', va='center', weight='bold',
                       color='black')
    
    # Set axis labels and ticks
    ax.set_xlim(0, len(methods) * cell_width)
    ax.set_ylim(0, len(k_values) * cell_height)
    
    # Set ticks
    # Methods on top
    ax.set_xticks([i * cell_width + cell_width / 2 for i in range(len(methods))])
    ax.set_xticklabels([method_display_names.get(m, m) for m in methods], rotation=0, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # K values from top to bottom: 2, 4, 6, 9, 12
    # Cells are drawn with k=2 at top (y=4), k=12 at bottom (y=0)
    # So ticks need to match: top tick for k=2, bottom tick for k=12
    ax.set_yticks([(len(k_values) - 1 - i) * cell_height + cell_height / 2 for i in range(len(k_values))])
    ax.set_yticklabels([f"k = {k}" for k in k_values])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel('Methods', fontsize=12, labelpad=10)
    ax.set_ylabel('k-values', fontsize=12, labelpad=10)
    if instance_name.startswith("SUMMARY_"):
        ax.set_title(f'Average Performance Heatmap: All Instances ({dissimilarity_type})', fontsize=14, pad=20)
    else:
        ax.set_title(f'Performance Heatmap: {instance_name} ({dissimilarity_type})', fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = BENCHMARK_DIR / f"{instance_name}_{dissimilarity_type}_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    
    plt.close()


def create_comparison_heatmap(instance_name: str, data: Dict):
    """
    Create a comparison heatmap showing which dissimilarity type performed better.
    
    For each (method, k) combination:
    - Green if spatial performed better (lower gap)
    - Purple if combined performed better (lower gap)
    - Saturation based on how good the better performance is
    
    Args:
        instance_name: Name of the instance
        data: Dictionary with {(method, k, dissimilarity): gap}
    """
    # Define k values and methods (full names as they appear in filenames)
    k_values = [2, 4, 6, 9, 12]
    methods = ['sk_ac_min', 'sk_ac_avg', 'sk_ac_complete', 'sk_kmeans', 'k_medoids_pyclustering', 'fcm']
    
    # Mapping for display names (simplified for readability)
    method_display_names = {
        'sk_ac_min': 'ac_min',
        'sk_ac_avg': 'ac_avg',
        'sk_ac_complete': 'ac_max',
        'sk_kmeans': 'k_means',
        'k_medoids_pyclustering': 'k_medoids',
        'fcm': 'fcm'
    }
    
    # Collect all best gaps (minimum of spatial and combined for each cell)
    best_gaps = []
    cell_info = {}  # {(method, k): (best_gap, winner_type, spatial_gap, combined_gap)}
    
    for method in methods:
        for k in k_values:
            spatial_gap = data.get((method, k, 'spatial'))
            combined_gap = data.get((method, k, 'combined'))
            
            if spatial_gap is not None and combined_gap is not None:
                # Both exist - compare them
                if spatial_gap <= combined_gap:
                    best_gap = spatial_gap
                    winner = 'spatial'
                else:
                    best_gap = combined_gap
                    winner = 'combined'
                best_gaps.append(best_gap)
                cell_info[(method, k)] = (best_gap, winner, spatial_gap, combined_gap)
            elif spatial_gap is not None:
                # Only spatial exists
                best_gap = spatial_gap
                winner = 'spatial'
                best_gaps.append(best_gap)
                cell_info[(method, k)] = (best_gap, winner, spatial_gap, None)
            elif combined_gap is not None:
                # Only combined exists
                best_gap = combined_gap
                winner = 'combined'
                best_gaps.append(best_gap)
                cell_info[(method, k)] = (best_gap, winner, None, combined_gap)
            else:
                # Neither exists
                cell_info[(method, k)] = (None, None, None, None)
    
    # Normalize best gaps for saturation
    if best_gaps:
        gap_min = min(best_gaps)
        gap_max = max(best_gaps)
        gap_range = gap_max - gap_min if gap_max > gap_min else 1
    else:
        gap_min = gap_max = gap_range = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cell dimensions
    cell_width = 1.0
    cell_height = 1.0
    
    # Create custom colormaps
    colors_green = [(0.95, 1.0, 0.95), (0.2, 0.7, 0.2)]  # very light green to lighter saturated green
    cmap_green = LinearSegmentedColormap.from_list('spatial', colors_green, N=256)
    
    colors_purple = [(0.98, 0.95, 1.0), (0.7, 0.3, 0.7)]  # very light purple to lighter saturated purple
    cmap_purple = LinearSegmentedColormap.from_list('combined', colors_purple, N=256)
    
    # Draw cells
    for i, k in enumerate(k_values):
        for j, method in enumerate(methods):
            # Cell position (bottom-left corner)
            x = j * cell_width
            y = (len(k_values) - 1 - i) * cell_height
            
            # Get cell info
            info = cell_info.get((method, k), (None, None, None, None))
            best_gap, winner, spatial_gap, combined_gap = info
            
            # Draw cell
            if best_gap is not None and winner is not None:
                # Normalize: 0 = best (lowest gap), 1 = worst (highest gap)
                normalized = (best_gap - gap_min) / gap_range if gap_range > 0 else 0
                # Lower gap = more saturated (darker color) = use 1.0 - normalized
                saturation = 1.0 - normalized
                
                # Choose color based on winner
                if winner == 'spatial':
                    color = cmap_green(saturation)
                else:  # combined
                    color = cmap_purple(saturation)
                
                cell_rect = patches.Rectangle(
                    (x, y), cell_width, cell_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(cell_rect)
                
                # Add text labels - better one in bold black, other in normal grey
                center_x = x + cell_width / 2
                center_y = y + cell_height / 2
                
                if spatial_gap is not None and combined_gap is not None:
                    # Both exist - show both with appropriate formatting
                    if winner == 'spatial':
                        # Spatial won - bold black
                        ax.text(center_x, center_y + 0.1, f"S: {spatial_gap:.2f}%",
                               fontsize=8, ha='center', va='center', weight='bold',
                               color='black')
                        # Combined lost - darker grey
                        ax.text(center_x, center_y - 0.1, f"C: {combined_gap:.2f}%",
                               fontsize=8, ha='center', va='center', weight='normal',
                               color=(0.4, 0.4, 0.4))
                    else:  # combined won
                        # Spatial lost - darker grey
                        ax.text(center_x, center_y + 0.1, f"S: {spatial_gap:.2f}%",
                               fontsize=8, ha='center', va='center', weight='normal',
                               color=(0.4, 0.4, 0.4))
                        # Combined won - bold black
                        ax.text(center_x, center_y - 0.1, f"C: {combined_gap:.2f}%",
                               fontsize=8, ha='center', va='center', weight='bold',
                               color='black')
                elif spatial_gap is not None:
                    # Only spatial exists
                    ax.text(center_x, center_y, f"S: {spatial_gap:.2f}%",
                           fontsize=8, ha='center', va='center', weight='bold',
                           color='black')
                elif combined_gap is not None:
                    # Only combined exists
                    ax.text(center_x, center_y, f"C: {combined_gap:.2f}%",
                           fontsize=8, ha='center', va='center', weight='bold',
                           color='black')
            else:
                # Grey out if missing
                cell_rect = patches.Rectangle(
                    (x, y), cell_width, cell_height,
                    facecolor='lightgray',
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(cell_rect)
    
    # Set axis labels and ticks
    ax.set_xlim(0, len(methods) * cell_width)
    ax.set_ylim(0, len(k_values) * cell_height)
    
    # Set ticks
    # Methods on top
    ax.set_xticks([i * cell_width + cell_width / 2 for i in range(len(methods))])
    ax.set_xticklabels([method_display_names.get(m, m) for m in methods], rotation=0, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # K values from top to bottom: 2, 4, 6, 9, 12
    ax.set_yticks([(len(k_values) - 1 - i) * cell_height + cell_height / 2 for i in range(len(k_values))])
    ax.set_yticklabels([f"k = {k}" for k in k_values])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel('Methods', fontsize=12, labelpad=10)
    ax.set_ylabel('k-values', fontsize=12, labelpad=10)
    if instance_name.startswith("SUMMARY_"):
        ax.set_title(f'Best Average Performance Comparison: All Instances (Green=Spatial, Purple=Combined)', fontsize=14, pad=20)
    else:
        ax.set_title(f'Best Performance Comparison: {instance_name} (Green=Spatial, Purple=Combined)', fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = BENCHMARK_DIR / f"{instance_name}_comparison_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison heatmap to {output_path}")
    
    plt.close()


def create_summary_heatmaps(instances: list):
    """
    Create summary heatmaps showing average performance across all instances.
    
    Args:
        instances: List of all instance names
    """
    # Collect all data across all instances
    all_data = {}  # {(method, k, dissimilarity): [gap1, gap2, ...]}
    
    for instance_name in instances:
        data = collect_dri_data(instance_name)
        for key, gap in data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(gap)
    
    # Calculate average gaps
    avg_data = {}
    for key, gaps in all_data.items():
        if gaps:
            avg_data[key] = sum(gaps) / len(gaps)
    
    if not avg_data:
        print("  No data found for summary heatmaps!")
        return
    
    # Create summary heatmaps using the same functions but with aggregated data
    print(f"  Found {len(avg_data)} aggregated data points")
    print(f"  Creating summary heatmaps...")
    create_heatmap("SUMMARY_ALL_INSTANCES", avg_data, 'spatial')
    create_heatmap("SUMMARY_ALL_INSTANCES", avg_data, 'combined')
    create_comparison_heatmap("SUMMARY_ALL_INSTANCES", avg_data)
    print(f"  ✓ Summary heatmaps created")


def create_dr_summary_heatmaps():
    """
    Create summary heatmaps for DR output across all instances.
    """
    # Discover all DR instances
    instances = discover_dr_instances()
    
    if not instances:
        print("  No instances found in benchmark_dr_output folder!")
        return
    
    print(f"  Found {len(instances)} DR instances: {', '.join(instances)}")
    
    # Collect all data across all instances
    all_data = {}  # {(method, k, dissimilarity): [gap1, gap2, ...]}
    
    for instance_name in instances:
        data = collect_dr_data(instance_name)
        for key, gap in data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(gap)
    
    # Calculate average gaps
    avg_data = {}
    for key, gaps in all_data.items():
        if gaps:
            avg_data[key] = sum(gaps) / len(gaps)
    
    if not avg_data:
        print("  No data found for DR summary heatmaps!")
        return
    
    # Create summary heatmaps using the same functions but with aggregated data
    print(f"  Found {len(avg_data)} aggregated data points")
    print(f"  Creating DR summary heatmaps...")
    create_heatmap("SUMMARY_DR_ALL_INSTANCES", avg_data, 'spatial')
    create_heatmap("SUMMARY_DR_ALL_INSTANCES", avg_data, 'combined')
    create_comparison_heatmap("SUMMARY_DR_ALL_INSTANCES", avg_data)
    print(f"  ✓ DR summary heatmaps created")


def main():
    # Discover all instances
    instances = discover_instances()
    
    if not instances:
        print("No instances found in benchmark_dri_output folder!")
        return
    
    print(f"Found {len(instances)} instances: {', '.join(instances)}")
    print("\n" + "="*80 + "\n")
    
    # Process each instance
    for instance_name in instances:
        print(f"Processing instance: {instance_name}")
        print(f"Collecting data for {instance_name}...")
        
        # Find BKS for this instance
        bks = find_local_bks(instance_name)
        if bks is not None:
            print(f"  BKS found: {bks:.2f}")
        else:
            print(f"  Warning: No BKS found for {instance_name}")
        
        data = collect_dri_data(instance_name)
        
        if not data:
            print(f"  No data found for {instance_name}, skipping...\n")
            continue
        
        print(f"  Found {len(data)} data points")
        
        print(f"  Creating heatmaps for {instance_name}...")
        create_heatmap(instance_name, data, 'spatial')
        create_heatmap(instance_name, data, 'combined')
        create_comparison_heatmap(instance_name, data)
        print(f"  ✓ Completed {instance_name}\n")
    
    print("="*80)
    print(f"Done! Processed {len(instances)} instance(s).")
    
    # Create summary heatmaps across all instances
    print("\n" + "="*80)
    print("Creating summary heatmaps across all instances...")
    create_summary_heatmaps(instances)
    print("="*80)
    print("All done!")
    
    # Create summary heatmaps across all instances
    print("\n" + "="*80)
    print("Creating summary heatmaps across all instances...")
    create_summary_heatmaps(instances)
    print("="*80)
    print("All done!")


if __name__ == "__main__":
    import sys
    
    # Check if user wants DR summary heatmaps
    if len(sys.argv) > 1 and sys.argv[1] == "--dr":
        print("="*80)
        print("Creating DR summary heatmaps across all instances...")
        create_dr_summary_heatmaps()
        print("="*80)
        print("Done!")
    else:
        main()

