#!/usr/bin/env python3
"""
Analysis script for challenge test run results.

Tasks:
1. Extract route pool statistics by mode, method, solver, and stage for each instance
2. Create convergence graphs showing improvements (DRI vs SCP phases)
"""

import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Configuration
LOG_DIR = Path(__file__).parent / "logs"
OUTPUT_DIR = Path(__file__).parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class RoutePoolStats:
    """Statistics for routes in the pool by category."""
    mode: str
    method: str
    solver: str
    stage: str
    count: int


@dataclass
class Improvement:
    """Record of an improvement event."""
    timestamp: str
    time_seconds: float
    cost: float
    phase: str  # "DRI" or "SCP"
    iteration: Optional[int] = None


def parse_timestamp(line: str) -> Optional[float]:
    """Parse timestamp from log line and convert to seconds since start."""
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if not match:
        return None
    
    try:
        dt = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        return dt.timestamp()
    except:
        return None


def extract_route_pool_summary(log_file: Path) -> List[RoutePoolStats]:
    """Extract route pool summary from log file."""
    stats = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the last ROUTE POOL SUMMARY section
    in_summary = False
    summary_lines = []
    
    for i, line in enumerate(lines):
        if '[ROUTE POOL SUMMARY]' in line:
            in_summary = True
            summary_lines = []
        elif in_summary:
            if line.strip() == '' or line.startswith('['):
                # End of summary section
                break
            summary_lines.append(line)
    
    # Parse summary lines
    for line in summary_lines:
        # Format: "  149 routes | ('VB', 'k_medoids_pyclustering', 'filo1', 'post_ls')"
        match = re.search(r'(\d+)\s+routes\s*\|\s*\(([^)]+)\)', line)
        if match:
            count = int(match.group(1))
            params = match.group(2).strip()
            # Parse tuple: 'VB', 'k_medoids_pyclustering', 'filo1', 'post_ls'
            parts = [p.strip().strip("'\"") for p in params.split(',')]
            if len(parts) >= 4:
                mode = parts[0]
                method = parts[1]
                solver = parts[2]
                stage = parts[3]
                stats.append(RoutePoolStats(mode, method, solver, stage, count))
    
    return stats


def extract_improvements(log_file: Path) -> List[Improvement]:
    """Extract improvement events from log file."""
    improvements = []
    start_time = None
    best_cost = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Get start time
        if start_time is None:
            ts = parse_timestamp(line)
            if ts:
                start_time = ts
        
        # Check for improvements
        ts = parse_timestamp(line)
        if ts is None:
            continue
        
        # DRI improvements (VB/RB iterations) - explicitly marked
        if 'IMPROVED-VB/RB' in line:
            match = re.search(r'best_cost=(\d+(?:\.\d+)?)', line)
            if match:
                cost = float(match.group(1))
                # Get iteration number from previous lines
                iteration = None
                for j in range(max(0, i-5), i):
                    iter_match = re.search(r'ITERATION (\d+)', lines[j])
                    if iter_match:
                        iteration = int(iter_match.group(1))
                        break
                
                time_seconds = ts - start_time if start_time else 0
                improvements.append(Improvement(
                    timestamp=datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                    time_seconds=time_seconds,
                    cost=cost,
                    phase="DRI",
                    iteration=iteration
                ))
                best_cost = cost
        
        # SCP improvements - explicitly marked
        elif 'IMPROVED-SCP' in line:
            match = re.search(r'best_cost=(\d+(?:\.\d+)?)', line)
            if match:
                cost = float(match.group(1))
                time_seconds = ts - start_time if start_time else 0
                improvements.append(Improvement(
                    timestamp=datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                    time_seconds=time_seconds,
                    cost=cost,
                    phase="SCP",
                    iteration=None
                ))
                best_cost = cost
        
        # SCP improvements - check if cost decreases after SCP (fallback for implicit improvements)
        elif 'SCP' in line and 'route_pool=' in line:
            # Look ahead for SCP result
            scp_cost_before = best_cost
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j]
                next_ts = parse_timestamp(next_line)
                
                # Check for SCP result
                if 'SCP-NO-IMPROVEMENT' in next_line:
                    match = re.search(r'cost=(\d+(?:\.\d+)?)\s*\(best=(\d+(?:\.\d+)?)\)', next_line)
                    if match:
                        scp_cost = float(match.group(1))
                        scp_best = float(match.group(2))
                        # If best improved, it's an SCP improvement
                        if scp_best < scp_cost_before if scp_cost_before else False:
                            time_seconds = ts - start_time if start_time else 0
                            improvements.append(Improvement(
                                timestamp=datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                                time_seconds=time_seconds,
                                cost=scp_best,
                                phase="SCP",
                                iteration=None
                            ))
                            best_cost = scp_best
                    break
                elif 'SCP' in next_line and 'best_cost=' in next_line:
                    # SCP improved
                    match = re.search(r'best_cost=(\d+(?:\.\d+)?)', next_line)
                    if match:
                        scp_best = float(match.group(1))
                        if scp_best < scp_cost_before if scp_cost_before else False:
                            time_seconds = ts - start_time if start_time else 0
                            improvements.append(Improvement(
                                timestamp=datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                                time_seconds=time_seconds,
                                cost=scp_best,
                                phase="SCP",
                                iteration=None
                            ))
                            best_cost = scp_best
                    break
        
        # Track best cost from NO-IMPROVEMENT messages
        elif 'NO-IMPROVEMENT' in line:
            match = re.search(r'best_cost=(\d+(?:\.\d+)?)', line)
            if match:
                best_cost = float(match.group(1))
    
    return improvements


def extract_cost_timeline(log_file: Path) -> List[Tuple[float, float]]:
    """Extract cost over time from log file."""
    timeline = []
    start_time = None
    best_cost = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        ts = parse_timestamp(line)
        if ts is None:
            continue
        
        if start_time is None:
            start_time = ts
        
        # Extract cost from various log messages
        cost = None
        
        # From IMPROVED-VB/RB
        if 'IMPROVED-VB/RB' in line:
            match = re.search(r'best_cost=(\d+(?:\.\d+)?)', line)
            if match:
                cost = float(match.group(1))
                best_cost = cost
        
        # From NO-IMPROVEMENT
        elif 'NO-IMPROVEMENT' in line:
            match = re.search(r'best_cost=(\d+(?:\.\d+)?)', line)
            if match:
                cost = float(match.group(1))
                best_cost = cost
        
        # From SCP messages
        elif 'SCP' in line:
            match = re.search(r'cost=(\d+(?:\.\d+)?)\s*\(best=(\d+(?:\.\d+)?)\)', line)
            if match:
                cost = float(match.group(2))  # Use best cost
                best_cost = cost
        
        # Use current best cost if available
        if cost is None and best_cost is not None:
            cost = best_cost
        
        if cost is not None:
            time_seconds = ts - start_time if start_time else 0
            timeline.append((time_seconds, cost))
    
    return timeline


def analyze_instance(log_file: Path) -> Dict:
    """Analyze a single instance log file."""
    instance_name = log_file.stem
    
    print(f"Analyzing {instance_name}...")
    
    # Extract route pool stats
    route_pool_stats = extract_route_pool_summary(log_file)
    
    # Extract improvements
    improvements = extract_improvements(log_file)
    
    # Extract cost timeline
    cost_timeline = extract_cost_timeline(log_file)
    
    return {
        'instance': instance_name,
        'route_pool_stats': route_pool_stats,
        'improvements': improvements,
        'cost_timeline': cost_timeline
    }


def create_convergence_graph(instance_name: str, cost_timeline: List[Tuple[float, float]], 
                            improvements: List[Improvement], output_path: Path):
    """Create convergence graph for an instance."""
    if not cost_timeline:
        print(f"  No timeline data for {instance_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cost timeline
    times, costs = zip(*cost_timeline) if cost_timeline else ([], [])
    ax.plot(times, costs, 'b-', alpha=0.3, linewidth=0.5, label='Cost timeline')
    
    # Mark improvements
    dri_times = [imp.time_seconds for imp in improvements if imp.phase == "DRI"]
    dri_costs = [imp.cost for imp in improvements if imp.phase == "DRI"]
    scp_times = [imp.time_seconds for imp in improvements if imp.phase == "SCP"]
    scp_costs = [imp.cost for imp in improvements if imp.phase == "SCP"]
    
    if dri_times:
        ax.scatter(dri_times, dri_costs, c='green', marker='o', s=50, 
                  label='DRI Improvement', zorder=5)
    
    if scp_times:
        ax.scatter(scp_times, scp_costs, c='red', marker='s', s=50, 
                  label='SCP Improvement', zorder=5)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title(f'Convergence: {instance_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis (lower cost is better)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Challenge Test Run Analysis")
    print("=" * 80)
    
    # Find all log files
    log_files = sorted(LOG_DIR.glob("*.log"))
    print(f"\nFound {len(log_files)} log files")
    
    if not log_files:
        print("No log files found!")
        return
    
    # Analyze each instance
    all_results = []
    for log_file in log_files:
        try:
            result = analyze_instance(log_file)
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing {log_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully analyzed {len(all_results)} instances")
    
    # ========================================================================
    # TASK 1: Route Pool Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: Route Pool Statistics")
    print("=" * 80)
    
    # Per-instance statistics
    per_instance_stats = {}
    for result in all_results:
        instance = result['instance']
        stats = result['route_pool_stats']
        per_instance_stats[instance] = stats
        
        print(f"\n{instance}:")
        print(f"  Total route pool entries: {len(stats)}")
        
        # Group by mode, method, solver, stage
        by_mode = Counter()
        by_method = Counter()
        by_solver = Counter()
        by_stage = Counter()
        
        for stat in stats:
            by_mode[stat.mode] += stat.count
            by_method[stat.method] += stat.count
            by_solver[stat.solver] += stat.count
            by_stage[stat.stage] += stat.count
        
        print(f"  By mode: {dict(by_mode)}")
        print(f"  By method: {dict(by_method)}")
        print(f"  By solver: {dict(by_solver)}")
        print(f"  By stage: {dict(by_stage)}")
    
    # Aggregate statistics across all instances
    print("\n" + "-" * 80)
    print("AGGREGATE STATISTICS (All Instances)")
    print("-" * 80)
    
    aggregate_by_mode = Counter()
    aggregate_by_method = Counter()
    aggregate_by_solver = Counter()
    aggregate_by_stage = Counter()
    aggregate_by_combination = Counter()
    
    for stats in per_instance_stats.values():
        for stat in stats:
            aggregate_by_mode[stat.mode] += stat.count
            aggregate_by_method[stat.method] += stat.count
            aggregate_by_solver[stat.solver] += stat.count
            aggregate_by_stage[stat.stage] += stat.count
            key = (stat.mode, stat.method, stat.solver, stat.stage)
            aggregate_by_combination[key] += stat.count
    
    print(f"\nBy Mode:")
    for mode, count in aggregate_by_mode.most_common():
        print(f"  {mode}: {count}")
    
    print(f"\nBy Method:")
    for method, count in aggregate_by_method.most_common():
        print(f"  {method}: {count}")
    
    print(f"\nBy Solver:")
    for solver, count in aggregate_by_solver.most_common():
        print(f"  {solver}: {count}")
    
    print(f"\nBy Stage:")
    for stage, count in aggregate_by_stage.most_common():
        print(f"  {stage}: {count}")
    
    print(f"\nBy Combination (mode, method, solver, stage):")
    for (mode, method, solver, stage), count in aggregate_by_combination.most_common(20):
        print(f"  ({mode}, {method}, {solver}, {stage}): {count}")
    
    # Save route pool statistics to JSON
    stats_output = {
        'per_instance': {
            inst: [
                {
                    'mode': s.mode,
                    'method': s.method,
                    'solver': s.solver,
                    'stage': s.stage,
                    'count': s.count
                }
                for s in stats
            ]
            for inst, stats in per_instance_stats.items()
        },
        'aggregate': {
            'by_mode': dict(aggregate_by_mode),
            'by_method': dict(aggregate_by_method),
            'by_solver': dict(aggregate_by_solver),
            'by_stage': dict(aggregate_by_stage),
            'by_combination': {
                f"{mode}|{method}|{solver}|{stage}": count
                for (mode, method, solver, stage), count in aggregate_by_combination.items()
            }
        }
    }
    
    stats_file = OUTPUT_DIR / "route_pool_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nRoute pool statistics saved to: {stats_file}")
    
    # ========================================================================
    # TASK 2: Convergence Graphs
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 2: Creating Convergence Graphs")
    print("=" * 80)
    
    graphs_dir = OUTPUT_DIR / "convergence_graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    for result in all_results:
        instance = result['instance']
        cost_timeline = result['cost_timeline']
        improvements = result['improvements']
        
        print(f"\nCreating graph for {instance}...")
        print(f"  Timeline points: {len(cost_timeline)}")
        print(f"  Improvements: {len(improvements)}")
        print(f"    DRI: {sum(1 for imp in improvements if imp.phase == 'DRI')}")
        print(f"    SCP: {sum(1 for imp in improvements if imp.phase == 'SCP')}")
        
        graph_path = graphs_dir / f"{instance}_convergence.png"
        create_convergence_graph(instance, cost_timeline, improvements, graph_path)
        print(f"  Saved to: {graph_path}")
    
    print(f"\nAll convergence graphs saved to: {graphs_dir}")
    
    # Summary of improvements
    print("\n" + "-" * 80)
    print("IMPROVEMENT SUMMARY")
    print("-" * 80)
    
    total_dri = sum(sum(1 for imp in r['improvements'] if imp.phase == 'DRI') 
                    for r in all_results)
    total_scp = sum(sum(1 for imp in r['improvements'] if imp.phase == 'SCP') 
                    for r in all_results)
    
    print(f"Total DRI improvements: {total_dri}")
    print(f"Total SCP improvements: {total_scp}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
