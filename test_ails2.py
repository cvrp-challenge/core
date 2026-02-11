#!/usr/bin/env python3
"""
Test script for AILS2 solver integration.

Tests:
1. Basic solver registration
2. Full instance solving
3. Cluster subproblem solving
4. Different solver options
5. Error handling
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from master.routing.solver import solve
from master.routing.solver_ails2 import (
    solve_cluster_with_ails2,
    _resolve_jar,
    _check_java,
)


def test_java_available():
    """Test 1: Check if Java is available."""
    print("=" * 60)
    print("Test 1: Checking Java availability")
    print("=" * 60)
    java_available = _check_java()
    if java_available:
        print("✓ Java is available")
    else:
        print("✗ Java is not available - install Java JDK")
        return False
    print()
    return True


def test_jar_resolution():
    """Test 2: Check if JAR file can be found."""
    print("=" * 60)
    print("Test 2: Resolving AILS2 JAR file")
    print("=" * 60)
    try:
        jar_path = _resolve_jar()
        print(f"✓ Found JAR: {jar_path}")
        print(f"  File exists: {jar_path.exists()}")
        print(f"  File size: {jar_path.stat().st_size / 1024:.1f} KB")
    except FileNotFoundError as e:
        print(f"✗ JAR file not found: {e}")
        return False
    print()
    return True


def test_solver_registration():
    """Test 3: Check if solver is registered."""
    print("=" * 60)
    print("Test 3: Checking solver registration")
    print("=" * 60)
    from master.routing.solver import _SOLVER_REGISTRY
    
    if "ails2" in _SOLVER_REGISTRY:
        print("✓ AILS2 is registered in solver registry")
    else:
        print("✗ AILS2 is NOT registered")
        print(f"  Available solvers: {list(_SOLVER_REGISTRY.keys())}")
        return False
    print()
    return True


def test_full_instance_solving():
    """Test 4: Solve a full instance."""
    print("=" * 60)
    print("Test 4: Solving full instance")
    print("=" * 60)
    
    # Try to find a small test instance
    test_instances = [
        "instances/test-instances/x/X-n101-k25.vrp",
        "instances/test-instances/x/X-n110-k13.vrp",
        "instances/challenge-instances/XL-n1048-k237.vrp",
    ]
    
    instance_path = None
    for inst in test_instances:
        path = Path(__file__).parent / inst
        if path.exists():
            instance_path = inst
            break
    
    if not instance_path:
        print("⚠ No test instance found, skipping full instance test")
        print()
        return True
    
    print(f"Using instance: {instance_path}")
    
    try:
        result = solve(
            instance=instance_path,
            solver="ails2",
            solver_options={
                "max_runtime": 10.0,  # Short timeout for testing
                "rounded": True,
                "best": 0.0,
            }
        )
        
        print(f"✓ Solved successfully!")
        print(f"  Cost: {result.cost}")
        print(f"  Runtime: {result.runtime:.2f}s")
        print(f"  Feasible: {result.feasible}")
        print(f"  Number of routes: {len(result.metadata.get('routes_vrplib', []))}")
        
        if result.metadata.get('routes_vrplib'):
            routes = result.metadata['routes_vrplib']
            print(f"  First route: {routes[0] if routes else 'None'}")
        
    except Exception as e:
        print(f"✗ Failed to solve instance: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_cluster_solving():
    """Test 5: Solve a cluster subproblem."""
    print("=" * 60)
    print("Test 5: Solving cluster subproblem")
    print("=" * 60)
    
    # Try to find a small test instance
    test_instances = [
        "instances/test-instances/x/X-n101-k25.vrp",
        "instances/test-instances/x/X-n110-k13.vrp",
    ]
    
    instance_name = None
    for inst in test_instances:
        path = Path(__file__).parent / inst
        if path.exists():
            instance_name = path.name
            break
    
    if not instance_name:
        print("⚠ No test instance found, skipping cluster test")
        print()
        return True
    
    print(f"Using instance: {instance_name}")
    print("Cluster customers: [2, 3, 4, 5, 6, 7, 8]")
    
    try:
        result = solve_cluster_with_ails2(
            instance_name=instance_name,
            cluster_customers=[2, 3, 4, 5, 6, 7, 8],
            max_runtime=10.0,
            rounded=True,
            keep_tmp=False,  # Clean up temp files
        )
        
        print(f"✓ Cluster solved successfully!")
        print(f"  Cost: {result.cost}")
        print(f"  Runtime: {result.runtime:.2f}s")
        print(f"  Feasible: {result.feasible}")
        print(f"  Number of routes: {len(result.routes_global)}")
        
        if result.routes_global:
            print(f"  Routes (global IDs):")
            for i, route in enumerate(result.routes_global[:3], 1):  # Show first 3
                print(f"    Route {i}: {route}")
            if len(result.routes_global) > 3:
                print(f"    ... and {len(result.routes_global) - 3} more routes")
        
    except Exception as e:
        print(f"✗ Failed to solve cluster: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_solver_options():
    """Test 6: Test different solver options."""
    print("=" * 60)
    print("Test 6: Testing solver options")
    print("=" * 60)
    
    test_instances = [
        "instances/test-instances/x/X-n101-k25.vrp",
        "instances/test-instances/x/X-n110-k13.vrp",
    ]
    
    instance_path = None
    for inst in test_instances:
        path = Path(__file__).parent / inst
        if path.exists():
            instance_path = inst
            break
    
    if not instance_path:
        print("⚠ No test instance found, skipping options test")
        print()
        return True
    
    print(f"Testing with instance: {instance_path}")
    
    # Test with iteration limit instead of time limit
    try:
        result = solve(
            instance=instance_path,
            solver="ails2",
            solver_options={
                "no_improvement": 100,  # Use iterations instead of time
                "rounded": True,
            }
        )
        print(f"✓ Solved with iteration limit")
        print(f"  Cost: {result.cost}")
        print(f"  Runtime: {result.runtime:.2f}s")
    except Exception as e:
        print(f"✗ Failed with iteration limit: {e}")
        return False
    
    print()
    return True


def test_cluster_via_solver_api():
    """Test 7: Test cluster solving via main solver API."""
    print("=" * 60)
    print("Test 7: Cluster solving via main solver API")
    print("=" * 60)
    
    test_instances = [
        "instances/test-instances/x/X-n101-k25.vrp",
        "instances/test-instances/x/X-n110-k13.vrp",
    ]
    
    instance_path = None
    for inst in test_instances:
        path = Path(__file__).parent / inst
        if path.exists():
            instance_path = inst
            break
    
    if not instance_path:
        print("⚠ No test instance found, skipping cluster API test")
        print()
        return True
    
    print(f"Using instance: {instance_path}")
    
    try:
        result = solve(
            instance=instance_path,
            solver="ails2",
            solver_options={
                "cluster_nodes": [2, 3, 4, 5, 6, 7, 8],  # Customer nodes
                "max_runtime": 10.0,
                "rounded": True,
            }
        )
        
        print(f"✓ Cluster solved via main API!")
        print(f"  Cost: {result.cost}")
        print(f"  Runtime: {result.runtime:.2f}s")
        print(f"  Feasible: {result.feasible}")
        
        routes = result.metadata.get('routes_vrplib', [])
        print(f"  Number of routes: {len(routes)}")
        
        if routes:
            print(f"  First route: {routes[0]}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AILS2 Solver Integration Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Java Availability", test_java_available),
        ("JAR Resolution", test_jar_resolution),
        ("Solver Registration", test_solver_registration),
        ("Full Instance Solving", test_full_instance_solving),
        ("Cluster Solving", test_cluster_solving),
        ("Solver Options", test_solver_options),
        ("Cluster via API", test_cluster_via_solver_api),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AILS2 integration is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
