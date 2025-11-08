# master/utils/loader.py

import os
import vrplib
from typing import Any, Dict
from functools import lru_cache


@lru_cache(maxsize=32)
def load_instance(instance_name: str) -> Dict[str, Any]:
    """
    Loads a CVRP instance using vrplib and returns its data dictionary.
    Caches up to 32 recently used instances in memory for fast reuse.

    Automatically searches for the instance in:
        core/instances/test-instances/x
        core/instances/test-instances/xl
    """
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    instances_root = os.path.join(core_root, "instances", "test-instances")

    # Try both possible subfolders
    for sub in ("x", "xl"):
        p = os.path.join(instances_root, sub, instance_name)
        if os.path.exists(p):
            return vrplib.read_instance(p)

    raise FileNotFoundError(
        f"Instance '{instance_name}' not found in: "
        f"[{os.path.join(instances_root, 'x')}, {os.path.join(instances_root, 'xl')}]"
    )


if __name__ == "__main__":
    # Example standalone test
    instance = load_instance("X-n101-k25.vrp")
    print(f"Loaded instance '{instance['name']}' with {len(instance['node_coord']) - 1} customers.")
    print(f"Vehicle capacity Q = {instance['capacity']}")
