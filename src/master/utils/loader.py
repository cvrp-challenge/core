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
        core/instances/challenge-instances
    """
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../"))

    # Define all search locations (order matters!)
    search_paths = [
        os.path.join(core_root, "instances", "test-instances", "x"),
        os.path.join(core_root, "instances", "test-instances", "xl"),
        os.path.join(core_root, "instances", "challenge-instances"),
    ]

    for path in search_paths:
        p = os.path.join(path, instance_name)
        if os.path.exists(p):
            return vrplib.read_instance(p)

    raise FileNotFoundError(
        f"Instance '{instance_name}' not found in any of:\n  "
        + "\n  ".join(search_paths)
    )

