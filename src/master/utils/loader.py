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
    
    Args:
        instance_name: Either a full path to the instance file, or just the filename.
                      If a full path is provided, only the basename will be used for searching.
    """
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../"))

    # Extract just the filename from instance_name (in case a full path is provided)
    instance_filename = os.path.basename(instance_name)

    # Define all search locations (order matters!)
    search_paths = [
        os.path.join(core_root, "instances", "test-instances", "x"),
        os.path.join(core_root, "instances", "test-instances", "xl"),
        os.path.join(core_root, "instances", "challenge-instances"),
    ]

    for path in search_paths:
        p = os.path.join(path, instance_filename)
        if os.path.exists(p):
            return vrplib.read_instance(p)

    raise FileNotFoundError(
        f"Instance '{instance_filename}' not found in any of:\n  "
        + "\n  ".join(search_paths)
    )

