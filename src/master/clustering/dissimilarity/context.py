from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple
from utils.loader import load_instance
from clustering.dissimilarity.polar_coordinates import compute_polar_angle
from clustering.dissimilarity.spatial import compute_lambda

@dataclass
class DRIContext:
    instance_name: str

    @cached_property
    def instance(self) -> dict:
        return load_instance(self.instance_name)

    @cached_property
    def coords(self) -> Dict[int, Tuple[float, float]]:
        coords_full = self.instance["node_coord"]
        return {i: coords_full[i] for i in range(1, len(coords_full))}

    @cached_property
    def demands(self) -> Dict[int, int]:
        d = self.instance["demand"]
        return {i: d[i] for i in range(1, len(d))}

    @cached_property
    def Q(self) -> int:
        return self.instance["capacity"]

    @cached_property
    def angles(self) -> Dict[int, float]:
        # reuse existing implementation that accepts a provided instance
        return compute_polar_angle(self.instance_name, self.instance)

    @cached_property
    def lam(self) -> float:
        return compute_lambda(self.coords)
