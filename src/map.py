from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Keyframe:
    index: int
    pose_T_wc: np.ndarray
    image: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None


@dataclass
class MapPoint:
    position: np.ndarray
    observations: int = 1
    descriptor: Optional[np.ndarray] = None


@dataclass
class LocalMap:
    keyframes: List[Keyframe] = field(default_factory=list)
    map_points: List[MapPoint] = field(default_factory=list)

    def add_keyframe(self, keyframe: Keyframe) -> None:
        self.keyframes.append(keyframe)

    def add_map_point(self, point: MapPoint) -> None:
        self.map_points.append(point)

    def add_points(self, points: List[np.ndarray]) -> None:
        for point in points:
            self.map_points.append(MapPoint(position=point))

    def add_points_with_desc(self, points: List[np.ndarray], descriptors: List[np.ndarray]) -> None:
        for point, desc in zip(points, descriptors):
            self.map_points.append(MapPoint(position=point, descriptor=desc))

    def sample_points(self, max_points: int) -> List[np.ndarray]:
        if max_points <= 0 or not self.map_points:
            return []
        if len(self.map_points) <= max_points:
            return [p.position for p in self.map_points]
        step = max(1, len(self.map_points) // max_points)
        return [self.map_points[i].position for i in range(0, len(self.map_points), step)][:max_points]

    def stats(self) -> dict:
        return {"keyframes": len(self.keyframes), "map_points": len(self.map_points)}
