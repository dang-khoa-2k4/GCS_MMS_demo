"""
environment.py - Environment definition for motion planning demo.

Defines the workspace bounds and obstacle polygons.
The free space is implicitly workspace minus union(obstacles).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from problem_data import DEFAULT_PROBLEM


@dataclass
class Environment:
    """
    Represents the 2D planning environment.
    
    Attributes:
        workspace: Polygon defining the outer boundary
        obstacles: List of obstacle polygons
        free_space: Shapely geometry of collision-free space (computed)
    """
    workspace: Polygon
    obstacles: List[Polygon]
    free_space: Optional[Polygon] = None
    
    def __post_init__(self):
        """Compute free space after initialization."""
        if self.obstacles:
            obstacle_union = unary_union(self.obstacles)
            self.free_space = self.workspace.difference(obstacle_union)
        else:
            self.free_space = self.workspace
    
    def is_collision_free(self, point: np.ndarray) -> bool:
        """
        Check if a point is in collision-free space.
        
        Args:
            point: 2D position [x, y]
            
        Returns:
            True if point is in free space
        """
        p = Point(point[0], point[1])
        return self.free_space.contains(p) or self.free_space.boundary.contains(p)
    
    def is_path_collision_free(self, path: np.ndarray, n_samples: int = 100) -> bool:
        """
        Check if a path is collision-free by sampling.
        
        Args:
            path: Array of shape (N, 2) representing waypoints
            n_samples: Number of interpolation samples
            
        Returns:
            True if all sampled points are collision-free
        """
        for i in range(len(path) - 1):
            for t in np.linspace(0, 1, n_samples):
                point = (1 - t) * path[i] + t * path[i + 1]
                if not self.is_collision_free(point):
                    return False
        return True
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get workspace bounds.
        
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        bounds = self.workspace.bounds
        return bounds[0], bounds[2], bounds[1], bounds[3]


def create_default_environment() -> Environment:
    """
    Create the default environment from the problem specification.
    
    The obstacles are simple convex polygons defined by the given vertices.
    """
    return create_environment_from_vertices(
        DEFAULT_PROBLEM.workspace_vertices,
        DEFAULT_PROBLEM.obstacle_vertices
    )


def create_simple_environment() -> Environment:
    """
    Create a simpler environment for testing.
    Single rectangular obstacle in the middle.
    """
    workspace = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    obstacle = Polygon([(1.5, 1.5), (3.5, 1.5), (3.5, 3.5), (1.5, 3.5)])
    
    return Environment(workspace=workspace, obstacles=[obstacle])


def create_environment_from_vertices(workspace_vertices: List[List[float]],
                                     obstacles_vertices: List[List[List[float]]]) -> Environment:
    """Build an environment from raw vertex lists."""
    workspace = Polygon(np.asarray(workspace_vertices, dtype=np.float64))
    obstacles = [
        Polygon(np.asarray(vertices, dtype=np.float64))
        for vertices in obstacles_vertices
    ]
    return Environment(workspace=workspace, obstacles=obstacles)
