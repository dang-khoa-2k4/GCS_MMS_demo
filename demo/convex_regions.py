"""
convex_regions.py - Convex region representation and operations.

Handles:
- Polytope representation (V-rep and H-rep)
- Intersection computation
- Membership checking with safety margin
- Perspective set approximation via Big-M
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from shapely.geometry import Polygon, Point
import warnings

from problem_data import DEFAULT_PROBLEM


@dataclass
class ConvexRegion:
    """
    Represents a convex polytope in 2D.
    
    Supports both vertex representation (V-rep) and halfspace representation (H-rep).
    H-rep: {x : A @ x <= b}
    
    Attributes:
        vertices: Array of shape (n_vertices, 2) in CCW order
        A: Halfspace matrix of shape (n_constraints, 2)
        b: Halfspace vector of shape (n_constraints,)
        index: Region index in the graph
        label: Optional string label
    """
    vertices: np.ndarray
    A: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    index: int = -1
    label: str = ""
    
    def __post_init__(self):
        """Convert V-rep to H-rep if not provided."""
        self.vertices = np.array(self.vertices, dtype=np.float64)
        
        # Ensure vertices are in CCW order
        self._ensure_ccw()
        
        # Compute H-representation if not provided
        if self.A is None or self.b is None:
            self._compute_halfspaces()
    
    def _ensure_ccw(self):
        """Ensure vertices are in counter-clockwise order."""
        # Compute signed area
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i, 0] * self.vertices[j, 1]
            area -= self.vertices[j, 0] * self.vertices[i, 1]
        
        # If clockwise (negative area), reverse
        if area < 0:
            self.vertices = self.vertices[::-1]
    
    def _compute_halfspaces(self):
        """
        Compute H-representation from vertices.
        
        For a polygon with CCW vertices, each edge defines a halfspace
        with the interior on the left side.
        """
        n = len(self.vertices)
        A_list = []
        b_list = []
        
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            
            # Edge direction
            edge = p2 - p1
            
            # Outward normal (perpendicular to edge, pointing right for CCW)
            # For interior on left: normal points outward
            normal = np.array([edge[1], -edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            
            # Halfspace: normal * (x - p1) <= 0
            # => normal * x <= normal * p1
            A_list.append(normal)
            b_list.append(np.dot(normal, p1))
        
        self.A = np.array(A_list)
        self.b = np.array(b_list)
    
    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if point is inside the region.
        
        Args:
            point: 2D point [x, y]
            margin: Safety margin (positive = shrink region)
            
        Returns:
            True if A @ point <= b - margin
        """
        point = np.asarray(point).flatten()[:2]
        return np.all(self.A @ point <= self.b - margin)
    
    def contains_batch(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Check containment for multiple points.
        
        Args:
            points: Array of shape (N, 2)
            margin: Safety margin
            
        Returns:
            Boolean array of shape (N,)
        """
        # A @ points.T has shape (n_constraints, N)
        violations = self.A @ points.T - (self.b - margin).reshape(-1, 1)
        return np.all(violations <= 0, axis=0)
    
    def get_centroid(self) -> np.ndarray:
        """Compute centroid of the polygon."""
        return np.mean(self.vertices, axis=0)
    
    def get_shapely_polygon(self) -> Polygon:
        """Convert to Shapely polygon."""
        return Polygon(self.vertices)
    
    def get_interior_point(self) -> np.ndarray:
        """
        Find a strictly interior point using LP.
        
        Solves: max s s.t. A @ x + s * 1 <= b
        """
        n_constraints = len(self.b)
        
        # Variables: [x, y, s]
        c = np.array([0, 0, -1])  # maximize s
        
        # Constraints: A @ [x,y] + s <= b
        A_ub = np.hstack([self.A, np.ones((n_constraints, 1))])
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=self.b, bounds=[(-100, 100), (-100, 100), (None, None)])
            if result.success and result.x[2] > 0:
                return result.x[:2]
        except Exception:
            pass
        
        # Fallback to centroid
        return self.get_centroid()


def compute_intersection(region1: ConvexRegion, region2: ConvexRegion) -> Optional[ConvexRegion]:
    """
    Compute intersection of two convex regions.
    
    For touching regions (shared boundary), returns a "virtual" intersection
    region representing the shared edge/point buffered slightly.
    
    Args:
        region1, region2: ConvexRegion objects
        
    Returns:
        New ConvexRegion representing intersection, or None if disjoint
    """
    poly1 = region1.get_shapely_polygon()
    poly2 = region2.get_shapely_polygon()
    
    intersection = poly1.intersection(poly2)
    
    if intersection.is_empty:
        return None
    
    # For proper area intersection
    if intersection.geom_type == 'Polygon' and intersection.area > 1e-10:
        vertices = np.array(intersection.exterior.coords[:-1])
        A = np.vstack([region1.A, region2.A])
        b = np.hstack([region1.b, region2.b])
        return ConvexRegion(vertices=vertices, A=A, b=b)
    
    # For touching regions (LineString or Point intersection),
    # create a small buffer around the shared boundary
    if intersection.geom_type in ['LineString', 'MultiLineString', 'Point', 'MultiPoint', 'GeometryCollection']:
        # Buffer the intersection slightly to create a valid region
        buffered = intersection.buffer(0.01, cap_style='square')
        
        # Clip to both original regions
        clipped = buffered.intersection(poly1).intersection(poly2.buffer(0.01))
        
        if clipped.is_empty or clipped.area < 1e-10:
            # Fall back: use centroid of intersection as point
            centroid = intersection.centroid
            # Create tiny square region around centroid
            eps = 0.01
            vertices = np.array([
                [centroid.x - eps, centroid.y - eps],
                [centroid.x + eps, centroid.y - eps],
                [centroid.x + eps, centroid.y + eps],
                [centroid.x - eps, centroid.y + eps]
            ])
            # Use combined H-rep
            A = np.vstack([region1.A, region2.A])
            b = np.hstack([region1.b, region2.b])
            return ConvexRegion(vertices=vertices, A=A, b=b)
        
        if clipped.geom_type == 'Polygon':
            vertices = np.array(clipped.exterior.coords[:-1])
            A = np.vstack([region1.A, region2.A])
            b = np.hstack([region1.b, region2.b])
            return ConvexRegion(vertices=vertices, A=A, b=b)
    
    return None


def regions_intersect(region1: ConvexRegion, region2: ConvexRegion, 
                      min_area: float = 1e-6) -> bool:
    """
    Check if two regions have non-trivial intersection OR touch.
    
    For motion planning, regions that share a boundary are considered
    adjacent since the robot can transition between them at the shared edge.
    
    Args:
        region1, region2: Regions to check
        min_area: Minimum intersection area to consider non-empty
        
    Returns:
        True if intersection has area >= min_area OR regions touch
    """
    poly1 = region1.get_shapely_polygon()
    poly2 = region2.get_shapely_polygon()
    
    # Check if they touch (share boundary) or properly intersect
    if poly1.touches(poly2) or poly1.intersects(poly2):
        intersection = poly1.intersection(poly2)
        # Accept if they touch (boundary intersection) or have area overlap
        if not intersection.is_empty:
            return True
    
    return False


def get_intersection_halfspaces(region1: ConvexRegion, region2: ConvexRegion) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get combined H-representation for intersection Q1 intersect Q2.
    
    Returns:
        (A, b) where intersection = {x : A @ x <= b}
    """
    A = np.vstack([region1.A, region2.A])
    b = np.hstack([region1.b, region2.b])
    return A, b


def create_regions_from_vertices_list(vertices_list: List[np.ndarray]) -> List[ConvexRegion]:
    """
    Create ConvexRegion objects from list of vertex arrays.
    
    Args:
        vertices_list: List of arrays, each of shape (n_v, 2)
        
    Returns:
        List of ConvexRegion objects
    """
    regions = []
    for i, vertices in enumerate(vertices_list):
        region = ConvexRegion(vertices=vertices, index=i, label=f"R{i}")
        regions.append(region)
    return regions


def create_buffered_regions_from_vertices_list(
    vertices_list: List[np.ndarray],
    workspace_vertices: np.ndarray,
    buffer_size: float
) -> List[ConvexRegion]:
    """
    Expand region polygons by a small buffer and clip them to the workspace.
    """
    workspace = Polygon(np.asarray(workspace_vertices, dtype=np.float64))
    expanded_list = []
    
    for verts in vertices_list:
        poly = Polygon(np.asarray(verts, dtype=np.float64))
        buffered = poly.buffer(buffer_size, join_style='mitre')
        clipped = buffered.intersection(workspace)
        if clipped.geom_type == 'Polygon':
            expanded_list.append(np.array(clipped.exterior.coords[:-1]))
        else:
            expanded_list.append(np.asarray(verts, dtype=np.float64))
    
    return create_regions_from_vertices_list(expanded_list)


def get_default_regions() -> List[ConvexRegion]:
    """
    Get the default safe convex regions from problem specification.
    
    Note: Original regions only touch at boundaries. For unicycle dynamics
    to be feasible, we expand each region slightly to create overlap.
    This is a common practice in motion planning to ensure dynamic feasibility.
    """
    original_vertices_list = [
        np.asarray(vertices, dtype=np.float64)
        for vertices in DEFAULT_PROBLEM.region_vertices
    ]
    return create_buffered_regions_from_vertices_list(
        original_vertices_list,
        np.asarray(DEFAULT_PROBLEM.workspace_vertices, dtype=np.float64),
        DEFAULT_PROBLEM.region_buffer
    )


def get_original_regions() -> List[ConvexRegion]:
    """
    Get the original (touching only) safe convex regions.
    These may not be dynamically feasible for unicycle model.
    """
    vertices_list = [
        np.asarray(vertices, dtype=np.float64)
        for vertices in DEFAULT_PROBLEM.region_vertices
    ]
    return create_regions_from_vertices_list(vertices_list)


class PerspectiveApproximation:
    """
    Approximation of perspective/homogenization for on/off geometry.
    
    For a convex set C, the perspective set is:
        C_tilde = {(x, lambda) : lambda > 0, x/lambda in C}
    
    When lambda in {0, 1} (binary):
        - lambda = 1: x in C
        - lambda = 0: x = 0 (off-state)
    
    We approximate using Big-M constraints:
        A @ x <= b * lambda + M * (1 - lambda)  [membership when active]
        -M * lambda <= x <= M * lambda          [zero when inactive]
    
    This is NOT equivalent to true perspective (weaker relaxation)
    but maintains correct binary semantics.
    """
    
    def __init__(self, region: ConvexRegion, big_M: float = 20.0):
        """
        Args:
            region: The convex region C
            big_M: Big-M constant for constraint encoding
        """
        self.region = region
        self.big_M = big_M
        self.A = region.A
        self.b = region.b
    
    def get_constraints_active(self, x: np.ndarray) -> np.ndarray:
        """
        Get constraint violations when lambda = 1 (active).
        
        Returns:
            violations = A @ x - b (should all be <= 0)
        """
        return self.A @ x - self.b
    
    def get_big_M_constraints(self) -> Dict:
        """
        Return Big-M constraint matrices for MINLP formulation.
        
        For binary lambda and continuous x:
            A @ x <= b + M * (1 - lambda)   [1]
            x >= -M * lambda                [2]
            x <= M * lambda                 [3]
        
        Returns:
            Dictionary with constraint components
        """
        return {
            'A': self.A,
            'b': self.b,
            'M': self.big_M,
            'n_constraints': len(self.b),
            'dim': self.A.shape[1]
        }
