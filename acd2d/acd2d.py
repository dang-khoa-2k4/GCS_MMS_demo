"""
Python wrapper for ACD2D using subprocess
Provides interface to approximate convex decomposition functionality
"""

import subprocess
import tempfile
import os
import re
from unittest import result
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union


class ACD2D:
    """
    Wrapper class for ACD2D approximate convex decomposition
    Uses subprocess to call the compiled ACD2D CLI executable
    """
    
    def __init__(self, acd2d_executable: Optional[str] = None):
        """
        Initialize ACD2D wrapper
        
        Args:
            acd2d_executable: Path to ACD2D executable. If None, searches in default locations
        """
        if acd2d_executable is None:
            # Search for executable in common locations
            search_paths = [
                "./acd2d",
                "./build/acd2d",
                "../build/acd2d",
                str(Path(__file__).parent / "build" / "acd2d"),
                str(Path(__file__).parent / "acd2d"),
            ]
            for path in search_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    self.executable = os.path.abspath(path)
                    break
            else:
                raise FileNotFoundError(
                    "ACD2D executable not found. Please build it first:\n"
                    "  cd build && cmake .. && make\n"
                    "Or provide explicit path to executable."
                )
        else:
            if not os.path.isfile(acd2d_executable):
                raise FileNotFoundError(f"ACD2D executable not found: {acd2d_executable}")
            self.executable = os.path.abspath(acd2d_executable)
            
        self.tau = 0.0
        self.alpha = 0.0
        self.beta = 1.0
        self.measure = "hybrid1"
        self.verbose = False
        
        print(f"ACD2D wrapper initialized with executable: {self.executable}")
        
    def set_parameters(self, tau: float, alpha: float = 0.0, beta: float = 1.0, 
                      measure: str = "hybrid1", verbose: bool = False):
        """
        Set decomposition parameters
        
        Args:
            tau: Concavity tolerance (0.0 - 1.0). Higher = more aggressive decomposition
            alpha: Weight for concavity in cut direction (default: 0.0)
            beta: Weight for distance in cut direction (default: 1.0)
            measure: Concavity measure type (hybrid1, hybrid2, sp, sl)
            verbose: Enable verbose output
        """
        if not 0.0 <= tau <= 1.0:
            raise ValueError("Tau must be in range [0.0, 1.0]")
            
        valid_measures = ["hybrid1", "hybrid2", "sp", "sl"]
        if measure not in valid_measures:
            raise ValueError(f"Measure must be one of {valid_measures}")
            
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.measure = measure
        self.verbose = verbose
        
    def decompose_file(self, input_file: str, output_file: Optional[str] = None) -> dict:
        """
        Decompose polygon from .poly file
        
        Args:
            input_file: Path to input .poly file
            output_file: Path to output file (optional, temp file used if None)
            
        Returns:
            Dictionary with keys:
                - 'polygons': List of numpy arrays, each containing vertices
                - 'cuts': List of tuples (v1, v2) representing cut lines
                - 'num_polygons': Number of decomposed polygons
                - 'num_cuts': Number of cuts made
                - 'output_file': Path to output file
        """
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Create temporary output file if needed
        if output_file is None:
            temp_fd, output_file = tempfile.mkstemp(suffix='.poly', text=True)
            os.close(temp_fd)
            temp_output = True
        else:
            temp_output = False
            
        # Build command
        cmd = [
            self.executable,
            "-i", input_file,
            "-o", output_file,
            "-t", str(self.tau),
            "-a", str(self.alpha),
            "-b", str(self.beta),
            "-m", self.measure
        ]
        
        if self.verbose:
            cmd.append("-v")
        
        try:
            # Run ACD2D
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.verbose:
                print(result.stdout)
            
            # Parse output
            num_polygons, num_cuts = self._parse_stats(result.stdout)
            polygons = self._parse_output_file(output_file)
            cuts = self._parse_cuts(result.stdout)
            
            return {
                'polygons': polygons,
                'cuts': cuts,
                'num_polygons': num_polygons,
                'num_cuts': num_cuts,
                'output_file': output_file if not temp_output else None
            }
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ACD2D failed:\n{e.stderr}")
            
        finally:
            if temp_output and os.path.exists(output_file):
                os.remove(output_file)
                
    def decompose_polygon(self, vertices: Union[np.ndarray, List], 
                     holes: Optional[List[np.ndarray]] = None,
                     keep_temp_files: bool = False) -> dict:
        """
        Decompose polygon from numpy array or list
        
        Args:
            vertices: Nx2 numpy array or list of [x,y] coordinates (outer boundary)
            holes: Optional list of Nx2 arrays representing holes
            keep_temp_files: If True, keep temporary .poly files for debugging
            
        Returns:
            Dictionary with decomposition results (same as decompose_file)
        """
        vertices = np.asarray(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("Vertices must be Nx2 array")
        
        # Create temporary input file in /tmp or current directory
        if keep_temp_files:
            # Save to current directory with timestamp
            import time
            timestamp = int(time.time())
            input_file = f"acd2d_input_{timestamp}.poly"
            with open(input_file, 'w') as f:
                self._write_poly_file(f, vertices, holes)
            print(f"Input file saved to: {input_file}")
        else:
            # Use system temp directory
            temp_fd, input_file = tempfile.mkstemp(suffix='.poly', text=True)
            with os.fdopen(temp_fd, 'w') as f:
                self._write_poly_file(f, vertices, holes)
        
        try:
            result = self.decompose_file(input_file)
            
            if keep_temp_files and 'output_file' in result and result['output_file']:
                print(f"Output file: {result['output_file']}")
            
            return result
            
        finally:
            if not keep_temp_files and os.path.exists(input_file):
                os.remove(input_file)

    # Helper function to ensure counter-clockwise order
    def ensure_ccw(self, vertices):
        """Ensure vertices are in counter-clockwise order"""
        vertices = np.asarray(vertices)
        # Calculate signed area
        area = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        
        # If area is negative, vertices are clockwise - reverse them
        if area < 0:
            return vertices[::-1]
        return vertices

    # Helper function to ensure CLOCKWISE order for holes
    def ensure_cw(self, vertices):
        """Ensure vertices are in clockwise order (for holes in ACD2D)"""
        vertices = np.asarray(vertices)
        # Calculate signed area
        area = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        
        # If area is positive, vertices are counter-clockwise - reverse them
        if area > 0:
            return vertices[::-1]
        return vertices

    # Helper to check if polygon is valid
    def is_valid_polygon(self, vertices):
        """Check if polygon is valid (no self-intersection)"""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(vertices)
            return True
        except:
            return False

    @staticmethod
    def _as_vertex_array(vertices: Union[np.ndarray, List]) -> np.ndarray:
        """
        Normalize polygon-like input to an Nx2 vertex array.

        Accepts either a raw vertex array/list or a Shapely Polygon.
        """
        if hasattr(vertices, "exterior") and hasattr(vertices.exterior, "coords"):
            return np.asarray(vertices.exterior.coords[:-1], dtype=np.float64)
        return np.asarray(vertices, dtype=np.float64)


    def merge_boundary_obstacles_to_workspace(self, workspace, obstacles, tolerance=1e-6):
        """
        Merge obstacles that touch or overlap workspace boundary into the workspace as notches.
        
        Args:
            workspace: np.array of workspace vertices (CCW order)
            obstacles: list of np.array obstacle vertices (CW order)
            tolerance: distance tolerance for considering a point on the boundary
        
        Returns:
            new_workspace: modified workspace with notches
            remaining_obstacles: obstacles that don't touch the boundary
        """
        from shapely.geometry import Polygon, LineString, Point
        from shapely.ops import unary_union
        
        workspace_poly = Polygon(workspace)
        workspace_boundary = LineString(list(workspace) + [workspace[0]])
        
        remaining_obstacles = []
        boundary_obstacles = []
        
        for obs in obstacles:
            obs_poly = Polygon(obs)
            
            # Check if obstacle intersects or touches workspace boundary
            if obs_poly.touches(workspace_boundary) or obs_poly.intersects(workspace_boundary):
                # Check if any vertex of obstacle is on workspace boundary
                touches_boundary = False
                for vertex in obs:
                    point = Point(vertex)
                    if workspace_boundary.distance(point) < tolerance:
                        touches_boundary = True
                        break
                
                if touches_boundary:
                    boundary_obstacles.append(obs_poly)
                else:
                    remaining_obstacles.append(obs)
            else:
                remaining_obstacles.append(obs)
        
        # If there are boundary obstacles, merge them with workspace
        if boundary_obstacles:
            # Subtract boundary obstacles from workspace to create notches
            new_workspace_poly = workspace_poly
            for obs_poly in boundary_obstacles:
                new_workspace_poly = new_workspace_poly.difference(obs_poly)
            
            # Extract new workspace boundary vertices
            if new_workspace_poly.geom_type == 'Polygon':
                new_workspace = np.array(new_workspace_poly.exterior.coords[:-1])
            else:
                # Handle MultiPolygon case (shouldn't happen but just in case)
                print("Warning: workspace became MultiPolygon, using largest component")
                largest = max(new_workspace_poly.geoms, key=lambda p: p.area)
                new_workspace = np.array(largest.exterior.coords[:-1])
        else:
            new_workspace = workspace
        
        return new_workspace, remaining_obstacles

    def decompose_to_polygons(self,
                              vertices: Union[np.ndarray, List],
                              holes: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], dict]:
        """
        Decompose polygon and return the convex polygon vertex lists.

        Args:
            vertices: Outer boundary vertices
            holes: List of hole vertices

        Returns:
            Tuple of:
                - List of decomposed convex polygons as vertex arrays
                - Raw decomposition result dictionary
        """
        if holes is None:
            holes = []

        vertices = self._as_vertex_array(vertices)
        holes = [self._as_vertex_array(hole) for hole in holes]

        workspace, obstacles = self.merge_boundary_obstacles_to_workspace(vertices, holes)
        workspace = self.ensure_ccw(workspace) if self.is_valid_polygon(workspace) else None
        obstacles = [self.ensure_cw(obs) for obs in obstacles] if all(self.is_valid_polygon(obs) for obs in obstacles) else []

        result = self.decompose_polygon(workspace, obstacles, True)
        polygons = [
            np.asarray(poly, dtype=np.float64)
            for poly in result.get('polygons', [])
            if len(poly) >= 3
        ]

        print(f"Decomposed into {result['num_polygons']} polygons with {result['num_cuts']} cuts.")
        for i, poly in enumerate(polygons):
            print(f"\nPolygon {i+1}:")
            print(poly)

        return polygons, result

    def decompose_to_hpolyhedron(self, vertices: Union[np.ndarray, List], 
                             holes: Optional[List[np.ndarray]] = None):
        """
        Decompose polygon and convert to HPolyhedron list
        
        Args:
            vertices: Outer boundary vertices
            holes: List of hole vertices
            
        Returns:
            List of HPolyhedron objects for GCS
        """
        from pydrake.geometry.optimization import HPolyhedron
        from scipy.spatial import ConvexHull

        polygons, result = self.decompose_to_polygons(vertices, holes)
        regions = []
        for i, poly in enumerate(polygons):
            try:
                hull = ConvexHull(poly)
                A = hull.equations[:, :-1]
                b = - hull.equations[:, -1]
                regions.append(HPolyhedron(A, b))
            except Exception as e:
                print(f"Warning: Could not create region {i}: {e}")
        
        return regions, result
        
    def _write_poly_file(self, f, vertices: np.ndarray, holes: Optional[List[np.ndarray]] = None):
        """Write polygon to .poly file format"""
        num_chains = 1 + (len(holes) if holes else 0)
        f.write(f"{num_chains}\n")
        
        # Write outer boundary
        n = len(vertices)
        f.write(f"{n} out\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]}\n")
        f.write(" ".join(str(i+1) for i in range(n)) + "\n")
        
        # Write holes
        if holes:
            for hole in holes:
                n = len(hole)
                f.write(f"{n} in\n")
                for v in hole:
                    f.write(f"{v[0]} {v[1]}\n")
                f.write(" ".join(str(i+1) for i in range(n)) + "\n")
        
    def _parse_output_file(self, filename: str) -> List[np.ndarray]:
        """Parse decomposed polygons from output file"""
        polygons = []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
                
            # First line: number of chains
            try:
                num_chains = int(line)
            except ValueError:
                i += 1
                continue
            i += 1
            
            for _ in range(num_chains):
                if i >= len(lines):
                    break
                    
                # Chain header: "n_vertices type"
                parts = lines[i].strip().split()
                if len(parts) < 1:
                    i += 1
                    continue
                    
                n_vertices = int(parts[0])
                i += 1
                
                # Read vertices
                vertices = []
                for _ in range(n_vertices):
                    if i >= len(lines):
                        break
                    coords = list(map(float, lines[i].strip().split()))
                    if len(coords) >= 2:
                        vertices.append(coords[:2])
                    i += 1
                    
                # Skip vertex order line
                if i < len(lines):
                    i += 1
                
                if vertices:
                    polygons.append(np.array(vertices))
                    
        return polygons


    def _parse_stats(self, stdout: str) -> Tuple[int, int]:
        """Parse statistics from stdout"""
        num_polygons = 0
        num_cuts = 0
        
        for line in stdout.split('\n'):
            if 'Number of output polygons:' in line:
                num_polygons = int(line.split(':')[1].strip())
            elif 'Number of cuts:' in line:
                num_cuts = int(line.split(':')[1].strip())
                
        return num_polygons, num_cuts
        
    def _parse_cuts(self, stdout: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Parse cut lines from verbose output"""
        cuts = []
        
        for line in stdout.split('\n'):
            # Format: "  Cut 1: (x1, y1) -> (x2, y2)"
            if 'Cut' in line and '->' in line:
                try:
                    # Extract coordinates using regex
                    pattern = r'\(([-\d.]+),\s*([-\d.]+)\)\s*->\s*\(([-\d.]+),\s*([-\d.]+)\)'
                    match = re.search(pattern, line)
                    if match:
                        x1, y1, x2, y2 = map(float, match.groups())
                        v1 = np.array([x1, y1])
                        v2 = np.array([x2, y2])
                        cuts.append((v1, v2))
                except:
                    pass
                    
        return cuts


def example_usage():
    """Example usage of ACD2D wrapper"""
    
    # Initialize wrapper
    acd = ACD2D()
    
    # Set parameters
    acd.set_parameters(tau=0.05, measure="hybrid1", verbose=True)
    
    # Example 1: Decompose from file
    print("\n=== Example 1: Decompose from file ===")
    try:
        result = acd.decompose_file("test_env/hole3.poly")
        print(f"Decomposed into {result['num_polygons']} polygons")
        print(f"Made {result['num_cuts']} cuts")
        
        for i, poly in enumerate(result['polygons']):
            print(f"Polygon {i+1}: {poly.shape[0]} vertices")
    except FileNotFoundError:
        print("Test file not found, skipping...")
    
    # Example 2: Decompose from numpy array
    print("\n=== Example 2: Decompose from array ===")
    
    # Create an L-shaped polygon
    vertices = np.array([
        [0, 0],
        [2, 0],
        [2, 1],
        [1, 1],
        [1, 2],
        [0, 2]
    ])
    
    result = acd.decompose_polygon(vertices)
    print(f"Decomposed into {result['num_polygons']} polygons")
    print(f"Made {result['num_cuts']} cuts")
    
    # Print decomposed polygons
    for i, poly in enumerate(result['polygons']):
        print(f"\nPolygon {i+1}:")
        print(poly)
    
    # Print cuts
    if result['cuts']:
        print("\nCut lines:")
        for i, (v1, v2) in enumerate(result['cuts'], 1):
            print(f"  Cut {i}: {v1} -> {v2}")


if __name__ == "__main__":
    example_usage()
