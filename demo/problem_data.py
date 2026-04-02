"""
problem_data.py - Geometry presets for demo problems.

This module is intentionally data-only so new environments can be added
without touching the application pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ProblemPresetSpec:
    """Static geometry and default boundary conditions for a demo problem."""
    key: str
    name: str
    description: str
    workspace_vertices: List[List[float]]
    obstacle_vertices: List[List[List[float]]]
    region_vertices: List[List[List[float]]]
    default_start_state: List[float]
    default_goal_state: List[float]
    region_buffer: float = 0.05


DEFAULT_PROBLEM = ProblemPresetSpec(
    key="default",
    name="Default Workspace",
    description="Baseline 5x5 workspace with hand-crafted convex safe regions.",
    workspace_vertices=[
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 5.0],
        [0.0, 5.0],
    ],
    obstacle_vertices=[
        [
            [3.4, 2.6],
            [3.4, 4.6],
            [2.4, 4.6],
            [2.4, 2.6],
            [1.4, 2.2],
            [3.8, 0.2],
            [4.8, 1.2],
        ],
        [
            [1.4, 2.8],
            [2.2, 2.8],
            [2.2, 4.6],
            [1.4, 4.6],
        ],
        [
            [1.0, 2.6],
            [1.0, 5.0],
            [0.4, 5.0],
            [0.4, 2.6],
        ],
        [
            [1.0, 2.4],
            [1.0, 0.0],
            [0.4, 0.0],
            [0.4, 2.4],
        ],
        [
            [3.8, 3.0],
            [3.8, 5.0],
            [4.4, 5.0],
            [4.4, 3.0],
        ],
        [
            [3.8, 2.8],
            [3.8, 2.6],
            [5.0, 2.6],
            [5.0, 2.8],
        ],
    ],
    region_vertices=[
        [[0.4, 0.0], [0.4, 5.0], [0.0, 5.0], [0.0, 0.0]],
        [[0.4, 2.4], [1.0, 2.4], [1.0, 2.6], [0.4, 2.6]],
        [[1.4, 2.2], [1.4, 4.6], [1.0, 4.6], [1.0, 2.2]],
        [[1.4, 2.2], [2.4, 2.6], [2.4, 2.8], [1.4, 2.8]],
        [[2.2, 2.8], [2.4, 2.8], [2.4, 4.6], [2.2, 4.6]],
        [[1.4, 2.2], [1.0, 2.2], [1.0, 0.0], [3.8, 0.0], [3.8, 0.2]],
        [[3.8, 4.6], [3.8, 5.0], [1.0, 5.0], [1.0, 4.6]],
        [[5.0, 0.0], [5.0, 1.2], [4.8, 1.2], [3.8, 0.2], [3.8, 0.0]],
        [[3.4, 2.6], [4.8, 1.2], [5.0, 1.2], [5.0, 2.6]],
        [[3.4, 2.6], [3.8, 2.6], [3.8, 4.6], [3.4, 4.6]],
        [[3.8, 2.8], [4.4, 2.8], [4.4, 3.0], [3.8, 3.0]],
        [[5.0, 2.8], [5.0, 5.0], [4.4, 5.0], [4.4, 2.8]],
    ],
    default_start_state=[0.2, 0.2, 0.785],
    default_goal_state=[4.8, 4.8, 0.785],
)


PROBLEM_PRESETS: Dict[str, ProblemPresetSpec] = {
    DEFAULT_PROBLEM.key: DEFAULT_PROBLEM,
}
