"""
visualization.py - Visualization utilities for GCS-MMS demo.

Provides:
- Environment and region plotting
- Trajectory visualization
- Animation generation
- Result summary figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from typing import List, Dict, Tuple, Optional
import os
import textwrap

from convex_regions import ConvexRegion
from graph_builder import RegionGraph, SOURCE, TARGET
from optimizer import OptimizationResult


def setup_plot_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight'
    })


def _polygon_vertices(polygon_like) -> np.ndarray:
    """Convert a shapely polygon or raw vertex array into plot-ready vertices."""
    if hasattr(polygon_like, "exterior"):
        vertices = np.asarray(polygon_like.exterior.coords, dtype=np.float64)
        if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
            return vertices[:-1]
        return vertices
    return np.asarray(polygon_like, dtype=np.float64)


def plot_environment(ax, workspace_bounds: Tuple[float, float, float, float],
                     regions: List[ConvexRegion],
                     start_pos: np.ndarray, goal_pos: np.ndarray,
                     obstacles: Optional[List[object]] = None,
                     highlight_regions: Optional[List[int]] = None,
                     show_labels: bool = True,
                     alpha: float = 0.3):
    """
    Plot the planning environment with convex regions.
    
    Args:
        ax: Matplotlib axes
        workspace_bounds: (x_min, x_max, y_min, y_max)
        regions: List of ConvexRegion objects
        start_pos: Start position [x, y]
        goal_pos: Goal position [x, y]
        obstacles: Obstacle polygons to overlay
        highlight_regions: Region indices to highlight (path)
        show_labels: Whether to show region labels
        alpha: Transparency for regions
    """
    x_min, x_max, y_min, y_max = workspace_bounds
    
    # Plot workspace boundary
    workspace_rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='black', facecolor='#f5f5f5'
    )
    ax.add_patch(workspace_rect)
    
    # Colormap for regions
    cmap = plt.cm.get_cmap('tab10')
    
    # Plot regions
    for region in regions:
        color = cmap(region.index % 10)
        
        # Determine if highlighted
        if highlight_regions is not None and region.index in highlight_regions:
            face_alpha = 0.5
            edge_width = 2.5
            edge_color = 'darkblue'
        else:
            face_alpha = alpha
            edge_width = 1.0
            edge_color = color
        
        poly = patches.Polygon(
            region.vertices,
            closed=True,
            facecolor=(*color[:3], face_alpha),
            edgecolor=edge_color,
            linewidth=edge_width
        )
        ax.add_patch(poly)

        # Label
        if show_labels:
            centroid = region.get_centroid()
            ax.text(centroid[0], centroid[1], f'R{region.index}',
                   ha='center', va='center', fontsize=8,
                   fontweight='bold' if highlight_regions and region.index in highlight_regions else 'normal')

    # Overlay obstacles explicitly so blocked space is easy to see.
    for obstacle_idx, obstacle in enumerate(obstacles or []):
        obstacle_patch = patches.Polygon(
            _polygon_vertices(obstacle),
            closed=True,
            facecolor='#404040',
            edgecolor='#f5f5f5',
            linewidth=1.2,
            hatch='///',
            alpha=0.95,
            zorder=4,
            label='Obstacle' if obstacle_idx == 0 else None,
        )
        ax.add_patch(obstacle_patch)

    # Plot start and goal
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, 
            label='Start', zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(goal_pos[0], goal_pos[1], 'r^', markersize=15,
            label='Goal', zorder=10, markeredgecolor='darkred', markeredgewidth=2)
    
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')


def _collect_mesh_points(result: OptimizationResult,
                         fallback_points_per_segment: int = 5
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect interior mesh points and their physical times.

    Uses exact solver mesh samples when available, with a trajectory-based
    fallback for older results.
    """
    mesh_points = []
    mesh_times = []
    cumulative_time = 0.0

    if result.mesh_samples:
        for mesh_positions, mesh_tau, delta in result.mesh_samples:
            if len(mesh_positions) > 2:
                for idx in range(1, len(mesh_positions) - 1):
                    mesh_points.append(mesh_positions[idx])
                    mesh_times.append(cumulative_time + mesh_tau[idx] * delta)
            cumulative_time += delta
    else:
        for traj, tau, delta in result.trajectories:
            sample_count = min(fallback_points_per_segment, len(traj))
            if sample_count > 2:
                indices = np.linspace(0, len(traj) - 1, sample_count, dtype=int)
                indices = np.unique(indices)
                for idx in indices[1:-1]:
                    mesh_points.append(traj[idx, :2])
                    mesh_times.append(cumulative_time + tau[idx] * delta)
            cumulative_time += delta

    if not mesh_points:
        return np.empty((0, 2)), np.empty((0,))

    return np.array(mesh_points), np.array(mesh_times)


def plot_trajectory(ax, result: OptimizationResult, regions: List[ConvexRegion],
                    show_mesh_points: bool = True,
                    show_interface_points: bool = True,
                    show_heading: bool = True,
                    trajectory_color: str = '#2196F3',
                    linewidth: float = 2.5):
    """
    Plot the optimized trajectory.
    
    Args:
        ax: Matplotlib axes
        result: OptimizationResult from optimizer
        regions: List of ConvexRegion objects
        show_mesh_points: Whether to show mesh sampling points
        show_interface_points: Whether to show interface points
        show_heading: Whether to show heading arrows
        trajectory_color: Color for trajectory line
        linewidth: Line width
    """
    if not result.success or not result.trajectories:
        return
    
    # Collect all trajectory segments
    all_positions = []
    all_times = []
    cumulative_time = 0.0
    
    for traj, tau, delta in result.trajectories:
        positions = traj[:, :2]
        times = tau * delta + cumulative_time
        
        all_positions.append(positions)
        all_times.append(times)
        cumulative_time += delta
    
    # Plot trajectory segments with color gradient
    for i, (positions, times) in enumerate(zip(all_positions, all_times)):
        # Main trajectory line
        ax.plot(positions[:, 0], positions[:, 1],
                color=trajectory_color, linewidth=linewidth,
                solid_capstyle='round', zorder=5)
        
        # Heading arrows (sample a few points)
        if show_heading and i == 0:  # Only first segment start
            traj = result.trajectories[i][0]
            pos = traj[0, :2]
            theta = traj[0, 2]
            
            arrow_len = 0.15
            dx = arrow_len * np.cos(theta)
            dy = arrow_len * np.sin(theta)
            ax.arrow(pos[0], pos[1], dx, dy, head_width=0.08,
                    head_length=0.04, fc='green', ec='darkgreen', zorder=6)
    
    # Interface points
    if show_interface_points and result.interface_points:
        interface_pts = np.array(result.interface_points)
        ax.scatter(interface_pts[:, 0], interface_pts[:, 1],
                  c='#FF9800', s=100, marker='D', zorder=7,
                  edgecolors='darkorange', linewidths=1.5,
                  label='Interface Points')
    
    # Mesh points (sample from trajectories)
    if show_mesh_points:
        mesh_pts, _ = _collect_mesh_points(result)
        if len(mesh_pts) > 0:
            ax.scatter(mesh_pts[:, 0], mesh_pts[:, 1],
                      c='#9C27B0', s=30, marker='o', zorder=6,
                      alpha=0.6, label='Mesh Points')


def plot_graph_structure(ax, graph: RegionGraph, 
                         highlight_path: Optional[List[str]] = None):
    """
    Plot the graph structure overlaid on regions.
    
    Args:
        ax: Matplotlib axes
        graph: RegionGraph object
        highlight_path: Path to highlight
    """
    import networkx as nx
    
    # Compute node positions
    pos = {}
    pos[SOURCE] = graph.start_pos
    pos[TARGET] = graph.goal_pos
    
    for region in graph.regions:
        node_id = f"R{region.index}"
        pos[node_id] = region.get_centroid()
    
    # Draw edges
    for u, v in graph.graph.edges():
        if highlight_path and u in highlight_path and v in highlight_path:
            # Check if consecutive in path
            try:
                idx_u = highlight_path.index(u)
                idx_v = highlight_path.index(v)
                if abs(idx_u - idx_v) == 1:
                    color = 'blue'
                    width = 2.5
                    alpha = 0.8
                else:
                    color = 'gray'
                    width = 0.5
                    alpha = 0.3
            except ValueError:
                color = 'gray'
                width = 0.5
                alpha = 0.3
        else:
            color = 'gray'
            width = 0.5
            alpha = 0.3
        
        p1 = pos[u]
        p2 = pos[v]
        
        ax.annotate("", xy=p2, xytext=p1,
                   arrowprops=dict(arrowstyle="->", color=color,
                                  lw=width, alpha=alpha,
                                  connectionstyle="arc3,rad=0.1"))


def create_result_figure(result: OptimizationResult, graph: RegionGraph,
                         workspace_bounds: Tuple[float, float, float, float],
                         start_pos: np.ndarray, goal_pos: np.ndarray,
                         obstacles: Optional[List[object]] = None,
                         title: str = "GCS-MMS Motion Planning Result",
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create comprehensive result figure.
    
    Args:
        result: OptimizationResult
        graph: RegionGraph
        workspace_bounds: Environment bounds
        start_pos, goal_pos: Start and goal positions
        obstacles: Obstacle polygons to overlay
        title: Figure title
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=figsize)
    
    # Main plot: environment + trajectory
    ax_main = fig.add_subplot(2, 2, (1, 3))
    
    # Plot environment
    plot_environment(ax_main, workspace_bounds, graph.regions,
                     start_pos, goal_pos,
                     obstacles=obstacles,
                     highlight_regions=result.path_regions if result.success else None)
    
    # Plot trajectory
    if result.success:
        plot_trajectory(ax_main, result, graph.regions)
    
    ax_main.set_title(f"{title}\n{'Success' if result.success else 'Failed'}")
    ax_main.set_xlabel("x [m]")
    ax_main.set_ylabel("y [m]")
    
    # Info panel
    ax_info = fig.add_subplot(2, 2, 2)
    ax_info.axis('off')

    path_labels = []
    for node_id in result.path:
        if node_id.startswith("R"):
            region_idx = int(node_id[1:])
            delta = result.time_durations.get(region_idx)
            if delta is not None:
                path_labels.append(f"{node_id} [{delta:.2f}s]")
                continue
        path_labels.append(node_id)

    path_text = ' -> '.join(path_labels) if path_labels else 'N/A'
    path_lines = textwrap.wrap(path_text, width=30, break_long_words=False) or ['N/A']

    info_text = [
        "Optimization",
        f"Status  {'OK' if result.success else 'FAIL'}",
        f"Cost    {result.total_cost:.4f}",
        f"Solve   {result.solve_time:.3f} s",
        f"Paths   {result.n_paths_evaluated}",
        f"Defect  {result.defect_norm:.2e}",
        f"Gap     {result.max_connection_gap:.2e}",
        f"Viol    {result.constraint_violation:.2e}",
    ]

    if result.max_integrality_gap > 0.0:
        info_text.append(f"IntGap  {result.max_integrality_gap:.2e}")

    info_text.extend([
        "",
        "Path",
        *path_lines,
    ])

    if result.solver_status:
        info_text.extend([
            "",
            "Solver",
            *textwrap.wrap(result.solver_status, width=30, break_long_words=False),
        ])

    if result.success:
        total_time = sum(result.time_durations.values())
        info_text.extend([
            "",
            f"Duration {total_time:.3f} s",
        ])

    ax_info.text(0.05, 0.95, '\n'.join(info_text),
                transform=ax_info.transAxes,
                fontfamily='monospace', fontsize=8.5,
                verticalalignment='top')
    
    # Trajectory profile
    ax_profile = fig.add_subplot(2, 2, 4)
    
    if result.success and result.trajectories:
        # Plot velocity profile
        times = []
        velocities = []
        cumulative_t = 0.0
        
        for traj, tau, delta in result.trajectories:
            for i in range(len(traj) - 1):
                dt = (tau[i+1] - tau[i]) * delta
                if dt > 1e-10:
                    pos_diff = traj[i+1, :2] - traj[i, :2]
                    vel = np.linalg.norm(pos_diff) / dt
                    times.append(cumulative_t + tau[i] * delta)
                    velocities.append(vel)
            cumulative_t += delta
        
        ax_profile.plot(times, velocities, 'b-', linewidth=1.5, label='|v|')
        ax_profile.set_xlabel("Time [s]")
        ax_profile.set_ylabel("Velocity [m/s]")
        ax_profile.set_title("Velocity Profile")
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend()
    else:
        ax_profile.text(0.5, 0.5, "No trajectory data",
                       ha='center', va='center', transform=ax_profile.transAxes)
        ax_profile.set_title("Velocity Profile")
    
    plt.tight_layout()
    return fig


def create_animation(result: OptimizationResult, graph: RegionGraph,
                     workspace_bounds: Tuple[float, float, float, float],
                     start_pos: np.ndarray, goal_pos: np.ndarray,
                     filename: str,
                     obstacles: Optional[List[object]] = None,
                     fps: int = 30,
                     duration: float = 5.0,
                     show_mesh_points: bool = True) -> None:
    """
    Create trajectory animation as GIF.
    
    Args:
        result: OptimizationResult
        graph: RegionGraph
        workspace_bounds: Environment bounds
        start_pos, goal_pos: Start and goal positions
        obstacles: Obstacle polygons to overlay
        filename: Output filename
        fps: Frames per second
        duration: Animation duration in seconds
        show_mesh_points: Whether to display interior mesh points in the GIF
    """
    if not result.success:
        print("Cannot create animation: optimization failed")
        return
    if not result.trajectories:
        print("Cannot create animation: no trajectory data available")
        return
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Static background
    plot_environment(ax, workspace_bounds, graph.regions,
                     start_pos, goal_pos,
                     obstacles=obstacles,
                     highlight_regions=result.path_regions)
    
    # Collect trajectory points with time
    all_points = []
    all_times = []
    all_headings = []
    cumulative_time = 0.0
    
    for traj, tau, delta in result.trajectories:
        for i in range(len(traj)):
            all_points.append(traj[i, :2])
            all_times.append(cumulative_time + tau[i] * delta)
            all_headings.append(traj[i, 2] if len(traj[i]) > 2 else 0)
        cumulative_time += delta
    
    all_points = np.array(all_points)
    all_times = np.array(all_times)
    all_headings = np.array(all_headings)
    mesh_points, mesh_times = _collect_mesh_points(result)
    
    # Total trajectory time
    total_time = all_times[-1]
    n_frames = max(2, int(fps * duration))
    
    # Animation elements
    robot_marker, = ax.plot([], [], 'bo', markersize=12, zorder=10)
    trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=5)
    mesh_scatter = ax.scatter([], [], c='#9C27B0', s=35, marker='o',
                              zorder=6, alpha=0.75, label='Mesh Points')
    heading_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        robot_marker.set_data([], [])
        trail_line.set_data([], [])
        mesh_scatter.set_offsets(np.empty((0, 2)))
        heading_arrow.set_position((0, 0))
        heading_arrow.xy = (0, 0)
        time_text.set_text('')
        return robot_marker, trail_line, mesh_scatter, heading_arrow, time_text
    
    def animate(frame):
        # Current animation time (mapped to trajectory time)
        t_anim = frame / (n_frames - 1) * duration
        t_traj = t_anim / duration * total_time
        
        # Find position at current time
        idx = np.searchsorted(all_times, t_traj, side='right') - 1
        idx = max(0, min(idx, len(all_times) - 2))
        
        # Interpolate
        t0, t1 = all_times[idx], all_times[idx + 1]
        alpha = (t_traj - t0) / (t1 - t0 + 1e-10)
        alpha = np.clip(alpha, 0, 1)
        
        pos = (1 - alpha) * all_points[idx] + alpha * all_points[idx + 1]
        heading = (1 - alpha) * all_headings[idx] + alpha * all_headings[idx + 1]
        
        # Update robot marker
        robot_marker.set_data([pos[0]], [pos[1]])
        
        # Update trail (all points up to current)
        trail_idx = idx + 1
        trail_line.set_data(all_points[:trail_idx, 0], all_points[:trail_idx, 1])

        # Show mesh points that have been reached so far
        if show_mesh_points and len(mesh_points) > 0:
            visible_mesh = mesh_points[mesh_times <= t_traj + 1e-10]
            mesh_scatter.set_offsets(visible_mesh if len(visible_mesh) > 0 else np.empty((0, 2)))
        else:
            mesh_scatter.set_offsets(np.empty((0, 2)))
        
        # Update heading arrow
        arrow_len = 0.2
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        heading_arrow.set_position((pos[0], pos[1]))
        heading_arrow.xy = (pos[0] + dx, pos[1] + dy)
        
        # Update time text
        time_text.set_text(f't = {t_traj:.2f} s')
        
        return robot_marker, trail_line, mesh_scatter, heading_arrow, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_frames, interval=1000/fps, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"Animation saved to {filename}")


def save_result_summary(result: OptimizationResult, filename: str):
    """
    Save result summary to JSON file.
    
    Args:
        result: OptimizationResult
        filename: Output filename
    """
    import json
    
    summary = {
        'success': result.success,
        'solver_status': result.solver_status,
        'total_cost': float(result.total_cost),
        'solve_time': float(result.solve_time),
        'n_paths_evaluated': result.n_paths_evaluated,
        'path': result.path,
        'path_regions': result.path_regions,
        'defect_norm': float(result.defect_norm),
        'max_connection_gap': float(result.max_connection_gap),
        'max_integrality_gap': float(result.max_integrality_gap),
        'constraint_violation': float(result.constraint_violation),
        'time_durations': {str(k): float(v) for k, v in result.time_durations.items()},
        'interface_points': [pt.tolist() for pt in result.interface_points],
    }
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {filename}")
