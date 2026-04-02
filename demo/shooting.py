"""
shooting.py - Multiple shooting blocks for trajectory optimization.

Implements:
- Local IVP on each region with normalized time
- Endpoint map F_v(s_v^-, w_v, Delta_v) = s_v^+
- Defect constraints: s_v^+ - F_v(...) = 0
- Safety constraints via mesh point sampling
- Local cost computation with quadrature
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import casadi as ca

from dynamics import (DynamicsModel, UnicycleModel, ControlParameterization,
                      RK4Integrator, create_casadi_integrator, 
                      create_casadi_trajectory_sampler)
from convex_regions import ConvexRegion


@dataclass
class ShootingBlockConfig:
    """Configuration for a multiple shooting block."""
    region_index: int
    n_integration_steps: int = 20
    n_mesh_points: int = 10
    n_control_segments: int = 2
    safety_margin: float = 0.02
    delta_min: float = 0.1
    delta_max: float = 10.0


@dataclass
class ShootingBlock:
    """
    Multiple shooting block for a single convex region.
    
    Represents the local trajectory segment with:
    - Entry state s_v^-
    - Exit state s_v^+
    - Control parameters w_v
    - Time duration Delta_v
    - Defect constraint: s_v^+ = F_v(s_v^-, w_v, Delta_v)
    - Safety constraints: A_v @ pi(x_v(tau_k)) <= b_v - epsilon_v
    
    Attributes:
        region: ConvexRegion object
        dynamics: Dynamics model
        config: Block configuration
        control_param: Control parameterization
        integrator: RK4 integrator
        F_casadi: CasADi endpoint map function
        mesh_sampler: CasADi trajectory sampling function
    """
    region: ConvexRegion
    dynamics: DynamicsModel
    config: ShootingBlockConfig
    control_param: ControlParameterization = field(init=False)
    integrator: RK4Integrator = field(init=False)
    F_casadi: Callable = field(init=False)
    mesh_sampler: Callable = field(init=False)
    
    def __post_init__(self):
        """Initialize integrators and CasADi functions."""
        # Control parameterization
        self.control_param = ControlParameterization(
            n_u=self.dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=self.config.n_control_segments
        )
        
        # NumPy integrator for evaluation
        self.integrator = RK4Integrator(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=self.config.n_integration_steps
        )
        
        # CasADi endpoint map
        self.F_casadi = create_casadi_integrator(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=self.config.n_integration_steps
        )
        
        # CasADi mesh sampler
        self.mesh_sampler = create_casadi_trajectory_sampler(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=self.config.n_integration_steps,
            n_mesh=self.config.n_mesh_points
        )
    
    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.dynamics.n_x
    
    @property
    def n_u(self) -> int:
        """Control dimension."""
        return self.dynamics.n_u
    
    @property
    def n_w(self) -> int:
        """Control parameter dimension."""
        return self.control_param.n_w
    
    def endpoint_map(self, s_minus: np.ndarray, w: np.ndarray, 
                     delta: float) -> np.ndarray:
        """
        Compute endpoint map F_v(s_v^-, w_v, Delta_v).
        
        Args:
            s_minus: Entry state
            w: Control parameters
            delta: Time duration
            
        Returns:
            Exit state s_plus
        """
        return self.integrator.integrate(s_minus, w, delta)
    
    def compute_defect(self, s_minus: np.ndarray, s_plus: np.ndarray,
                       w: np.ndarray, delta: float) -> np.ndarray:
        """
        Compute defect: s_plus - F_v(s_minus, w, delta).
        
        Should be zero for valid trajectory.
        """
        F_result = self.endpoint_map(s_minus, w, delta)
        return s_plus - F_result
    
    def check_safety(self, s_minus: np.ndarray, w: np.ndarray,
                     delta: float) -> Tuple[bool, np.ndarray]:
        """
        Check safety constraints at mesh points.
        
        Args:
            s_minus: Entry state
            w: Control parameters
            delta: Time duration
            
        Returns:
            (is_safe, violations) where violations is (n_mesh, n_constraints)
        """
        # Sample trajectory at mesh points
        mesh_states = self.integrator.sample_trajectory_at_mesh(
            s_minus, w, delta, self.config.n_mesh_points
        )
        
        # Check containment at each mesh point
        margin = self.config.safety_margin
        violations = []
        
        for state in mesh_states:
            pos = self.dynamics.project_to_position(state)
            # A @ pos - (b - margin) should be <= 0
            violation = self.region.A @ pos - (self.region.b - margin)
            violations.append(violation)
        
        violations = np.array(violations)
        is_safe = np.all(violations <= 0)
        
        return is_safe, violations
    
    def compute_local_cost(self, s_minus: np.ndarray, w: np.ndarray,
                           delta: float, cost_weights: Dict[str, float]) -> float:
        """
        Compute local cost J_v using trapezoidal quadrature.
        
        J_v = a * Delta_v + integral[w_L * ||q_dot||^2 + w_E * ||u||^2] dtau
        
        Args:
            s_minus: Entry state
            w: Control parameters
            delta: Time duration
            cost_weights: Dict with 'a', 'w_L', 'w_E'
            
        Returns:
            Local cost value
        """
        a = cost_weights.get('a', 1.0)
        w_L = cost_weights.get('w_L', 1.0)
        w_E = cost_weights.get('w_E', 1.0)
        
        # Time cost
        time_cost = a * delta
        
        # Get trajectory
        traj, tau_vals = self.integrator.integrate_with_trajectory(s_minus, w, delta)
        
        # Compute running cost via trapezoidal rule
        running_cost = 0.0
        n_pts = len(traj)
        
        for i in range(n_pts - 1):
            tau0 = tau_vals[i]
            tau1 = tau_vals[i + 1]
            dt = tau1 - tau0
            
            # Approximate velocity as finite difference (in physical time)
            pos0 = self.dynamics.project_to_position(traj[i])
            pos1 = self.dynamics.project_to_position(traj[i + 1])
            
            # Physical time velocity: dq/dt = (1/Delta) * dq/dtau
            dq_dtau = (pos1 - pos0) / (dt + 1e-12)
            q_dot = dq_dtau / (delta + 1e-12)  # Physical velocity
            
            # Control at midpoint
            tau_mid = 0.5 * (tau0 + tau1)
            u = self.control_param.evaluate(tau_mid, w)
            
            # Running cost integrand
            # Note: the integral is integral[Delta * (w_L / Delta^2 * ||dq/dtau||^2
            #     + w_E * ||u||^2)] dtau
            #     = integral[(w_L / Delta) * ||dq/dtau||^2 + Delta * w_E * ||u||^2] dtau
            velocity_cost = w_L * np.dot(q_dot, q_dot)
            control_cost = w_E * np.dot(u, u)
            
            # Trapezoidal contribution (multiply by delta for time transformation)
            integrand = velocity_cost + control_cost
            running_cost += delta * dt * integrand
        
        return time_cost + running_cost


class ShootingBlockManager:
    """
    Manages multiple shooting blocks across all regions.
    
    Coordinates:
    - Block creation
    - Variable indexing
    - Constraint generation
    - Cost aggregation
    """
    
    def __init__(self, regions: List[ConvexRegion], dynamics: DynamicsModel,
                 config: Dict):
        """
        Args:
            regions: List of ConvexRegion objects
            dynamics: Dynamics model
            config: Configuration dictionary
        """
        self.regions = regions
        self.dynamics = dynamics
        self.config = config
        
        # Create shooting blocks
        self.blocks: Dict[int, ShootingBlock] = {}
        
        for region in regions:
            block_config = ShootingBlockConfig(
                region_index=region.index,
                n_integration_steps=config.get('n_integration_steps', 20),
                n_mesh_points=config.get('n_mesh_points', 10),
                n_control_segments=config.get('n_segments', 2),
                safety_margin=config.get('safety_margin', 0.02),
                delta_min=config.get('delta_min', 0.1),
                delta_max=config.get('delta_max', 10.0)
            )
            
            self.blocks[region.index] = ShootingBlock(
                region=region,
                dynamics=dynamics,
                config=block_config
            )
    
    def get_block(self, region_index: int) -> ShootingBlock:
        """Get shooting block for a region."""
        return self.blocks[region_index]
    
    def get_variable_dimensions(self) -> Dict:
        """
        Get dimensions of decision variables per block.
        
        Returns:
            Dict with dimensions for each variable type
        """
        block = list(self.blocks.values())[0]  # All blocks have same dimensions
        
        return {
            'n_x': block.n_x,           # Entry/exit state dimension
            'n_w': block.n_w,           # Control parameter dimension
            'n_delta': 1,               # Time duration
            'n_rho': 1,                 # Cost epigraph variable
            'n_per_block': 2 * block.n_x + block.n_w + 2  # s-, s+, w, delta, rho
        }
    
    def evaluate_path(self, path_regions: List[int], 
                      variables: Dict[int, Dict[str, np.ndarray]],
                      start_state: np.ndarray,
                      goal_state: np.ndarray,
                      cost_weights: Dict[str, float]) -> Dict:
        """
        Evaluate a complete path through regions.
        
        Args:
            path_regions: List of region indices in path order
            variables: Dict mapping region index to {'s_minus', 's_plus', 'w', 'delta'}
            start_state: Initial state
            goal_state: Target state
            cost_weights: Cost function weights
            
        Returns:
            Dict with trajectory data, costs, violations
        """
        result = {
            'trajectories': [],
            'mesh_states': [],
            'defects': [],
            'safety_violations': [],
            'local_costs': [],
            'total_cost': 0.0,
            'is_feasible': True
        }
        
        for i, region_idx in enumerate(path_regions):
            block = self.blocks[region_idx]
            var = variables[region_idx]
            
            s_minus = var['s_minus']
            s_plus = var['s_plus']
            w = var['w']
            delta = var['delta']
            
            # Compute trajectory
            traj, tau = block.integrator.integrate_with_trajectory(s_minus, w, delta)
            result['trajectories'].append((traj, tau, delta))
            
            # Compute defect
            defect = block.compute_defect(s_minus, s_plus, w, delta)
            result['defects'].append(defect)
            
            if np.linalg.norm(defect) > 1e-4:
                result['is_feasible'] = False
            
            # Check safety
            is_safe, violations = block.check_safety(s_minus, w, delta)
            result['safety_violations'].append(violations)
            
            if not is_safe:
                result['is_feasible'] = False
            
            # Compute local cost
            local_cost = block.compute_local_cost(s_minus, w, delta, cost_weights)
            result['local_costs'].append(local_cost)
            result['total_cost'] += local_cost
        
        # Check interface continuity
        for i in range(len(path_regions) - 1):
            curr_idx = path_regions[i]
            next_idx = path_regions[i + 1]
            
            s_plus_curr = variables[curr_idx]['s_plus']
            s_minus_next = variables[next_idx]['s_minus']
            
            interface_error = np.linalg.norm(s_plus_curr - s_minus_next)
            if interface_error > 1e-4:
                result['is_feasible'] = False
        
        # Check boundary conditions
        first_idx = path_regions[0]
        last_idx = path_regions[-1]
        
        start_error = np.linalg.norm(variables[first_idx]['s_minus'][:2] - start_state[:2])
        goal_error = np.linalg.norm(variables[last_idx]['s_plus'][:2] - goal_state[:2])
        
        if start_error > 1e-4 or goal_error > 1e-4:
            result['is_feasible'] = False
        
        return result


def create_casadi_local_cost(dynamics: DynamicsModel,
                             control_param: ControlParameterization,
                             n_steps: int = 20) -> Callable:
    """
    Create CasADi function for local cost computation.
    
    J_v = a * Delta + integral[Delta * (w_L / Delta^2 * ||dq/dtau||^2 + w_E * ||u||^2)] dtau
        = a * Delta + sum_i Delta * dt * [(w_L / Delta) * ||dq/dtau||^2 + w_E * ||u||^2]
        
    Returns:
        Function(s_minus, w, delta, a, w_L, w_E) -> cost
    """
    # Symbolic variables
    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)
    a = ca.MX.sym('a', 1)
    w_L = ca.MX.sym('w_L', 1)
    w_E = ca.MX.sym('w_E', 1)
    
    dt = 1.0 / n_steps
    
    # Time cost
    cost = a * delta
    
    # Integrate and accumulate running cost
    x = s_minus
    
    for i in range(n_steps):
        tau = i * dt
        tau_mid = tau + 0.5 * dt
        tau_end = tau + dt
        
        # Get control
        u_start = control_param.evaluate_casadi(tau, w)
        u_mid = control_param.evaluate_casadi(tau_mid, w)
        u_end = control_param.evaluate_casadi(tau_end, w)
        
        # Current position
        pos_start = dynamics.project_to_position_casadi(x)
        
        # RK4 step
        k1 = delta * dynamics.f_casadi(x, u_start)
        k2 = delta * dynamics.f_casadi(x + 0.5 * dt * k1, u_mid)
        k3 = delta * dynamics.f_casadi(x + 0.5 * dt * k2, u_mid)
        k4 = delta * dynamics.f_casadi(x + dt * k3, u_end)
        
        x_next = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Position after step
        pos_end = dynamics.project_to_position_casadi(x_next)
        
        # Velocity (physical time)
        dq_dtau = (pos_end - pos_start) / dt
        # Note: actual velocity is dq/dt = (1 / Delta) * dq/dtau
        # ||dq/dt||^2 = (1 / Delta^2) * ||dq/dtau||^2
        
        # Control cost
        u = u_mid
        
        # Running cost: Delta * dt * [(w_L / Delta) * ||dq/dtau||^2 / Delta + w_E * ||u||^2]
        #             = dt * [(w_L / Delta) * ||dq/dtau||^2 + Delta * w_E * ||u||^2]
        velocity_sq = ca.dot(dq_dtau, dq_dtau)
        control_sq = ca.dot(u, u)
        
        running = w_L * velocity_sq / (delta + 1e-6) + delta * w_E * control_sq
        cost = cost + dt * running
        
        x = x_next
    
    F = ca.Function('local_cost', [s_minus, w, delta, a, w_L, w_E], [cost],
                   ['s_minus', 'w', 'delta', 'a', 'w_L', 'w_E'], ['cost'])
    
    return F
