import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler
from src.warp_visualizer import WarpVisualizer


def wavy_flow_field(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    r"""Computes a custom wavy analytical flow field.

    .. math::
        u_x = 1.0 + 0.5 \sin(y \cdot \pi), \quad u_y = 0.5 \sin(x \cdot \pi)

    Args:
        position: Position vector $(x, y)$. Units: [m].
        config: Simulation configuration.

    Returns:
        jnp.ndarray: Velocity vector $(u, v)$. Units: [m/s].
    """
    coord_x, coord_y = position
    vel_x = 1.0 + 0.5 * jnp.sin(coord_y * 2.0 * jnp.pi / 2.0)
    vel_y = 0.5 * jnp.sin(coord_x * 2.0 * jnp.pi / 2.0)
    return jnp.array([vel_x, vel_y])


def sinusoidal_temperature_field(position: jnp.ndarray, config: SimConfig) -> float:
    """Computes a custom sinusoidal carrier temperature field.

    Args:
        position: Position vector $(x, y)$. Units: [m].
        config: Simulation configuration.

    Returns:
        float: Local fluid temperature. Units: [K].
    """
    coord_x, coord_y = position
    base_temp = 300.0
    fluctuation = 50.0 * (jnp.sin(coord_x * 2.0) * jnp.cos(coord_y * 2.0) + 1.0)
    return base_temp + fluctuation


def run_simulation_example() -> None:
    """Orchestrates a simulation and visualization showcase using the Warp engine."""

    # Configuration Setup
    sim_config = SimConfig(
        d_particle=350e-6,
        rho_particle=1000.0,
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=1.0,
        alpha=1.0,
        g=0.0,
        cp_particle=4184.0,
        cp_fluid=1005.0,
        k_fluid=0.026,
        T_room_ref=300.0,
        T_wall=350.0,
        RH_room=0.65,
        enable_turbulence=True,
        turbulence_intensity=0.5,
    )

    active_forces = ForceConfig(gravity=False, undisturbed_flow=True, drag=True)

    # Spatial Boundaries
    x_bounds = (-2.0, 10.0)
    y_bounds = (-4.0, 4.0)
    domain_manager = BoundaryManager(
        x_bounds=x_bounds, y_bounds=y_bounds, periodic=True
    )

    # State Initialization
    particle_count = 5000
    initial_pos_x = np.random.uniform(x_bounds[0], x_bounds[0] + 0.5, particle_count)
    initial_pos_y = np.random.uniform(y_bounds[0], y_bounds[1], particle_count)
    initial_positions = jnp.array(np.stack([initial_pos_x, initial_pos_y], axis=1))

    # Initialize state variables
    initial_velocities = jax.vmap(lambda p: wavy_flow_field(p, sim_config))(
        initial_positions
    )
    initial_temperatures = jnp.full((particle_count,), sim_config.T_room_ref - 100)
    initial_masses = jnp.full((particle_count,), sim_config.m_particle_init)
    initial_active_mask = jnp.ones((particle_count,), dtype=bool)

    particle_state = ParticleState(
        position=initial_positions,
        velocity=initial_velocities,
        temperature=initial_temperatures,
        mass=initial_masses,
        active=initial_active_mask,
    )

    # Time Discretization
    simulation_end_time = 20.0
    time_step = 0.0005
    evaluation_times = jnp.array(np.arange(0.0, simulation_end_time, time_step))
    rng_seed = jnp.array([777, 888], dtype=jnp.uint32)

    # Execution: Physics Solver
    print("Initiating Custom Fields Simulation...")
    simulation_history = run_simulation_euler(
        particle_state,
        evaluation_times,
        sim_config,
        active_forces,
        domain_manager,
        wavy_flow_field,
        sinusoidal_temperature_field,
        rng_seed,
    )

    # Execution: Visualizer
    print("Initiating Video Generation via Warp Atomic Splatting...")
    viewport_bounds = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])

    warp_visualizer = WarpVisualizer(
        sim_config,
        simulation_history,
        evaluation_times,
        wavy_flow_field,
        sinusoidal_temperature_field,
    )

    warp_visualizer.generate_video(
        "warp_custom_fields.mp4",
        bounds=viewport_bounds,
        width=800,
        height=450,
        fps=120,
        slow_mo_factor=1.0,
    )
    print("Showcase Complete.")


if __name__ == "__main__":
    run_simulation_example()
