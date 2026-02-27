import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import flow_cylinder_potential, temp_constant
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler
from src.warp_visualizer import WarpVisualizer


def run_hybrid_collision_simulation() -> None:
    """Orchestrates the execution of a collision scenario using Warp Native Hash Grid."""
    config = SimConfig(
        d_particle=0.1,
        rho_particle=100.0,
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=10.0,
        alpha=1.0,
        g=0.0,  # Zero gravity to keep streams straight
        R_cylinder=0.5,
        enable_turbulence=False,
        turbulence_intensity=0.0,
        enable_collisions=True,
        collision_restitution=0.9,
        collision_engine="warp",
    )

    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    x_bounds = (-3.0, 3.0)
    y_bounds = (-2.0, 2.0)
    boundary_manager = BoundaryManager(
        x_bounds=x_bounds, y_bounds=y_bounds, cylinder_collision=True, periodic=False
    )

    # Initialization: Two Single-File Lines
    particles_per_stream = 100
    spacing = config.d_particle * 3.0

    target_pos = jnp.array([-0.5, 0.0])

    # Stream 1
    start_1 = jnp.array([-3.0, 1.0])
    dir_1 = target_pos - start_1
    dir_1 = dir_1 / jnp.linalg.norm(dir_1)
    indices = jnp.arange(particles_per_stream)
    pos_1_start = start_1 - (dir_1 * spacing * indices[:, None])
    vel_1 = jnp.tile(dir_1 * 8.0, (particles_per_stream, 1))

    # Stream 2
    start_2 = jnp.array([-3.0, -1.0])
    dir_2 = target_pos - start_2
    dir_2 = dir_2 / jnp.linalg.norm(dir_2)
    pos_2_start = start_2 - (dir_2 * spacing * indices[:, None])
    vel_2 = jnp.tile(dir_2 * 8.0, (particles_per_stream, 1))

    # Combine Initial Arrays
    combined_pos = jnp.concatenate([pos_1_start, pos_2_start])
    combined_vel = jnp.concatenate([vel_1, vel_2])

    total_particles = combined_pos.shape[0]
    initial_mass = jnp.full((total_particles,), config.m_particle_init)

    # Populate missing fields to satisfy strict types
    initial_temp = jnp.full((total_particles,), config.T_room_ref)
    initial_active = jnp.ones((total_particles,), dtype=bool)

    particle_state = ParticleState(
        position=combined_pos,
        velocity=combined_vel,
        temperature=initial_temp,
        mass=initial_mass,
        active=initial_active,
    )

    # Simulation Discretization
    simulation_end_time = 1.0
    time_step = 0.00001
    evaluation_times = jnp.array(np.arange(0.0, simulation_end_time, time_step))

    rng_seed = jnp.array([123, 456], dtype=jnp.uint32)
    print("Initiating Thin Stream Collision Fast XLA Simulation...")
    print(f"Total Particles: {total_particles}")
    print(f"Collision Target: {target_pos}")

    # Standard XLA compiled solver (incorporates Warp collision backend seamlessly)
    simulation_history = run_simulation_euler(
        particle_state,
        evaluation_times,
        config,
        force_config,
        boundary_manager,
        flow_cylinder_potential,
        temp_constant,
        rng_seed,
    )

    # Visualization
    print("Rendering Simulation Output...")
    viewport_bounds = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])

    visualizer = WarpVisualizer(
        config,
        simulation_history,
        evaluation_times,
        flow_cylinder_potential,
        temp_constant,
    )

    visualizer.generate_video(
        "warp_collision_thin_streams.mp4",
        bounds=viewport_bounds,
        width=800,
        height=500,
        fps=30,
        slow_mo_factor=10.0,
    )
    print("Simulation Complete. Output saved to warp_collision_thin_streams.mp4")


if __name__ == "__main__":
    run_hybrid_collision_simulation()
