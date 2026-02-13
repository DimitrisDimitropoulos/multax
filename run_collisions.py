import src.patch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import flow_cylinder_potential
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler


def main():
    config = SimConfig(
        d_particle=0.1,
        rho_particle=100.0,
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=10.0,
        alpha=1.0,
        g=0.0,  # Zero gravity to keep streams straight
        R_cylinder=0.5,
        enable_turbulence=False,  # Disable turbulence to keep aim steady
        turbulence_intensity=0.0,
        enable_collisions=True,
        collision_restitution=0.9,
    )

    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    x_lim = (-3.0, 3.0)
    y_lim = (-2.0, 2.0)
    bounds = BoundaryManager(
        x_bounds=x_lim, y_bounds=y_lim, cylinder_collision=True, periodic=False
    )

    # Initialization: Two Single-File Lines
    n_per_stream = 100
    spacing = config.d_particle * 3.0  # Space them out so they arrive sequentially

    # Start far left, outside the view if possible, or just edge
    # Target: (-0.5, 0.0) -> Just in front of cylinder
    target = jnp.array([-0.5, 0.0])
    start_1 = jnp.array([-3.0, 1.0])
    # Direction vector
    dir_1 = target - start_1
    dir_1 = dir_1 / jnp.linalg.norm(dir_1)
    # Create line of positions BACKWARDS from start point
    # So the first particle is at start, the last is far behind
    indices = jnp.arange(n_per_stream)
    pos_1_start = start_1 - (dir_1 * spacing * indices[:, None])
    vel_1 = jnp.tile(dir_1 * 8.0, (n_per_stream, 1))  # Fast speed

    # Stream 2: Bottom-Left, aiming up-right
    start_2 = jnp.array([-3.0, -1.0])
    dir_2 = target - start_2
    dir_2 = dir_2 / jnp.linalg.norm(dir_2)
    pos_2_start = start_2 - (dir_2 * spacing * indices[:, None])
    vel_2 = jnp.tile(dir_2 * 8.0, (n_per_stream, 1))

    # Combine
    pos = jnp.concatenate([pos_1_start, pos_2_start])
    vel = jnp.concatenate([vel_1, vel_2])

    n_particles = pos.shape[0]
    m_p = config.m_particle_init
    mass = jnp.full((n_particles,), m_p)

    initial_state = ParticleState(position=pos, velocity=vel, mass=mass)

    # Simulation
    t_end = 1  # Longer time for the long tail to arrive
    dt = 0.00001
    t_eval = jnp.array(np.arange(0.0, t_end, dt))

    sim_key = jnp.array([123, 456], dtype=jnp.uint32)
    print("Running Thin Stream Collision Simulation...")
    print(f"N Particles: {n_particles}")
    print(f"Collision Target: {target}")

    history = run_simulation_euler(
        initial_state,
        t_eval,
        config,
        force_config,
        bounds,
        flow_cylinder_potential,
        sim_key,
    )

    # Visualization
    print("Generating Video...")
    from src.jax_visualizer import JAXVisualizer

    flat_bounds = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    viz = JAXVisualizer(config, history, t_eval, flow_cylinder_potential)
    viz.generate_video(
        "collision_thin_streams.mp4",
        bounds=flat_bounds,
        width=800,
        height=500,
        fps=30,
        slow_mo_factor=10.0,
    )
    print("Done. Saved to collision_thin_streams.mp4")


if __name__ == "__main__":
    main()
