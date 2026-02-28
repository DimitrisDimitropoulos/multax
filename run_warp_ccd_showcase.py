import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import temp_constant
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler
from src.warp_visualizer import WarpVisualizer


def flow_zero(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    r"""Returns a zero flow velocity field.

    Args:
        position (jnp.ndarray): Position vector. Units: [m].
        config (SimConfig): Simulation configuration.

    Returns:
        jnp.ndarray: Zero velocity vector. Units: [m/s].
    """
    return jnp.zeros(2)


def run_newtons_cradle_showcase() -> None:
    r"""Showcases the CCD-LCP solver's ability to perfectly transfer momentum
    across multiple consecutive collisions (like a Newton's Cradle).

    Demonstrates momentum transfer where:
    - A high-speed heavy projectile strikes a stationary chain of particles.
    - The momentum propagates through the chain.
    - The final particle hits a target cluster.
    """

    config = SimConfig(
        d_particle=0.1,
        rho_particle=100.0,
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=0.0,
        alpha=1.0,
        g=0.0,  # Zero gravity for pure kinetic transfer
        R_cylinder=0.0,
        enable_turbulence=False,
        turbulence_intensity=0.0,
        enable_collisions=True,
        collision_engine="warp_ccd",  # The precise LCP engine
        collision_restitution=1.0,  # Perfectly elastic collisions
    )

    force_config = ForceConfig(gravity=False, undisturbed_flow=False, drag=False)

    x_bounds = (-2.0, 2.0)
    y_bounds = (-1.0, 1.0)
    boundary_manager = BoundaryManager(
        x_bounds=x_bounds, y_bounds=y_bounds, cylinder_collision=False, periodic=False
    )

    # Scenario Setup: A High-Speed Projectile hitting a stationary chain

    # The Projectile (Left side, moving fast right)
    projectile_pos = jnp.array([[-1.5, 0.0]])
    projectile_vel = jnp.array([[5.0, 0.0]])
    projectile_mass = jnp.array([config.m_particle_init * 2.0])  # Double mass
    projectile_temp = jnp.array([400.0])  # Hot (Red)

    # The Stationary Chain (Center, perfectly aligned and touching)
    # Several particles touching edge-to-edge
    chain_length = 5
    start_x = -0.5
    chain_x = start_x + jnp.arange(chain_length) * config.d_particle
    chain_y = jnp.zeros(chain_length)
    chain_pos = jnp.stack([chain_x, chain_y], axis=1)

    chain_vel = jnp.zeros((chain_length, 2))
    chain_mass = jnp.full((chain_length,), config.m_particle_init)  # Standard mass
    chain_temp = jnp.full((chain_length,), 300.0)  # Cold (Blue)

    # The Target Cluster (Right side, a small pyramid)
    # A tiny pyramid of particles waiting to be hit by the chain's momentum
    pyramid_x = jnp.array(
        [0.5, 0.5 + config.d_particle * 0.866, 0.5 + config.d_particle * 0.866]
    )
    pyramid_y = jnp.array([0.0, config.d_particle * 0.5, -config.d_particle * 0.5])
    pyramid_pos = jnp.stack([pyramid_x, pyramid_y], axis=1)

    pyramid_vel = jnp.zeros((3, 2))
    pyramid_mass = jnp.full((3,), config.m_particle_init)
    pyramid_temp = jnp.full((3,), 350.0)  # Warm (Green/Yellow)

    # Combine all classes
    initial_pos = jnp.concatenate([projectile_pos, chain_pos, pyramid_pos], axis=0)
    initial_vel = jnp.concatenate([projectile_vel, chain_vel, pyramid_vel], axis=0)
    initial_mass = jnp.concatenate([projectile_mass, chain_mass, pyramid_mass], axis=0)
    initial_temp = jnp.concatenate([projectile_temp, chain_temp, pyramid_temp], axis=0)

    total_particles = initial_pos.shape[0]
    initial_active = jnp.ones((total_particles,), dtype=bool)

    particle_state = ParticleState(
        position=initial_pos,
        velocity=initial_vel,
        temperature=initial_temp,
        mass=initial_mass,
        active=initial_active,
    )

    simulation_end_time = 0.8
    time_step = 0.0005
    evaluation_times = jnp.array(np.arange(0.0, simulation_end_time, time_step))
    rng_seed = jnp.array([123, 456], dtype=jnp.uint32)

    print("Initiating CCD-LCP Multi-Class Collision Showcase...")
    print(f"Engine: {config.collision_engine.upper()}")
    print("Scenario: Heavy Projectile -> Stationary Chain -> Target Pyramid")

    simulation_history = run_simulation_euler(
        particle_state,
        evaluation_times,
        config,
        force_config,
        boundary_manager,
        flow_zero,
        temp_constant,
        rng_seed,
    )

    print("Rendering Simulation Output...")
    viewport_bounds = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])

    visualizer = WarpVisualizer(
        config, simulation_history, evaluation_times, flow_zero, temp_constant
    )

    visualizer.generate_video(
        "warp_ccd_showcase.mp4",
        bounds=viewport_bounds,
        width=1200,
        height=400,
        fps=60,
        slow_mo_factor=5.0,  # Slow motion to see the momentum transfer clearly
    )
    print("Simulation Complete. Output saved to warp_ccd_showcase.mp4")


if __name__ == "__main__":
    run_newtons_cradle_showcase()
