import src.patch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import flow_cylinder_potential, temp_constant
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler


def main():
    config = SimConfig(
        d_particle=0.0001,
        rho_particle=2650.0,  # RHO_SAND
        rho_fluid=1.225,  # RHO_AIR
        mu_fluid=1.81e-5,  # MU_AIR
        U_0=10.0,
        alpha=1.0,
        g=-9.81,
        R_cylinder=0.5,
        enable_turbulence=True,
        turbulence_intensity=0.2,
    )

    force_config = ForceConfig(gravity=False, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    x_lim = (-3.0, 3.0)
    y_lim = (-2.0, 2.0)
    bounds = BoundaryManager(
        x_bounds=x_lim, y_bounds=y_lim, cylinder_collision=True, periodic=False
    )

    # Initialization
    n_particles = 500
    pos_x = np.full((n_particles,), -3.5)
    pos_y = np.linspace(-0.0, 0.0, n_particles)
    pos = jnp.array(np.stack([pos_x, pos_y], axis=1))
    vel = jax.vmap(lambda p: flow_cylinder_potential(p, config))(pos)
    # Initialize Mass
    m_p = config.m_particle_init
    mass = jnp.full((n_particles,), m_p)
    initial_state = ParticleState(position=pos, velocity=vel, mass=mass)

    # Simulation
    t_end = 1.0
    dt = 0.005
    t_eval = jnp.array(np.arange(0.0, t_end, dt))
    key = jnp.array([123, 456], dtype=jnp.uint32)
    print("Running Cylinder Simulation...")
    history = run_simulation_euler(
        initial_state,
        t_eval,
        config,
        force_config,
        bounds,
        flow_cylinder_potential,
        temp_constant,
        key,
    )

    # Visualization
    print("Generating Video (JAX Rasterizer)...")
    from src.jax_visualizer import JAXVisualizer

    flat_bounds = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    viz = JAXVisualizer(config, history, t_eval, flow_cylinder_potential, temp_constant)
    viz.generate_video(
        "cylinder_flow.mp4",
        bounds=flat_bounds,
        width=800,
        height=500,
        fps=10,
        slow_mo_factor=5.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
