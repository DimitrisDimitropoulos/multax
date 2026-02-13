import src.patch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import flow_wall_stagnation
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler


def main():
    config = SimConfig(
        d_particle=50e-6,  # 50 microns (Realistic Mist)
        rho_particle=1000.0,  # Water
        rho_fluid=1.225,  # Air
        mu_fluid=1.81e-5,
        U_0=1.3,
        alpha=1.0,
        g=-9.81,
        # Thermal Properties (User Inputs)
        cp_particle=4184.0,  # Water J/(kg K)
        cp_fluid=1005.0,  # Air J/(kg K)
        k_fluid=0.026,  # Air W/(m K)
        # Scenario Specifics
        wall_x=0.0,
        T_wall=273.15 + 30.0,
        T_gradient_slope=20.0,
        RH_room=0.5,
        T_room_ref=293.15,
        enable_turbulence=True,
        turbulence_intensity=0.2,
        evap_cutoff_ratio=0.1,
    )

    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    # Wall is at x=0
    x_lim = (-3.0, 1.0)
    y_lim = (-2.0, 2.0)
    bounds = BoundaryManager(
        x_bounds=x_lim, y_bounds=y_lim, wall_collision=True, periodic=False
    )

    # Initialization
    n_particles = 500
    # Line upstream
    pos_x = np.full((n_particles,), -2.5)
    pos_y = np.linspace(-0.5, 0.5, n_particles)
    pos = jnp.array(np.stack([pos_x, pos_y], axis=1))
    vel = jax.vmap(lambda p: flow_wall_stagnation(p, config))(pos)
    # Init temp to room temp
    temp = jnp.full((n_particles,), config.T_room_ref)
    # Init mass (assuming d_particle is initial diameter)
    mass = jnp.full((n_particles,), config.m_particle_init)
    # Init active
    active = jnp.ones((n_particles,), dtype=bool)
    initial_state = ParticleState(
        position=pos, velocity=vel, temperature=temp, mass=mass, active=active
    )

    # Simulation
    t_end = 5.0
    dt = 0.0001
    t_eval = jnp.array(np.arange(0.0, t_end, dt))
    key = jnp.array([42, 42], dtype=jnp.uint32)
    print("Running Wall Thermal Simulation...")
    history = run_simulation_euler(
        initial_state, t_eval, config, force_config, bounds, flow_wall_stagnation, key
    )

    # Visualization
    print("Generating Video (JAX Rasterizer)...")
    from src.jax_visualizer import JAXVisualizer

    # Flatten bounds for rasterizer: x_min, x_max, y_min, y_max
    flat_bounds = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    viz = JAXVisualizer(config, history, t_eval, flow_wall_stagnation)
    viz.generate_video(
        "wall_thermal.mp4",
        bounds=flat_bounds,
        width=800,
        height=450,
        fps=30,
        slow_mo_factor=1.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
