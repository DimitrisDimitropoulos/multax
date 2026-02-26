import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler
from src.jax_visualizer import JAXVisualizer


def custom_flow(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """A custom wavy flow field."""
    x, y = position
    ux = 1.0 + 0.5 * jnp.sin(y * 2.0 * jnp.pi / 2.0)
    uy = 0.5 * jnp.sin(x * 2.0 * jnp.pi / 2.0)
    return jnp.array([ux, uy])


def custom_temp(position: jnp.ndarray, config: SimConfig) -> float:
    """A custom sinusoidal temperature field."""
    x, y = position
    # Temperature varies between 300K and 400K
    return 300.0 + 50.0 * (jnp.sin(x * 2.0) * jnp.cos(y * 2.0) + 1.0)


def main():
    config = SimConfig(
        d_particle=150e-6,
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

    force_config = ForceConfig(gravity=False, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    x_lim = (-1.0, 5.0)
    y_lim = (-2.0, 2.0)
    bounds = BoundaryManager(x_bounds=x_lim, y_bounds=y_lim, periodic=True)

    # Initialization
    n_particles = 1500
    pos_x = np.random.uniform(x_lim[0], x_lim[0] + 0.5, n_particles)
    pos_y = np.random.uniform(y_lim[0], y_lim[1], n_particles)
    pos = jnp.array(np.stack([pos_x, pos_y], axis=1))

    # Init velocity from flow field
    vel = jax.vmap(lambda p: custom_flow(p, config))(pos)

    # Init temp to room temp
    temp = jnp.full((n_particles,), config.T_room_ref - 100)
    mass = jnp.full((n_particles,), config.m_particle_init)
    active = jnp.ones((n_particles,), dtype=bool)

    initial_state = ParticleState(
        position=pos, velocity=vel, temperature=temp, mass=mass, active=active
    )

    # Simulation
    t_end = 6.0
    dt = 0.0005
    t_eval = jnp.array(np.arange(0.0, t_end, dt))
    key = jnp.array([777, 888], dtype=jnp.uint32)

    print("Running Custom Fields Simulation...")
    history = run_simulation_euler(
        initial_state,
        t_eval,
        config,
        force_config,
        bounds,
        custom_flow,
        custom_temp,
        key,
    )

    # Visualization
    print("Generating Video...")
    flat_bounds = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    viz = JAXVisualizer(config, history, t_eval, custom_flow, custom_temp)
    viz.generate_video(
        "custom_fields.mp4",
        bounds=flat_bounds,
        width=800,
        height=450,
        fps=120,
        slow_mo_factor=1.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
