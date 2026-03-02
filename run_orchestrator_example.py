import src.patch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import h5py
import os
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import flow_cylinder_potential, temp_constant
from src.boundary import BoundaryManager

from src.orchestrator import run_orchestrated_simulation, generate_orchestrated_video


def main():
    config = SimConfig(
        d_particle=0.0001,
        rho_particle=2650.0,
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=10.0,
        alpha=1.0,
        g=-9.81,
        R_cylinder=0.5,
        enable_turbulence=True,
        turbulence_intensity=0.2,
    )

    force_config = ForceConfig(gravity=False, undisturbed_flow=True, drag=True)

    x_lim = (-3.0, 3.0)
    y_lim = (-2.0, 2.0)
    bounds = BoundaryManager(
        x_bounds=x_lim, y_bounds=y_lim, cylinder_collision=True, periodic=False
    )

    n_particles: int = int(1e5)
    pos_x = np.full((n_particles,), -3.5)
    pos_y = np.linspace(-0.0, 0.0, n_particles)
    pos = jnp.array(np.stack([pos_x, pos_y], axis=1))
    vel = jax.vmap(lambda p: flow_cylinder_potential(p, config))(pos)
    m_p = config.m_particle_init
    mass = jnp.full((n_particles,), m_p)
    initial_state = ParticleState(position=pos, velocity=vel, mass=mass)

    t_end = 1.5
    dt = 0.005
    key = jnp.array([123, 456], dtype=jnp.uint32)
    db_path = "cylinder_chunked.h5"

    target_memory_mb = 1000.0

    if os.path.exists(db_path):
        os.remove(db_path)

    print(
        f"Running Orchestrated Cylinder Simulation (Memory Target: {target_memory_mb} MB)..."
    )
    run_orchestrated_simulation(
        initial_state=initial_state,
        config=config,
        force_config=force_config,
        boundary_manager=bounds,
        flow_func=flow_cylinder_potential,
        temp_func=temp_constant,
        master_rng_key=key,
        t_end=t_end,
        dt=dt,
        target_memory_mb=target_memory_mb,
        db_path=db_path,
    )

    # Visualization
    print("Reading from DB Out-Of-Core and Generating Video...")

    flat_bounds = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])

    # We figure out t_eval array simply mathematically:
    t_eval = jnp.arange(0.0, t_end, dt)

    generate_orchestrated_video(
        db_path=db_path,
        output_path="cylinder_flow_orchestrator.mp4",
        config=config,
        t_eval=t_eval,
        bounds=flat_bounds,
        flow_func=flow_cylinder_potential,
        temp_func=temp_constant,
        width=800,
        height=500,
        fps=10,
        slow_mo_factor=5.0,
        use_warp=False,  # Set to True to test Warp rasterizer!
    )
    print("Done.")


if __name__ == "__main__":
    main()
