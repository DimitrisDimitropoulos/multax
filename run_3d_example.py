import os
import jax
import jax.numpy as jnp
import numpy as np

from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler
from src.vtu_exporter import export_simulation_to_vtu


def chimney_flow(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    r"""Constant velocity at X and Z with high turbulence."""
    # The user requested a constant velocity at x and z.
    # The solver will automatically superimpose random turbulence
    # based on config.turbulence_intensity and config.U_0.
    V_wind_x = 30.0
    V_wind_z = 15.0
    return jnp.array([V_wind_x, 0.0, V_wind_z])


def chimney_temp(position: jnp.ndarray, config: SimConfig) -> float:
    r"""Temperature profile: hot at the origin (chimney tip), cooling away."""
    x, y, z = position[0], position[1], position[2]

    T_exhaust = 500.0  # Hot gases at chimney tip
    T_ambient = config.T_room_ref

    # Distance from the origin (where chimney tip is)
    dist = jnp.sqrt(x**2 + y**2 + z**2)

    # Exponential cooling as it travels away from the origin
    length_scale = 20.0
    temp_rise = (T_exhaust - T_ambient) * jnp.exp(-dist / length_scale)

    return T_ambient + temp_rise


def main() -> None:
    # Initialize 3D Configuration
    config = SimConfig(
        dim=3,
        d_particle=0.001,  # 1mm particles
        rho_particle=1500.0,  # Ash/dust density
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=5.0,  # Reference wind speed for turbulence scaling
        alpha=1.0,
        g=-9.81,
        enable_turbulence=True,
        turbulence_intensity=0.8,  # VERY HIGH turbulence
        enable_collisions=False,
    )

    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Setup 3D Boundaries (Large domain to observe long carry over)
    x_lim = (-5.0, 500.0)  # Long downwind fetch
    y_lim = (-40.0, 40.0)
    z_lim = (-1.0, 100.0)  # Upward carry

    bounds = BoundaryManager(
        x_bounds=x_lim,
        y_bounds=y_lim,
        z_bounds=z_lim,
        periodic=False,
    )

    # Initialize Particles
    n_particles = 10000  # Lots of particles for nice visualization
    np.random.seed(42)

    # Inject particles at the tip of the chimney exactly at the XY plane (Z=0)
    R_chimney = 2.0
    r_inj = np.random.uniform(0.0, R_chimney, n_particles)
    theta_inj = np.random.uniform(0, 2 * np.pi, n_particles)

    pos_x = r_inj * np.cos(theta_inj)
    pos_y = r_inj * np.sin(theta_inj)
    pos_z = np.zeros(n_particles)  # Exactly on the XY plane

    pos = jnp.array(np.stack([pos_x, pos_y, pos_z], axis=1))
    vel = jax.vmap(lambda p: chimney_flow(p, config))(pos)

    # Let's give them a slight initial extra upward/outward burst optionally,
    # or just let the flow/turbulence carry them. We'll add some initial velocity variation
    vel = vel + jnp.array(np.random.normal(scale=1.5, size=(n_particles, 3)))

    mass = jnp.full((n_particles,), config.m_particle_init)

    temp = jax.vmap(lambda p: chimney_temp(p, config))(pos)

    initial_state = ParticleState(
        position=pos, velocity=vel, mass=mass, temperature=temp
    )

    # Execute Simulation Engine
    t_end = 15.0  # Very long duration to see where they land/fly
    dt = 0.01  # Larger timestep
    key = jnp.array([123, 456], dtype=jnp.uint32)
    t_eval = jnp.arange(0.0, t_end, dt)

    print("Running 3D Constant Crosswind Chimney Simulation...")
    history = run_simulation_euler(
        initial_state=initial_state,
        t_eval=t_eval,
        config=config,
        force_config=force_config,
        boundary_manager=bounds,
        flow_func=chimney_flow,
        temp_func=chimney_temp,
        master_rng_key=key,
    )

    t_eval_np = np.array(t_eval)

    # Export VTU Sequence
    output_dir = "vtu_output_chimney"
    export_simulation_to_vtu(
        output_dir=output_dir,
        history=history,
        config=config,
        t_eval=t_eval_np,
        flow_func=chimney_flow,
        temp_func=chimney_temp,
        stride=10,  # Save every 10th frame to keep data size manageable
    )


if __name__ == "__main__":
    main()
