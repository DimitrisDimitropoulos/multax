import jax
import jax.numpy as jnp
import numpy as np
from src.parcel import ParcelState, DPMConfig
from src.grid import DPMGrid
from src.flow import flow_wall_stagnation
from src.boundary import BoundaryManager
from src.config import ForceConfig
from src.dpm_solver import run_dpm_simulation
from src.dpm_visualizer import DPMVisualizer


def main():
    config = DPMConfig(
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
        T_wall=273.15 + 45.0,
        T_gradient_slope=20.0,
        RH_room=0.5,
        T_room_ref=293.15,
        enable_turbulence=True,
        turbulence_intensity=0.2,
        evap_cutoff_ratio=0.1,
        # DPM Specifics
        total_mass_flow_rate=5e-4,  # kg/s (e.g. spray nozzle)
        n_streams=1000,  # Number of computational parcels
    )
    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    x_lim = (-3.0, 1.0)
    y_lim = (-2.0, 2.0)
    bounds = BoundaryManager(
        x_bounds=x_lim, y_bounds=y_lim, wall_collision=True, periodic=False
    )

    # Initialization (Parcel Injection)
    n_streams = config.n_streams
    # Vertical line upstream at x = -2.5
    pos_x = np.full((n_streams,), -2.5)
    pos_y = np.linspace(-0.5, 0.5, n_streams)
    pos = jnp.array(np.stack([pos_x, pos_y], axis=1))
    # Velocity: Equilibrium with flow
    vel = jax.vmap(lambda p: flow_wall_stagnation(p, config))(pos)
    # Temp: Room Temp
    temp = jnp.full((n_streams,), config.T_room_ref)
    # Mass: Based on d_particle
    m_p_init = config.m_particle_init
    mass = jnp.full((n_streams,), m_p_init)
    active = jnp.ones((n_streams,), dtype=bool)

    # Calculate Flow Rate per Parcel (Real Particles / sec)
    # Total Particles / sec = Total Mass Flow / Mass per Particle
    total_particles_per_sec = config.total_mass_flow_rate / m_p_init
    flow_rate_per_parcel = total_particles_per_sec / n_streams

    flow_rates = jnp.full((n_streams,), flow_rate_per_parcel)
    stream_ids = jnp.arange(n_streams, dtype=jnp.int32)

    initial_parcels = ParcelState(
        position=pos,
        velocity=vel,
        temperature=temp,
        mass=mass,
        active=active,
        flow_rate=flow_rates,
        stream_id=stream_ids,
    )

    # Simulation
    t_end = 5.0
    dt = 0.0001
    t_eval = jnp.array(np.arange(0.0, t_end, dt))
    key = jnp.array([101, 202], dtype=jnp.uint32)
    print(f"Running Steady-State DPM Simulation with {n_streams} streams...")
    print(f"Total Mass Flow: {config.total_mass_flow_rate} kg/s")
    history, grid = run_dpm_simulation(
        initial_parcels,
        t_eval,
        config,
        force_config,
        bounds,
        flow_wall_stagnation,
        key,
        grid_resolution=(200, 200),
        grid_bounds=(x_lim, y_lim),
    )
    print("Simulation complete.")

    # Visualization
    print("Plotting results...")
    viz = DPMVisualizer(grid, config, history=history, t_eval=t_eval)
    viz.plot_fields("dpm_wall_thermal.png")
    viz.plot_trajectories("dpm_trajectories.png")
    print("Done.")


if __name__ == "__main__":
    main()
