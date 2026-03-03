import os
import jax
import jax.numpy as jnp
import numpy as np

from src.parcel import ParcelState, DPMConfig
from src.boundary import BoundaryManager
from src.config import ForceConfig
from src.dpm_solver import run_dpm_simulation
from src.vtu_exporter import export_dpm_grid_vti, export_simulation_to_vtu


def flow_3d_chamber(position: jnp.ndarray, config: DPMConfig) -> jnp.ndarray:
    """
    Computes a simple 3D uniform flow in the X direction through a chamber.

    Args:
        position: Array of particle positions, shape (3, N).
        config: DPM configuration object containing flow parameters.

    Returns:
        jnp.ndarray: Array of fluid velocities at particle positions,
                     shape (3, N).
    """
    vel_x = jnp.full_like(position[0], config.U_0)
    vel_y = jnp.zeros_like(position[0])
    vel_z = jnp.zeros_like(position[0])
    return jnp.array([vel_x, vel_y, vel_z])


def temp_3d_chamber(position: jnp.ndarray, config: DPMConfig) -> jnp.ndarray:
    """
    Computes a chamber temperature with spatial gradients in X and Z directions.

    The chamber starts at ambient temperature and gets warmer progressively
    downstream (X) and outwards from the center (Z) to delay boiling.

    Args:
        position: Array of particle positions, shape (3, N).
        config: DPM configuration object containing temperature parameters.

    Returns:
        jnp.ndarray: Array of fluid temperatures at particle positions,
                     shape (N,).
    """
    x = position[0]
    z = position[2]

    # Reference temperatures
    t_ambient = 293.15
    t_hot = config.T_room_ref

    # Base ramp in the X direction (from 0.5m to 2.5m downstream)
    ramp_x = jnp.clip((x - 0.5) / 2.0, 0.0, 1.0)

    # Secondary ramp in the Z direction (outwards from center 0 to +/- 5m)
    ramp_z = jnp.clip(jnp.abs(z) / 5.0, 0.0, 1.0)

    # Combine ramps to create a 3D gradient effect
    ramp_combined = jnp.clip(ramp_x + 0.5 * ramp_z, 0.0, 1.0)

    return t_ambient + ramp_combined * (t_hot - t_ambient)


def main() -> None:
    # Initialize Configuration for 3D Mist Spray

    config = DPMConfig(
        dim=3,
        d_particle=80e-6,
        rho_particle=1000.0,
        rho_fluid=1.225,
        mu_fluid=1.81e-5,
        U_0=20.0,
        alpha=1.0,
        g=-9.81,
        cp_particle=4184.0,
        cp_fluid=1005.0,
        k_fluid=0.026,
        wall_x=200.0,
        T_wall=273.15 + 130.0,
        T_gradient_slope=2.0,
        RH_room=0.5,
        T_room_ref=273.15 + 150.0,
        enable_turbulence=True,
        turbulence_intensity=0.15,
        evap_cutoff_ratio=0.1,
        total_mass_flow_rate=1e-3,
        n_streams=1000,
    )
    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Set up 3D Domain boundaries (extend X to prevent bouncing backwards over 3 seconds!)
    x_lim = (0.0, 30.0)
    y_lim = (-5.0, 5.0)
    z_lim = (-5.0, 5.0)
    bounds = BoundaryManager(
        x_bounds=x_lim,
        y_bounds=y_lim,
        z_bounds=z_lim,
        wall_collision=False,
        periodic=False,
    )

    # Initialize conical spray nozzle injection
    n_streams = config.n_streams
    np.random.seed(42)

    # Inject at origin (0, 0, 0)
    pos_x = np.zeros(n_streams)
    pos_y = np.zeros(n_streams)
    pos_z = np.zeros(n_streams)
    pos = jnp.array(np.stack([pos_x, pos_y, pos_z], axis=1))

    # Conical spray velocities
    spray_velocity_magnitude = 9.0
    cone_angle = np.pi / 6  # 30 degrees half-angle

    angles_theta = np.random.uniform(0, 2 * np.pi, n_streams)
    # Cosine distribution for solid angle
    angles_phi = np.arccos(np.random.uniform(np.cos(cone_angle), 1.0, n_streams))

    vel_x = spray_velocity_magnitude * np.cos(angles_phi)
    vel_y = spray_velocity_magnitude * np.sin(angles_phi) * np.cos(angles_theta)
    vel_z = spray_velocity_magnitude * np.sin(angles_phi) * np.sin(angles_theta)

    vel = jnp.array(np.stack([vel_x, vel_y, vel_z], axis=1))

    # Fluid temperature initially
    temp = jnp.full((n_streams,), 293.15)  # Injected cold

    m_p_init = config.m_particle_init
    mass = jnp.full((n_streams,), m_p_init)
    active = jnp.ones((n_streams,), dtype=bool)

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

    # Execute Engine
    t_end = 2  # Allow more time for slow mist
    dt = 0.00005  # oarse for speed, DPM tracks steady lines
    t_eval = jnp.array(np.arange(0.0, t_end, dt))
    key = jnp.array([101, 202], dtype=jnp.uint32)

    print(f"Running Steady-State 3D DPM Simulation with {n_streams} streams...")
    history, grid_3d = run_dpm_simulation(
        initial_parcels,
        t_eval,
        config,
        force_config,
        bounds,
        flow_3d_chamber,
        temp_3d_chamber,
        key,
        grid_resolution=(100, 50, 50),
        grid_bounds=(x_lim, y_lim, z_lim),
    )
    print("Simulation complete.")

    output_dir = "vtu_output_dpm_3d"
    os.makedirs(output_dir, exist_ok=True)

    # Extract to Paraview
    grid_path = os.path.join(output_dir, "dpm_grid.vti")
    export_dpm_grid_vti(grid_path, grid_3d)

    # Export Sequence Animation
    anim_dir = os.path.join(output_dir, "animation")
    export_simulation_to_vtu(
        output_dir=anim_dir,
        history=history,
        config=config,
        t_eval=np.array(t_eval),
        flow_func=flow_3d_chamber,
        temp_func=temp_3d_chamber,
        stride=150,  # Save every 5 frames for manageable sequence
        bounds=(x_lim, y_lim, z_lim),
    )

    print(f"3D DPM Outputs saved to '{output_dir}/'")


if __name__ == "__main__":
    main()
