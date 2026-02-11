import src.patch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from src.config import SimConfig, ForceConfig
from src.state import ParticleState
from src.flow import flow_cellular
from src.boundary import BoundaryManager
from src.solver import run_simulation_euler
from src.visualizer import Visualizer


def main():
    # Fix alpha (domain size) and derive U_0 based on Maxey parameters.
    config = SimConfig.from_maxey(
        W=0.5,  # Settling velocity ratio
        A=0.5,  # Inertia parameter
        alpha=1.0,  # Fixed flow scale (Domain size controls this)
        rho_particle=2650.0,  # Sand
        rho_fluid=1.225,  # Air
        mu_fluid=1.81e-5,  # Air
        g=-9.81,
        # Additional flags
        enable_turbulence=False,
        turbulence_intensity=0.2,
    )

    print("Maxey Config Generated:")
    print(f"  Alpha (Fixed): {config.alpha:.4f} m")
    print(f"  U_0 (Derived): {config.U_0:.4f} m/s")
    print(f"  d_p (Derived): {config.d_particle:.6e} m")
    print(f"  Stk (Calc):    {config.get_stokes_number():.4f}")

    force_config = ForceConfig(gravity=True, undisturbed_flow=True, drag=True)

    # Domain & Boundaries
    L = 4 * np.pi * config.alpha
    bounds = BoundaryManager(x_bounds=(0.0, L), y_bounds=(0.0, L), periodic=True)

    # Initialization
    n_particles = 10000
    grid_side = int(np.sqrt(n_particles))
    gx = np.linspace(0.1, L - 0.1, grid_side)
    gy = np.linspace(0.1, L - 0.1, grid_side)
    mx, my = np.meshgrid(gx, gy)
    pos = jnp.array(np.stack([mx.ravel(), my.ravel()], axis=1))

    # Recalculate actual n_particles (in case of rounding)
    actual_n = pos.shape[0]
    print(f"Initializing {actual_n} particles on a {grid_side}x{grid_side} grid...")

    vel = jax.vmap(lambda p: flow_cellular(p, config))(pos)

    # Initialize Mass
    m_p = config.m_particle_init
    mass = jnp.full((actual_n,), m_p)

    initial_state = ParticleState(position=pos, velocity=vel, mass=mass)

    t_end = 10.0
    dt = 0.005
    t_eval = jnp.array(np.arange(0.0, t_end, dt))
    key = jnp.array([0, 0], dtype=jnp.uint32)
    # Print the stokes number
    stokes = config.get_stokes_number()
    print(f"Calculated Stokes Number: {stokes:.4f}")
    print("Running Cellular Flow Simulation...")
    history = run_simulation_euler(
        initial_state, t_eval, config, force_config, bounds, flow_cellular, key
    )

    print("Generating Video (JAX Rasterizer)...")
    from src.jax_visualizer import JAXVisualizer

    flat_bounds = (0.0, L, 0.0, L)
    viz = JAXVisualizer(config, history, t_eval, flow_cellular)
    viz.generate_video(
        "cellular_flow.mp4",
        bounds=flat_bounds,
        width=600,
        height=600,
        fps=20,
        slow_mo_factor=1.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
