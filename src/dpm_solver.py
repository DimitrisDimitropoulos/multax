import jax
from jax import lax
import jax.numpy as jnp
from typing import Tuple

from src.state import ParticleState
from src.parcel import ParcelState, DPMConfig
from src.grid import DPMGrid
from src.solver import run_simulation_euler
from src.config import ForceConfig
from src.boundary import BoundaryManager
from src.flow import FlowFunc, TempFunc


def run_dpm_simulation(
    initial_parcels: ParcelState,
    t_eval: jnp.ndarray,
    config: DPMConfig,
    force_config: ForceConfig,
    boundary_manager: BoundaryManager,
    flow_func: FlowFunc,
    temp_func: TempFunc,
    master_rng_key: jax.Array,
    grid_resolution: Tuple[int, int] = (100, 100),
    grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
) -> Tuple[ParticleState, DPMGrid]:
    """
    Runs a steady-state Discrete Parcel Method simulation.

    1. Evolves the parcel trajectories using the unsteady solver.
    2. Accumulates statistics onto an Eulerian grid based on parcel history.

    Args:
        initial_parcels: Initial state of the parcels (ParcelState).
        t_eval: Time evaluation points.
        config: DPM Configuration.
        ...

    Returns:
        history: The trajectory history (ParticleState).
        grid: The populated DPMGrid with statistics.
    """

    # Run Trajectory Simulation
    # Note: run_simulation_euler returns ParticleState history.
    # It downgrades ParcelState to ParticleState in the loop, which is fine
    # because flow_rate is constant.
    # NOTE: We must explicitly downcast the input to ParticleState so that
    # lax.scan's input and output carry types match (ParticleState -> ParticleState).

    initial_base_state = ParticleState(
        position=initial_parcels.position,
        velocity=initial_parcels.velocity,
        temperature=initial_parcels.temperature,
        mass=initial_parcels.mass,
        active=initial_parcels.active,
    )

    print("  > Computing Trajectories...")
    history = run_simulation_euler(
        initial_base_state,
        t_eval,
        config,
        force_config,
        boundary_manager,
        flow_func,
        temp_func,
        master_rng_key,
    )

    # Grid
    if grid_bounds is None:
        # Auto-detect bounds from boundary manager if possible, or config
        # Default to Config Alpha or provided bounds
        L = config.alpha * 10  # heuristic fallback
        x_b = boundary_manager.x_bounds if boundary_manager.x_bounds else (0.0, L)
        y_b = boundary_manager.y_bounds if boundary_manager.y_bounds else (0.0, L)
    else:
        x_b, y_b = grid_bounds

    nx, ny = grid_resolution
    initial_grid = DPMGrid.create(x_b, y_b, nx, ny)

    # Accumulate Statistics
    dt = t_eval[1] - t_eval[0]

    # The solver computes diameter internally but doesn't store it in state, only mass.
    # We can recompute diameter from mass history.
    def get_diameter(mass):
        # d = (6m / (pi * rho))^(1/3)
        return jnp.cbrt((6.0 * mass) / (jnp.pi * config.rho_particle))

    # Vectorized accumulation function
    def accumulate_step(carry_grid, i):
        # Extract state at time step i
        # history is a PyTree with leading axis time (T, N, ...)
        step_state = jax.tree_util.tree_map(lambda x: x[i], history)
        step_diam = get_diameter(step_state.mass)

        # Calculate Evaporation Rate (dm/dt)
        # (mass[i] - mass[i-1]) / dt (Backward)
        prev_mass = jax.lax.dynamic_index_in_dim(
            history.mass, jnp.maximum(0, i - 1), axis=0, keepdims=False
        )
        dm_dt = (step_state.mass - prev_mass) / dt
        # If i=0, dm_dt is 0.
        dm_dt = jnp.where(i == 0, 0.0, dm_dt)  # kg/s

        # Use initial flow rates as weights
        # initial_parcels.flow_rate has shape (N,)

        new_grid = carry_grid.accumulate(
            position=step_state.position,
            diameter=step_diam,
            temperature=step_state.temperature,
            velocity=step_state.velocity,
            evap_rate=dm_dt,
            active=step_state.active,
            dt=dt,
            weights=initial_parcels.flow_rate,
        )
        return new_grid, None

    print("  > Accumulating Grid Statistics...")
    final_grid, _ = lax.scan(accumulate_step, initial_grid, jnp.arange(len(t_eval)))

    return history, final_grid
