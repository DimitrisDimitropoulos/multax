import jax
import jax.numpy as jnp
from jax import lax, random
from functools import partial
from typing import Tuple

from src.state import ParticleState
from src.config import PhysicsConfig, ForceConfig
from src.boundary import BoundaryManager
from src.flow import FlowFunc, TempFunc
from src.physics import total_force, calculate_rates
from src.collisions import resolve_collisions as jax_resolve_collisions
from src.warp_collisions import resolve_collisions as warp_resolve_collisions


def equations_of_motion(
    state: ParticleState,
    config: PhysicsConfig,
    force_config: ForceConfig,
    flow_func: FlowFunc,
    temp_func: TempFunc,
    rng_key: jax.Array,
) -> Tuple[jnp.ndarray, float, float]:
    r"""Computes the instantaneous derivatives for a single particle.

    Calculates acceleration, temperature rate of change, and mass rate of change based on
    active forces and thermodynamic models.

    .. math::
        \mathbf{a} = \frac{\sum \mathbf{F}}{m_{eff}}, \quad m_{eff} = m_p + \frac{1}{2}m_f

    Args:
        state (ParticleState): Current state of the particle.
        config (PhysicsConfig): Simulation configuration.
        force_config (ForceConfig): Active forces configuration.
        flow_func (FlowFunc): Flow field function.
        temp_func (TempFunc): Temperature field function.
        rng_key (jax.Array): PRNG key for stochastic processes.

    Returns:
        Tuple[jnp.ndarray, float, float]: A tuple containing:
            - Acceleration vector :math:`\mathbf{a}`. Units: [m/s^2].
            - Temperature rate :math:`dT/dt`. Units: [K/s].
            - Mass rate :math:`dm/dt`. Units: [kg/s].
    """
    # Use mass if available, else standard
    # Put a cap on mass to prevent division with zero
    safe_mass = jnp.maximum(state.mass, 1e-16)
    # If mass is essentially constant (default), this calculation is redundant but safe
    # Used for phase change scenarios where mass can vary significantly
    current_d = jnp.cbrt((6.0 * safe_mass) / (jnp.pi * config.rho_particle))
    force, u_eff = total_force(
        state, config, force_config, flow_func, rng_key, current_d, safe_mass
    )

    # Effective Inertial Mass (Particle + Added Mass)
    # The mass that the fluid would have if there was no particles there, the displaced fluid mass
    m_fluid_disp = (jnp.pi * current_d**3 / 6) * config.rho_fluid
    inertial_mass = safe_mass + 0.5 * m_fluid_disp
    accel = force / inertial_mass

    # Thermo eqs
    dm_dt, dT_dt = calculate_rates(state, u_eff, config, current_d, temp_func)

    return accel, dT_dt, dm_dt


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def run_simulation_euler(
    initial_state: ParticleState,
    t_eval: jnp.ndarray,
    config: PhysicsConfig,
    force_config: ForceConfig,
    boundary_manager: BoundaryManager,
    flow_func: FlowFunc,
    temp_func: TempFunc,
    master_rng_key: jax.Array,
) -> ParticleState:
    r"""Runs the full simulation using a Semi-Implicit Euler integrator.

    Orchestrates the time-stepping loop, applying physics, thermodynamics, and boundary conditions.
    Handles the lifecycle of particles (active vs. evaporated).

    Args:
        initial_state (ParticleState): Initial state of all particles.
        t_eval (jnp.ndarray): Array of time points to evaluate. Units: [s].
        config (PhysicsConfig): Simulation configuration.
        force_config (ForceConfig): Active forces configuration.
        boundary_manager (BoundaryManager): Boundary condition logic.
        flow_func (FlowFunc): Flow field function.
        temp_func (TempFunc): Temperature field function.
        master_rng_key (jax.Array): Master PRNG key.

    Returns:
        ParticleState: The full history of the simulation state (stacked along time axis).
    """
    dt = t_eval[1] - t_eval[0]

    def step_fn(carry, t):
        """
        Basically the core of the simulation loop, but vectorized over all
        particles and JIT-compiled.

        carry: (ParticleState, rng_key)
        returns: (new_carry, ParticleState)
        """
        state, key = carry
        step_key, next_key = random.split(key)
        # Split key for N particles
        particle_keys = random.split(step_key, state.position.shape[0])

        # Calculate Derivatives (Vectorized over particles)
        # vmap over (State, Keys) -> (Accel, dT, dm)
        # We need to reshape/structure the vmap correctly.
        # State is a Pytree of arrays (N, ...).

        def single_particle_deriv(s, k):
            return equations_of_motion(s, config, force_config, flow_func, temp_func, k)

        # vmap over state (0) and keys (0). config/force_config/flow_func are captured constants.
        accel, dT_dt, dm_dt = jax.vmap(single_particle_deriv, in_axes=(0, 0))(
            state, particle_keys
        )

        # Evaporation Cutoff, if the particle has lost 90% of its mass declare
        # it gone
        cutoff_mass = config.m_particle_init * (config.evap_cutoff_ratio**3)
        # Particle is active if it was active AND mass > cutoff
        # Active status/tag to prevent NaNs from particles that have evaporated away (mass ~ 0)
        # Also makes the visualizarer logic cleaner
        new_active = state.active & (state.mass > cutoff_mass)

        # Zero out updates for inactive particles to prevent NaNs
        accel = jnp.where(new_active[:, None], accel, 0.0)
        dT_dt = jnp.where(new_active, dT_dt, 0.0)
        dm_dt = jnp.where(new_active, dm_dt, 0.0)

        # Semi-implicit Euler for better coupling with stochastic noise
        # update velocity first, then position with new velocity
        new_vel = state.velocity + accel * dt
        new_pos = state.position + new_vel * dt
        new_temp = state.temperature + dT_dt * dt
        new_mass = state.mass + dm_dt * dt

        # Freeze position/velocity for inactive particles (keep old state or just stop updating)
        # We let them freeze in place to avoid them flying off with NaNs, if we make them zero or None
        new_pos = jnp.where(new_active[:, None], new_pos, state.position)
        new_vel = jnp.where(
            new_active[:, None], new_vel, jnp.zeros_like(state.velocity)
        )

        temp_state = ParticleState(new_pos, new_vel, new_temp, new_mass, new_active)

        # Resolve Particle-Particle Collisions
        if config.collision_engine == "warp":
            temp_state = warp_resolve_collisions(temp_state, config)
        else:
            temp_state = jax_resolve_collisions(temp_state, config)

        final_state = boundary_manager.apply(temp_state, config)
        return (final_state, next_key), final_state

    _, history = lax.scan(step_fn, (initial_state, master_rng_key), t_eval)
    return history
