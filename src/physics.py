import jax
import jax.numpy as jnp
from jax import random
from src.state import ParticleState
from src.config import SimConfig, ForceConfig
from src.flow import FlowFunc
from typing import Tuple


def material_derivative(
    position: jnp.ndarray, config: SimConfig, flow_func: FlowFunc
) -> jnp.ndarray:
    """Computes (u . grad) u"""
    velocity_func = lambda p: flow_func(p, config)
    jacobian_matrix = jax.jacobian(velocity_func)(position)
    velocity_vector = velocity_func(position)
    return jacobian_matrix @ velocity_vector


def get_turbulent_velocity(
    mean_vel: jnp.ndarray, key: jax.Array, config: SimConfig
) -> jnp.ndarray:
    if not config.enable_turbulence:
        return mean_vel
    u_mag = jnp.linalg.norm(mean_vel)
    k_val = config.turbulence_intensity * u_mag
    sigma = jnp.sqrt((2.0 / 3.0) * k_val)
    noise = random.normal(key, shape=mean_vel.shape) * sigma
    return mean_vel + noise


def get_fluid_temperature(position: jnp.ndarray, config: SimConfig) -> float:
    """
    Calculates fluid temperature, for the matrix fluid, based on proximity to a
    heated wall.
    T(x) = T_wall - gradient * dist
    """
    dist = config.wall_x - position[0]
    dist = jnp.maximum(dist, 0.0)
    return config.T_wall - config.T_gradient_slope * dist


def gravity_force(config: SimConfig, current_mass: float) -> jnp.ndarray:
    return jnp.array([0.0, current_mass * config.g])


def undisturbed_flow_force(
    state: ParticleState, config: SimConfig, flow_func: FlowFunc, current_d: float
) -> jnp.ndarray:
    """
    F_undisturbed = m_fluid * Du/Dt
    Also includes buoyancy (Archimedes): - m_fluid * g
    """
    m_fluid_curr = (jnp.pi * current_d**3 / 6) * config.rho_fluid
    fluid_accel = material_derivative(state.position, config, flow_func)
    buoyancy = -m_fluid_curr * jnp.array([0.0, config.g])
    return m_fluid_curr * fluid_accel + buoyancy


def drag_force(
    state: ParticleState, u_effective: jnp.ndarray, config: SimConfig, current_d: float
) -> jnp.ndarray:
    """Stokes Drag: F = 3 * pi * mu * d * (u_fluid - v_particle)"""
    drag_coeff = 3 * jnp.pi * config.mu_fluid * current_d
    rel_vel = state.velocity - u_effective
    # Force acts opposite to relative velocity (v_part - u_fluid)
    # Standard formula: F_drag = 0.5 * rho * A * Cd * |v-u|(u-v).
    # Stokes: F_drag = 3 pi mu d (u - v)
    return drag_coeff * (u_effective - state.velocity)


def total_force(
    state: ParticleState,
    config: SimConfig,
    force_config: ForceConfig,
    flow_func: FlowFunc,
    rng_key: jax.Array,
    current_d: float,
    current_mass: float,
) -> jnp.ndarray:
    u_mean = flow_func(state.position, config)
    u_eff = get_turbulent_velocity(u_mean, rng_key, config)
    tf = jnp.zeros(2)
    if force_config.gravity:
        tf += gravity_force(config, current_mass)
    if force_config.undisturbed_flow:
        tf += undisturbed_flow_force(state, config, flow_func, current_d)
    if force_config.drag:
        tf += drag_force(state, u_eff, config, current_d)
    return tf, u_eff


def calculate_rates(
    state: ParticleState, u_effective: jnp.ndarray, config: SimConfig, current_d: float
) -> Tuple[float, float]:
    """
    Calculates dm/dt (evaporation) and dT/dt (heat transfer).
    """
    # Safety: If temperature is near 0, assume thermal/mass transfer is disabled
    # or uninitialized to avoid numerical instability
    is_active = state.temperature > 1.0

    T_fluid = get_fluid_temperature(state.position, config)
    T_part = state.temperature
    rel_vel = state.velocity - u_effective
    rel_speed = jnp.linalg.norm(rel_vel)
    reynolds = (config.rho_fluid * rel_speed * current_d) / config.mu_fluid
    prandtl = config.get_prandtl_number()

    # Ranz-Marshall for Nusselt
    nu = 2.0 + 0.6 * jnp.sqrt(reynolds) * jnp.cbrt(prandtl)
    # Q_dot = Nu * pi * d * k_f * (T_f - T_p)
    q_conv = nu * jnp.pi * current_d * config.k_fluid * (T_fluid - T_part)

    # Phase Change
    # Schmidt Number
    D_AB = 2.6e-5
    sc = config.mu_fluid / (config.rho_fluid * D_AB)
    # Sherwood Number (analogous to Nusselt for mass transfer)
    sh = 2.0 + 0.6 * jnp.sqrt(reynolds) * jnp.cbrt(sc)

    # Saturation Pressure (Magnus formula) - only safe for T > -40C roughly
    # We clip T to a safe range for the formula even if is_active is false to
    # avoid NaNs in the branch
    # Assume below -73C is not relevant for our scenario and can cause numerical issues
    T_safe = jnp.maximum(T_part, 200.0)
    T_cel = T_safe - 273.15
    p_sat = 610.78 * jnp.exp((17.27 * T_cel) / (T_cel + 237.3))

    M_WATER = 18.015e-3
    M_AIR = 28.97e-3
    P_ATM = 101325.0
    L_VAP = 2.26e6

    # Mass Fraction
    omega_surf = (M_WATER / M_AIR) * (p_sat / P_ATM)

    T_room_cel = config.T_room_ref - 273.15
    p_sat_room = 610.78 * jnp.exp((17.27 * T_room_cel) / (T_room_cel + 237.3))
    omega_inf = config.RH_room * (M_WATER / M_AIR) * (p_sat_room / P_ATM)

    # dm/tt = Sh * pi * d * rho_fluid * D_AB * (omega_inf - omega_surf)
    dm_dt_calc = (
        sh * jnp.pi * current_d * config.rho_fluid * D_AB * (omega_inf - omega_surf)
    )

    q_total = q_conv + dm_dt_calc * L_VAP
    dT_dt_calc = q_total / (state.mass * config.cp_particle)

    is_too_small = state.mass < (config.m_particle_init * config.evap_cutoff_ratio**3)

    # Drop inactive particles or those that are too small to avoid numerical issues
    dm_dt = jnp.where(is_active & ~is_too_small, dm_dt_calc, 0.0)
    dT_dt = jnp.where(is_active & ~is_too_small, dT_dt_calc, 0.0)

    return dm_dt, dT_dt
