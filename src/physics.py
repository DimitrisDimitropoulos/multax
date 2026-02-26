import jax
import jax.numpy as jnp
from jax import random
from src.state import ParticleState
from src.config import PhysicsConfig, ForceConfig
from src.flow import FlowFunc, TempFunc
from typing import Tuple


def material_derivative(
    position: jnp.ndarray, config: PhysicsConfig, flow_func: FlowFunc
) -> jnp.ndarray:
    r"""Computes the material derivative of the flow velocity at a given position.

    Calculates :math:`\frac{D\mathbf{u}}{Dt} = (\mathbf{u} \cdot \nabla) \mathbf{u}`.
    Note that the time-dependent term :math:`\frac{\partial \mathbf{u}}{\partial t}` is
    assumed to be zero (steady flow).

    Args:
        position (jnp.ndarray): Position vector :math:`\mathbf{x}`. Units: [m].
        config (PhysicsConfig): Simulation configuration.
        flow_func (FlowFunc): Function returning flow velocity field.

    Returns:
        jnp.ndarray: Acceleration vector. Units: [m/s^2].
    """
    velocity_func = lambda p: flow_func(p, config)
    jacobian_matrix = jax.jacobian(velocity_func)(position)
    velocity_vector = velocity_func(position)
    return jacobian_matrix @ velocity_vector


def get_turbulent_velocity(
    mean_vel: jnp.ndarray, key: jax.Array, config: PhysicsConfig
) -> jnp.ndarray:
    r"""Calculates the effective velocity including a stochastic turbulent component.

    Adds a fluctuating component :math:`\mathbf{u}'` to the mean velocity :math:`\mathbf{u}`.
    The fluctuation is modeled as Gaussian noise with standard deviation :math:`\sigma` derived
    from the turbulent kinetic energy :math:`k`.

    .. math::
        \mathbf{u}_{eff} = \mathbf{u} + \mathbf{u}' \\
        \sigma = \sqrt{\frac{2}{3} k}, \quad k \approx I \cdot |\mathbf{u}|

    Args:
        mean_vel (jnp.ndarray): Mean flow velocity vector. Units: [m/s].
        key (jax.Array): PRNG key for stochastic generation.
        config (PhysicsConfig): Simulation configuration.

    Returns:
        jnp.ndarray: Effective velocity vector. Units: [m/s].
    """
    if not config.enable_turbulence:
        return mean_vel
    u_mag = jnp.linalg.norm(mean_vel)
    k_val = (3 / 2) * (config.turbulence_intensity * u_mag) ** (2)
    sigma = jnp.sqrt((2.0 / 3.0) * k_val)
    noise = random.normal(key, shape=mean_vel.shape) * sigma
    return mean_vel + noise


def gravity_force(config: PhysicsConfig, current_mass: float) -> jnp.ndarray:
    r"""Calculates the gravitational force acting on the particle.

    .. math::
        \mathbf{F}_g = m_p \mathbf{g}

    Args:
        config (PhysicsConfig): Simulation configuration.
        current_mass (float): Current mass of the particle. Units: [kg].

    Returns:
        jnp.ndarray: Gravitational force vector. Units: [N].
    """
    return jnp.array([0.0, current_mass * config.g])


def undisturbed_flow_force(
    state: ParticleState, config: PhysicsConfig, flow_func: FlowFunc, current_d: float
) -> jnp.ndarray:
    r"""Calculates the force due to the undisturbed flow (pressure gradient + buoyancy).

    .. math::
        \mathbf{F}_{undisturbed} = m_f \frac{D\mathbf{u}}{Dt} - m_f \mathbf{g}

    Args:
        state (ParticleState): Current state of the particle.
        config (PhysicsConfig): Simulation configuration.
        flow_func (FlowFunc): Function defining the flow field.
        current_d (float): Current particle diameter. Units: [m].

    Returns:
        jnp.ndarray: Undisturbed flow force vector. Units: [N].
    """
    m_fluid_curr = (jnp.pi * current_d**3 / 6) * config.rho_fluid
    fluid_accel = material_derivative(state.position, config, flow_func)
    buoyancy = -m_fluid_curr * jnp.array([0.0, config.g])
    return m_fluid_curr * fluid_accel + buoyancy


def drag_force(
    state: ParticleState, u_effective: jnp.ndarray, config: PhysicsConfig, current_d: float
) -> jnp.ndarray:
    r"""Calculates the Stokes drag force acting on the particle.

    .. math::
        \mathbf{F}_{drag} = 3 \pi \mu d_p (\mathbf{u}_{eff} - \mathbf{v}_p)

    Args:
        state (ParticleState): Current state of the particle.
        u_effective (jnp.ndarray): Effective fluid velocity (mean + turbulence). Units: [m/s].
        config (PhysicsConfig): Simulation configuration.
        current_d (float): Current particle diameter. Units: [m].

    Returns:
        jnp.ndarray: Drag force vector. Units: [N].
    """
    drag_coeff = 3 * jnp.pi * config.mu_fluid * current_d
    rel_vel = state.velocity - u_effective
    # Force acts opposite to relative velocity (v_part - u_fluid)
    # Standard formula: F_drag = 0.5 * rho * A * Cd * |v-u|(u-v).
    # Stokes: F_drag = 3 pi mu d (u - v)
    return drag_coeff * (u_effective - state.velocity)


def total_force(
    state: ParticleState,
    config: PhysicsConfig,
    force_config: ForceConfig,
    flow_func: FlowFunc,
    rng_key: jax.Array,
    current_d: float,
    current_mass: float,
) -> jnp.ndarray:
    r"""Calculates the total force acting on the particle.

    Sums the enabled forces (gravity, undisturbed flow, drag).

    Args:
        state (ParticleState): Current state of the particle.
        config (PhysicsConfig): Simulation configuration.
        force_config (ForceConfig): Configuration enabling/disabling specific forces.
        flow_func (FlowFunc): Function defining the flow field.
        rng_key (jax.Array): PRNG key for turbulence.
        current_d (float): Current particle diameter. Units: [m].
        current_mass (float): Current particle mass. Units: [kg].

    Returns:
        jnp.ndarray: Total force vector. Units: [N].
    """
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
    state: ParticleState,
    u_effective: jnp.ndarray,
    config: PhysicsConfig,
    current_d: float,
    temp_func: TempFunc,
) -> Tuple[float, float]:
    r"""Calculates the rate of change of mass (evaporation) and temperature (heat transfer).

    Uses the Ranz-Marshall correlation for the Nusselt and Sherwood numbers.

    .. math::
        \frac{dm}{dt} = Sh \pi d_p \rho_f D_{AB} (\omega_{\infty} - \omega_{surf}) \\
        \frac{dT}{dt} = \frac{\dot{Q}_{conv} + \dot{m} L_{vap}}{m c_p}

    Args:
        state (ParticleState): Current state of the particle.
        u_effective (jnp.ndarray): Effective fluid velocity. Units: [m/s].
        config (PhysicsConfig): Simulation configuration.
        current_d (float): Current particle diameter. Units: [m].
        temp_func (TempFunc): Function returning the fluid temperature field.

    Returns:
        Tuple[float, float]: A tuple containing:
            - dm_dt: Rate of mass change. Units: [kg/s].
            - dT_dt: Rate of temperature change. Units: [K/s].
    """
    # Safety: If temperature is near 0, assume thermal/mass transfer is disabled
    # or uninitialized to avoid numerical instability
    is_active = state.temperature > 1.0

    T_fluid = temp_func(state.position, config)
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
    D_AB = config.D_ref
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

    # Mass Fraction
    omega_surf = (config.M_dispersed / config.M_continuous) * (p_sat / config.P_atm)

    T_room_cel = config.T_room_ref - 273.15
    p_sat_room = 610.78 * jnp.exp((17.27 * T_room_cel) / (T_room_cel + 237.3))
    omega_inf = (
        config.RH_room
        * (config.M_dispersed / config.M_continuous)
        * (p_sat_room / config.P_atm)
    )

    # dm/tt = Sh * pi * d * rho_fluid * D_AB * (omega_inf - omega_surf)
    dm_dt_calc = (
        sh * jnp.pi * current_d * config.rho_fluid * D_AB * (omega_inf - omega_surf)
    )

    q_total = q_conv + dm_dt_calc * config.latent_heat
    dT_dt_calc = q_total / (state.mass * config.cp_particle)

    is_too_small = state.mass < (config.m_particle_init * config.evap_cutoff_ratio**3)

    # Drop inactive particles or those that are too small to avoid numerical issues
    dm_dt = jnp.where(is_active & ~is_too_small, dm_dt_calc, 0.0)
    dT_dt = jnp.where(is_active & ~is_too_small, dT_dt_calc, 0.0)

    return dm_dt, dT_dt
