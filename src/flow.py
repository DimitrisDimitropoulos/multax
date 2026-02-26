import jax.numpy as jnp
from src.config import SimConfig
from typing import Callable

FlowFunc = Callable[[jnp.ndarray, SimConfig], jnp.ndarray]
TempFunc = Callable[[jnp.ndarray, SimConfig], float]


def flow_cellular(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    r"""Computes velocity for a cellular flow field.

    .. math::
        u_x = U_0 \cos(x/\alpha) \cos(y/\alpha) \\
        u_y = U_0 \sin(x/\alpha) \sin(y/\alpha)

    Args:
        position (jnp.ndarray): Position vector :math:`(x, y)`. Units: [m].
        config (SimConfig): Simulation configuration.

    Returns:
        jnp.ndarray: Velocity vector :math:`(u_x, u_y)`. Units: [m/s].
    """
    x, y = position
    ux = config.U_0 * jnp.cos(x / config.alpha) * jnp.cos(y / config.alpha)
    uy = config.U_0 * jnp.sin(x / config.alpha) * jnp.sin(y / config.alpha)
    return jnp.array([ux, uy])


def flow_cylinder_potential(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    r"""Computes velocity for potential flow around a cylinder.

    Calculates the flow around a cylinder of radius :math:`R` centered at origin.

    Args:
        position (jnp.ndarray): Position vector :math:`(x, y)`. Units: [m].
        config (SimConfig): Simulation configuration.

    Returns:
        jnp.ndarray: Velocity vector :math:`(u_x, u_y)`. Units: [m/s].
    """
    x, y = position
    r2 = x**2 + y**2 + 1e-9
    r = jnp.sqrt(r2)
    r_safe = jnp.maximum(r, config.R_cylinder)
    r2_safe = r_safe**2
    R2 = config.R_cylinder**2
    factor = R2 / r2_safe
    ux = config.U_0 * (1 - factor * ((x**2 - y**2) / r2_safe))
    uy = config.U_0 * (-factor * ((2 * x * y) / r2_safe))
    return jnp.array([ux, uy])


def flow_wall_stagnation(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    r"""Computes velocity for stagnation flow impinging on a wall.

    .. math::
        u_x = A (x_{wall} - x) \\
        u_y = A y

    Where :math:`A = U_0 / \alpha` is the strain rate.

    Args:
        position (jnp.ndarray): Position vector :math:`(x, y)`. Units: [m].
        config (SimConfig): Simulation configuration.

    Returns:
        jnp.ndarray: Velocity vector :math:`(u_x, u_y)`. Units: [m/s].
    """
    x, y = position
    # A = U0 / alpha (Strain rate)
    A = config.U_0 / config.alpha
    # Flow comes from left to right, wall is at wall_x
    ux = A * (config.wall_x - x)
    vy = A * y
    return jnp.array([ux, vy])


FLOW_REGISTRY = {
    "cellular": flow_cellular,
    "cylinder": flow_cylinder_potential,
    "wall": flow_wall_stagnation,
}


def temp_constant(position: jnp.ndarray, config: SimConfig) -> float:
    r"""Computes a constant fluid temperature.

    Args:
        position (jnp.ndarray): Position vector :math:`(x, y)`. Units: [m].
        config (SimConfig): Simulation configuration.

    Returns:
        float: Constant fluid temperature. Units: [K].
    """
    return config.T_room_ref


def temp_wall_gradient(position: jnp.ndarray, config: SimConfig) -> float:
    r"""Calculates the fluid temperature at a specific position with a wall gradient.

    Assumes a linear temperature gradient from a heated wall.

    Args:
        position (jnp.ndarray): Particle position vector. Units: [m].
        config (SimConfig): Simulation configuration containing wall parameters.

    Returns:
        float: Fluid temperature. Units: [K].
    """
    dist = config.wall_x - position[0]
    dist = jnp.maximum(dist, 0.0)
    return config.T_wall - config.T_gradient_slope * dist


TEMP_REGISTRY = {
    "constant": temp_constant,
    "wall_gradient": temp_wall_gradient,
}
