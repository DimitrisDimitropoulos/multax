import jax.numpy as jnp
from src.config import SimConfig
from typing import Callable

FlowFunc = Callable[[jnp.ndarray, SimConfig], jnp.ndarray]


def flow_cellular(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """Cellular flow field: u = U0 cos(x/a)cos(y/a), v = U0 sin(x/a)sin(y/a)"""
    x, y = position
    ux = config.U_0 * jnp.cos(x / config.alpha) * jnp.cos(y / config.alpha)
    uy = config.U_0 * jnp.sin(x / config.alpha) * jnp.sin(y / config.alpha)
    return jnp.array([ux, uy])


def flow_cylinder_potential(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """Potential flow around a cylinder centered at (0,0)."""
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
    """Stagnation flow impinging on a wall at x = wall_x."""
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
