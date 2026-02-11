import jax
import jax.numpy as jnp
from dataclasses import dataclass
from src.state import ParticleState
from src.config import SimConfig
from typing import Tuple


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoundaryManager:
    """
    Manages boundary conditions for the simulation.
    """

    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]
    periodic: bool = False
    cylinder_collision: bool = False
    wall_collision: bool = False

    def tree_flatten(self):
        return (), self.__dict__

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def apply(self, state: ParticleState, config: SimConfig) -> ParticleState:
        pos = state.position
        vel = state.velocity

        # We need to handle current radius if mass is changing
        # If mass is constant, use config.r_particle
        # If mass is varying, calculate radius

        # Check if mass is effectively constant (all equal to
        # config.m_particle_init)
        # However, strictly for collision logic, we usually assume the
        # 'current' size matters. Let's calculate current radius per particle.
        # r = (3m / 4pi rho)^(1/3) = (6m / pi rho)^(1/3) / 2

        safe_mass = jnp.maximum(state.mass, 1e-16)
        current_d = jnp.cbrt((6.0 * safe_mass) / (jnp.pi * config.rho_particle))
        current_r = current_d / 2.0

        # Periodic Boundary
        if self.periodic:
            x_min, x_max = self.x_bounds
            y_min, y_max = self.y_bounds
            width = x_max - x_min
            height = y_max - y_min

            new_x = (pos[..., 0] - x_min) % width + x_min
            new_y = (pos[..., 1] - y_min) % height + y_min
            pos = jnp.stack([new_x, new_y], axis=-1)

        # Elastic Collision with Cylinder
        if self.cylinder_collision:
            # Distance from origin
            dist = jnp.linalg.norm(pos, axis=-1)
            # Simple scalar radius sum
            # Note: This logic assumes particles are outside the cylinder.
            collision_dist = config.R_cylinder + current_r
            is_collision = dist < collision_dist
            # Normal vector at collision point (pointing outwards from cylinder)
            normal = pos / (dist[..., None] + 1e-10)  # Do not divide by zero
            # Reflect Velocity: v_new = v - 2(v dot n)n
            v_dot_n = jnp.sum(
                vel * normal, axis=-1, keepdims=True
            )  # To leave reduced axis in the res
            vel_reflected = vel - 2 * v_dot_n * normal
            # Project Position
            pos_projected = normal * collision_dist[..., None]
            # Apply ONLY where collision occurred
            # Expand dims for scalar boolean mask
            mask = is_collision[..., None]
            pos = jnp.where(mask, pos_projected, pos)
            vel = jnp.where(mask, vel_reflected, vel)

        # Elastic Collision with Wall
        if self.wall_collision:
            # Wall at x = config.wall_x
            # Particle is to the left (x < wall_x)
            # Collision if x + r > wall_x
            penetration = (pos[..., 0] + current_r) - config.wall_x
            is_collision = penetration > 0
            # Reflect Velocity (Elastic X reflection)
            vel_x_new = -vel[..., 0]
            vel_new = jnp.stack([vel_x_new, vel[..., 1]], axis=-1)
            # Project Position
            pos_x_new = config.wall_x - current_r
            pos_new = jnp.stack([pos_x_new, pos[..., 1]], axis=-1)
            mask = is_collision[..., None]
            pos = jnp.where(mask, pos_new, pos)
            vel = jnp.where(mask, vel_new, vel)

        return ParticleState(
            position=pos,
            velocity=vel,
            temperature=state.temperature,
            mass=state.mass,
            active=state.active,
        )
