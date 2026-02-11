import jax.numpy as jnp
from dataclasses import dataclass
import jax


@dataclass
@jax.tree_util.register_pytree_node_class
class ParticleState:
    """
    Represents the state of a system of particles.

    Attributes:
        position (jnp.ndarray): Shape (N, 2)
        velocity (jnp.ndarray): Shape (N, 2)
        temperature (jnp.ndarray): Shape (N,). Optional (zeros if unused).
        mass (jnp.ndarray): Shape (N,). Optional (const/zeros if unused).
    """

    position: jnp.ndarray
    velocity: jnp.ndarray
    temperature: jnp.ndarray = None
    mass: jnp.ndarray = None
    active: jnp.ndarray = None  # Boolean mask (1=active, 0=inactive)

    def __post_init__(self):
        # Ensure JAX arrays
        if self.temperature is None and self.position is not None:
            self.temperature = jnp.zeros(self.position.shape[0])
        if self.mass is None and self.position is not None:
            self.mass = jnp.ones(self.position.shape[0])
        if self.active is None and self.position is not None:
            self.active = jnp.ones(self.position.shape[0], dtype=bool)

    def tree_flatten(self):
        children = (
            self.position,
            self.velocity,
            self.temperature,
            self.mass,
            self.active,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
