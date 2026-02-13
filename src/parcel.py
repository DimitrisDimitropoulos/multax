import jax.numpy as jnp
from dataclasses import dataclass
import jax
from src.state import ParticleState
from src.config import SimConfig


@dataclass
@jax.tree_util.register_pytree_node_class
class ParcelState(ParticleState):
    """
    Represents a computational parcel (collection of particles).
    Inherits from ParticleState.

    New Attributes:
        flow_rate (jnp.ndarray): Number of real particles per second this parcel represents [1/s].
                                 Shape (N,).
        stream_id (jnp.ndarray): Identifier for the injection stream. Shape (N,).
    """

    flow_rate: jnp.ndarray = None
    stream_id: jnp.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        if self.flow_rate is None and self.position is not None:
            self.flow_rate = jnp.ones(self.position.shape[0])
        if self.stream_id is None and self.position is not None:
            self.stream_id = jnp.zeros(self.position.shape[0], dtype=jnp.int32)

    def tree_flatten(self):
        # In order to platten the tree we need to return the fields in the
        # correct order manually
        children = (
            self.position,
            self.velocity,
            self.temperature,
            self.mass,
            self.active,
            self.flow_rate,
            self.stream_id,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Unpack in order
        return cls(
            position=children[0],
            velocity=children[1],
            temperature=children[2],
            mass=children[3],
            active=children[4],
            flow_rate=children[5],
            stream_id=children[6],
        )


@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class DPMConfig(SimConfig):
    """
    Extended configuration for Discrete Parcel Method.

    Attributes:
        total_mass_flow_rate (float): Total mass flow rate of dispersed phase to inject [kg/s].
        n_streams (int): Number of discrete streams (parcels) to approximate the flow.
    """

    total_mass_flow_rate: float = 0.01
    n_streams: int = 100

    @property
    def mass_flow_per_stream(self) -> float:
        """Mass flow rate assigned to each stream [kg/s]."""
        return self.total_mass_flow_rate / self.n_streams

    @property
    def particles_per_second_per_stream(self) -> float:
        """Number of real particles per second per stream [1/s]."""
        # dot_N = dot_m / m_particle
        return self.mass_flow_per_stream / self.m_particle_init

    # We need to re-implement tree_flatten/unflatten because dataclass inheritance
    # with custom flattening can be tricky if the base class hardcodes fields.

    def tree_flatten(self):
        return (), tuple(sorted(self.__dict__.items()))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(aux_data))
