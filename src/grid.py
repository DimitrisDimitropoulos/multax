import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple


@dataclass
@jax.tree_util.register_pytree_node_class
class DPMGrid:
    """
    Represents the Eulerian grid for collecting DPM statistics.

    Attributes:
        x_min (float): Minimum X coordinate.
        x_max (float): Maximum X coordinate.
        y_min (float): Minimum Y coordinate.
        y_max (float): Maximum Y coordinate.
        nx (int): Number of cells in X direction.
        ny (int): Number of cells in Y direction.

        # Accumulated Fields
        residence_time (jnp.ndarray): Total time spent by all parcels in each cell [s].
                                      Shape: (nx, ny)
        weighted_diameter (jnp.ndarray): Accumulator for time-weighted diameter sum [m * s].
                                         Shape: (nx, ny)
        visit_count (jnp.ndarray): Number of discrete time steps recorded in each cell.
                                   Shape: (nx, ny)
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    # Buffers
    residence_time: jnp.ndarray
    weighted_diameter: jnp.ndarray
    weighted_temperature: jnp.ndarray
    weighted_velocity: jnp.ndarray  # Shape (nx, ny, 2)
    weighted_evap_rate: jnp.ndarray  # Shape (nx, ny)
    visit_count: jnp.ndarray

    @classmethod
    def create(
        cls,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        nx: int,
        ny: int,
    ):
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        shape = (nx, ny)
        return cls(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            nx=nx,
            ny=ny,
            residence_time=jnp.zeros(shape),
            weighted_diameter=jnp.zeros(shape),
            weighted_temperature=jnp.zeros(shape),
            weighted_velocity=jnp.zeros(shape + (2,)),
            weighted_evap_rate=jnp.zeros(shape),
            visit_count=jnp.zeros(shape, dtype=jnp.int32),
        )

    def get_cell_indices(
        self, position: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Maps positions (N, 2) to grid indices (N,).
        Clamps to boundary.
        """
        x = position[:, 0]
        y = position[:, 1]
        # Normalized coordinates [0, 1]
        u = (x - self.x_min) / (self.x_max - self.x_min)
        v = (y - self.y_min) / (self.y_max - self.y_min)
        # Map to indices
        ix = jnp.floor(u * self.nx).astype(jnp.int32)
        iy = jnp.floor(v * self.ny).astype(jnp.int32)
        # Clamp
        ix = jnp.clip(ix, 0, self.nx - 1)
        iy = jnp.clip(iy, 0, self.ny - 1)

        return ix, iy

    def accumulate(
        self,
        position: jnp.ndarray,
        diameter: jnp.ndarray,
        temperature: jnp.ndarray,
        velocity: jnp.ndarray,
        evap_rate: jnp.ndarray,
        active: jnp.ndarray,
        dt: float,
        weights: jnp.ndarray = None,
    ):
        """
        Updates the grid with a batch of particle states.

        Args:
            position: (N, 2) positions
            diameter: (N,) particle diameters
            temperature: (N,) particle temperature
            velocity: (N, 2) particle velocity
            evap_rate: (N,) dm/dt (should be positive for evaporation magnitude if preferred, or raw dm/dt)
            active: (N,) boolean mask
            dt: Time step duration
            weights: (N,) optional flow rate weights for each parcel.
        """
        ix, iy = self.get_cell_indices(position)

        if weights is None:
            weights = jnp.ones_like(diameter)

        # We only accumulate for active particles
        # Contribution to residence time = dt * weight
        time_contrib = jnp.where(active, dt * weights, 0.0)
        # Weighted contributions for time-averaged fields
        diam_contrib = diameter * time_contrib
        temp_contrib = temperature * time_contrib
        evap_contrib = evap_rate * time_contrib
        # Broadcasting time_contrib for velocity (N, 1) * (N, 2)
        vel_contrib = velocity * time_contrib[:, None]
        visit_contrib = jnp.where(active, 1, 0)
        new_res_time = self.residence_time.at[ix, iy].add(time_contrib)
        new_weight_diam = self.weighted_diameter.at[ix, iy].add(diam_contrib)
        new_weight_temp = self.weighted_temperature.at[ix, iy].add(temp_contrib)
        new_weight_vel = self.weighted_velocity.at[ix, iy].add(vel_contrib)
        new_weight_evap = self.weighted_evap_rate.at[ix, iy].add(evap_contrib)
        new_visits = self.visit_count.at[ix, iy].add(visit_contrib)

        return DPMGrid(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            nx=self.nx,
            ny=self.ny,
            residence_time=new_res_time,
            weighted_diameter=new_weight_diam,
            weighted_temperature=new_weight_temp,
            weighted_velocity=new_weight_vel,
            weighted_evap_rate=new_weight_evap,
            visit_count=new_visits,
        )

    def get_mean_diameter(self):
        """Returns the time-weighted mean diameter field."""
        safe_time = jnp.where(self.residence_time > 1e-10, self.residence_time, 1.0)
        return self.weighted_diameter / safe_time

    def get_mean_temperature(self):
        """Returns the time-weighted mean temperature field."""
        safe_time = jnp.where(self.residence_time > 1e-10, self.residence_time, 1.0)
        return self.weighted_temperature / safe_time

    def get_mean_velocity(self):
        """Returns the time-weighted mean velocity field (nx, ny, 2)."""
        safe_time = jnp.where(self.residence_time > 1e-10, self.residence_time, 1.0)
        return self.weighted_velocity / safe_time[..., None]

    def get_mean_evap_rate(self):
        """Returns the time-weighted mean evaporation rate [kg/s]."""
        safe_time = jnp.where(self.residence_time > 1e-10, self.residence_time, 1.0)
        return self.weighted_evap_rate / safe_time

    def tree_flatten(self):
        children = (
            self.residence_time,
            self.weighted_diameter,
            self.weighted_temperature,
            self.weighted_velocity,
            self.weighted_evap_rate,
            self.visit_count,
        )
        aux_data = (self.x_min, self.x_max, self.y_min, self.y_max, self.nx, self.ny)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            x_min=aux_data[0],
            x_max=aux_data[1],
            y_min=aux_data[2],
            y_max=aux_data[3],
            nx=aux_data[4],
            ny=aux_data[5],
            residence_time=children[0],
            weighted_diameter=children[1],
            weighted_temperature=children[2],
            weighted_velocity=children[3],
            weighted_evap_rate=children[4],
            visit_count=children[5],
        )
