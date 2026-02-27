import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from src.state import ParticleState
from src.config import SimConfig

wp.init()


@wp.kernel
def build_grid_kernel(
    positions: wp.array(dtype=wp.float32, ndim=2),
    cell_size: float,
    grid_dim: int,
    grid_head: wp.array(dtype=int),
    grid_next: wp.array(dtype=int),
):
    r"""Hashes particles into a spatial grid using atomic exchanges.

    Constructs a linked list for each spatial grid cell, allowing for efficient
    neighbor searches during collision resolution.

    .. math::
        c_x = \lfloor x / s \rfloor, \quad c_y = \lfloor y / s \rfloor \\
        h = (c_x \cdot 73856093 \oplus c_y \cdot 19349663) \pmod D

    Args:
        positions (wp.array): Particle position vectors :math:`\mathbf{x}`. Units: [m].
        cell_size (float): The physical size of each grid cell :math:`s`. Units: [m].
        grid_dim (int): The total number of buckets in the hash grid :math:`D`.
        grid_head (wp.array): Array storing the index of the first particle in each cell.
        grid_next (wp.array): Array storing the index of the next particle in the linked list.
    """
    tid = wp.tid()
    x = positions[tid, 0]
    y = positions[tid, 1]

    # Calculate cell coordinates
    cx = int(wp.floor(x / cell_size))
    cy = int(wp.floor(y / cell_size))

    # Simple hash function for 2D cell
    hash_val = (cx * 73856093 ^ cy * 19349663) % grid_dim
    if hash_val < 0:
        hash_val += grid_dim

    # Atomically insert into linked list.
    # We use (tid + 1) to represent pointers so 0 can mean null/empty.
    # Note: array is initialized to 0 from JAX.
    old_head = wp.atomic_exch(grid_head, hash_val, tid + 1)
    grid_next[tid] = old_head


@wp.kernel
def resolve_collisions_hash_kernel(
    positions_in: wp.array(dtype=wp.float32, ndim=2),
    velocities_in: wp.array(dtype=wp.float32, ndim=2),
    masses: wp.array(dtype=wp.float32),
    active_mask: wp.array(dtype=wp.int32),
    grid_head: wp.array(dtype=int),
    grid_next: wp.array(dtype=int),
    cell_size: float,
    grid_dim: int,
    particle_radius: float,
    restitution: float,
    positions_out: wp.array(dtype=wp.float32, ndim=2),
    velocities_out: wp.array(dtype=wp.float32, ndim=2),
):
    r"""Resolves particle collisions natively using the constructed hash grid.

    Calculates the resulting velocities after perfectly elastic or inelastic collisions
    based on the restitution coefficient :math:`e`. Applies an impulse :math:`J`
    along the collision normal :math:`\mathbf{n}`.

    .. math::
        J = \frac{-(1 + e) \mathbf{v}_{rel} \cdot \mathbf{n}}{\frac{1}{m_i} + \frac{1}{m_j}} \\
        \Delta \mathbf{v}_i = \frac{J \mathbf{n}}{m_i}

    Args:
        positions_in (wp.array): Input particle positions. Units: [m].
        velocities_in (wp.array): Input particle velocities. Units: [m/s].
        masses (wp.array): Particle masses. Units: [kg].
        active_mask (wp.array): Array indicating active particles (1) vs inactive (0).
        grid_head (wp.array): Hash grid cell heads.
        grid_next (wp.array): Hash grid linked list next pointers.
        cell_size (float): The physical size of each grid cell. Units: [m].
        grid_dim (int): The total number of buckets in the hash grid.
        particle_radius (float): The interaction radius for collision detection. Units: [m].
        restitution (float): The coefficient of restitution :math:`e` (0 to 1).
        positions_out (wp.array): Output array for updated particle positions. Units: [m].
        velocities_out (wp.array): Output array for updated particle velocities. Units: [m/s].
    """
    tid = wp.tid()

    # Pass-through for inactive particles
    if active_mask[tid] == 0:
        positions_out[tid, 0] = positions_in[tid, 0]
        positions_out[tid, 1] = positions_in[tid, 1]
        velocities_out[tid, 0] = velocities_in[tid, 0]
        velocities_out[tid, 1] = velocities_in[tid, 1]
        return

    pos_i_x = positions_in[tid, 0]
    pos_i_y = positions_in[tid, 1]
    vel_i_x = velocities_in[tid, 0]
    vel_i_y = velocities_in[tid, 1]
    mass_i = masses[tid]

    delta_vel_x = float(0.0)
    delta_vel_y = float(0.0)

    # Search 3x3 neighborhood
    cx_base = int(wp.floor(pos_i_x / cell_size))
    cy_base = int(wp.floor(pos_i_y / cell_size))

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            cx = cx_base + dx
            cy = cy_base + dy

            hash_val = (cx * 73856093 ^ cy * 19349663) % grid_dim
            if hash_val < 0:
                hash_val += grid_dim

            # Read head pointer (0 means empty, otherwise idx = pointer - 1)
            neighbor_ptr = grid_head[hash_val]

            while neighbor_ptr != 0:
                neighbor_idx = neighbor_ptr - 1

                # Check collision condition
                if neighbor_idx != tid and active_mask[neighbor_idx] != 0:
                    pos_j_x = positions_in[neighbor_idx, 0]
                    pos_j_y = positions_in[neighbor_idx, 1]

                    dist_x = pos_i_x - pos_j_x
                    dist_y = pos_i_y - pos_j_y
                    dist_sq = dist_x * dist_x + dist_y * dist_y
                    dist = wp.sqrt(dist_sq)

                    if dist > 0.0 and dist < particle_radius:
                        normal_x = dist_x / dist
                        normal_y = dist_y / dist

                        vel_j_x = velocities_in[neighbor_idx, 0]
                        vel_j_y = velocities_in[neighbor_idx, 1]

                        rel_vel_x = vel_i_x - vel_j_x
                        rel_vel_y = vel_i_y - vel_j_y
                        vel_normal = rel_vel_x * normal_x + rel_vel_y * normal_y

                        # Apply velocity impulse if approaching
                        if vel_normal < 0.0:
                            mass_j = masses[neighbor_idx]
                            inv_mass_sum = 1.0 / mass_i + 1.0 / mass_j
                            impulse_mag = (
                                -(1.0 + restitution) * vel_normal / inv_mass_sum
                            )

                            delta_vel_x += normal_x * impulse_mag / mass_i
                            delta_vel_y += normal_y * impulse_mag / mass_i

                # Move to next node in the linked list
                neighbor_ptr = grid_next[neighbor_idx]

    # Clamp the total velocity change to avoid massive impulses from overlapping clusters
    max_delta_vel = 50.0  # [m/s] safety limit
    delta_vel_sq = delta_vel_x * delta_vel_x + delta_vel_y * delta_vel_y
    if delta_vel_sq > max_delta_vel * max_delta_vel:
        scale_factor_vel = max_delta_vel / wp.sqrt(delta_vel_sq)
        delta_vel_x *= scale_factor_vel
        delta_vel_y *= scale_factor_vel

    # Write out updated state
    positions_out[tid, 0] = pos_i_x
    positions_out[tid, 1] = pos_i_y
    velocities_out[tid, 0] = vel_i_x + delta_vel_x
    velocities_out[tid, 1] = vel_i_y + delta_vel_y


# JAX bindings via Warp FFI interface
build_grid_jax = jax_kernel(
    build_grid_kernel, num_outputs=2, in_out_argnames=["grid_head", "grid_next"]
)

resolve_collisions_jax = jax_kernel(resolve_collisions_hash_kernel, num_outputs=2)


def resolve_collisions(state: ParticleState, config: SimConfig) -> ParticleState:
    r"""A pure XLA-compatible collision resolver utilizing Warp.

    Replaces the original JAX-only Morton code sorting with a fast, thread-safe
    atomic spatial hash grid executed natively via `wp.jax_kernel`.

    Args:
        state (ParticleState): The current state of all particles.
        config (SimConfig): The simulation configuration parameters.

    Returns:
        ParticleState: The new state of particles after resolving collisions.
    """
    if not config.enable_collisions:
        return state

    active_mask = state.active.astype(jnp.int32)
    num_particles = state.position.shape[0]

    # Dynamically scale the grid dimension to ~2x the number of particles to minimize hash collisions
    grid_dim = int(num_particles * 2)
    grid_dim = max(grid_dim, 1024)

    cell_size = float(config.d_particle)
    # Initialize empty linked-list buffers (0 = null)
    grid_head = jnp.zeros((grid_dim,), dtype=jnp.int32)
    grid_next = jnp.zeros((num_particles,), dtype=jnp.int32)

    # Build the hash grid natively on GPU
    grid_head_out, grid_next_out = build_grid_jax(
        state.position, cell_size, grid_dim, grid_head, grid_next
    )

    # Query the grid and resolve collisions
    new_pos, new_vel = resolve_collisions_jax(
        state.position,
        state.velocity,
        state.mass,
        active_mask,
        grid_head_out,
        grid_next_out,
        cell_size,
        grid_dim,
        float(config.d_particle),
        float(config.collision_restitution),
    )

    return ParticleState(
        position=new_pos,
        velocity=new_vel,
        temperature=state.temperature,
        mass=state.mass,
        active=state.active,
    )
