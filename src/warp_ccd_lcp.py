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
    r"""Broad Phase: Hashes particles into a spatial grid.

    Constructs a linked list for each spatial grid cell, allowing for efficient
    neighbor searches during collision resolution.

    .. math::
        c_x = \lfloor x / s \rfloor, \quad c_y = \lfloor y / s \rfloor \\
        h = (c_x \cdot 73856093 + c_y \cdot 19349663) \pmod D

    Args:
        positions (wp.array): Particle position vectors :math:`\mathbf{x}`. Units: [m].
        cell_size (float): The physical size of each grid cell :math:`s`. Units: [m].
        grid_dim (int): The total number of buckets in the hash grid :math:`D`.
        grid_head (wp.array): Array storing the index of the first particle in each cell.
        grid_next (wp.array): Array storing the index of the next particle in the linked list.
    """
    tid = wp.tid()
    cx = int(wp.floor(positions[tid, 0] / cell_size))
    cy = int(wp.floor(positions[tid, 1] / cell_size))

    # Simple hash function for 2D cell
    hash_val = (cx * 73856093 + cy * 19349663) % grid_dim
    if hash_val < 0:
        hash_val += grid_dim

    old_head = wp.atomic_exch(grid_head, hash_val, tid + 1)
    grid_next[tid] = old_head


@wp.kernel
def resolve_ccd_jacobi_kernel(
    positions: wp.array(dtype=wp.float32, ndim=2),
    velocities_original: wp.array(dtype=wp.float32, ndim=2),
    velocities_in: wp.array(dtype=wp.float32, ndim=2),
    masses: wp.array(dtype=wp.float32),
    active_mask: wp.array(dtype=wp.int32),
    grid_head: wp.array(dtype=int),
    grid_next: wp.array(dtype=int),
    cell_size: float,
    grid_dim: int,
    particle_radius: float,
    dt_arr: wp.array(dtype=wp.float32),
    restitution: float,
    relaxation: float,
    velocities_out: wp.array(dtype=wp.float32, ndim=2),
):
    r"""CCD + Jacobi LCP Solver: Finds TOI and applies relaxed impulses locally.

    Performs Continuous Collision Detection (CCD) by solving a quadratic equation
    for the Time Of Impact (TOI). Resolves collisions via a relaxed Jacobi iteration
    for the Linear Complementarity Problem (LCP).

    .. math::
        \mathbf{x}_{rel}(t) = (\mathbf{x}_{i,0} - \mathbf{x}_{j,0}) + t (\mathbf{v}_{i,0} - \mathbf{v}_{j,0}) \\
        ||\mathbf{x}_{rel}(t_{toi})||^2 = (2r)^2 \\
        J = \frac{\Delta \mathbf{v}_{n}}{m_{i}^{-1} + m_{j}^{-1}} \cdot \omega

    Args:
        positions (wp.array): Particle positions at the end of the time step. Units: [m].
        velocities_original (wp.array): Velocities before collision resolution. Units: [m/s].
        velocities_in (wp.array): Current iterative guess for post-collision velocities. Units: [m/s].
        masses (wp.array): Particle masses. Units: [kg].
        active_mask (wp.array): Mask indicating active particles.
        grid_head (wp.array): Hash grid cell heads.
        grid_next (wp.array): Hash grid linked list next pointers.
        cell_size (float): Grid cell size. Units: [m].
        grid_dim (int): Hash grid dimensions.
        particle_radius (float): Particle interaction radius. Units: [m].
        dt_arr (wp.array): Time step array (length 1). Units: [s].
        restitution (float): Coefficient of restitution :math:`e`.
        relaxation (float): Jacobi relaxation factor :math:`\omega`.
        velocities_out (wp.array): Updated post-collision velocities for the next iteration. Units: [m/s].
    """
    tid = wp.tid()

    if active_mask[tid] == 0:
        velocities_out[tid, 0] = velocities_in[tid, 0]
        velocities_out[tid, 1] = velocities_in[tid, 1]
        return

    dt = dt_arr[0]

    end_pos_i = wp.vec2(positions[tid, 0], positions[tid, 1])
    vel_i_orig = wp.vec2(velocities_original[tid, 0], velocities_original[tid, 1])
    vel_i_curr = wp.vec2(velocities_in[tid, 0], velocities_in[tid, 1])

    start_pos_i = end_pos_i - vel_i_orig * dt
    mass_i = masses[tid]

    cx_base = int(wp.floor(end_pos_i[0] / cell_size))
    cy_base = int(wp.floor(end_pos_i[1] / cell_size))

    min_dist = particle_radius * 2.0
    min_dist_sq = min_dist * min_dist

    delta_vel = wp.vec2(0.0, 0.0)

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            cx = cx_base + dx
            cy = cy_base + dy

            hash_val = (cx * 73856093 + cy * 19349663) % grid_dim
            if hash_val < 0:
                hash_val += grid_dim

            neighbor_ptr = grid_head[hash_val]

            while neighbor_ptr != 0:
                neighbor_idx = neighbor_ptr - 1

                if tid != neighbor_idx and active_mask[neighbor_idx] != 0:
                    end_pos_j = wp.vec2(
                        positions[neighbor_idx, 0], positions[neighbor_idx, 1]
                    )
                    vel_j_orig = wp.vec2(
                        velocities_original[neighbor_idx, 0],
                        velocities_original[neighbor_idx, 1],
                    )
                    vel_j_curr = wp.vec2(
                        velocities_in[neighbor_idx, 0], velocities_in[neighbor_idx, 1]
                    )

                    start_pos_j = end_pos_j - vel_j_orig * dt

                    rel_pos = start_pos_i - start_pos_j
                    rel_vel_orig = vel_i_orig - vel_j_orig

                    a = wp.dot(rel_vel_orig, rel_vel_orig)
                    b = 2.0 * wp.dot(rel_pos, rel_vel_orig)
                    c = wp.dot(rel_pos, rel_pos) - min_dist_sq

                    toi = float(-1.0)
                    is_hit = False

                    if c < 0.0:
                        is_hit = True
                        toi = 0.0
                    elif b < 0.0 and a > 1e-8:
                        disc = b * b - 4.0 * a * c
                        if disc >= 0.0:
                            t1 = (-b - wp.sqrt(disc)) / (2.0 * a)
                            if t1 >= 0.0 and t1 <= dt:
                                is_hit = True
                                toi = t1

                    if is_hit:
                        hit_pos_i = start_pos_i + vel_i_orig * toi
                        hit_pos_j = start_pos_j + vel_j_orig * toi

                        normal = hit_pos_i - hit_pos_j
                        dist_at_hit = wp.length(normal)

                        if dist_at_hit > 1e-7:
                            normal = normal / dist_at_hit
                        else:
                            normal = wp.vec2(1.0, 0.0)

                        # Restitution bias is based strictly on ORIGINAL approach velocity
                        vn_orig = wp.dot(rel_vel_orig, normal)

                        # We only resolve if they were originally approaching
                        if vn_orig < 0.0:
                            bias = -restitution * vn_orig

                            # Current relative velocity
                            rel_vel_curr = vel_i_curr - vel_j_curr
                            vn_curr = wp.dot(rel_vel_curr, normal)

                            # Required delta to reach target bias
                            delta_vn = bias - vn_curr

                            # We only push apart, never pull together (LCP constraint)
                            if delta_vn > 0.0:
                                mass_j = masses[neighbor_idx]
                                inv_mass_sum = (1.0 / mass_i) + (1.0 / mass_j)

                                impulse_mag = delta_vn / inv_mass_sum
                                impulse_mag *= relaxation

                                impulse_vec = normal * impulse_mag
                                delta_vel += impulse_vec / mass_i

                neighbor_ptr = grid_next[neighbor_idx]

    velocities_out[tid, 0] = vel_i_curr[0] + delta_vel[0]
    velocities_out[tid, 1] = vel_i_curr[1] + delta_vel[1]


# JAX XLA BINDINGS
build_grid_jax = jax_kernel(
    build_grid_kernel, num_outputs=2, in_out_argnames=["grid_head", "grid_next"]
)

resolve_ccd_jax = jax_kernel(
    resolve_ccd_jacobi_kernel, num_outputs=1, in_out_argnames=["velocities_out"]
)


def resolve_collisions_ccd_lcp(
    state: ParticleState, config: SimConfig, dt: float
) -> ParticleState:
    r"""A highly accurate collision resolver using CCD and relaxed Jacobi.

    Args:
        state (ParticleState): Current state of the particles.
        config (SimConfig): Simulation configuration.
        dt (float): Time step size. Units: [s].

    Returns:
        ParticleState: Updated state of the particles.
    """
    if not config.enable_collisions:
        return state

    active_mask = state.active.astype(jnp.int32)
    num_particles = state.position.shape[0]

    grid_dim = max(int(num_particles * 2), 1024)
    cell_size = float(config.d_particle)
    grid_head = jnp.zeros((grid_dim,), dtype=jnp.int32)
    grid_next = jnp.zeros((num_particles,), dtype=jnp.int32)

    # Execute Broad Phase
    grid_head, grid_next = build_grid_jax(
        state.position, cell_size, grid_dim, grid_head, grid_next
    )

    dt_arr = jnp.array([dt], dtype=jnp.float32)

    num_jacobi_iterations = 10
    # Under-relaxation factor guarantees the parallel Jacobi solver converges
    # Maybe ake this configurable in the future
    relaxation_factor = 0.5

    def jacobi_iteration(vels, _):
        out_vels = resolve_ccd_jax(
            state.position,
            state.velocity,  # velocities_original
            vels,  # velocities_in
            state.mass,
            active_mask,
            grid_head,
            grid_next,
            cell_size,
            grid_dim,
            float(config.d_particle),
            dt_arr,
            float(config.collision_restitution),
            float(relaxation_factor),
            vels,  # velocities_out is populated in-place
        )
        if isinstance(out_vels, (list, tuple)):
            out_vels = out_vels[0]
        return out_vels, None

    # Loop the solver natively in XLA pipe it in lax scan to avoid tranfers and
    # solving the whole systme in python for loops
    final_velocities, _ = jax.lax.scan(
        jacobi_iteration, state.velocity, jnp.arange(num_jacobi_iterations)
    )

    return ParticleState(
        position=state.position,
        velocity=final_velocities,
        temperature=state.temperature,
        mass=state.mass,
        active=state.active,
    )
