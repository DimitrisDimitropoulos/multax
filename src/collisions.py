import jax
import jax.numpy as jnp
from src.state import ParticleState
from src.config import SimConfig


def get_morton_code(
    position: jnp.ndarray, min_bound: float, max_bound: float
) -> jnp.ndarray:
    """Computes the Morton code (Z-order curve) for 2D positions.

    Maps 2D coordinates to a 1D integer that preserves locality.

    Args:
        position (jnp.ndarray): Shape (N, 2)
        min_bound (float): Minimum coordinate value (for normalization)
        max_bound (float): Maximum coordinate value

    Returns:
        jnp.ndarray: Array of integer codes.
    """
    # Normalize to [0, 1]
    norm_pos = (position - min_bound) / (max_bound - min_bound)
    norm_pos = jnp.clip(norm_pos, 0.0, 1.0)

    # Quantize to 16-bit integers (allows for 32-bit Morton code)
    # 2^16 = 65536 grid cells along each axis
    quantized = (norm_pos * 65535).astype(jnp.uint32)
    x = quantized[:, 0]
    y = quantized[:, 1]

    # Interleave bits (Magic numbers for bit spreading)
    # See "Bit Twiddling Hacks" for Morton Codes
    def spread_bits(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    xx = spread_bits(x)
    yy = spread_bits(y)

    return xx | (yy << 1)


def resolve_collisions(state: ParticleState, config: SimConfig) -> ParticleState:
    """Resolves particle-particle collisions using Spatial Hashing (Morton Codes) + Sweep & Prune.

    Includes both velocity impulse (elastic) and positional correction (projection) to prevent sinking.

    Args:
        state (ParticleState): Current particle state.
        config (SimConfig): Simulation configuration.

    Returns:
        ParticleState: Updated state with modified velocities and positions.
    """
    if not config.enable_collisions:
        return state

    # Broad Phase: Spatial Hashing with Morton Codes
    # We assume a reasonable bounding box for the hash, e.g. [-10, 10]
    morton_codes = get_morton_code(state.position, -10.0, 10.0)

    # Sort particles
    perm = jnp.argsort(
        morton_codes
    )  # Use argsort for stable sorting and to get permutation indices
    sorted_pos = state.position[perm]
    sorted_vel = state.velocity[perm]
    sorted_mass = state.mass[perm]
    sorted_active = state.active[perm]

    # Inverse permutation to restore order
    inv_perm = jnp.argsort(perm)

    # Narrow Phase: Sweep and Prune
    K = 32  # Tunable parameter for now not user inputed

    def compute_updates_for_particle(i, carry):
        pos_i = sorted_pos[i]
        vel_i = sorted_vel[i]
        mass_i = sorted_mass[i]
        active_i = sorted_active[i]

        # Accumulate changes
        # tuple: (velocity_delta, position_delta)
        accum_init = (jnp.zeros(2), jnp.zeros(2))

        def check_neighbor(j_offset, accum):
            acc_dv, acc_dx = accum
            # Map offset [0, 2K] to index shift [-K, K]
            shift = j_offset - K
            j = i + shift
            # Skip self and OOB
            is_self = shift == 0
            is_valid = (j >= 0) & (j < sorted_pos.shape[0]) & (~is_self)

            # Clamp for safe memory access
            j_clamped = jnp.clip(j, 0, sorted_pos.shape[0] - 1)
            pos_j = sorted_pos[j_clamped]
            vel_j = sorted_vel[j_clamped]
            mass_j = sorted_mass[j_clamped]
            active_j = sorted_active[j_clamped]
            # Physics Check
            delta_x = pos_i - pos_j
            dist_sq = jnp.sum(delta_x**2)
            dist = jnp.sqrt(dist_sq + 1e-9)

            # Diameter sum
            min_dist = config.d_particle

            # Collision detected
            collision_mask = (dist < min_dist) & active_i & active_j & is_valid

            # Normal Vector
            normal = delta_x / dist

            # Velocity impulse calculation
            rel_vel = vel_i - vel_j
            v_normal = jnp.dot(rel_vel, normal)
            # Only bounce if moving towards each other
            approaching = v_normal < 0.0
            # Elastic Impulse (Coefficient of Restitution e=0.9)
            # J = -(1+e) * v_normal / (1/m1 + 1/m2)
            inv_mass_sum = 1.0 / mass_i + 1.0 / mass_j
            restitution = config.collision_restitution
            j_impulse = -(1.0 + restitution) * v_normal / inv_mass_sum
            impulse_vec = j_impulse * normal

            # Apply velocity update if colliding AND approaching
            # Note: We apply force P_i = J, P_j = -J.
            # dv_i = J / m_i
            dv_update = jnp.where(
                collision_mask & approaching, impulse_vec / mass_i, 0.0
            )

            # Projection to prevent sinking: Move particles apart along normal
            # direction
            # Move particles apart so they don't stay overlapping
            overlap = min_dist - dist
            # Relaxation factor to avoid jitter (0.5 to 0.8 is standard, 0.2 safer for piles)
            correction_mag = 0.2 * overlap
            dx_update_vec = correction_mag * normal
            # Apply position update if colliding (regardless of velocity)
            dx_update = jnp.where(collision_mask, dx_update_vec, 0.0)

            return (acc_dv + dv_update, acc_dx + dx_update)

        dv, dx = jax.lax.fori_loop(0, 2 * K + 1, check_neighbor, accum_init)
        return vel_i + dv, pos_i + dx

    # Vectorize
    new_sorted_vel, new_sorted_pos = jax.vmap(
        compute_updates_for_particle, in_axes=(0, None)
    )(jnp.arange(sorted_pos.shape[0]), None)

    # Restore original order
    final_vel = new_sorted_vel[inv_perm]
    final_pos = new_sorted_pos[inv_perm]

    return ParticleState(
        position=final_pos,
        velocity=final_vel,
        temperature=state.temperature,
        mass=state.mass,
        active=state.active,
    )
