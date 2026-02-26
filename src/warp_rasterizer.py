import warp as wp
import jax
import jax.numpy as jnp
from typing import Tuple, Any

# Initialize Warp Engine
wp.init()


@wp.func
def turbo_colormap_warp(scalar_value: wp.float32) -> wp.vec3:
    """Approximates the Turbo colormap using piecewise linear interpolation.

    Maps a normalized scalar value [0, 1] to an RGB color vector.
    The gradient transitions: Blue -> Cyan -> Green -> Yellow -> Red.

    Args:
        scalar_value: Normalized input value in range [0, 1].

    Returns:
        wp.vec3: RGB color vector.
    """
    x_clamped = wp.clamp(scalar_value, 0.0, 1.0) * 4.0

    red_channel = wp.clamp(x_clamped - 2.0, 0.0, 1.0)
    if x_clamped > 3.0:
        red_channel = 1.0

    green_channel = 0.0
    if x_clamped < 1.0:
        green_channel = x_clamped
    elif x_clamped >= 1.0 and x_clamped < 3.0:
        green_channel = 1.0
    else:
        green_channel = 1.0 - (x_clamped - 3.0)

    blue_channel = 0.0
    if x_clamped < 1.0:
        blue_channel = 1.0
    elif x_clamped >= 1.0 and x_clamped < 2.0:
        blue_channel = 1.0 - (x_clamped - 1.0)

    return wp.vec3(red_channel, green_channel, blue_channel)


@wp.kernel
def splat_particles_kernel(
    positions: wp.array(dtype=wp.float32, ndim=2),
    temperatures: wp.array(dtype=wp.float32),
    active_mask: wp.array(dtype=wp.int32),
    masses: wp.array(dtype=wp.float32),
    resolution_width: int,
    resolution_height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    reference_mass: float,
    glow_sigma: float,
    t_min: float,
    t_max: float,
    output_buffer: wp.array(dtype=wp.float32, ndim=3),
):
    r"""Atomic splatting kernel that maps each particle to the screen buffer.

    Implements a Gaussian splatting algorithm where each particle thread
    contributes light to a local window of pixels using atomic additions.

    .. math::
        I(x, y) = \sum_{p} w_p \cdot \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_p\|^2}{2\sigma^2}\right)

    Args:
        positions: Particle positions $(N, 2)$. Units: [m].
        temperatures: Particle temperatures $(N,)$. Units: [K].
        active_mask: Boolean mask for active particles $(N,)$.
        masses: Particle masses $(N,)$. Units: [kg].
        resolution_width: Output buffer width in pixels.
        resolution_height: Output buffer height in pixels.
        x_min: Minimum x-boundary of the viewport. Units: [m].
        x_max: Maximum x-boundary of the viewport. Units: [m].
        y_min: Minimum y-boundary of the viewport. Units: [m].
        y_max: Maximum y-boundary of the viewport. Units: [m].
        reference_mass: Base mass used for scaling intensity. Units: [kg].
        glow_sigma: Standard deviation of the Gaussian splat. Units: [pixels].
        t_min: Minimum temperature for colormap normalization. Units: [K].
        t_max: Maximum temperature for colormap normalization. Units: [K].
        output_buffer: Accumulation RGB buffer $(H, W, 3)$.
    """
    thread_idx = wp.tid()

    if active_mask[thread_idx] == 0:
        return

    pos_x = positions[thread_idx, 0]
    pos_y = positions[thread_idx, 1]

    # Viewport Culling
    if pos_x < x_min or pos_x > x_max or pos_y < y_min or pos_y > y_max:
        return

    # Projection: Map spatial coordinates to screen indices
    norm_x = (pos_x - x_min) / (x_max - x_min)
    norm_y = (pos_y - y_min) / (y_max - y_min)

    pixel_x = int(norm_x * float(resolution_width - 1))
    pixel_y = int(norm_y * float(resolution_height - 1))

    # Weighting: Scale intensity by particle mass (surface area approximation)
    particle_mass = masses[thread_idx]
    safe_mass = wp.max(particle_mass, 0.0)
    size_factor = wp.pow(safe_mass / reference_mass, 0.667)

    # Base weight tuned for atomic accumulation
    base_weight = 40.0 * 0.05
    weight = base_weight * size_factor

    # Shading: Map temperature to color
    particle_temp = temperatures[thread_idx]
    norm_temp = (particle_temp - t_min) / (t_max - t_min)
    particle_color = turbo_colormap_warp(norm_temp)

    # Splatting: Local window accumulation
    splat_radius = int(4.0 * glow_sigma) | 1
    half_radius = splat_radius / 2

    for offset_y in range(-half_radius, half_radius + 1):
        for offset_x in range(-half_radius, half_radius + 1):
            target_x = pixel_x + offset_x
            target_y = pixel_y + offset_y

            if (
                target_x >= 0
                and target_x < resolution_width
                and target_y >= 0
                and target_y < resolution_height
            ):
                distance_squared = float(offset_x * offset_x + offset_y * offset_y)
                gaussian_weight = wp.exp(
                    -distance_squared / (2.0 * glow_sigma * glow_sigma)
                )
                final_intensity = weight * gaussian_weight

                # Thread-safe color accumulation
                wp.atomic_add(
                    output_buffer,
                    target_y,
                    target_x,
                    0,
                    particle_color[0] * final_intensity,
                )
                wp.atomic_add(
                    output_buffer,
                    target_y,
                    target_x,
                    1,
                    particle_color[1] * final_intensity,
                )
                wp.atomic_add(
                    output_buffer,
                    target_y,
                    target_x,
                    2,
                    particle_color[2] * final_intensity,
                )


def rasterize_frame_warp(
    positions: jnp.ndarray,
    temperatures: jnp.ndarray,
    active_mask: jnp.ndarray,
    masses: jnp.ndarray,
    resolution: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    reference_mass: float = 1.0,
    glow_sigma: float = 0.8,
    t_min: float = 280.0,
    t_max: float = 400.0,
) -> jnp.ndarray:
    """Orchestrates the Warp-based atomic splatting for a single frame.

    Utilizes DLPack for zero-copy memory sharing between JAX and Warp.

    Args:
        positions: JAX array of particle positions $(N, 2)$. Units: [m].
        temperatures: JAX array of particle temperatures $(N,)$. Units: [K].
        active_mask: JAX boolean mask for active particles $(N,)$.
        masses: JAX array of particle masses $(N,)$. Units: [kg].
        resolution: Output image resolution (Height, Width).
        bounds: Viewport bounds (x_min, x_max, y_min, y_max). Units: [m].
        reference_mass: Reference mass for size scaling. Units: [kg].
        glow_sigma: Splat blur standard deviation. Units: [pixels].
        t_min: Min temperature for colormap. Units: [K].
        t_max: Max temperature for colormap. Units: [K].

    Returns:
        jnp.ndarray: Rasterized RGB image buffer $(H, W, 3)$.
    """
    height, width = resolution
    x_min, x_max, y_min, y_max = bounds

    # Interoperability: Map JAX arrays to Warp natively
    # Convert mask to int32 for Warp compatibility
    active_mask_int = active_mask.astype(jnp.int32)

    warp_positions = wp.from_dlpack(positions)
    warp_temperatures = wp.from_dlpack(temperatures)
    warp_active_mask = wp.from_dlpack(active_mask_int)
    warp_masses = wp.from_dlpack(masses)

    # Initialize accumulation buffer on the same device
    warp_output = wp.zeros(
        (height, width, 3), dtype=wp.float32, device=warp_positions.device
    )
    particle_count = positions.shape[0]

    # Compute: Launch Atomic Splatting Kernel
    wp.launch(
        kernel=splat_particles_kernel,
        dim=particle_count,
        inputs=[
            warp_positions,
            warp_temperatures,
            warp_active_mask,
            warp_masses,
            width,
            height,
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
            float(reference_mass),
            float(glow_sigma),
            float(t_min),
            float(t_max),
            warp_output,
        ],
        device=warp_positions.device,
    )

    wp.synchronize()

    # Back to JAX: Retrieve accumulation buffer
    output_jax = jax.dlpack.from_dlpack(warp_output)

    # Post-Processing: Tone-mapping and clamping
    # Uses tanh for soft saturation of high-intensity regions
    processed_image = jnp.tanh(output_jax * 3.0)

    # Orientation: Flip Y-axis to match Cartesian viewport
    return processed_image[::-1, :, :]
