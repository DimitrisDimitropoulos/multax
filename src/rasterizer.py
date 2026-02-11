import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


def make_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    """Creates a 2D Gaussian kernel."""
    # Ensure correct centering: range should be [-(N-1)/2, ..., +(N-1)/2]
    # jnp.arange(size) gives 0..size-1
    # Shift by (size-1)/2
    coords = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2.0
    xx, yy = jnp.meshgrid(coords, coords)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / jnp.sum(kernel)


def turbo_colormap(x):
    """
    Approximate Turbo colormap logic in JAX.
    x: value in [0, 1]
    Returns: (..., 3) RGB
    """
    # Simple thermal gradient: Blue -> Cyan -> Green -> Yellow -> Red
    # 0.0 -> Blue (0, 0, 1)
    # 0.25 -> Cyan (0, 1, 1)
    # 0.5 -> Green (0, 1, 0)
    # 0.75 -> Yellow (1, 1, 0)
    # 1.0 -> Red (1, 0, 0)

    x = jnp.clip(x, 0.0, 1.0) * 4.0
    # R channel
    # 0-1: 0, 1-2: 0, 2-3: (x-2), 3-4: 1
    r = jnp.clip(x - 2.0, 0.0, 1.0)
    r = jnp.where(x > 3.0, 1.0, r)
    # G channel
    # 0-1: x, 1-2: 1, 2-3: 1, 3-4: 1-(x-3)
    g = jnp.where(x < 1.0, x, 0.0)
    g = jnp.where((x >= 1.0) & (x < 3.0), 1.0, g)
    g = jnp.where(x >= 3.0, 1.0 - (x - 3.0), g)
    # B channel
    # 0-1: 1, 1-2: 1-(x-1), 2-3: 0, 3-4: 0
    b = jnp.where(x < 1.0, 1.0, 0.0)
    b = jnp.where((x >= 1.0) & (x < 2.0), 1.0 - (x - 1.0), b)
    return jnp.stack([r, g, b], axis=-1)


@partial(jax.jit, static_argnums=(4, 5, 6))
def rasterize_frame_jax(
    positions: jnp.ndarray,  # (N, 2)
    temperatures: jnp.ndarray,  # (N,)
    active_mask: jnp.ndarray,  # (N,)
    masses: jnp.ndarray,  # (N,)
    resolution: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    reference_mass: float = 1.0,
    glow_sigma: float = 0.8,
) -> jnp.ndarray:
    """
    Rasterizes particles into a glowing heatmap.
    Returns: (H, W, 3) RGB array with float values 0..1
    """
    H, W = resolution
    x_min, x_max, y_min, y_max = bounds
    # Coordinate Mapping -> Grid Indices
    # Filter out-of-bounds AND inactive particles
    mask_x = (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max)
    mask_y = (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max)
    valid_mask = mask_x & mask_y & active_mask

    # Normalize 0..1
    xn = (jnp.clip(positions[:, 0], x_min, x_max) - x_min) / (x_max - x_min)
    yn = (jnp.clip(positions[:, 1], y_min, y_max) - y_min) / (y_max - y_min)
    ix = (xn * (W - 1)).astype(jnp.int32)
    iy = (yn * (H - 1)).astype(jnp.int32)

    # Accumulate Density and Weighted Temp
    grid_density = jnp.zeros((H, W))
    grid_w_temp = jnp.zeros((H, W))
    flat_indices = iy * W + ix

    # Scale intensity by size (cross-sectional area approx m^(2/3))
    # Avoid div by zero
    size_factor = (jnp.maximum(masses, 0.0) / reference_mass) ** 0.667
    base_weight = 40.0
    weights_density = jnp.where(valid_mask, base_weight * size_factor, 0.0)
    weights_temp = jnp.where(valid_mask, temperatures, 0.0)
    grid_density = (
        grid_density.ravel().at[flat_indices].add(weights_density).reshape(H, W)
    )
    grid_w_temp = (
        grid_w_temp.ravel()
        .at[flat_indices]
        .add(weights_temp * weights_density)
        .reshape(H, W)
    )

    # Apply Glow (Convolution)
    # Kernel size proportional to sigma
    k_size = int(4 * glow_sigma) | 1
    kernel = make_gaussian_kernel(k_size, glow_sigma)
    density_smooth = jax.scipy.signal.convolve2d(grid_density, kernel, mode="same")
    temp_smooth = jax.scipy.signal.convolve2d(grid_w_temp, kernel, mode="same")

    # Color Mapping
    # Recalculate average temp (weighted)
    avg_temp = temp_smooth / (density_smooth + 1e-6)

    # Normalize Temp
    # This is a fixed approach, would it be better to normalize based on
    # observed min/max in the frame? For now, let's use a fixed range for
    # better color consistency across frames.
    t_min, t_max = 280.0, 400.0
    t_norm = (avg_temp - t_min) / (t_max - t_min)
    rgb_color = turbo_colormap(t_norm)

    #  Intensity / Alpha
    # Map density to brightness.
    # Higher sensitivity for small sigma: tanh(x * gain)
    intensity = jnp.tanh(density_smooth * 3.0)
    image = rgb_color * intensity[..., None]
    # Flip Y axis to match Cartesian (Matplotlib origin=lower)
    # Image origin is usually top-left.
    return image[::-1, :, :]
