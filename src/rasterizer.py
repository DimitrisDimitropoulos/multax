import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


def make_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    r"""Creates a 2D Gaussian kernel.

    .. math::
        G(x, y) = \frac{1}{Z} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)

    Args:
        size (int): Size of the kernel side in pixels (NxN).
        sigma (float): Standard deviation :math:`\sigma` of the Gaussian.

    Returns:
        jnp.ndarray: Normalized 2D Gaussian kernel array.
    """
    # Ensure correct centering: range should be [-(N-1)/2, ..., +(N-1)/2]
    # jnp.arange(size) gives 0..size-1
    # Shift by (size-1)/2
    coords = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2.0
    xx, yy = jnp.meshgrid(coords, coords)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / jnp.sum(kernel)


def turbo_colormap(x):
    """Approximates the Turbo colormap using piecewise linear interpolation.

    Maps a normalized scalar value [0, 1] to an RGB color.
    Gradient: Blue -> Cyan -> Green -> Yellow -> Red.

    Args:
        x (jnp.ndarray): Normalized input values in range [0, 1].

    Returns:
        jnp.ndarray: RGB colors array (..., 3).
    """
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


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def rasterize_frame_jax(
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
    r"""Rasterizes particles into a glowing heatmap.

    Uses a Gaussian kernel splatting technique to accumulate particle density and
    temperature-weighted density onto a grid. The resulting fields are smoothed
    and combined to produce a colored heatmap where brightness indicates density/size
    and color indicates temperature.

    Args:
        positions (jnp.ndarray): Particle positions :math:`(N, 2)`. Units: [m].
        temperatures (jnp.ndarray): Particle temperatures :math:`(N,)`. Units: [K].
        active_mask (jnp.ndarray): Boolean mask for active particles :math:`(N,)`.
        masses (jnp.ndarray): Particle masses :math:`(N,)`. Units: [kg].
        resolution (Tuple[int, int]): Output image resolution (H, W).
        bounds (Tuple[float, float, float, float]): Viewport bounds (x_min, x_max, y_min, y_max).
        reference_mass (float, optional): Reference mass for scaling particle size. Units: [kg].
        glow_sigma (float, optional): Standard deviation for Gaussian glow in pixels.
        t_min (float, optional): Min temperature for colormap normalization. Units: [K].
        t_max (float, optional): Max temperature for colormap normalization. Units: [K].

    Returns:
        jnp.ndarray: Rasterized frame as an RGB array (H, W, 3) with values 0..1.
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
