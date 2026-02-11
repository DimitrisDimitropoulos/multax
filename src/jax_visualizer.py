import numpy as np
import subprocess
import os
import sys
import time
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List
from src.state import ParticleState
from src.config import SimConfig
from src.rasterizer import rasterize_frame_jax
from src.physics import get_fluid_temperature


class JAXVisualizer:
    def __init__(
        self,
        config: SimConfig,
        history: ParticleState,
        time_array: np.ndarray,
        flow_func: Callable,
    ):
        """Initializes the JAX visualizer.

        Args:
            config (SimConfig): Simulation configuration.
            history (ParticleState): Full history of particle states.
            time_array (np.ndarray): Array of time points corresponding to history.
            flow_func (Callable): Function defining the flow field for streamlines.
        """
        self.config = config
        self.pos_history = jax.device_put(history.position)
        self.temp_history = jax.device_put(history.temperature)
        # Handle active mask (default to all true if missing for backward compatibility)
        if history.active is not None:
            self.active_history = jax.device_put(history.active)
        else:
            self.active_history = jnp.ones(history.position.shape[:2], dtype=bool)

        self.mass_history = jax.device_put(history.mass)
        self.time = time_array
        self.flow_func = flow_func

    def _generate_background(
        self, width: int, height: int, bounds: Tuple[float, float, float, float]
    ) -> jnp.ndarray:
        """Generates a static background image with streamlines and temperature field.

        Args:
            width (int): Width of the background image in pixels.
            height (int): Height of the background image in pixels.
            bounds (Tuple[float, float, float, float]): Plot bounds (x_min, x_max, y_min, y_max).

        Returns:
            jnp.ndarray: Background image as an RGB array (H, W, 3) with float values 0..1.
        """
        x_min, x_max, y_min, y_max = bounds
        # Setup Figure
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor("black")  # Ensure full figure is black
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis("off")
        ax.set_facecolor("black")

        # Generate Grid for Streamlines
        density = 50
        x = np.linspace(x_min, x_max, density)
        y = np.linspace(y_min, y_max, density)
        X, Y = np.meshgrid(x, y)

        # Calculate Velocity
        grid_pos = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Safe wrapper for flow func to avoid JAX/Numpy issues
        # We run it in JAX but pull result to numpy
        vel_jax = jax.vmap(lambda p: self.flow_func(p, self.config))(
            jnp.array(grid_pos)
        )
        vel = np.array(vel_jax)
        U = vel[:, 0].reshape(X.shape)
        V = vel[:, 1].reshape(Y.shape)

        # Plot Temperature Field if config indicates a thermal scenario
        # Check if T_wall is significantly different from Room Temp, implying a gradient
        # Later make the T_wall some char temperature to make more problem agnostic
        if abs(self.config.T_wall - self.config.T_room_ref) > 1.0:
            # Get the temperature of the carrrier with a
            # based on the continuous function of the temperature
            # grid_pos shape is (N_points, 2)
            T_field_jax = jax.vmap(lambda p: get_fluid_temperature(p, self.config))(
                jnp.array(grid_pos)
            )
            T_field = np.array(T_field_jax).reshape(X.shape)

            # Use hot or YlOrRd colormap
            # Alpha 0.3 so streamlines are visible
            ax.imshow(
                T_field,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                cmap="YlOrRd",
                alpha=0.4,
                aspect="auto",
            )

        # Streamlines in faint gray with some alpha
        strm = ax.streamplot(
            x, y, U, V, color="#333333", density=1.5, linewidth=1.0, arrowsize=1.0
        )
        strm.lines.set_alpha(0.5)

        # Draw Cylinder/Walls
        if self.config.R_cylinder > 0 and "cylinder" in str(self.flow_func.__name__):
            circle = plt.Circle(
                (0, 0), self.config.R_cylinder, color="#111111", ec="#555555"
            )
            ax.add_patch(circle)
        if self.config.wall_x != 0 and "wall" in str(self.flow_func.__name__):
            ax.axvline(self.config.wall_x, color="#555555", linewidth=3)
        fig.canvas.draw()

        try:
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        except AttributeError:
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        w, h = fig.canvas.get_width_height()

        # Reshape logic
        if data.size == w * h * 4:
            data = data.reshape((h, w, 4))
            data = data[:, :, :3]  # Drop Alpha
        elif data.size == w * h * 3:
            data = data.reshape((h, w, 3))
        else:
            # Fallback if dimensions mismatch (e.g. HiDPI)
            # Re-read knowing size
            depth = data.size // (w * h)
            data = data.reshape((h, w, depth))[:, :, :3]

        plt.close(fig)

        # Normalize to 0..1 float
        return jax.device_put(data.astype(np.float32) / 255.0)

    def generate_video(
        self,
        output_path: str,
        bounds: Tuple[float, float, float, float],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        slow_mo_factor: float = 1.0,
    ):
        """Generates and saves a video of the simulation using FFMPEG.

        Renders frames using the JAX rasterizer and pipes them to FFMPEG.

        Args:
            output_path (str): Path to save the output video (e.g., 'output.mp4').
            bounds (Tuple[float, float, float, float]): Viewport bounds (x_min, x_max, y_min, y_max).
            width (int, optional): Video width in pixels. Defaults to 1280.
            height (int, optional): Video height in pixels. Defaults to 720.
            fps (int, optional): Frames per second. Defaults to 30.
            slow_mo_factor (float, optional): Factor to slow down the video time relative to simulation time.
                                              >1.0 is slower, <1.0 is faster. Defaults to 1.0.
        """
        print(f"JAX Devices: {jax.devices()}")
        start_time = time.time()

        # Debug Checks
        all_x = self.pos_history[:, :, 0]
        if jnp.any(jnp.isnan(all_x)):
            print(
                "CRITICAL WARNING: Simulation history contains NaNs! Video will be black."
            )

        print("Pre-rendering background...")
        #  Generate BG
        bg_raw = self._generate_background(width, height, bounds)
        #  Force Resize to exact JAX dimensions to guarantee shape match
        # jax.image.resize expects (H, W, C)
        bg_image = jax.image.resize(bg_raw, (height, width, 3), method="cubic")

        # Frame Logic
        total_sim_duration = self.time[-1] - self.time[0]
        target_video_duration = total_sim_duration * slow_mo_factor
        num_video_frames = int(target_video_duration * fps)
        sim_indices = np.linspace(0, len(self.time) - 1, num_video_frames).astype(int)
        print(f"Video Gen: {num_video_frames} frames. Output: {output_path}")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",  # Use CPU for guaranteed robustness as previous attempts crashed
            "-b:v",
            "5M",  # Bitrate
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            output_path,
        ]

        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        m_init = float(self.config.m_particle_init)

        # Dynamic Temperature Bounds for Visualization
        # Start at Room Temp, go up to Wall Temp (or slightly higher/lower buffer)
        t_min = float(min(self.config.T_room_ref, self.config.T_wall)) - 5.0
        t_max = float(max(self.config.T_room_ref, self.config.T_wall)) + 5.0

        # Pre-compile render function (vmap outside loop)
        render_batch_fn = jax.jit(
            jax.vmap(
                lambda p, t, a, m: rasterize_frame_jax(
                    p, t, a, m, (height, width), bounds, m_init, 0.8, t_min, t_max
                )
            )
        )

        chunk_size = 10
        for i in range(0, num_video_frames, chunk_size):
            end = min(i + chunk_size, num_video_frames)
            batch_idxs = sim_indices[i:end]

            batch_pos = self.pos_history[batch_idxs]
            batch_temp = self.temp_history[batch_idxs]
            batch_active = self.active_history[batch_idxs]
            batch_mass = self.mass_history[batch_idxs]

            # Execute Render
            batch_particles = render_batch_fn(
                batch_pos, batch_temp, batch_active, batch_mass
            )

            # Composite
            batch_particles = jnp.nan_to_num(batch_particles, nan=0.0)
            batch_final = jnp.clip(bg_image + batch_particles, 0.0, 1.0)

            cpu_bytes = np.array((batch_final * 255).astype(jnp.uint8)).tobytes()

            try:
                process.stdin.write(cpu_bytes)
            except (BrokenPipeError, OSError):
                print("FFMPEG pipe broken!")
                break

            sys.stdout.write(f"\rProgress: {end}/{num_video_frames}")
            sys.stdout.flush()

        process.stdin.close()
        process.wait()

        duration = time.time() - start_time
        print(f"\nDone in {duration:.2f}s.")

        # Log to file
        with open("video_generation.log", "a") as f:
            f.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {output_path}: {num_video_frames} frames, {width}x{height}, {fps}fps, SlowMo={slow_mo_factor}x, GenTime={duration:.2f}s\n"
            )
