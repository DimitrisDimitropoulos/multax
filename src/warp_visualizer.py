import numpy as np
import subprocess
import time
import sys
import jax
import jax.numpy as jnp
from typing import Tuple, Callable
from src.state import ParticleState
from src.config import SimConfig
from src.flow import TempFunc
from src.jax_visualizer import JAXVisualizer
from src.warp_rasterizer import rasterize_frame_warp


class WarpVisualizer(JAXVisualizer):
    """High-performance visualizer using NVIDIA Warp's atomic splatting engine.

    Inherits from JAXVisualizer to utilize its background and streamline
    generation logic while replacing the rasterization pipeline.
    """

    def generate_video(
        self,
        output_path: str,
        bounds: Tuple[float, float, float, float],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        slow_mo_factor: float = 1.0,
    ) -> None:
        """Generates and encodes a simulation video using Warp-based rendering.

        Args:
            output_path: Destination path for the MP4 file.
            bounds: Viewport spatial boundaries (x_min, x_max, y_min, y_max). Units: [m].
            width: Video resolution width in pixels.
            height: Video resolution height in pixels.
            fps: Video frames per second.
            slow_mo_factor: Temporal scaling factor (>1.0 is slower).
        """
        print(f"JAX Devices identified: {jax.devices()}")
        start_timestamp = time.time()

        # Step: Generate static background with streamlines
        print("Pre-rendering simulation background...")
        raw_background = self._generate_background(width, height, bounds)
        background_image = jax.image.resize(
            raw_background, (height, width, 3), method="cubic"
        )

        # Step: Temporal mapping
        total_duration = self.time[-1] - self.time[0]
        video_duration = total_duration * slow_mo_factor
        total_frames = int(video_duration * fps)

        # Interpolate simulation time indices to video frames
        time_indices = np.linspace(0, len(self.time) - 1, total_frames).astype(int)
        print(f"Video Generation: {total_frames} frames. Target: {output_path}")

        # Step: Initialize FFMPEG pipeline
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
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
            "libx264",
            "-b:v",
            "5M",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            output_path,
        ]

        process = subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stderr=None, text=False
        )

        # Step: Configure visualization constants
        initial_mass = float(self.config.m_particle_init)
        temp_min = float(min(self.config.T_room_ref, self.config.T_wall)) - 5.0
        temp_max = float(max(self.config.T_room_ref, self.config.T_wall)) + 5.0

        # Step: Iterative Frame Processing
        for frame_idx in range(total_frames):
            sim_idx = time_indices[frame_idx]

            # Extract instantaneous particle states
            pos = self.pos_history[sim_idx]
            temp = self.temp_history[sim_idx]
            active = self.active_history[sim_idx]
            mass = self.mass_history[sim_idx]

            # Delegate rendering to Warp Atomic Engine
            rendered_particles = rasterize_frame_warp(
                pos,
                temp,
                active,
                mass,
                (height, width),
                bounds,
                initial_mass,
                0.8,
                temp_min,
                temp_max,
            )

            # Composition: Add particles over the static background
            rendered_particles = jnp.nan_to_num(rendered_particles, nan=0.0)
            final_frame = jnp.clip(background_image + rendered_particles, 0.0, 1.0)

            # Data Transfer: Pipe raw bytes to FFMPEG
            pixel_bytes = np.array((final_frame * 255).astype(jnp.uint8)).tobytes()

            try:
                process.stdin.write(pixel_bytes)
            except (BrokenPipeError, OSError):
                print("\nFFMPEG communication pipe collapsed.")
                break

            # Logging Progress
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                sys.stdout.write(
                    f"\rProgress: {frame_idx + 1}/{total_frames} frames rendered."
                )
                sys.stdout.flush()

        # Split lines correctly before final report
        sys.stdout.write("\n")
        sys.stdout.flush()

        process.stdin.close()
        process.wait()

        elapsed_time = time.time() - start_timestamp
        print(f"Video encoding completed in {elapsed_time:.2f}s.")
