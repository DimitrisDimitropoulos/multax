import math
import jax
import jax.numpy as jnp
import h5py
import numpy as np
from typing import Optional

from src.state import ParticleState
from src.solver import run_simulation_euler
from src.config import PhysicsConfig, ForceConfig
from src.boundary import BoundaryManager
from src.flow import FlowFunc, TempFunc
from src.ffmpeg_utils import get_ffmpeg_cmd


def _calculate_chunk_size(
    n_particles: int, dt: float, t_end: float, target_memory_mb: float
) -> int:
    """
    Calculates the number of steps per chunk based on memory limit.
    A ParticleState contains:
     - position (N, 2) float32 -> 8 bytes
     - velocity (N, 2) float32 -> 8 bytes
     - temperature (N,) float32 -> 4 bytes
     - mass (N,) float32 -> 4 bytes
     - active (N,) bool -> 1 byte
    Total = 25 bytes per particle per step.

    Args:
        n_particles: Number of particles
        dt: Time step size
        t_end: End time
        target_memory_mb: Target memory chunk size in MB

    Returns:
        Number of steps per chunk
    """
    bytes_per_particle = 8 + 8 + 4 + 4 + 1
    bytes_per_step = n_particles * bytes_per_particle

    target_bytes = target_memory_mb * 1024 * 1024
    steps_per_chunk = max(1, math.floor(target_bytes / bytes_per_step))

    # We shouldn't use more steps than the total simulation length
    total_steps = math.ceil(t_end / dt)
    return min(steps_per_chunk, total_steps)


def _save_chunk_to_h5(
    file: h5py.File, chunk_history: ParticleState, start_idx: int, n_steps_chunk: int
):
    """
    Saves a single history chunk to the open HDF5 database.

    Args:
        file: Open h5py File object in write/append mode
        chunk_history: ParticleState history for the current chunk
        start_idx: Starting time index for appending
        n_steps_chunk: Number of steps strictly simulated in this chunk
    """
    # Create datasets if they don't exist
    if "position" not in file:
        n_particles = chunk_history.position.shape[1]

        # Maxshape=(None, N, ...) allows unlimited appending along 0-th dimension
        file.create_dataset(
            "position",
            shape=(0, n_particles, chunk_history.position.shape[2]),
            maxshape=(None, n_particles, chunk_history.position.shape[2]),
            dtype=np.float32,
            chunks=True,
        )
        file.create_dataset(
            "velocity",
            shape=(0, n_particles, chunk_history.velocity.shape[2]),
            maxshape=(None, n_particles, chunk_history.velocity.shape[2]),
            dtype=np.float32,
            chunks=True,
        )
        file.create_dataset(
            "temperature",
            shape=(0, n_particles),
            maxshape=(None, n_particles),
            dtype=np.float32,
            chunks=True,
        )
        file.create_dataset(
            "mass",
            shape=(0, n_particles),
            maxshape=(None, n_particles),
            dtype=np.float32,
            chunks=True,
        )
        file.create_dataset(
            "active",
            shape=(0, n_particles),
            maxshape=(None, n_particles),
            dtype=bool,
            chunks=True,
        )

    # Resize and append
    new_len = start_idx + n_steps_chunk

    # Resize all datasets
    for key in ["position", "velocity", "temperature", "mass", "active"]:
        file[key].resize(
            (new_len, chunk_history.position.shape[1], *file[key].shape[2:])
        )

    # Assign values - explicitly convert JAX array to numpy and grab the slice of actual steps
    # We take slice [:n_steps_chunk] because history could be padded if we jitted a fixed size
    file["position"][start_idx:new_len] = np.array(
        chunk_history.position[:n_steps_chunk]
    )
    file["velocity"][start_idx:new_len] = np.array(
        chunk_history.velocity[:n_steps_chunk]
    )
    file["temperature"][start_idx:new_len] = np.array(
        chunk_history.temperature[:n_steps_chunk]
    )
    file["mass"][start_idx:new_len] = np.array(chunk_history.mass[:n_steps_chunk])
    file["active"][start_idx:new_len] = np.array(chunk_history.active[:n_steps_chunk])


def run_orchestrated_simulation(
    initial_state: ParticleState,
    config: PhysicsConfig,
    force_config: ForceConfig,
    boundary_manager: BoundaryManager,
    flow_func: FlowFunc,
    temp_func: TempFunc,
    master_rng_key: jax.Array,
    t_end: float,
    dt: float,
    target_memory_mb: float = 100.0,
    db_path: str = "simulation_history.h5",
):
    """
    Orchestrates solving the simulation by chunking the time array based on
    memory limits, leveraging jax.jit on identical chunk sizes, and saving
    the result iteratively to an HDF5 Database.

    Args:
        initial_state: Initial state of all particles.
        config: Simulation configuration.
        force_config: Active forces configuration.
        boundary_manager: Boundary condition logic.
        flow_func: Flow field function.
        temp_func: Temperature field function.
        master_rng_key: Master PRNG key.
        t_end: Total time to run simulation
        dt: Time step
        target_memory_mb: Max memory allowable for history in MB
        db_path: Where to save the HDF5 file
    """
    n_particles = initial_state.position.shape[0]
    total_steps = math.ceil(t_end / dt)

    # Figure out the max steps we can run to respect memory limit
    chunk_steps = _calculate_chunk_size(n_particles, dt, t_end, target_memory_mb)

    print(
        f"Orchestrator: Simulating {total_steps} total steps in chunks of {chunk_steps} steps."
    )
    print(f"Target DB: {db_path}")

    current_state = initial_state
    current_key = master_rng_key

    # We will jit compile one variant of our time evaluation length
    # to make sure JAX is fast. We'll use a standard chunk size, and
    # optionally pad the very last slice if it doesn't fit evenly.
    standard_t_eval = jnp.arange(0, chunk_steps * dt, dt)

    with h5py.File(db_path, "w") as f:
        # Loop over chunks
        global_step = 0
        while global_step < total_steps:
            steps_left = total_steps - global_step
            n_steps_this_chunk = min(chunk_steps, steps_left)

            # To avoid re-jitting the last imperfect block,
            # we always provide exactly `chunk_steps` to the solver, but
            # we will only save `n_steps_this_chunk` to the DB and update internal clocks

            # Generate a new split key
            step_key, current_key = jax.random.split(current_key)

            start_t = global_step * dt
            # Provide exact block array (though relative time doesn't affect
            # standard ODE equations directly in
            # this solver, we provide it
            # relatively starting 0 up to
            # chunk_steps)
            t_eval_chunk = standard_t_eval + start_t

            print(
                f"  > Simulating chunk [{global_step}:{global_step + n_steps_this_chunk}]..."
            )
            history_chunk = run_simulation_euler(
                current_state,
                t_eval_chunk,
                config,
                force_config,
                boundary_manager,
                flow_func,
                temp_func,
                step_key,
            )

            # Save the valid block to HDF5
            print(f"  > Saving to HDF5...")
            _save_chunk_to_h5(f, history_chunk, global_step, n_steps_this_chunk)

            # the new `current_state` is the last *valid* step inside this chunk
            valid_last_idx = n_steps_this_chunk - 1
            current_state = ParticleState(
                position=history_chunk.position[valid_last_idx],
                velocity=history_chunk.velocity[valid_last_idx],
                temperature=history_chunk.temperature[valid_last_idx],
                mass=history_chunk.mass[valid_last_idx],
                active=history_chunk.active[valid_last_idx],
            )

            global_step += n_steps_this_chunk

    print("Orchestration complete.")


def generate_orchestrated_video(
    db_path: str,
    output_path: str,
    config: PhysicsConfig,
    t_eval: np.ndarray,
    bounds: tuple,
    flow_func: FlowFunc,
    temp_func: TempFunc = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    slow_mo_factor: float = 1.0,
    use_warp: bool = False,
):
    """
    Reads the out-of-core HDF5 DB chunk by chunk and renders the video frames
    through FFMPEG to prevent system memory from being exhausted.
    """
    import subprocess
    import sys
    import time
    from src.jax_visualizer import JAXVisualizer

    start_time = time.time()

    # We initialize JAXVisualizer with a dummy zero-state just to reuse
    # the reliable background renderer and colormap parameters.
    dummy_state = ParticleState(
        position=jnp.zeros((1, 1, 2)),
        velocity=jnp.zeros((1, 1, 2)),
        temperature=jnp.zeros((1, 1)),
        mass=jnp.zeros((1, 1)),
        active=jnp.zeros((1, 1), dtype=bool),
    )
    dummy_viz = JAXVisualizer(config, dummy_state, t_eval, flow_func, temp_func)

    print("Pre-rendering background...")
    bg_raw = dummy_viz._generate_background(width, height, bounds)
    bg_image = jax.image.resize(bg_raw, (height, width, 3), method="cubic")

    total_sim_duration = t_eval[-1] - t_eval[0]
    target_video_duration = total_sim_duration * slow_mo_factor
    num_video_frames = int(target_video_duration * fps)
    sim_indices = np.linspace(0, len(t_eval) - 1, num_video_frames).astype(int)

    print(f"Video Gen: {num_video_frames} frames. Output: {output_path}")

    cmd = get_ffmpeg_cmd(width, height, fps, output_path, use_nvenc_if_available=True)

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    m_init = float(config.m_particle_init)
    t_min = float(min(config.T_room_ref, config.T_wall)) - 5.0
    t_max = float(max(config.T_room_ref, config.T_wall)) + 5.0

    if use_warp:
        from src.warp_rasterizer import rasterize_frame_warp

        def get_frame(p, t, a, m):
            return rasterize_frame_warp(
                p, t, a, m, (height, width), bounds, m_init, 0.8, t_min, t_max
            )
    else:
        from src.rasterizer import rasterize_frame_jax

        render_batch_fn = jax.jit(
            jax.vmap(
                lambda p, t, a, m: rasterize_frame_jax(
                    p, t, a, m, (height, width), bounds, m_init, 0.8, t_min, t_max
                )
            )
        )

    with h5py.File(db_path, "r") as f:
        # Pointers to datasets on disk
        dset_pos = f["position"]
        dset_temp = f["temperature"]
        dset_active = f["active"]
        dset_mass = f["mass"]

        chunk_size = 10 if not use_warp else 1
        for i in range(0, num_video_frames, chunk_size):
            end = min(i + chunk_size, num_video_frames)
            batch_idxs = sim_indices[i:end].tolist()  # discrete slice indices

            # Read explicitly only the requested frame slice from disk into RAM, then to GPU
            batch_pos = jax.device_put(jnp.array(dset_pos[batch_idxs]))
            batch_temp = jax.device_put(jnp.array(dset_temp[batch_idxs]))
            batch_active = jax.device_put(jnp.array(dset_active[batch_idxs]))
            batch_mass = jax.device_put(jnp.array(dset_mass[batch_idxs]))

            if use_warp:
                # Warp relies on single frame parsing
                batch_particles = get_frame(
                    batch_pos[0], batch_temp[0], batch_active[0], batch_mass[0]
                )
                batch_particles = jnp.expand_dims(
                    batch_particles, 0
                )  # [1, H, W, 3] to match loop
            else:
                batch_particles = render_batch_fn(
                    batch_pos, batch_temp, batch_active, batch_mass
                )

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
