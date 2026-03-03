import os
import jax.numpy as jnp
import numpy as np
from src.state import ParticleState
from src.config import SimConfig
from typing import Callable


def format_array_to_xml_string(arr: np.ndarray, num_components: int) -> str:
    """Formats a flat numpy array into a space-separated string for XML."""
    if arr is None or len(arr) == 0:
        return ""
    # Ensure it's flattened
    flat_arr = arr.flatten()
    return " ".join(map(str, flat_arr))


def export_single_frame_to_vtu(
    filename: str,
    points: np.ndarray,
    velocity: np.ndarray,
    temperature: np.ndarray,
    mass: np.ndarray,
    active: np.ndarray,
):
    """
    Exports a single frame of particle data to a VTK Unstructured Grid (.vtu) XML file.

    Args:
        filename (str): The output file path.
        points (np.ndarray): (N, 3) array of positions. Ensure 2D is padded to 3D.
        velocity (np.ndarray): (N, 3) array of velocities.
        temperature (np.ndarray): (N,) array of temperatures.
        mass (np.ndarray): (N,) array of masses.
        active (np.ndarray): (N,) array of active mask components.
    """
    num_particles = points.shape[0]

    # VTK requires 3D points
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)), mode="constant")
    if velocity.shape[1] == 2:
        velocity = np.pad(velocity, ((0, 0), (0, 1)), mode="constant")

    points_str = format_array_to_xml_string(points, 3)
    velocity_str = format_array_to_xml_string(velocity, 3)
    temperature_str = format_array_to_xml_string(temperature, 1)
    mass_str = format_array_to_xml_string(mass, 1)
    active_str = format_array_to_xml_string(active.astype(int), 1)

    # VTK vertex cell structure
    # Cells require 'connectivity', 'offsets', and 'types' (type 1 is VTK_VERTEX)
    connectivity = " ".join(map(str, range(num_particles)))
    offsets = " ".join(map(str, range(1, num_particles + 1)))
    types = " ".join(["1"] * num_particles)

    xml_content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{num_particles}" NumberOfCells="{num_particles}">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {connectivity}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {offsets}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {types}
        </DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float32" Name="Velocity" NumberOfComponents="3" format="ascii">
          {velocity_str}
        </DataArray>
        <DataArray type="Float32" Name="Temperature" format="ascii">
          {temperature_str}
        </DataArray>
        <DataArray type="Float32" Name="Mass" format="ascii">
          {mass_str}
        </DataArray>
        <DataArray type="Int32" Name="Active" format="ascii">
          {active_str}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    with open(filename, "w") as f:
        f.write(xml_content)


def export_fluid_grid_to_vti(
    filename: str,
    flow_func: Callable,
    temp_func: Callable,
    config: SimConfig,
    resolution: int = 30,
    bounds: tuple = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
):
    """
    Exports the background carrier phase (velocity, temperature) to a .vti regular grid.

    Args:
        filename: Output .vti file path.
        flow_func: Simulation flow function.
        temp_func: Simulation temperature function.
        config: SimConfig.
        resolution: Number of samples per dimension.
        bounds: Tuple of ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    """
    import jax
    import jax.numpy as jnp

    print(f"Generating Background VTK Image Data: {filename}")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    dx = (x_max - x_min) / max(1, resolution - 1)
    dy = (y_max - y_min) / max(1, resolution - 1)
    dz = (z_max - z_min) / max(1, resolution - 1)

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    zs = np.linspace(z_min, z_max, resolution)

    # VTK ImageData expects data in fortan-like ordering for X, Y, Z iteration natively,
    # but meshgrid ij creates exactly that shape structure natively when unrolled
    # Fastest varying is X, then Y, then Z.
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    pts_jax = jnp.array(pts)

    vel_jax = jax.vmap(lambda p: flow_func(p, config))(pts_jax)
    vel_np = np.array(vel_jax)

    temperature_str = ""
    if temp_func is not None:
        temp_jax = jax.vmap(lambda p: temp_func(p, config))(pts_jax)
        temp_np = np.array(temp_jax)
        temperature_str = f"""<DataArray type="Float32" Name="FluidTemperature" format="ascii">
          {format_array_to_xml_string(temp_np, 1)}
        </DataArray>"""

    velocity_str = format_array_to_xml_string(vel_np, 3)
    num_points = resolution**3

    xml_content = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {resolution - 1} 0 {resolution - 1} 0 {resolution - 1}" Origin="{x_min} {y_min} {z_min}" Spacing="{dx} {dy} {dz}">
    <Piece Extent="0 {resolution - 1} 0 {resolution - 1} 0 {resolution - 1}">
      <PointData Vectors="FluidVelocity" Scalars="FluidTemperature">
        <DataArray type="Float32" Name="FluidVelocity" NumberOfComponents="3" format="ascii">
          {velocity_str}
        </DataArray>
        {temperature_str}
      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
"""
    with open(filename, "w") as f:
        f.write(xml_content)


def export_simulation_to_vtu(
    output_dir: str,
    history: ParticleState,
    config: SimConfig,
    t_eval: np.ndarray,
    flow_func: Callable,
    temp_func: Callable = None,
    stride: int = 1,
    bounds: tuple = None,
):
    """
    Exports a recorded simulation state history to a sequence of VTU files.

    Args:
        output_dir: Directory where files will be created.
        history: JAX ParticleState containing full history (T, N, D).
        config: SimConfig object.
        t_eval: The time array mapping.
        flow_func: Function to evaluate the background flow field dynamically.
        temp_func: Function representing fluid temperature.
        stride: Framerate temporal subsampling.
    """
    print(f"Exporting VTU Sequence to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if bounds is None:
        bounds = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))

    grid_filename = os.path.join(output_dir, "fluid_background.vti")
    export_fluid_grid_to_vti(
        grid_filename, flow_func, temp_func, config, resolution=30, bounds=bounds
    )

    num_steps = history.position.shape[0]
    frame_idx = 0

    for t_idx in range(0, num_steps, stride):
        pos = np.array(history.position[t_idx])
        vel = np.array(history.velocity[t_idx])
        temp = np.array(history.temperature[t_idx])
        mass = np.array(history.mass[t_idx])
        active = np.array(history.active[t_idx])

        filename = os.path.join(output_dir, f"particles_{frame_idx:04d}.vtu")
        export_single_frame_to_vtu(filename, pos, vel, temp, mass, active)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Exported frame {frame_idx} out of {num_steps // stride}")

    print("VTU sequence export complete!")


def export_dpm_grid_vti(
    filename: str,
    grid_3d: "DPMGrid3D",
) -> None:
    """Exports the accumulated Eulerian 3D DPM fields.

    The exported volume grid includes statistics such as concentration,
    temperature, and evaporation rate in a .vti format suitable for Paraview.

    Args:
        filename: The output file path for the .vti file.
        grid_3d: The DPMGrid3D instance containing the accumulated Eulerian fields.
    """
    print(f"Exporting DPM Volume Grid to {filename}")
    x_min, x_max = grid_3d.x_min, grid_3d.x_max
    y_min, y_max = grid_3d.y_min, grid_3d.y_max
    z_min, z_max = grid_3d.z_min, grid_3d.z_max

    nx, ny, nz = grid_3d.nx, grid_3d.ny, grid_3d.nz

    dx = (x_max - x_min) / max(1, nx)
    dy = (y_max - y_min) / max(1, ny)
    dz = (z_max - z_min) / max(1, nz)

    cell_vol = dx * dy * dz

    # Calculate fields
    res_time = np.array(grid_3d.residence_time)
    concentration = res_time / cell_vol

    mean_temp = np.array(grid_3d.get_mean_temperature())
    mean_diam = np.array(grid_3d.get_mean_diameter())
    mean_evap = np.array(grid_3d.get_mean_evap_rate()) * 1000.0  # g/s
    mean_vel = np.array(grid_3d.get_mean_velocity())  # (nx, ny, nz, 3)

    # VTK ImageData expects data in fortan-like ordering for X, Y, Z iteration natively
    # Best way to ensure correct layout is transposing from (x, y, z) into standard flat layout
    concentration_flat = concentration.transpose(2, 1, 0).flatten()
    mean_temp_flat = mean_temp.transpose(2, 1, 0).flatten()
    mean_diam_flat = mean_diam.transpose(2, 1, 0).flatten()
    mean_evap_flat = mean_evap.transpose(2, 1, 0).flatten()

    # reshape vel to flat layout correctly
    vel_x = mean_vel[..., 0].transpose(2, 1, 0).flatten()
    vel_y = mean_vel[..., 1].transpose(2, 1, 0).flatten()
    vel_z = mean_vel[..., 2].transpose(2, 1, 0).flatten()

    vel_flat = np.column_stack([vel_x, vel_y, vel_z])

    xml_content = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="{x_min} {y_min} {z_min}" Spacing="{dx} {dy} {dz}">
    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">
      <CellData Vectors="MeanVelocity" Scalars="Concentration">
        <DataArray type="Float32" Name="Concentration" format="ascii">
          {format_array_to_xml_string(concentration_flat, 1)}
        </DataArray>
        <DataArray type="Float32" Name="MeanTemperature" format="ascii">
          {format_array_to_xml_string(mean_temp_flat, 1)}
        </DataArray>
        <DataArray type="Float32" Name="MeanDiameter" format="ascii">
          {format_array_to_xml_string(mean_diam_flat, 1)}
        </DataArray>
        <DataArray type="Float32" Name="MeanEvapRate" format="ascii">
          {format_array_to_xml_string(mean_evap_flat, 1)}
        </DataArray>
        <DataArray type="Float32" Name="MeanVelocity" NumberOfComponents="3" format="ascii">
          {format_array_to_xml_string(vel_flat, 3)}
        </DataArray>
      </CellData>
    </Piece>
  </ImageData>
</VTKFile>
"""
    with open(filename, "w") as f:
        f.write(xml_content)


def export_dpm_trajectories_vtu(
    filename: str,
    history: ParticleState,
    active_mask: np.ndarray = None,
    max_trajectories: int = 2000,
) -> None:
    """Exports steady state DPM parcel trajectories as continuous polylines.
    -BROKEN-

    Args:
        filename: Output path for the .vtu file.
        history: ParticleState history.
        active_mask: Optional boolean mask to filter active trajectories.
        max_trajectories: Maximum number of trajectories to export to preserve performance.
    """
    print(f"Exporting DPM Trajectories to {filename}")
    # History shape: (T, N, D)
    pos = np.array(history.position)
    active = np.array(history.active)

    T, N, _ = pos.shape

    # Restrict to max trajectories for performance.
    n_export = min(N, max_trajectories)
    indices = np.linspace(0, N - 1, n_export, dtype=int)

    all_points = []
    offsets = []
    connectivity = []

    current_offset = 0
    valid_streaks_total = 0

    for idx_ct, p_idx in enumerate(indices):
        p_pos = pos[:, p_idx, :]
        p_act = active[:, p_idx]

        # Only keep points where the particle was active
        valid_idxs = np.where(p_act)[0]
        if len(valid_idxs) < 2:
            continue

        valid_pos = p_pos[valid_idxs]

        # Determine continuous streaks
        # Because it's DPM, usually it's active from start to end continuously until it evaporates
        num_points = len(valid_pos)

        all_points.append(valid_pos)

        stream_connectivity = np.arange(current_offset, current_offset + num_points)
        connectivity.append(stream_connectivity)
        current_offset += num_points
        offsets.append(current_offset)
        valid_streaks_total += 1

    if valid_streaks_total == 0:
        print("No valid trajectories found to export.")
        return

    all_points = np.concatenate(all_points, axis=0)

    # Pad 2D trajectories to 3D for VTK validity
    if all_points.shape[1] == 2:
        all_points = np.pad(all_points, ((0, 0), (0, 1)), mode="constant")

    points_str = format_array_to_xml_string(all_points, 3)

    connectivity_flat = np.concatenate(connectivity)
    connectivity_str = " ".join(map(str, connectivity_flat))
    offsets_str = " ".join(map(str, offsets))

    # VTK_POLY_LINE is type 4
    types_str = " ".join(["4"] * valid_streaks_total)

    num_total_points = all_points.shape[0]

    xml_content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{num_total_points}" NumberOfCells="{valid_streaks_total}">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    with open(filename, "w") as f:
        f.write(xml_content)
