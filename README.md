# Multax: a JAX Multiphase Flow Solver and Visualization Engine

A high-performance multiphase flow solver implemented in JAX, featuring a custom GPU-accelerated visualization engine. This project leverages functional programming and PyTrees to simulate complex particle-fluid interactions with high efficiency.

## Core Features

- **JAX-Powered Physics:** Fully differentiable and hardware-accelerated (CPU/GPU/TPU) physics engine.
- **Advanced Force Models:** Includes drag, lift, buoyancy, and added mass forces.
- **Thermodynamics:** Supports heat transfer and phase changes.
- **Collision Engine:** Efficient particle-wall collision resolution (and particle-particle).
- **Custom Rasterization:** A bespoke visualization engine that uses JAX to rasterize particle data into high-quality frames/videos.

## Project Structure

### Source Code (`src/`)

The core logic is modularized for clarity and extensibility:

- `solver.py`: Core time-integration and equations of motion (EOM).
- `flow.py`: Definitions for analytical flow fields (e.g., Cellular, Potential Cylinder Flow).
- `physics.py`: Detailed physical models for forces and thermal rates.
- `collisions.py`: Algorithms for collision detection and response.
- `state.py`: PyTree-based state management for JAX compatibility.
- `config.py`: Simulation and force model configurations.
- `grid.py` & `patch.py`: Spatial indexing for optimized neighborhood searches.
- `rasterizer.py` & `jax_visualizer.py`: The custom high-performance rendering pipeline.
- `dpm_solver.py`: Discrete Phase Model (DPM) specific implementations.

### Examples

The root directory contains several `run_*.py` scripts that demonstrate the solver in different scenarios:

- `run_cellular.py`: Particles trapped in cellular vortex structures.


https://github.com/user-attachments/assets/236a50c0-3bf2-4525-a11a-0bd5d8af1ac7




- `run_cylinder.py`: Classic flow around a cylinder with particle trajectories.



https://github.com/user-attachments/assets/78975051-d741-4532-b010-74a31a251327



- `run_wall_thermal.py`: Simulation of thermal boundary layers and their effect on particles.




https://github.com/user-attachments/assets/12d82512-b39f-4200-8b61-903d276802b8





- `run_dpm_wall.py`: Focuses on thermal interactions between particles and walls.


<img width="2400" height="1800" alt="dpm_trajectories" src="https://github.com/user-attachments/assets/35405f87-751e-4e71-8c22-a06f1bd9557c" />


- `run_collisions.py`: Demonstrates the efficiency and accuracy of the collision engine.




https://github.com/user-attachments/assets/284bee9f-30cd-4815-8059-0e6d27f701da




These scripts serve as reference implementations for setting up your own simulations.

## Usage

Most simulations can be executed directly using Python:

```bash
python run_cylinder.py
```

Ensure you have JAX and its dependencies installed to take full advantage of the acceleration.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

