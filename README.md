# Multax: a JAX Multiphase Flow Solver and Visualization Engine

A high-performance multiphase flow solver implemented in JAX, featuring a custom GPU-accelerated visualization engine. This project leverages functional programming, PyTrees, and specialized GPU kernels to simulate complex particle-fluid interactions with high efficiency.

## Core Features

- **JAX-Powered Physics:** Fully differentiable and hardware-accelerated (CPU/GPU/TPU) physics engine compiled via XLA. Allows parallel integration of equations of motion for millions of particles.
- **Advanced Force Models:** Includes drag (Stokes with corrections), gravity, undisturbed flow (pressure gradient), and added mass forces.
- **Thermodynamics & Phase Change:** Supports heat transfer (convection via Ranz-Marshall) and particle vaporization/evaporation based on Fick's law of diffusion and the Magnus equation.
- **Discrete Phase Model (DPM):** Simulates steady-state multiphase flows using representative computational parcels for efficient Eulerian-Lagrangian coupling. By tracking the statistical trajectories of these parcels, the engine generates dense Eulerian fields (e.g., local phase concentrations, mean particle diameters, and volumetric evaporation rates) allowing deep insight into the macroscopic behavior of the spray or dispersion.
- **Versatile Phenomenological Modeling:** Designed with extreme modularity in mind. Users can effortlessly construct simulations from a diverse pool of phenomena—combining custom analytical flow fields (`FlowFunc`), temperature gradients (`TempFunc`), and selectively enabling physical forces (drag, lift, gravity, added mass) to rapidly prototype completely novel multiphase scenarios.
- **Hardware Scalability & HDF5 Orchestration (Overcoming RAM Limits):** Engineered to leverage lower-end hardware gracefully while scaling to massive datasets. The `orchestrator.py` chunks the JAX time-integration into memory-safe blocks that fit within consumer GPU VRAM. State history is streamed sequentially into an out-of-core **HDF5 database**. Later, FFMPEG fetches and renders video frames chunk-by-chunk directly from disk, fully preventing Out-Of-Memory (OOM) crashes even for simulations with millions of time steps.

## Advanced Engines

### Collision Engine (Warp)
Particle-particle and particle-wall collisions are resolved using **NVIDIA Warp** (`warp_collisions.py`, `warp_ccd_lcp.py`).
- Utilizes fast, thread-safe atomic spatial hash grids executed natively on the GPU.
- Achieves $\mathcal{O}(N)$ neighborhood searches without costly CPU-GPU memory transfers.
- Supports both simple restitution-based impulses and advanced **Continuous Collision Detection (CCD)** coupled with **Linear Complementarity Problem (LCP)** solvers. CCD sweeps particle trajectories temporally to completely eradicate "tunneling" effects for high-speed projectiles, while the LCP solver mathematically guarantees stable, non-penetrating resting states for simultaneous, multi-body contacts (e.g., dense particle clustering or settling).
- Integrates seamlessly with JAX arrays via Warp's `jax_experimental` FFI.

### Custom Rasterization
A bespoke visualization engine built with JAX and Warp (`rasterizer.py`, `warp_rasterizer.py`, `warp_visualizer.py`). 
- Uses Gaussian splatting to project particle size and temperature directly onto an image grid. 
- Massively parallelized on the GPU, enabling real-time generation of high-quality frames and videos.

## Project Structure

### Source Code (`src/`)

The core logic is modularized for clarity and extensibility:

- `solver.py`: Core time-integration (Symplectic Euler) and equations of motion (EOM).
- `flow.py`: Definitions for analytical flow fields (e.g., Cellular, Potential Cylinder Flow, Hiemenz wall flow).
- `physics.py`: Detailed physical models for forces, thermodynamics, and evaporation rates.
- `state.py`: PyTree-based state management for JAX compatibility.
- `config.py`: Simulation and force model configurations.
- `orchestrator.py`: Out-of-core HDF5 database manager for memory-safe execution on lower-end hardware.
- `warp_collisions.py` & `warp_ccd_lcp.py`: GPU-native collision detection and resolution using NVIDIA Warp.
- `rasterizer.py` & `warp_rasterizer.py`: Custom high-performance rendering pipelines.

### Examples

The codebase includes a large, robust, and highly diverse collection of examples (`run_*.py` scripts) that serve both as mathematical demonstrations and architectural templates. These comprehensively illustrate how to stack the engine's various modules to simulate complex physical scenarios:

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

- `run_custom_fields.py`: Showcases how to define arbitrary carrier velocity and temperature fields for specialized scenarios.



https://github.com/user-attachments/assets/70d26dca-e346-4bdf-9ff6-44f5041ab016



These scripts serve as reference implementations for setting up your own simulations.

## Usage

Most simulations can be executed directly using Python:

```bash
python run_cylinder.py
```

Ensure you have JAX, NVIDIA Warp, and their dependencies installed to take full advantage of the GPU acceleration.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

