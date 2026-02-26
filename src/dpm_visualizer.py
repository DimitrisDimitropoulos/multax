import matplotlib.pyplot as plt
import numpy as np
from src.grid import DPMGrid
from src.flow import FlowFunc, TempFunc
from src.config import SimConfig
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import jax.numpy as jnp


class DPMVisualizer:
    def __init__(
        self,
        grid: DPMGrid,
        config: SimConfig = None,
        history=None,
        t_eval=None,
        flow_func: FlowFunc = None,
        temp_func: TempFunc = None,
    ):
        self.grid = grid
        self.config = config
        self.history = history  # history is ParticleState (T, N)
        self.t_eval = t_eval
        self.flow_func = flow_func
        self.temp_func = temp_func

    def plot_fields(self, filename: str = "dpm_stats.png"):
        """
        Plots 2x2 Grid:
        1. Concentration [m^-3] + Streamlines
        2. Mean Temperature [K]
        3. Mean Diameter [m]
        4. Mean Evaporation Rate [g/s]
        """
        dx = (self.grid.x_max - self.grid.x_min) / self.grid.nx
        dy = (self.grid.y_max - self.grid.y_min) / self.grid.ny
        cell_vol = dx * dy

        res_time = np.array(self.grid.residence_time)
        concentration = res_time / cell_vol
        mean_temp = np.array(self.grid.get_mean_temperature())
        mean_diam = np.array(self.grid.get_mean_diameter())  # meters
        # Grid stores kg/s. Convert to g/s.
        mean_evap = np.array(self.grid.get_mean_evap_rate()) * 1000.0

        # Masking
        mask = res_time > 1e-10
        masked_conc = np.where(mask, concentration, np.nan)
        masked_temp = np.where(mask, mean_temp, np.nan)
        masked_diam = np.where(mask, mean_diam, np.nan)
        masked_evap = np.where(mask, mean_evap, np.nan)

        # Flow Field (Optimization: Downsample for Streamplot)
        nx_stream, ny_stream = 50, 50
        x_s = np.linspace(self.grid.x_min, self.grid.x_max, nx_stream)
        y_s = np.linspace(self.grid.y_min, self.grid.y_max, ny_stream)
        X_s, Y_s = np.meshgrid(x_s, y_s)

        if self.config and self.flow_func:
            import jax

            pts = np.stack([X_s.ravel(), Y_s.ravel()], axis=1)
            vmap_flow = jax.vmap(lambda p: self.flow_func(p, self.config))
            vel_vecs = vmap_flow(pts)
            U_s = vel_vecs[:, 0].reshape(X_s.shape)
            V_s = vel_vecs[:, 1].reshape(X_s.shape)
        else:
            U_s, V_s = None, None

        extent = [self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max]

        # Setup Figure 2x2
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Concentration
        vmin_c = (
            np.nanmin(masked_conc[masked_conc > 0])
            if np.any(masked_conc > 0)
            else 1e-10
        )
        vmax_c = np.nanmax(masked_conc) if np.any(masked_conc > 0) else 1.0

        im1 = axes[0, 0].imshow(
            masked_conc.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="inferno",
            norm=colors.LogNorm(vmin=vmin_c, vmax=vmax_c),
        )
        if U_s is not None:
            axes[0, 0].streamplot(
                X_s,
                Y_s,
                U_s,
                V_s,
                color="cyan",
                linewidth=0.8,
                density=1.0,
                arrowsize=1.0,
            )

        axes[0, 0].set_title("Particle Concentration [#/m^3] & Flow")
        plt.colorbar(im1, ax=axes[0, 0], label="Concentration")

        # Temperature
        im2 = axes[0, 1].imshow(
            masked_temp.T, origin="lower", extent=extent, aspect="auto", cmap="inferno"
        )
        axes[0, 1].set_title("Mean Particle Temperature [K]")
        plt.colorbar(im2, ax=axes[0, 1], label="Temp [K]")

        # Diametera
        im3 = axes[1, 0].imshow(
            masked_diam.T, origin="lower", extent=extent, aspect="auto", cmap="inferno"
        )
        axes[1, 0].set_title("Mean Particle Diameter [m]")
        plt.colorbar(im3, ax=axes[1, 0], label="Diameter [m]")

        # Evaporation Rate
        im4 = axes[1, 1].imshow(
            masked_evap.T, origin="lower", extent=extent, aspect="auto", cmap="inferno"
        )
        axes[1, 1].set_title("Mean Evaporation Rate [g/s]")
        plt.colorbar(im4, ax=axes[1, 1], label="Evap Rate [g/s]")

        for ax in axes.flat:
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            # Overlay Grid Lines (White, faint)
            x_edges = np.linspace(self.grid.x_min, self.grid.x_max, self.grid.nx + 1)
            y_edges = np.linspace(self.grid.y_min, self.grid.y_max, self.grid.ny + 1)

            ax.vlines(
                x_edges,
                self.grid.y_min,
                self.grid.y_max,
                colors="white",
                linestyles="-",
                linewidth=0.3,
                alpha=0.15,
            )
            ax.hlines(
                y_edges,
                self.grid.x_min,
                self.grid.x_max,
                colors="white",
                linestyles="-",
                linewidth=0.3,
                alpha=0.15,
            )

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved 4-panel DPM statistics plot to {filename}")

    def plot_trajectories(self, filename: str = "dpm_trajectories.png"):
        """
        Plots 4 subplots with Carrier Background (Flow & Temp)
        and Parcel Trajectories foreground colored by:
        1. Local Concentration [#/m^3]
        2. Temperature
        3. Diameter
        4. Evaporation Rate
        """
        if self.history is None or self.config is None:
            print("Cannot plot trajectories: missing history or config.")
            return

        # Prepare the background flow and temperature fields on a coarser grid
        # for visualization
        nx_bg, ny_bg = 50, 50
        x = np.linspace(self.grid.x_min, self.grid.x_max, nx_bg)
        y = np.linspace(self.grid.y_min, self.grid.y_max, ny_bg)
        X, Y = np.meshgrid(x, y)
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)

        import jax

        # Carrier Temp
        if self.temp_func:
            vmap_temp = jax.vmap(lambda p: self.temp_func(p, self.config))
            carrier_T = vmap_temp(pts).reshape(X.shape)
        else:
            carrier_T = np.zeros(X.shape)

        # Carrier Flow
        if self.flow_func:
            vmap_flow = jax.vmap(lambda p: self.flow_func(p, self.config))
            vel_vecs = vmap_flow(pts)
            U = vel_vecs[:, 0].reshape(X.shape)
            V = vel_vecs[:, 1].reshape(X.shape)
        else:
            U = np.zeros(X.shape)
            V = np.zeros(X.shape)

        # Extract Trajectory Data
        # history: ParticleState (T, N)
        pos = np.array(self.history.position)  # (T, N, 2)
        vel = np.array(self.history.velocity)  # (T, N, 2)
        temp = np.array(self.history.temperature)  # (T, N)
        mass = np.array(self.history.mass)  # (T, N)
        active = np.array(self.history.active)  # (T, N)

        # Derived Fields
        # Local Concentration
        # We need to map particle positions to grid indices
        dx = (self.grid.x_max - self.grid.x_min) / self.grid.nx
        dy = (self.grid.y_max - self.grid.y_min) / self.grid.ny
        cell_vol = dx * dy
        res_time = np.array(self.grid.residence_time)
        grid_conc = res_time / cell_vol

        # Flatten pos to list of points (T*N, 2)
        flat_pos = pos.reshape(-1, 2)
        # Use grid helper to get indices
        # Cast to JAX array for the helper
        jax_pos = jnp.array(flat_pos)
        ix_flat, iy_flat = self.grid.get_cell_indices(jax_pos)
        # Cast back to numpy for indexing numpy array
        ix_flat = np.array(ix_flat)
        iy_flat = np.array(iy_flat)
        # Sample concentration
        conc_flat = grid_conc[ix_flat, iy_flat]
        # Reshape back to (T, N)
        parcel_conc = conc_flat.reshape(pos.shape[0], pos.shape[1])

        # Diameter [m]
        diam = np.cbrt((6.0 * mass) / (np.pi * self.config.rho_particle))  # meters
        # C. Evap Rate
        # Backwards difference
        dM = np.diff(mass, axis=0, prepend=mass[0:1])
        # DT Logic
        if self.t_eval is not None and len(self.t_eval) > 1:
            dt = float(self.t_eval[1] - self.t_eval[0])
        else:
            print("Warning: t_eval not provided to visualizer. Using default dt=0.005.")
            dt = 0.005

        evap_rate = dM / dt * 1000.0  # g/s

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        titles = [
            "Local Concentration [#/m^3]",
            "Parcel Temperature [K]",
            "Parcel Diameter [m]",
            "Parcel Evap Rate [g/s]",
        ]
        datas = [parcel_conc, temp, diam, evap_rate]
        cmaps = ["inferno", "inferno", "inferno", "inferno"]

        # For Concentration, we might want log scale normalization, but
        # LineCollection norm logic is linear by default.
        # We can pass a LogNorm to LineCollection.
        for i, ax in enumerate(axes.flat):
            # Carrier Temp Contour as backgroundback
            im_bg = ax.contourf(X, Y, carrier_T, levels=20, cmap="Greys", alpha=0.5)

            # Streamlines background
            ax.streamplot(
                X, Y, U, V, color="gray", linewidth=0.5, density=0.8, arrowsize=0.8
            )

            # Trajectories
            n_plot = min(100, pos.shape[1])
            indices = np.linspace(0, pos.shape[1] - 1, n_plot, dtype=int)

            data_field = datas[i]
            cmap_name = cmaps[i]

            all_segments = []
            all_values = []

            for p_idx in indices:
                p_pos = pos[:, p_idx, :]
                p_act = active[:, p_idx]
                p_val = data_field[:, p_idx]

                valid_t = np.where(p_act[:-1] & p_act[1:])[0]

                if len(valid_t) == 0:
                    continue

                p1 = p_pos[valid_t]
                p2 = p_pos[valid_t + 1]
                segments = np.stack([p1, p2], axis=1)
                all_segments.append(segments)
                all_values.append(p_val[valid_t])

            if not all_segments:
                continue

            flat_segments = np.concatenate(all_segments, axis=0)
            flat_values = np.concatenate(all_values, axis=0)

            # Determine Normalization
            if i == 0:  # Concentration - Log Norm
                vmin = (
                    np.nanmin(flat_values[flat_values > 0])
                    if np.any(flat_values > 0)
                    else 1e-10
                )
                vmax = np.nanmax(flat_values) if np.any(flat_values > 0) else 1.0
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = plt.Normalize(np.nanmin(flat_values), np.nanmax(flat_values))

            lc = LineCollection(
                flat_segments, cmap=cmap_name, norm=norm, linewidth=1.0, alpha=0.8
            )
            lc.set_array(flat_values)

            ax.add_collection(lc)
            ax.set_xlim(self.grid.x_min, self.grid.x_max)
            ax.set_ylim(self.grid.y_min, self.grid.y_max)

            ax.set_title(titles[i])
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")

            plt.colorbar(lc, ax=ax, label=titles[i])

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved trajectory plot to {filename}")

