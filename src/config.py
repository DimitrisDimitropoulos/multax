from dataclasses import dataclass
from typing import Tuple, Optional
import jax


@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class SimConfig:
    """
    Configuration for the simulation parameters.
    Includes physical properties and scenario-specific parameters.
    """

    # Particle Properties
    d_particle: float
    rho_particle: float

    # Fluid Properties
    rho_fluid: float
    mu_fluid: float

    # Flow Parameters
    U_0: float
    alpha: float  # Length scale
    g: float

    # Thermal/Mass Transfer (Only for heat and phase change scenarios)
    k_fluid: float = 0.026
    cp_fluid: float = 1005.0
    cp_particle: float = 4184.0
    k_particle: float = 0.6

    # Scenario Specifics, like cylinder radius for flow around cylinder, wall temperature, etc.
    R_cylinder: float = 1.0
    wall_x: float = 0.0
    T_wall: float = 300.0
    T_gradient_slope: float = 0.0
    RH_room: float = 0.5
    T_room_ref: float = 293.15
    evap_cutoff_ratio: float = 0.1

    # Browinan motion parameters for a turbulence like effect (if enabled)
    enable_turbulence: bool = False
    turbulence_intensity: float = 0.1

    def tree_flatten(self):
        #  aux_data must be hashable
        # Convert dict to a sorted tuple of items
        # We leave 'children' empty so these remain static Python values
        # (avoiding Tracers)
        return (), tuple(sorted(self.__dict__.items()))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        #       Reconstruct from the tuple of items
        return cls(**dict(aux_data))

    @property
    def r_particle(self) -> float:
        return self.d_particle / 2.0

    @property
    def m_particle_init(self) -> float:
        return (3.14159 * self.d_particle**3 / 6) * self.rho_particle

    @property
    def m_fluid_init(self) -> float:
        return (3.14159 * self.d_particle**3 / 6) * self.rho_fluid

    def get_prandtl_number(self) -> float:
        return (self.cp_fluid * self.mu_fluid) / self.k_fluid

    def get_stokes_number(self) -> float:
        """
        Calculates the Stokes number (Stk).
        Stk = tau_p / tau_f
        tau_p = (rho_p * d^2) / (18 * mu)
        tau_f = L / U_0
        """
        # Characteristic Length L or alpha
        # Use R_cylinder if it's relevant (non-zero), otherwise alpha
        L = self.R_cylinder if self.R_cylinder > 0 else self.alpha
        tau_p = (self.rho_particle * self.d_particle**2) / (18 * self.mu_fluid)
        tau_f = L / self.U_0
        return tau_p / tau_f

    @classmethod
    def from_maxey(
        cls,
        W: float,
        A: float,
        U_0: float = 10.0,
        alpha: Optional[float] = None,
        rho_particle: float = 2650.0,  # Sand
        rho_fluid: float = 1.225,  # Air
        mu_fluid: float = 1.81e-5,  # Air
        g: float = -9.81,
        **kwargs,
    ) -> "SimConfig":
        """
        Reverse engineers simulation parameters from Maxey parameters W and A.

        Mode 1 (Default): Fix U_0, solve for d_particle and alpha.
        Mode 2: Fix alpha, solve for U_0 and d_particle.

        W: Settling velocity ratio = V_settling / U_0
        A: Inertia parameter (Stokes number related)

        Returns a SimConfig instance with calculated d_particle, alpha, and U_0
        (if needed).
        """
        import numpy as np

        g_mag = abs(g)

        if alpha is not None:
            # Mode 2: Fix Alpha, Solve U_0
            # U_0 = sqrt( (A * g * alpha) / W )
            U_0 = float(np.sqrt((A * g_mag * alpha) / W))
            # Now solve d_particle
            d_sq = (18 * mu_fluid * U_0 * W) / (g_mag * rho_particle)
            d_particle = float(np.sqrt(d_sq))

        else:
            # Mode 1: Fix U_0, Solve Alpha (Default)
            # 1. Calculate d_particle from W
            d_sq = (18 * mu_fluid * U_0 * W) / (g_mag * rho_particle)
            d_particle = float(np.sqrt(d_sq))
            # 2. Calculate alpha from A
            alpha = float((rho_particle * d_particle**2 * U_0) / (18 * mu_fluid * A))

        return cls(
            d_particle=d_particle,
            rho_particle=rho_particle,
            rho_fluid=rho_fluid,
            mu_fluid=mu_fluid,
            U_0=U_0,
            alpha=alpha,
            g=g,
            **kwargs,
        )


@dataclass(frozen=True)
class ForceConfig:
    """Enables or disables specific forces."""

    gravity: bool = True
    undisturbed_flow: bool = True
    drag: bool = True
