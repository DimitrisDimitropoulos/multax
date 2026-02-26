from dataclasses import dataclass
from typing import Tuple, Optional
import jax


@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class PhysicsConfig:
    """Universal physical properties independent of flow geometry.

    Attributes:
        d_particle (float): Particle diameter. Units: [m].
        rho_particle (float): Particle density. Units: [kg/m^3].
        rho_fluid (float): Fluid density. Units: [kg/m^3].
        mu_fluid (float): Fluid dynamic viscosity. Units: [Pa*s].
        g (float): Gravitational acceleration. Units: [m/s^2].
        k_fluid (float): Fluid thermal conductivity. Units: [W/(m*K)].
        cp_fluid (float): Fluid specific heat capacity. Units: [J/(kg*K)].
        cp_particle (float): Particle specific heat capacity. Units: [J/(kg*K)].
        k_particle (float): Particle thermal conductivity. Units: [W/(m*K)].
        M_dispersed (float): Molar mass of dispersed phase. Units: [kg/mol].
        M_continuous (float): Molar mass of continuous phase. Units: [kg/mol].
        latent_heat (float): Latent heat of vaporization. Units: [J/kg].
        P_atm (float): Atmospheric pressure. Units: [Pa].
        D_ref (float): Reference mass diffusivity. Units: [m^2/s].
        RH_room (float): Relative humidity of the far field (0.0 to 1.0).
        T_room_ref (float): Reference temperature of the far field. Units: [K].
        evap_cutoff_ratio (float): Ratio of initial diameter for removal.
        enable_turbulence (bool): Enable stochastic turbulence model.
        turbulence_intensity (float): Intensity of turbulence.
        enable_collisions (bool): Enable particle-particle collisions.
        collision_restitution (float): Restitution coefficient.
    """

    # Particle Properties
    d_particle: float
    rho_particle: float

    # Fluid Properties
    rho_fluid: float
    mu_fluid: float

    # Gravity
    g: float

    # Thermal/Mass Transfer
    k_fluid: float = 0.026
    cp_fluid: float = 1005.0
    cp_particle: float = 4184.0
    k_particle: float = 0.6

    # Phase Change Properties
    M_dispersed: float = 18.015e-3
    M_continuous: float = 28.97e-3
    latent_heat: float = 2.26e6
    P_atm: float = 101325.0
    D_ref: float = 2.6e-5

    # Environment
    RH_room: float = 0.5
    T_room_ref: float = 293.15
    evap_cutoff_ratio: float = 0.1

    # Simulation Flags
    enable_turbulence: bool = False
    turbulence_intensity: float = 0.1
    enable_collisions: bool = False
    collision_restitution: float = 0.9

    def tree_flatten(self):
        return (), tuple(sorted(self.__dict__.items()))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
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

    def get_stokes_number(self, L: float, U: float) -> float:
        r"""Calculates the generalized Stokes number (Stk).

        Uses the effective density (\rho_p + 0.5 \rho_f) to account for
        added mass.

        .. math::
            \tau_{eff} = \frac{(\rho_p + 0.5\rho_f) d_p^2}{18 \mu_f} \\
            Stk = \frac{\tau_{eff}}{L / U}

        Args:
            L (float): Characteristic length scale.
            U (float): Characteristic velocity scale.

        Returns:
            float: Stokes number (dimensionless).
        """
        rho_eff = self.rho_particle + 0.5 * self.rho_fluid
        tau_eff = (rho_eff * self.d_particle**2) / (18 * self.mu_fluid)
        tau_f = L / U
        return tau_eff / tau_f


@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class SimConfig(PhysicsConfig):
    """Scenario-specific configuration including legacy parameters.

    Attributes:
        U_0 (float): Characteristic flow velocity. Units: [m/s].
        alpha (float): Characteristic length scale. Units: [m].
        R_cylinder (float): Cylinder radius. Units: [m].
        wall_x (float): Wall X-coordinate. Units: [m].
        T_wall (float): Wall temperature. Units: [K].
        T_gradient_slope (float): Wall temp gradient. Units: [K/m].
    """

    # Flow Scales
    U_0: float = 1.0
    alpha: float = 1.0

    # Geometry-specific parameters
    R_cylinder: float = 0.0
    wall_x: float = 0.0
    T_wall: float = 300.0
    T_gradient_slope: float = 0.0

    def get_stokes_number(
        self, L: Optional[float] = None, U: Optional[float] = None
    ) -> float:
        """Helper for legacy examples or custom scales."""
        L_eff = (
            L
            if L is not None
            else (self.R_cylinder if self.R_cylinder > 0 else self.alpha)
        )
        U_eff = U if U is not None else self.U_0
        return super().get_stokes_number(L_eff, U_eff)

    @classmethod
    def from_maxey(
        cls,
        W: float,
        A: float,
        U_0: float = 10.0,
        alpha: Optional[float] = None,
        rho_particle: float = 2650.0,
        rho_fluid: float = 1.225,
        mu_fluid: float = 1.81e-5,
        g: float = -9.81,
        **kwargs,
    ) -> "SimConfig":
        """Reverse engineers parameters from Maxey W and A."""
        import numpy as np

        g_mag = abs(g)
        delta_rho = abs(rho_particle - rho_fluid)
        rho_eff = rho_particle + 0.5 * rho_fluid

        if alpha is not None:
            # Mode 2: Fix Alpha, Solve U_0
            U_0_sq = (g_mag * alpha * A * delta_rho) / (abs(W) * rho_eff)
            U_0 = float(np.sqrt(U_0_sq))
            d_sq = (18 * mu_fluid * abs(W) * U_0) / (g_mag * delta_rho)
            d_particle = float(np.sqrt(d_sq))
        else:
            # Mode 1: Fix U_0, Solve Alpha
            v_settle = abs(W) * U_0
            d_sq = (18 * mu_fluid * v_settle) / (g_mag * delta_rho)
            d_particle = float(np.sqrt(d_sq))
            tau = (rho_eff * d_particle**2) / (18 * mu_fluid)
            alpha = float(A * U_0 * tau)

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
    """Configuration for enabling/disabling specific forces."""

    gravity: bool = True
    undisturbed_flow: bool = True
    drag: bool = True
