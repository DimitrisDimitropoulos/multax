from dataclasses import dataclass
from typing import Tuple, Optional
import jax


@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class SimConfig:
    """Configuration for the simulation parameters.

    Includes physical properties, flow parameters, and scenario-specific settings.

    Attributes:
        d_particle (float): Particle diameter. Units: [m].
        rho_particle (float): Particle density. Units: [kg/m^3].
        rho_fluid (float): Fluid density. Units: [kg/m^3].
        mu_fluid (float): Fluid dynamic viscosity. Units: [Pa*s] or [kg/(m*s)].
        U_0 (float): Characteristic flow velocity. Units: [m/s].
        alpha (float): Characteristic length scale of the flow. Units: [m].
        g (float): Gravitational acceleration. Units: [m/s^2].
        k_fluid (float): Fluid thermal conductivity. Units: [W/(m*K)].
        cp_fluid (float): Fluid specific heat capacity. Units: [J/(kg*K)].
        cp_particle (float): Particle specific heat capacity. Units: [J/(kg*K)].
        k_particle (float): Particle thermal conductivity. Units: [W/(m*K)].
        M_dispersed (float): Molar mass of dispersed phase (e.g., Water). Units: [kg/mol].
        M_continuous (float): Molar mass of continuous phase (e.g., Air). Units: [kg/mol].
        latent_heat (float): Latent heat of vaporization. Units: [J/kg].
        P_atm (float): Atmospheric pressure. Units: [Pa].
        D_ref (float): Reference mass diffusivity. Units: [m^2/s].
        R_cylinder (float): Cylinder radius (if applicable). Units: [m].
        wall_x (float): X-coordinate of the vertical wall (if applicable). Units: [m].
        T_wall (float): Temperature of the heated wall. Units: [K].
        T_gradient_slope (float): Temperature gradient slope away from wall. Units: [K/m].
        RH_room (float): Relative humidity of the far field (0.0 to 1.0).
        T_room_ref (float): Reference temperature of the far field. Units: [K].
        evap_cutoff_ratio (float): Ratio of initial diameter below which particle is removed.
        enable_turbulence (bool): Whether to enable stochastic turbulence model.
        turbulence_intensity (float): Intensity of turbulence (fraction of mean velocity).
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

    # Phase Change Properties
    M_dispersed: float = 18.015e-3  # Molar mass of dispersed phase (e.g. Water)
    M_continuous: float = 28.97e-3  # Molar mass of continuous phase (e.g. Air)
    latent_heat: float = 2.26e6  # Latent heat of vaporization for water (L_vap)
    P_atm: float = 101325.0  # Atmospheric pressure
    D_ref: float = 2.6e-5  # Reference mass diffusivity (D_AB)

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

    # Collision parameters
    enable_collisions: bool = False
    collision_restitution: float = 0.9

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
        """Particle radius. Units: [m]."""
        return self.d_particle / 2.0

    @property
    def m_particle_init(self) -> float:
        """Initial particle mass. Units: [kg]."""
        return (3.14159 * self.d_particle**3 / 6) * self.rho_particle

    @property
    def m_fluid_init(self) -> float:
        """Mass of fluid displaced by initial particle volume. Units: [kg]."""
        return (3.14159 * self.d_particle**3 / 6) * self.rho_fluid

    def get_prandtl_number(self) -> float:
        """Calculates the Prandtl number of the fluid.

        Returns:
            float: Prandtl number (dimensionless).
        """
        return (self.cp_fluid * self.mu_fluid) / self.k_fluid

    def get_stokes_number(self) -> float:
        r"""Calculates the generalized Stokes number (Stk).

            Uses the effective density (\rho_p + 0.5 \rho_f) to account for 
            added mass, ensuring validity for both heavy particles (aerosols) 
            and light particles (bubbles).

            .. math::
                \tau_{eff} = \frac{(\rho_p + 0.5\rho_f) d_p^2}{18 \mu_f} \\
                Stk = \frac{\tau_{eff}}{L / U_0}

            Returns:
                float: Stokes number (dimensionless).
            """
        # Characteristic Length
        L = self.R_cylinder if self.R_cylinder > 0 else self.alpha
        # Effective Density: Particle Mass + Added Mass (0.5 * Fluid Mass)
        rho_eff = self.rho_particle + 0.5 * self.rho_fluid
        tau_eff = (rho_eff * self.d_particle**2) / (18 * self.mu_fluid)
        tau_f = L / self.U_0
        return tau_eff / tau_f

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
        """Reverse engineers simulation parameters from Maxey parameters W and A.

        Mode 1 (Default): Fix U_0, solve for d_particle and alpha.
        Mode 2: Fix alpha, solve for U_0 and d_particle.

        Args:
            W (float): Settling velocity ratio (V_settling / U_0).
            A (float): Inertia parameter (related to Stokes number).
            U_0 (float, optional): Characteristic velocity. Defaults to 10.0.
            alpha (float, optional): Characteristic length scale. If provided, U_0 is recalculated.
            rho_particle (float, optional): Particle density. Defaults to 2650.0.
            rho_fluid (float, optional): Fluid density. Defaults to 1.225.
            mu_fluid (float, optional): Fluid viscosity. Defaults to 1.81e-5.
            g (float, optional): Gravity. Defaults to -9.81.
            **kwargs: Additional SimConfig arguments.

        Returns:
            SimConfig: Configured simulation instance.
        """
        import numpy as np

        g_mag = abs(g)
        # Effective Density Difference for Settling (Buoyancy)
        delta_rho = abs(rho_particle - rho_fluid)

        # Effective Density for Inertia (Particle Mass + Added Mass)
        # m_eff = Vol * (rho_p + 0.5 * rho_f)
        rho_eff = rho_particle + 0.5 * rho_fluid

        if alpha is not None:
            # Mode 2: Fix Length Scale (alpha), Solve for U_0 and d_particle
            # This is a coupled system, we solve it algebraically.
            #  Stokes settling: W * U_0 = (g * delta_rho * d^2) / (18 * mu)
            #  Inertia param:   A = (alpha / U_0) * (1 / tau)
            #    where tau = (rho_eff * d^2) / (18 * mu)
            # Combine to eliminate d^2 and solve for U_0:
            # U_0^2 = (g * alpha * A * delta_rho) / (W * rho_eff)
            U_0_sq = (g_mag * alpha * A * delta_rho) / (abs(W) * rho_eff)
            U_0 = float(np.sqrt(U_0_sq))
            # Now solve for d_particle using the Settling velocity equation
            # d^2 = (18 * mu * W * U_0) / (g * delta_rho)
            d_sq = (18 * mu_fluid * abs(W) * U_0) / (g_mag * delta_rho)
            d_particle = float(np.sqrt(d_sq))

        else:
            # Mode 1: Fix U_0, Solve for d_particle and Length Scale (alpha)
            # Calculate d_particle from W (Settling)
            # W * U_0 = V_settle
            # d^2 = (18 * mu * V_settle) / (g * delta_rho)
            v_settle = abs(W) * U_0
            d_sq = (18 * mu_fluid * v_settle) / (g_mag * delta_rho)
            d_particle = float(np.sqrt(d_sq))
            # Calculate Relaxation Time (tau) using Effective Mass
            # tau = m_eff / (3 * pi * mu * d) = (rho_eff * d^2) / (18 * mu)
            tau = (rho_eff * d_particle**2) / (18 * mu_fluid)
            # Calculate alpha (Length Scale) from A
            # A = alpha / (U_0 * tau)  ->  alpha = A * U_0 * tau
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
    """Configuration for enabling/disabling specific forces.

    Attributes:
        gravity (bool): Enable gravitational force.
        undisturbed_flow (bool): Enable undisturbed flow force (pressure + buoyancy).
        drag (bool): Enable drag force.
    """

    gravity: bool = True
    undisturbed_flow: bool = True
    drag: bool = True
