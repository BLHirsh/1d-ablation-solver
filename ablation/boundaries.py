"""
Boundary condition types.

Each BC implements:
    flux(T_surface: float) -> float

returning the total heat flux INTO the surface [W/mm²].

Units: W/mm² (flux), °C (temperatures), K conversion where applicable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# Stefan-Boltzmann constant in mm units: 5.6704e-8 W/(m²·K⁴) / 1e6 mm²/m²
SIGMA_MM = 5.6704e-14   # [W/(mm²·K⁴)]


class BoundaryCondition:
    """Abstract base class. All classes must implement flux(T_surface)."""
    def flux(self, T_surface: float) -> float:
        raise NotImplementedError


# ── Primitive boundary conditions ────────────────────────────────────────────

@dataclass
class HeatFlux(BoundaryCondition):
    """Fixed incident heat flux [W/mm²].
    Flux is independent of surface T.

    q > 0: energy flows into the body.
    """
    q: float  # [W/mm²]

    def flux(self, T_surface: float) -> float:
        return self.q

    def __str__(self) -> str:
        return f"HeatFlux(q={self.q} W/mm²)"


@dataclass
class Convection(BoundaryCondition):
    """Newton's law of cooling: q = h·(T_env - T_s).

    h  [W/(mm²·K)]  | heat transfer coefficient
    T_env [°C]      | environment / adiabatic wall temperature
    """
    h: float
    T_env: float

    def flux(self, T_surface: float) -> float:
        return self.h * (self.T_env - T_surface)

    def __str__(self) -> str:
        return f"Convection(h={self.h} W/mm²K, T_env={self.T_env}°C)"


@dataclass
class Radiation(BoundaryCondition):
    """Grey-body radiation exchange: q = ε·σ·(T_env⁴ - T_s⁴).

    NOTE: Temperatures are converted to Kelvin internally.

    emissivity [0–1]  | surface emissivity
    T_env [°C]        | radiation source/sink temperature
    """
    emissivity: float
    T_env: float

    def flux(self, T_surface: float) -> float:
        Ts_K = T_surface + 273.15
        Te_K = self.T_env + 273.15
        return self.emissivity * SIGMA_MM * (Te_K**4 - Ts_K**4)

    def __str__(self) -> str:
        return f"Radiation(ε={self.emissivity}, T_env={self.T_env}°C)"


@dataclass
class Adiabatic(BoundaryCondition):
    """Zero-flux (insulated) boundary.

    Use for any surface with no external heat exchange.
    """
    def flux(self, T_surface: float) -> float:
        return 0.0

    def __str__(self) -> str:
        return "Adiabatic"


# ── Composite boundary condition ─────────────────────────────────────────────

@dataclass
class Combined(BoundaryCondition):
    """Linear superposition of multiple boundary conditions.
    """
    components: List[BoundaryCondition]

    def flux(self, T_surface: float) -> float:
        return sum(c.flux(T_surface) for c in self.components)

    def __str__(self) -> str:
        parts = ' + '.join(str(c) for c in self.components)
        return f"Combined({parts})"
