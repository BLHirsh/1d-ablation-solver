"""
Material property library.

Unit system: mm · s · J · g · °C  (consistent with solver)
  k          [W/(mm·K)]   thermal conductivity
  rho        [g/mm³]      density
  c_p        [J/(g·K)]    specific heat
  T_ablation [°C]         ablation/melt onset; use float('inf') for non-ablating
  L          [J/g]        effective latent heat of ablation (0 for non-ablating)

Every property is a callable function of temperature:
  k_func(T)   → k   [W/(mm·K)]
  rho_func(T) → rho [g/mm³]
  cp_func(T)  → c_p [J/(g·K)]
  L_func(T)   → L   [J/g]

Use _const(v) for properties that do not vary with temperature.
Use _make_interp(...) for tabulated piecewise-linear data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# ── Property function builders ───────────────────────────────────────────────

def _const(v: float) -> Callable[[float], float]:
    """Wrap a scalar into a constant-property function (returns v for any T)."""
    def f(T: float) -> float: return float(v)
    return f


def _make_interp(T_pts: list[float], v_pts: list[float],
                 scale: float = 1.0) -> Callable[[float], float]:
    """Piecewise-linear interpolation clamped to the endpoint values.

    Args:
        T_pts:  Temperature breakpoints [°C], strictly increasing.
        v_pts:  Property values at each breakpoint (before scaling).
        scale:  Multiply every v_pt by this factor (e.g. 1e-3 to convert
                W/(m·K) → W/(mm·K), or kg/m³ → g/mm³).
    """
    T_arr = np.array(T_pts, dtype=float)
    v_arr = np.array(v_pts, dtype=float) * scale

    def interp(T: float) -> float:
        return float(np.interp(T, T_arr, v_arr))

    return interp


# ── Material dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Material:
    """
    1-D material.

    All three property functions are required.  Use _const(v) for materials
    whose properties do not vary with temperature.

    Parameters
    ----------
    name        : display name
    k_func      : conductivity    k(T)  [W/(mm·K)]
    rho_func    : density         ρ(T)  [g/mm³]
    cp_func     : specific heat  cp(T)  [J/(g·K)]
    L_func      : latent heat     L(T)  [J/g]
    T_ablation  : ablation/melt onset   [°C]  (float('inf') → non-ablating)
    color       : hex colour for plots
    """
    name:       str
    k_func:     Callable[[float], float] = field(compare=False, hash=False, repr=False)
    rho_func:   Callable[[float], float] = field(compare=False, hash=False, repr=False)
    cp_func:    Callable[[float], float] = field(compare=False, hash=False, repr=False)
    L_func:     Callable[[float], float] = field(compare=False, hash=False, repr=False)
    T_ablation: float
    color:      str = '#888888'

    # ------------------------------------------------------------------
    # T-dependent property evaluation
    # ------------------------------------------------------------------

    def k_at(self, T: float) -> float:
        """Conductivity [W/(mm·K)] at T [°C]."""
        return self.k_func(T)

    def rho_at(self, T: float) -> float:
        """Density [g/mm³] at T [°C]."""
        return self.rho_func(T)

    def cp_at(self, T: float) -> float:
        """Specific heat [J/(g·K)] at T [°C]."""
        return self.cp_func(T)

    def rhocp_at(self, T: float) -> float:
        """Volumetric heat capacity [J/(mm³·K)] at T [°C]."""
        return self.rho_at(T) * self.cp_at(T)

    def alpha_at(self, T: float) -> float:
        """Thermal diffusivity [mm²/s] at T [°C]."""
        return self.k_at(T) / max(self.rhocp_at(T), 1e-30)

    def L_at(self, T: float) -> float:
        """Effective latent heat [J/g] at T [°C]."""
        return self.L_func(T)

    # ------------------------------------------------------------------
    # Scalar reference values (for display and quick checks)
    # k / rho / c_p at 20 °C; L at T_ablation (L has no meaning at ambient)
    # ------------------------------------------------------------------

    @property
    def k(self) -> float:
        """Reference conductivity [W/(mm·K)] at 20 °C."""
        return self.k_at(20.0)

    @property
    def rho(self) -> float:
        """Reference density [g/mm³] at 20 °C."""
        return self.rho_at(20.0)

    @property
    def c_p(self) -> float:
        """Reference specific heat [J/(g·K)] at 20 °C."""
        return self.cp_at(20.0)

    @property
    def rho_cp(self) -> float:
        """Reference volumetric heat capacity [J/(mm³·K)] at 20 °C."""
        return self.rhocp_at(20.0)

    @property
    def alpha(self) -> float:
        """Reference thermal diffusivity [mm²/s] at 20 °C."""
        return self.alpha_at(20.0)

    @property
    def L(self) -> float:
        """Effective latent heat [J/g] at T_ablation."""
        return self.L_at(self.T_ablation)

    @property
    def rho_L(self) -> float:
        """Volumetric latent heat [J/mm³] at T_ablation."""
        return self.rho_at(self.T_ablation) * self.L_at(self.T_ablation)

    def __repr__(self) -> str:
        return (f"Material({self.name!r}  α={self.alpha:.3g} mm²/s  "
                f"T_ab={self.T_ablation}°C  L={self.L:.0f} J/g)")


# ── Metals ───────────────────────────────────────────────────────────────────
# Source: Engineering Toolbox

_Al_k_func = _make_interp(
    T_pts=[-73,   0,  127,  327,  527],
    v_pts=[ 237, 236,  240,  232,  220],  # W/(m·K)
    scale=1e-3,
)
_Al_rho_func = _make_interp(
    T_pts=[-123.15, -73.15, -23.15,  33.15, 133.15, 333.15],
    v_pts=[   2726,   2719,   2710,   2701,   2681,   2639],  # kg/m³
    scale=1e-6,
)
_Al_cp_func = _make_interp(
    T_pts=[-123.15, -73.15, -23.15,  33.15, 133.15, 333.15],
    v_pts=[  0.683,  0.797,  0.859,  0.902,  0.949,  1.042],  # J/(g·K)
    scale=1.0,
)

ALUMINIUM = Material(
    name='Aluminium',
    k_func=_Al_k_func,
    rho_func=_Al_rho_func,
    cp_func=_Al_cp_func,
    L_func=_const(397.0),   # heat of fusion [J/g]
    T_ablation=660.0,        # solidus [°C]
    color='#4e9af1',
)


_Cu_k_func = _make_interp(
    T_pts=[-126.85, -76.85, -26.85,  36.85, 86.85, 126.85, 326.85, 526.85, 726.85, 926.85],
    v_pts=[ 480, 429,  413,  406,  401, 393, 379, 366, 352, 339],  # W/(m·K)
    scale=1e-3,
)

_Cu_rho_func = _make_interp(
    T_pts=[-126.85, -76.85, -26.85,  36.85, 86.85, 126.85, 326.85, 526.85, 726.85, 926.85],
    v_pts=[ 9009, 8992,  8973,  8951,  8930, 8884, 8787, 8642, 8568, 8458],  # kg/m³
    scale=1e-6,
)

_Cu_cp_func = _make_interp(
    T_pts=[-126.85, -76.85, -26.85,  36.85, 86.85, 126.85, 326.85, 526.85, 726.85, 926.85],
    v_pts=[ 0.254, 0.323,  0.357,  0.377,  0.386, 0.396, 0.431, 0.448, 0.446, 0.480],  # J/(g·K)
    scale=1.0,
)

COPPER = Material(
    name='Copper',
    k_func=_Cu_k_func,
    rho_func=_Cu_rho_func,
    cp_func=_Cu_cp_func,
    L_func=_const(205.0),   # heat of fusion [J/g]
    T_ablation=1085.0,
    color='#b87333',
)

# ── Ceramics / oxides ────────────────────────────────────────────────────────

ALUMINA = Material(
    name='Alumina (Al₂O₃)',
    k_func=_const(0.030),
    rho_func=_const(3.95e-3),
    cp_func=_const(0.880),
    L_func=_const(1000.0),   # https://www.azom.com/properties.aspx?ArticleID=52
    T_ablation=2072.0,
    color='#f17a4e',
)

SILICA = Material(
    name='Fused Silica (SiO₂)',
    k_func=_const(0.0014),
    rho_func=_const(2.20e-3),
    cp_func=_const(0.720),
    L_func=_const(156.4),    # https://www.sciencedirect.com/science/article/abs/pii/0016703782903830
    T_ablation=1600.0,
    color='#cce8ff',
)

# ── Ablative TPS materials ───────────────────────────────────────────────────

# PICA — Phenolic Impregnated Carbon Ablator, ~260 kg/m³
# Milos & Chen (2009), AIAA-2009-4232, virgin-material curve
_PICA_k_func = _make_interp(
    T_pts=[  20,  100,  200,  300,  400,  500,  600],
    v_pts=[0.22, 0.31, 0.33, 0.36, 0.40, 0.45, 0.50],    # W/(m·K)
    scale=1e-3,
)
_PICA_cp_func = _make_interp(
    T_pts=[  20,  100,  200,  300,  400,  500,  600],
    v_pts=[1000, 1060, 1140, 1220, 1300, 1370, 1420],     # J/(kg·K)
    scale=1e-3,
)

PICA = Material(
    name='PICA',
    k_func=_PICA_k_func,
    rho_func=_const(2.60e-4),   # 260 kg/m³ — virgin density
    cp_func=_PICA_cp_func,
    L_func=_const(1050.0),      # effective endothermic pyrolysis enthalpy [J/g]
    T_ablation=600.0,           # simplified pyrolysis onset [°C]
    color='#8b6914',
)

TACOT = Material(               # Jean Lachaud
    name='TACOT',
    k_func=_const(3.37e-4),     # 0.337 W/(m·K) at 300 K
    rho_func=_const(2.80e-4),   # 280 kg/m³
    cp_func=_const(1.80),       # J/(g·K) at 300 K
    L_func=_const(3500.0),     # effective endothermic pyrolysis enthalpy [J/g]
    T_ablation=650.0,
    color='#c8a96e',
)


# ── Library ──────────────────────────────────────────────────────────────────

LIBRARY: dict[str, Material] = {
    'aluminum':       ALUMINIUM,
    'aluminium':      ALUMINIUM,
    'copper':         COPPER,
    'alumina':        ALUMINA,
    'silica':         SILICA,
    'pica':           PICA,
    'tacot':          TACOT,
}


def get(name: str) -> Material:
    """Return a material by name (case-insensitive, spaces/hyphens → underscores)."""
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key not in LIBRARY:
        available = ', '.join(sorted(LIBRARY))
        raise KeyError(f"Material {name!r} not found. Available: {available}")
    return LIBRARY[key]
