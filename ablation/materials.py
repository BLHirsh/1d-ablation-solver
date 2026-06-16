"""
Material property library.

Unit system: SI — m · s · J · kg · K
  k          [W/(m·K)]    thermal conductivity
  rho        [kg/m³]      density
  c_p        [J/(kg·K)]   specific heat
  T_ablation [K]          ablation/melt onset; use float('inf') for non-ablating
  L          [J/kg]       effective latent heat of ablation (0 for non-ablating)

Every property is a callable function of temperature:
  k_func(T)   → k   [W/(m·K)]
  rho_func(T) → rho [kg/m³]
  cp_func(T)  → c_p [J/(kg·K)]
  L_func(T)   → L   [J/kg]

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
        T_pts:  Temperature breakpoints [K], strictly increasing.
        v_pts:  Property values at each breakpoint (before scaling).
        scale:  Multiply every v_pt by this factor for unit conversion.
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
    k_func      : conductivity    k(T)  [W/(m·K)]
    rho_func    : density         ρ(T)  [kg/m³]
    cp_func     : specific heat  cp(T)  [J/(kg·K)]
    L_func      : latent heat     L(T)  [J/kg]
    T_ablation  : ablation/melt onset   [K]  (float('inf') → non-ablating)
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
        """Conductivity [W/(m·K)] at T [K]."""
        return self.k_func(T)

    def rho_at(self, T: float) -> float:
        """Density [kg/m³] at T [K]."""
        return self.rho_func(T)

    def cp_at(self, T: float) -> float:
        """Specific heat [J/(kg·K)] at T [K]."""
        return self.cp_func(T)

    def rhocp_at(self, T: float) -> float:
        """Volumetric heat capacity [J/(m³·K)] at T [K]."""
        return self.rho_at(T) * self.cp_at(T)

    def alpha_at(self, T: float) -> float:
        """Thermal diffusivity [m²/s] at T [K]."""
        return self.k_at(T) / max(self.rhocp_at(T), 1e-30)

    def L_at(self, T: float) -> float:
        """Effective latent heat [J/kg] at T [K]."""
        return self.L_func(T)

    # ------------------------------------------------------------------
    # Scalar reference values (for display and quick checks)
    # k / rho / c_p at 293.15 K; L at T_ablation (L has no meaning at ambient)
    # ------------------------------------------------------------------

    @property
    def k(self) -> float:
        """Reference conductivity [W/(m·K)] at 293.15 K."""
        return self.k_at(293.15)

    @property
    def rho(self) -> float:
        """Reference density [kg/m³] at 293.15 K."""
        return self.rho_at(293.15)

    @property
    def c_p(self) -> float:
        """Reference specific heat [J/(kg·K)] at 293.15 K."""
        return self.cp_at(293.15)

    @property
    def rho_cp(self) -> float:
        """Reference volumetric heat capacity [J/(m³·K)] at 293.15 K."""
        return self.rhocp_at(293.15)

    @property
    def alpha(self) -> float:
        """Reference thermal diffusivity [m²/s] at 293.15 K."""
        return self.alpha_at(293.15)

    @property
    def L(self) -> float:
        """Effective latent heat [J/kg] at T_ablation."""
        return self.L_at(self.T_ablation)

    @property
    def rho_L(self) -> float:
        """Volumetric latent heat [J/m³] at T_ablation."""
        return self.rho_at(self.T_ablation) * self.L_at(self.T_ablation)

    def __repr__(self) -> str:
        return (f"Material({self.name!r}  α={self.alpha:.3g} m²/s  "
                f"T_ab={self.T_ablation} K  L={self.L:.0f} J/kg)")


# ── Metals ───────────────────────────────────────────────────────────────────
# Source: Engineering Toolbox

_Al_k_func = _make_interp(
    T_pts=[200.15, 273.15, 400.15, 600.15, 800.15],
    v_pts=[   237,    236,    240,    232,    220],  # W/(m·K)
    scale=1.0,
)
_Al_rho_func = _make_interp(
    T_pts=[150.00, 200.00, 250.00, 306.30, 406.30, 606.30],
    v_pts=[  2726,   2719,   2710,   2701,   2681,   2639],  # kg/m³
    scale=1.0,
)
_Al_cp_func = _make_interp(
    T_pts=[150.00, 200.00, 250.00, 306.30, 406.30, 606.30],
    v_pts=[ 0.683,  0.797,  0.859,  0.902,  0.949,  1.042],  # J/(g·K) → J/(kg·K)
    scale=1e3,
)

ALUMINIUM = Material(
    name='Aluminium',
    k_func=_Al_k_func,
    rho_func=_Al_rho_func,
    cp_func=_Al_cp_func,
    L_func=_const(397e3),    # heat of fusion [J/kg]
    T_ablation=933.15,        # solidus [K]
    color='#4e9af1',
)


_Cu_k_func = _make_interp(
    T_pts=[146.30, 196.30, 246.30, 310.00, 360.00, 400.00, 600.00, 800.00, 1000.00, 1200.00],
    v_pts=[   480,    429,    413,    406,    401,    393,    379,    366,     352,     339],  # W/(m·K)
    scale=1.0,
)

_Cu_rho_func = _make_interp(
    T_pts=[146.30, 196.30, 246.30, 310.00, 360.00, 400.00, 600.00, 800.00, 1000.00, 1200.00],
    v_pts=[  9009,   8992,   8973,   8951,   8930,   8884,   8787,   8642,    8568,    8458],  # kg/m³
    scale=1.0,
)

_Cu_cp_func = _make_interp(
    T_pts=[146.30, 196.30, 246.30, 310.00, 360.00, 400.00, 600.00, 800.00, 1000.00, 1200.00],
    v_pts=[ 0.254,  0.323,  0.357,  0.377,  0.386,  0.396,  0.431,  0.448,   0.446,   0.480],  # J/(g·K) → J/(kg·K)
    scale=1e3,
)

COPPER = Material(
    name='Copper',
    k_func=_Cu_k_func,
    rho_func=_Cu_rho_func,
    cp_func=_Cu_cp_func,
    L_func=_const(205e3),    # heat of fusion [J/kg]
    T_ablation=1358.15,
    color='#b87333',
)

# ── Ceramics / oxides ────────────────────────────────────────────────────────

ALUMINA = Material(
    name='Alumina (Al₂O₃)',
    k_func=_const(30.0),       # W/(m·K)
    rho_func=_const(3950.0),   # kg/m³
    cp_func=_const(880.0),     # J/(kg·K)
    L_func=_const(1e6),        # https://www.azom.com/properties.aspx?ArticleID=52  [J/kg]
    T_ablation=2345.15,        # [K]
    color='#f17a4e',
)

SILICA = Material(
    name='Fused Silica (SiO₂)',
    k_func=_const(1.4),        # W/(m·K)
    rho_func=_const(2200.0),   # kg/m³
    cp_func=_const(720.0),     # J/(kg·K)
    L_func=_const(156.4e3),    # https://www.sciencedirect.com/science/article/abs/pii/0016703782903830  [J/kg]
    T_ablation=1873.15,        # [K]
    color='#cce8ff',
)

# ── Ablative TPS materials ───────────────────────────────────────────────────

# PICA — Phenolic Impregnated Carbon Ablator, ~260 kg/m³
# Milos & Chen (2009), AIAA-2009-4232, virgin-material curve
_PICA_k_func = _make_interp(
    T_pts=[293.15, 373.15, 473.15, 573.15, 673.15, 773.15, 873.15],
    v_pts=[ 0.22,   0.31,   0.33,   0.36,   0.40,   0.45,   0.50],  # W/(m·K)
    scale=1.0,
)
_PICA_cp_func = _make_interp(
    T_pts=[293.15, 373.15, 473.15, 573.15, 673.15, 773.15, 873.15],
    v_pts=[  1000,   1060,   1140,   1220,   1300,   1370,   1420],  # J/(kg·K)
    scale=1.0,
)

PICA = Material(
    name='PICA',
    k_func=_PICA_k_func,
    rho_func=_const(260.0),      # 260 kg/m³ — virgin density
    cp_func=_PICA_cp_func,
    L_func=_const(1050e3),       # effective endothermic pyrolysis enthalpy [J/kg]
    T_ablation=873.15,           # simplified pyrolysis onset [K]
    color='#8b6914',
)

TACOT = Material(               # Jean Lachaud
    name='TACOT',
    k_func=_const(0.337),        # 0.337 W/(m·K) at 300 K
    rho_func=_const(280.0),      # 280 kg/m³
    cp_func=_const(1800.0),      # J/(kg·K) at 300 K
    L_func=_const(3.5e6),        # effective endothermic pyrolysis enthalpy [J/kg]
    T_ablation=923.15,           # [K]
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
