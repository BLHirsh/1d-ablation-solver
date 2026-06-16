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

STEEL_304 = Material(
    name='Stainless Steel 304',
    k_func=_const(22),          
    rho_func=_const(7200.0),   
    cp_func=_const(500.0),     # https://www.azom.com/properties.aspx?ArticleID=965 490-530 J/(kg·K) at 300 K
    L_func=_const(247e3),      # heat of fusion [J/kg]
    T_ablation=1670.0,         # https://www.researchgate.net/figure/Thermophysical-properties-of-304-stainless-steel-and-process-parameters-Nomenclature_tbl1_231114628
    color="#595454",
)

STEEL_316 = Material(
    name='Stainless Steel 316',
    k_func=_const(13),          
    rho_func=_const(7900.0),   # 7.87	8.07 g/cm³ at 20°C
    cp_func=_const(500.0),     # https://www.azom.com/properties.aspx?ArticleID=863 490-530 J/(kg·K) at 300 K
    L_func=_const(260e3),      # heat of fusion [J/kg]
    T_ablation=1648.0,         
    color="#4d4d4d",
)

STEEL_4240 = Material(
    name='Steel 4240',
    k_func=_const(51),          
    rho_func=_const(7800.0),   # 7.8 g/cm³ at 20°C
    cp_func=_const(470.0),     # https://www.makeitfrom.com/material-properties/SAE-AISI-4024-G40240-Molybdenum-Steel
    L_func=_const(250e3),      # heat of fusion [J/kg]
    T_ablation=1693.0,         
    color="#424424",
)

# Following from https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf 
_625_k_func = _make_interp(
    T_pts=[116.15, 144.15, 200.15, 255.15, 311.15, 366.15, 422.15, 477.15, 589.15, 700.15, 811.15, 922.15, 1033.15, 1144.15, 1255.15],
    v_pts=[   7.2,   7.5,    8.4,    9.2, 9.8, 10.1, 10.8, 12.5, 14.1, 15.7, 17.5, 19.0, 20.8, 22.8, 25.2],  # W/(m·K)
    scale=1.0,
) 

_625_cp_func = _make_interp(
    T_pts=[255.15, 294.15, 366.15, 477.15, 589.15, 700.15, 811.15, 922.15, 1033.15, 1144.15, 1255.15, 1366.15],
    v_pts=[402,   410,   427,   456,   481,   511,   536,   565,   590,   620,   645,   670], # J/(kg·K)
    scale=1.0,
)

INCONEL_625 = Material(
    name='Inconel 625',
    k_func=_625_k_func,
    rho_func=_const(8442),   
    cp_func=_625_cp_func,      
    L_func=_const(330e3),      # heat of fusion [J/kg] https://www.manifestalloys.com/blog/inconel-625-vs-718.html
    T_ablation=1570.0,         
    color="#39354e",
)

# From https://iopscience.iop.org/article/10.1088/1742-6596/1382/1/012175/pdf
_718_k_func = _make_interp(
    T_pts=[298, 400, 500, 600, 700, 800, 1100, 1200, 1300, 1400],
    v_pts=[9.94, 11.59, 13.24, 14.91, 16.61, 18.34, 22.72, 23.61, 24.47, 25.32], # W/(m·K)
    scale=1.0,
)

_718_cp_func = _make_interp(
    T_pts=[298, 400, 500, 600, 700, 800, 1100, 1200, 1300, 1400],
    v_pts=[425, 447, 468, 489, 510, 531, 635, 635, 635, 634], # J/(kg·K)
    scale=1.0,
)

INCONEL_718 = Material(
    name='Inconel 718',
    k_func=_718_k_func,
    rho_func=_const(8190.0),   
    cp_func=_718_cp_func,     
    L_func=_const(310e3),   # heat of fusion [J/kg] https://www.manifestalloys.com/blog/inconel-625-vs-718.html
    T_ablation=1533.15, #Solidus        
    color="#2b2b3c",
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
    'phenolic_canvas': PHENOLIC_CANVAS,
}


def get(name: str) -> Material:
    """Return a material by name (case-insensitive, spaces/hyphens → underscores)."""
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key not in LIBRARY:
        available = ', '.join(sorted(LIBRARY))
        raise KeyError(f"Material {name!r} not found. Available: {available}")
    return LIBRARY[key]
