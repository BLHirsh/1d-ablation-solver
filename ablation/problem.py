"""
Problem definition: geometry, layout, boundary conditions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .materials import Material
from .boundaries import BoundaryCondition, Adiabatic


@dataclass
class Layer:
    """One material layer in the stack."""
    material: Material
    thickness: float   # [m]


@dataclass
class Problem:
    """
    1-D ablation / conduction problem specification.

    Parameters
    ----------
    layers : list[Layer]
        Ordered left-to-right.
    left_bc : BoundaryCondition
        Boundary condition at x = 0.
    right_bc : BoundaryCondition
        Boundary condition at x = total_thickness. Default: Adiabatic.
    T_init : float
        Uniform initial temperature [K].
    nodes_per_m : float
        Grid resolution [nodes/m]. Increase for better spatial accuracy.
    r_target : float
        Target Fourier number for explicit time stepping (≤ 0.5).
        Lower values give smaller dt and higher accuracy.

    Use Problem.one_sided() or Problem.two_sided() for quick configuration.
    """
    layers: List[Layer]
    left_bc: BoundaryCondition
    right_bc: BoundaryCondition = field(default_factory=Adiabatic)
    T_init: float = 293.15
    nodes_per_m: float = 2000.0
    r_target: float = 0.40

    def __post_init__(self):
        if not self.layers:
            raise ValueError("Problem must have at least one layer.")
        for i, lay in enumerate(self.layers):
            if lay.thickness <= 0:
                raise ValueError(
                    f"Layer {i} ({lay.material.name!r}) thickness must be > 0; "
                    f"got {lay.thickness}."
                )
        if not isinstance(self.left_bc, BoundaryCondition):
            raise TypeError(f"left_bc must be a BoundaryCondition, got {type(self.left_bc)}.")
        if not isinstance(self.right_bc, BoundaryCondition):
            raise TypeError(f"right_bc must be a BoundaryCondition, got {type(self.right_bc)}.")
        if self.nodes_per_m <= 0:
            raise ValueError(f"nodes_per_m must be > 0; got {self.nodes_per_m}.")
        if not (0 < self.r_target <= 0.5):
            raise ValueError(
                f"r_target must be in (0, 0.5] for stability; got {self.r_target}."
            )

    # ── factory methods ──────────────────────────────────────────────────────
    @classmethod
    def one_sided(
        cls,
        material: Material,
        thickness: float,
        left_bc: BoundaryCondition,
        *,
        T_init: float = 293.15,
        nodes_per_m: float = 2000.0,
    ) -> 'Problem':
        """Single-layer slab, heated on one face, insulated on the other."""
        return cls(
            layers=[Layer(material, thickness)],
            left_bc=left_bc,
            right_bc=Adiabatic(),
            T_init=T_init,
            nodes_per_m=nodes_per_m,
        )

    @classmethod
    def two_sided(
        cls,
        material: Material,
        thickness: float,
        bc: BoundaryCondition,
        *,
        T_init: float = 293.15,
        nodes_per_m: float = 2000.0,
    ) -> 'Problem':
        """Single-layer slab with identical BCs on both faces."""
        return cls(
            layers=[Layer(material, thickness)],
            left_bc=bc,
            right_bc=bc,
            T_init=T_init,
            nodes_per_m=nodes_per_m,
        )

    @classmethod
    def bilayer(
        cls,
        mat_left: Material,
        t_left: float,
        mat_right: Material,
        t_right: float,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition | None = None,
        *,
        T_init: float = 293.15,
        nodes_per_m: float = 2000.0,
    ) -> 'Problem':
        """Two-layer slab with independent BCs on each face."""
        return cls(
            layers=[Layer(mat_left, t_left), Layer(mat_right, t_right)],
            left_bc=left_bc,
            right_bc=right_bc or Adiabatic(),
            T_init=T_init,
            nodes_per_m=nodes_per_m,
        )

    @classmethod
    def casing(
        cls,
        casing: Material,
        thickness: float,
        inside: Material,
        inside_thickness: float,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        *,
        T_init: float = 293.15,
        nodes_per_m: float = 2000.0,
    ) -> 'Problem':
        """Slab with interior material and exterior casing. BCs on both sides."""
        return cls(
            layers=[Layer(casing, thickness), Layer(inside, inside_thickness), Layer(casing, thickness)],
            left_bc=left_bc,
            right_bc=right_bc,
            T_init=T_init,
            nodes_per_m=nodes_per_m,
        )

    # ── properties / describe ────────────────────────────────────────────────
    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def n_nodes(self) -> int:
        return max(10, round(self.total_thickness * self.nodes_per_m))

    @property
    def dx(self) -> float:
        return self.total_thickness / self.n_nodes

    def describe(self) -> str:
        lines = [
            f"Problem ({len(self.layers)} layer{'s' if len(self.layers) > 1 else ''}, "
            f"{self.total_thickness:.4f} m total)",
            f"  Left BC  : {self.left_bc}",
            f"  Right BC : {self.right_bc}",
            f"  T_init   : {self.T_init} K",
            f"  Grid     : {self.n_nodes} nodes  ({self.dx:.6f} m/cell)",
        ]
        x = 0.0
        for i, layer in enumerate(self.layers):
            m = layer.material
            lines.append(
                f"  Layer {i+1}  : {m.name}  {x:.4f}–{x+layer.thickness:.4f} m  "
                f"α={m.alpha:.3g} m²/s  T_ab={m.T_ablation} K  L={m.L:.0f} J/kg"
            )
            x += layer.thickness
        return '\n'.join(lines)
