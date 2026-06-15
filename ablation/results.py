"""
Results container: accumulates time-series diagnostics from the solver.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .solver import Solver


@dataclass
class Results:
    """
    Time-series history of a simulation run.

    All per-unit-area quantities are per mm² of frontal area — i.e. the
    simulation is intrinsically 1-D, but energy and mass are reported as
    areal densities so they scale to any slab area.

    Attributes
    ----------
    times                 [s]        recording timestamps
    recession_left_mm     [mm]       ablation recession from left face
    recession_right_mm    [mm]       ablation recession from right face
    mass_ablated_left     [g/mm²]    cumulative areal mass removed from left
    mass_ablated_right    [g/mm²]    cumulative areal mass removed from right
    E_in                  [J/mm²]    cumulative external energy delivered
    E_stored              [J/mm²]    enthalpy currently stored in material
    E_ablated             [J/mm²]    enthalpy carried away by ablated mass
    energy_residual       [J/mm²]    |E_in - E_stored - E_ablated| (should be ~0)
    snapshots             (t, T[])   sparse temperature-profile history
    """

    times: List[float] = field(default_factory=list)
    recession_left_mm: List[float] = field(default_factory=list)
    recession_right_mm: List[float] = field(default_factory=list)
    mass_ablated_left: List[float] = field(default_factory=list)
    mass_ablated_right: List[float] = field(default_factory=list)
    E_in: List[float] = field(default_factory=list)
    E_stored: List[float] = field(default_factory=list)
    E_ablated: List[float] = field(default_factory=list)
    energy_residual: List[float] = field(default_factory=list)

    snapshots: List[Tuple[float, np.ndarray]] = field(default_factory=list)

    # Layer ablation events: (time, layer_index, message)
    layer_events: List[Tuple[float, int, str]] = field(default_factory=list)

    # Final state
    completed: bool = False
    t_final: float = 0.0
    fully_ablated: bool = False
    problem_description: str = ''

    # Steady-state detection (set by run() when convergence criterion met)
    is_steady_state: bool = False
    steady_state_time: float = 0.0

    # ── recording ────────────────────────────────────────────────────────────
    def record(self, solver: 'Solver') -> None:
        """Append one data point from the current solver state."""
        self.times.append(solver.t)
        self.recession_left_mm.append(solver.rec_left)
        self.recession_right_mm.append(solver.rec_right)
        self.mass_ablated_left.append(solver.mass_ablated_left)
        self.mass_ablated_right.append(solver.mass_ablated_right)
        ein = solver.E_in
        es  = solver.E_stored()
        ea  = solver.E_ablated
        self.E_in.append(ein)
        self.E_stored.append(es)
        self.E_ablated.append(ea)
        self.energy_residual.append(abs(ein - es - ea))

    def take_snapshot(self, solver: 'Solver') -> None:
        """Store a full temperature profile for later plotting."""
        self.snapshots.append((solver.t, solver.temp_profile().copy()))

    # ── derived properties ────────────────────────────────────────────────────
    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.times)

    @property
    def rec_left(self) -> np.ndarray:
        return np.asarray(self.recession_left_mm)

    @property
    def rec_right(self) -> np.ndarray:
        return np.asarray(self.recession_right_mm)

    @property
    def ablation_rate_left(self) -> np.ndarray:
        """Instantaneous left recession rate [mm/s], from finite differences."""
        t = self.t
        if len(t) < 2:
            return np.array([0.0])
        dt = np.diff(t)
        dr = np.diff(self.rec_left)
        rate = dr / np.where(dt > 0, dt, 1e-30)
        return np.concatenate([[rate[0]], rate])  # same length as times

    @property
    def ablation_rate_right(self) -> np.ndarray:
        """Instantaneous right recession rate [mm/s], from finite differences."""
        t = self.t
        if len(t) < 2:
            return np.array([0.0])
        dt = np.diff(t)
        dr = np.diff(self.rec_right)
        rate = dr / np.where(dt > 0, dt, 1e-30)
        return np.concatenate([[rate[0]], rate])

    @property
    def max_energy_residual_fraction(self) -> float:
        """Max |residual|/E_in over the run — energy conservation quality check."""
        residuals = np.asarray(self.energy_residual)
        ein = np.asarray(self.E_in)
        mask = ein > 0
        if not np.any(mask):
            return 0.0
        return float(np.max(residuals[mask] / ein[mask]))

    # ── export ───────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Return all time-series data as a plain dictionary of lists.

        Suitable for JSON serialisation, pandas DataFrame construction, etc.
        """
        return {
            'time_s':                  self.times,
            'recession_left_mm':       self.recession_left_mm,
            'recession_right_mm':      self.recession_right_mm,
            'ablation_rate_left_mm_s': self.ablation_rate_left.tolist(),
            'ablation_rate_right_mm_s':self.ablation_rate_right.tolist(),
            'mass_ablated_left_g_mm2': self.mass_ablated_left,
            'mass_ablated_right_g_mm2':self.mass_ablated_right,
            'E_in_J_mm2':              self.E_in,
            'E_stored_J_mm2':          self.E_stored,
            'E_ablated_J_mm2':         self.E_ablated,
            'energy_residual_J_mm2':   self.energy_residual,
        }

    def to_csv(self, path: str) -> None:
        """Write the time-series data to a CSV file.

        Args:
            path: File path for the output CSV (e.g. 'results.csv').
        """
        import csv
        data = self.to_dict()
        keys = list(data.keys())
        rows = zip(*[data[k] for k in keys])
        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(keys)
            writer.writerows(rows)

    # ── summary ──────────────────────────────────────────────────────────────
    def summary(self) -> str:
        if not self.times:
            return "No data recorded."
        rec_L  = self.recession_left_mm[-1]
        rec_R  = self.recession_right_mm[-1]
        mabl_L = self.mass_ablated_left[-1]
        mabl_R = self.mass_ablated_right[-1]
        ein    = self.E_in[-1]
        es     = self.E_stored[-1]
        ea     = self.E_ablated[-1]

        lines = [
            '─' * 55,
            f'  t_final            = {self.t_final:.4f} s',
            f'  Recession left     = {rec_L:.4f} mm',
            f'  Recession right    = {rec_R:.4f} mm',
            f'  Mass ablated left  = {mabl_L:.6f} g/mm²',
            f'  Mass ablated right = {mabl_R:.6f} g/mm²',
            f'  E_in               = {ein:.4f} J/mm²',
            f'  E_stored           = {es:.4f} J/mm²',
            f'  E_ablated          = {ea:.4f} J/mm²',
            f'  Ablation fraction  = {ea/max(ein,1e-30)*100:.1f}% of E_in',
            f'  Energy residual    = {self.energy_residual[-1]:.2e} J/mm²',
            f'  Max resid/E_in     = {self.max_energy_residual_fraction:.2e}',
            f'  Fully ablated      = {self.fully_ablated}',
        ]
        if self.is_steady_state:
            lines.append(f'  Steady state         = reached at t={self.steady_state_time:.3f} s')
        if self.layer_events:
            lines.append('  Layer events:')
            for t_evt, _, msg in self.layer_events:
                lines.append(f'    t={t_evt:.4f}s  {msg}')
        lines.append('─' * 55)
        return '\n'.join(lines)
