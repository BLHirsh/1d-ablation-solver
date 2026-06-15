"""
1-D finite-volume ablation / heat-conduction solver.

Physics
-------
Explicit forward-Euler finite-volume update with harmonic-mean (series-
resistance) face conductivities.

Stefan ablation condition:
    The ablating surface is pinned at T_ablation.  The surface energy balance
        rho * L * (ds/dt) = q_net_surface - q_conducted_into_solid      (Stefan)
    drives the recession rate.
        E_in  ≡  E_stored  +  E_ablated       (closed to floating-point precision)

Temperature-dependent properties:
    If a material supplies k_func(T) or cp_func(T), those are evaluated at
    the start-of-step temperature field each step.  k_arr and rhocp_arr are
    updated in-place; the time step dt is computed conservatively using the
    worst-case (highest) α over the full temperature range [T_init, T_ablation].

Boundary conditions:
    Any BoundaryCondition object.  The net flux q = bc.flux(T_surface) is
    evaluated at the surface temperature at the *start* of each step
    (explicit treatment), so convective and radiative BCs are fully supported.

Steady-state detection:
    run() monitors the maximum rate of temperature change.  When
    max|ΔT/Δt| < ss_tol [K/s] for ss_window consecutive record intervals,
    the simulation is flagged as steady-state.

Time step:
    dt = r_target * dx² / max(α_worst)
    where α_worst accounts for the highest thermal diffusivity each material
    can reach across its full operating temperature range.

Units: mm · s · J · g · °C
"""
from __future__ import annotations
import numpy as np
from .problem import Problem
from .results import Results


class Solver:
    """
    Finite-volume solver.  Instantiate with a Problem, then:
        • Call step() repeatedly (for animation / real-time monitoring), or
        • Call run(t_end) to obtain a complete Results object.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self._build_grid()
        self._init_state()

    # ── setup ────────────────────────────────────────────────────────────────
    def _build_grid(self):
        prob = self.problem
        n    = prob.n_nodes
        L    = prob.total_thickness
        dx   = L / n
        T0   = prob.T_init

        self.nodes        = n
        self.total_length = L
        self.dx           = dx

        # Per-cell material arrays (mutable; updated each step for T-dep mats)
        k_arr     = np.empty(n)
        rho_arr   = np.empty(n)
        cp_arr    = np.empty(n)
        Tmelt_arr = np.empty(n)
        rhoL_arr  = np.empty(n)
        mat_idx   = np.empty(n, dtype=int)
        # Worst-case α for each cell (max over operating temperature range)
        a_worst   = np.empty(n)

        layers      = prob.layers
        thicknesses = [lay.thickness for lay in layers]
        cell_cx     = (np.arange(n) + 0.5) * dx

        for j, layer in enumerate(layers):
            x_lo = sum(thicknesses[:j])
            x_hi = x_lo + layer.thickness
            mask = (cell_cx >= x_lo) & (cell_cx < x_hi)
            if j == len(layers) - 1:
                mask |= (cell_cx >= x_lo)
            m = layer.material
            k_arr[mask]     = m.k_at(T0)
            rho_arr[mask]   = m.rho_at(T0)
            cp_arr[mask]    = m.cp_at(T0)
            Tmelt_arr[mask] = m.T_ablation
            rhoL_arr[mask]  = m.rho_at(m.T_ablation) * m.L_at(m.T_ablation)
            mat_idx[mask]   = j

            # Worst-case α: evaluate at both T_init and T_ablation, take max.
            T_hi = min(float(m.T_ablation), 5000.0)
            a_worst[mask] = max(m.alpha_at(T0), m.alpha_at(T_hi))

        rhocp_arr = rho_arr * cp_arr
        Hs_arr    = self._compute_Hs(layers, mat_idx, T0, Tmelt_arr, n)
        Hfull_arr = Hs_arr + rhoL_arr

        self.k_arr     = k_arr
        self.rho_arr   = rho_arr
        self.cp_arr    = cp_arr
        self.Tmelt_arr = Tmelt_arr
        self.rhoL_arr  = rhoL_arr
        self.mat_idx   = mat_idx
        self.rhocp_arr = rhocp_arr
        self.Hs_arr    = Hs_arr
        self.Hfull_arr = Hfull_arr

        self.dt = prob.r_target * dx**2 / np.max(a_worst)

    @staticmethod
    def _compute_Hs(layers, mat_idx, T0, Tmelt_arr, n):
        """Sensible enthalpy to bring each cell from T_init to T_ablation [J/mm³].

        Always integrates rho(T)*cp(T) numerically so constant-property and
        tabulated materials are handled by the same code path.
        """
        Hs = np.empty(n)
        for i in range(n):
            m  = layers[mat_idx[i]].material
            dT = Tmelt_arr[i] - T0
            if dT > 0:
                Ts         = np.linspace(T0, Tmelt_arr[i], 64)
                rhocp_vals = np.array([m.rhocp_at(T) for T in Ts])
                Hs[i]      = np.trapezoid(rhocp_vals, Ts)
            else:
                Hs[i] = 0.0
        return Hs

    def _update_props(self, T: np.ndarray):
        """Update k, rho, cp, rhocp arrays from the current temperature field."""
        layers = self.problem.layers
        for i in range(self.iL, self.iR + 1):
            Ti = T[i]
            if np.isnan(Ti):
                continue
            m = layers[self.mat_idx[i]].material
            self.k_arr[i]     = m.k_at(Ti)
            self.rho_arr[i]   = m.rho_at(Ti)
            self.cp_arr[i]    = m.cp_at(Ti)
            self.rhocp_arr[i] = self.rho_arr[i] * self.cp_arr[i]

    def _init_state(self):
        n  = self.nodes

        self.H = np.zeros(n)

        self.iL = 0
        self.iR = n - 1

        self.frac_L = 1.0
        self.frac_R = 1.0

        self.mode_L = 'heat'
        self.mode_R = 'heat'

        self.t = 0.0
        self.rec_left  = 0.0
        self.rec_right = 0.0
        self.mass_ablated_left  = 0.0
        self.mass_ablated_right = 0.0
        self.E_in      = 0.0
        self.E_ablated = 0.0
        self.done      = False

        self.layer_events: list = []
        self._layer_logged = [False] * len(self.problem.layers)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _width(self, i: int) -> float:
        dx = self.dx
        if i == self.iL == self.iR and self.mode_L == 'ablate' and self.mode_R == 'ablate':
            return max(0.0, self.frac_L + self.frac_R - 1.0) * dx
        if i == self.iL and self.mode_L == 'ablate':
            return self.frac_L * dx
        if i == self.iR and self.mode_R == 'ablate':
            return self.frac_R * dx
        return dx

    def _temp(self, i: int) -> float:
        """Temperature of cell i [°C]."""
        T0 = self.problem.T_init
        if i == self.iL and self.mode_L == 'ablate':
            return self.Tmelt_arr[i]
        if i == self.iR and self.mode_R == 'ablate':
            return self.Tmelt_arr[i]
        return T0 + self.H[i] / self.rhocp_arr[i]

    def E_stored(self) -> float:
        """Current enthalpy of all remaining material [J/mm²]."""
        total = 0.0
        for i in range(self.iL, self.iR + 1):
            if (i == self.iL and self.mode_L == 'ablate') or \
               (i == self.iR and self.mode_R == 'ablate'):
                Heff = self.Hs_arr[i]
            else:
                Heff = self.H[i]
            total += Heff * self._width(i)
        return total

    def residual(self) -> float:
        return self.E_in - self.E_stored() - self.E_ablated

    def temp_profile(self) -> np.ndarray:
        """Temperature at every cell centre [°C]; NaN for ablated cells."""
        T = np.full(self.nodes, np.nan)
        for i in range(self.iL, self.iR + 1):
            T[i] = self._temp(i)
        return T

    def remaining_thickness(self) -> float:
        return self.total_length - self.rec_left - self.rec_right

    def surface_material(self, side: str):
        """Current Material object at the ablating surface (left or right)."""
        idx       = self.iL if side == 'L' else self.iR
        layer_idx = self.mat_idx[idx]
        return self.problem.layers[layer_idx].material

    def _check_layer_event(self, layer_idx: int) -> None:
        if self._layer_logged[layer_idx]:
            return
        gone = (self.iL > self.iR) or (
            not np.any(self.mat_idx[self.iL:self.iR + 1] == layer_idx)
        )
        if gone:
            self._layer_logged[layer_idx] = True
            name = self.problem.layers[layer_idx].material.name
            self.layer_events.append((self.t, layer_idx, f'{name} fully ablated'))

    # ── recession ────────────────────────────────────────────────────────────
    def _recede(self, side: str, budget: float):
        """
        Consume `budget` [J/mm²] of net surface energy as recession.

        Each ablated slice ds contributes:
            E_ablated += ds * Hfull   [J/mm²]
            mass_ablated += rho * ds  [g/mm²]

        The ledger E_in = E_stored + E_ablated closes by construction.
        """
        dx = self.dx
        while budget > 1e-18:
            if self.iL > self.iR:
                break
            i     = self.iL if side == 'L' else self.iR
            rho_L = self.rhoL_arr[i]
            Hfull = self.Hfull_arr[i]
            rho_i = self.rho_arr[i]
            frac  = self.frac_L if side == 'L' else self.frac_R
            w     = frac * dx
            e_rem = w * rho_L

            if budget < e_rem:
                ds = budget / rho_L
                if side == 'L':
                    self.frac_L            -= ds / dx
                    self.rec_left          += ds
                    self.mass_ablated_left += rho_i * ds
                else:
                    self.frac_R             -= ds / dx
                    self.rec_right          += ds
                    self.mass_ablated_right += rho_i * ds
                self.E_ablated += ds * Hfull
                budget = 0.0

            else:
                old_layer = self.mat_idx[i]
                if side == 'L':
                    self.rec_left          += w
                    self.mass_ablated_left += rho_i * w
                    self.E_ablated         += w * Hfull
                    budget                 -= e_rem
                    self.iL                += 1
                    self.frac_L             = 1.0
                    self.mode_L             = 'heat'
                    ni                      = self.iL
                else:
                    self.rec_right          += w
                    self.mass_ablated_right += rho_i * w
                    self.E_ablated          += w * Hfull
                    budget                  -= e_rem
                    self.iR                 -= 1
                    self.frac_R              = 1.0
                    self.mode_R              = 'heat'
                    ni                       = self.iR

                self._check_layer_event(old_layer)

                if self.iL <= self.iR and budget > 0.0:
                    self.H[ni] += budget / dx
                    budget = 0.0

        if budget > 1e-18:
            self.E_ablated += budget

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self):
        if self.done:
            return

        dt, dx = self.dt, self.dx
        iL, iR = self.iL, self.iR
        prob    = self.problem
        T_init  = prob.T_init

        # 1. Temperature at start of step ────────────────────────────────────
        T = np.full(self.nodes, np.nan)
        for i in range(iL, iR + 1):
            T[i] = self._temp(i)

        # 2. Update material properties for current T, then recompute T ────────
        self._update_props(T)
        for i in range(iL, iR + 1):
            T[i] = self._temp(i)

        # 3. Face fluxes  Phi[i] = flux cell i → i+1 [W/mm²]
        #    Harmonic-mean (series-resistance) conductance
        Phi = np.zeros(self.nodes)
        for i in range(iL, iR):
            Wi  = self._width(i)
            Wi1 = self._width(i + 1)
            G   = 1.0 / ((Wi / 2) / self.k_arr[i] + (Wi1 / 2) / self.k_arr[i + 1])
            Phi[i] = G * (T[i] - T[i + 1])

        # 4. Interior cell update (explicit FV) ──────────────────────────────
        for i in range(iL + 1, iR):
            self.H[i] += dt * (Phi[i - 1] - Phi[i]) / dx

        # 5. Boundary fluxes (start-of-step surface T)
        q_L = prob.left_bc.flux(T[iL])
        q_R = prob.right_bc.flux(T[iR])

        # 6. Surface cells ────────────────────────────────────────────────────
        if iL == iR:
            # Single remaining cell — no conduction term.
            i = iL
            if self.mode_L == 'ablate' or self.mode_R == 'ablate':
                self.mode_L = 'ablate'
                self.mode_R = 'ablate'
                if q_L > 0:
                    self._recede('L', q_L * dt)
                # Left recession may have consumed the last cell:
                if self.iL <= self.iR and q_R > 0:
                    self._recede('R', q_R * dt)
                # Fronts-met check: both faces together consumed the cell
                if self.iL <= self.iR and self.frac_L + self.frac_R <= 1.0:
                    self.iL = self.iR + 1
            else:
                self.H[i] += dt * (q_L + q_R) / dx
                Tnew = T_init + self.H[i] / self.rhocp_arr[i]
                if Tnew >= self.Tmelt_arr[i]:
                    self.mode_L = 'ablate'
                    self.mode_R = 'ablate'
                    excess = (self.H[i] - self.Hs_arr[i]) * dx
                    self.H[i] = self.Hs_arr[i]
                    if excess > 0:
                        self._recede('L', excess * 0.5)
                        self._recede('R', excess * 0.5)

        else:
            # Normal multi-cell path ──────────────────────────────────────────
            phi_inner = Phi[iL]

            if self.mode_L == 'heat':
                self.H[iL] += dt * (q_L - phi_inner) / dx
                Tnew = T_init + self.H[iL] / self.rhocp_arr[iL]
                if Tnew >= self.Tmelt_arr[iL]:
                    self.mode_L = 'ablate'
                    self.frac_L = 1.0
                    excess = (self.H[iL] - self.Hs_arr[iL]) * dx
                    self.H[iL] = self.Hs_arr[iL]
                    if excess > 0:
                        self._recede('L', excess)
            else:
                net = q_L - phi_inner
                if net >= 0:
                    self._recede('L', net * dt)
                else:
                    self.H[iL] = self.Hs_arr[iL] * self.frac_L + net * dt / dx
                    self.mode_L = 'heat'

            iL, iR = self.iL, self.iR

            if iL < iR:
                phi_inner = Phi[iR - 1]

                if self.mode_R == 'heat':
                    self.H[iR] += dt * (q_R + phi_inner) / dx
                    Tnew = T_init + self.H[iR] / self.rhocp_arr[iR]
                    if Tnew >= self.Tmelt_arr[iR]:
                        self.mode_R = 'ablate'
                        self.frac_R = 1.0
                        excess = (self.H[iR] - self.Hs_arr[iR]) * dx
                        self.H[iR] = self.Hs_arr[iR]
                        if excess > 0:
                            self._recede('R', excess)
                else:
                    q_cond = -phi_inner
                    net    = q_R - q_cond
                    if net >= 0:
                        self._recede('R', net * dt)
                    else:
                        self.H[iR] = self.Hs_arr[iR] * self.frac_R + net * dt / dx
                        self.mode_R = 'heat'

            elif iL == iR and self.mode_R == 'heat':
                # Left just consumed its boundary cell; credit the pre-step
                # conduction flux to iR before the next step's single-cell path.
                phi_inner = Phi[iR - 1]
                self.H[iR] += dt * (q_R + phi_inner) / dx

        # 7. Energy accounting ────────────────────────────────────────────────
        self.E_in += (q_L + q_R) * dt
        self.t    += dt

        # 8. Termination ──────────────────────────────────────────────────────
        if self.iL > self.iR or self.remaining_thickness() <= 0.0:
            for j in range(len(self.problem.layers)):
                if not self._layer_logged[j]:
                    self._layer_logged[j] = True
                    name = self.problem.layers[j].material.name
                    self.layer_events.append((self.t, j, f'{name} fully ablated'))
            self.done = True

    # ── run ──────────────────────────────────────────────────────────────────
    def run(
        self,
        t_end: float,
        record_every:   int   = 100,
        snapshot_every: int   = 2000,
        verbose:        bool  = False,
        print_every:    int   = 5000,
        ss_tol:         float = 0.05,   # [K/s]  steady-state temp-change threshold
        ss_window:      int   = 5,      # consecutive records below threshold → SS
    ) -> Results:
        """
        Run to t_end (or full ablation) and return a Results object.

        Parameters
        ----------
        t_end          Stop time [s].
        record_every   Steps between data-point recordings.
        snapshot_every Steps between full temperature-profile snapshots.
        verbose        Print progress to stdout.
        print_every    Steps between verbose prints.
        ss_tol         Steady-state detection: max |dT/dt| threshold [K/s].
                       Set to 0 to disable.
        ss_window      Number of consecutive sub-threshold records required
                       before declaring steady state.
        """
        results = Results(problem_description=self.problem.describe())
        n_step  = 0
        n_events_logged = 0
        _ss_count       = 0
        _H_prev         = None   # H snapshot from previous record for dT/dt estimate

        while not self.done and self.t < t_end:
            self.step()
            n_step += 1

            if n_step % record_every == 0:
                results.record(self)

                # Steady-state check: compare current H to previous snapshot
                if ss_tol > 0 and _H_prev is not None and not self.done:
                    dt_record   = record_every * self.dt
                    iL, iR      = self.iL, self.iR
                    dT_max = 0.0
                    for i in range(iL, iR + 1):
                        rcp = self.rhocp_arr[i]
                        if rcp > 0:
                            dT_max = max(dT_max,
                                         abs(self.H[i] - _H_prev[i]) / rcp / dt_record)
                    if dT_max < ss_tol:
                        _ss_count += 1
                        if _ss_count >= ss_window:
                            results.is_steady_state  = True
                            results.steady_state_time = self.t
                            if verbose:
                                print(f"  >>> Steady state detected at t={self.t:.3f} s"
                                      f"  (max|dT/dt|={dT_max:.3e} K/s)")
                    else:
                        _ss_count = 0
                _H_prev = self.H.copy()

            if n_step % snapshot_every == 0:
                results.take_snapshot(self)

            if verbose and n_step % print_every == 0:
                res_frac = self.residual() / max(self.E_in, 1e-30)
                print(f"  t={self.t:7.3f}s  "
                      f"rec_L={self.rec_left:6.3f}mm  "
                      f"rec_R={self.rec_right:6.3f}mm  "
                      f"rem={self.remaining_thickness():6.3f}mm  "
                      f"resid/E_in={res_frac:+.2e}")
            if verbose:
                for evt in self.layer_events[n_events_logged:]:
                    print(f"  >>> t={evt[0]:.4f}s: {evt[2]}")
                n_events_logged = len(self.layer_events)

        # Final record & snapshot
        results.record(self)
        results.take_snapshot(self)
        results.completed     = True
        results.t_final       = self.t
        results.fully_ablated = self.done
        results.layer_events  = list(self.layer_events)
        return results
