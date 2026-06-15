"""
Visualisation: live animation and post-run result plots.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from matplotlib.animation import PillowWriter
import numpy as np

if TYPE_CHECKING:
    from .results import Results
    from .problem import Problem
    from .solver import Solver


# ── shared helpers ────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = '', xlabel: str = '', ylabel: str = '') -> None:
    if title:
        ax.set_title(title, fontsize=10, fontweight='semibold', pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.22, lw=0.7, zorder=0)


def animate(
    solver: 'Solver',
    steps_per_frame: int = 60,
    t_end: float = 30.0,
    interval_ms: int = 30,
    gif_path: str | None = None,
):
    """
    Real-time animation of a running simulation.

    Layout
    ------
    Top    : temperature heatmap with ablation-front markers and sim state
    Middle : temperature profile + melt-temperature guidelines
    Bottom : recession / mass-ablated (left) + energy budget / residual (right)

    Parameters
    ----------
    solver           A freshly initialised Solver (will be stepped in-place).
    steps_per_frame  Simulation steps per animation frame.
    t_end            Stop time [s].
    interval_ms      Milliseconds between frames.
    gif_path         If provided, save the finished animation to this path as
                     a GIF (requires Pillow).  Pass None to skip saving.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec

    prob   = solver.problem
    layers = prob.layers
    n      = solver.nodes
    dx     = solver.dx
    L      = solver.total_length
    T_init = prob.T_init

    x_centres = (np.arange(n) + 0.5) * dx
    x_edges   = np.linspace(0, L, n + 1)

    T_max = max(lay.material.T_ablation for lay in layers
                if lay.material.T_ablation < 1e15)

    seen, unique_mats = set(), []
    for lay in layers:
        if lay.material.name not in seen:
            seen.add(lay.material.name)
            unique_mats.append(lay.material)

    # ── figure layout ─────────────────────────────────────────────────────────
    # Left 3/4:  heatmap (thicker strip) above temperature profile (dominant).
    # Right 1/4: recession and energy stacked equally.
    # Nested GridSpec keeps each column's row-heights independent.
    # subplots_adjust used instead of constrained_layout — the latter
    # recalculates geometry every frame and causes visible jitter.
    fig = plt.figure(figsize=(13, 9))
    fig.subplots_adjust(
        left=0.07, right=0.97, top=0.91, bottom=0.07,
        wspace=0.42,
    )

    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1],
                               wspace=0.42)

    left_gs  = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0],
        height_ratios=[1, 2.8], hspace=0.45,
    )
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1],
        height_ratios=[1, 1], hspace=0.55,
    )

    ax_heat = fig.add_subplot(left_gs[0])
    ax_temp = fig.add_subplot(left_gs[1])
    ax_rec  = fig.add_subplot(right_gs[0])
    ax_erg  = fig.add_subplot(right_gs[1])

    # ── static top title ──────────────────────────────────────────────────────
    fig.suptitle('1-D Ablation Solver', fontsize=12, fontweight='bold', y=0.975)

    # ── heatmap ───────────────────────────────────────────────────────────────
    x0 = 0.0
    for lay in layers:
        ax_heat.axvspan(x0, x0 + lay.thickness, alpha=0.08,
                        color=lay.material.color, zorder=0)
        x0 += lay.thickness

    x0 = 0.0
    for lay in layers[:-1]:
        x0 += lay.thickness
        ax_heat.axvline(x0, color='k', lw=0.8, ls=':', alpha=0.35, zorder=1)

    pcm = ax_heat.pcolormesh(
        x_edges, np.array([0.0, 1.0]),
        solver.temp_profile().reshape(1, -1),
        cmap='inferno', vmin=T_init, vmax=T_max,
        shading='flat', zorder=2,
    )
    plt.colorbar(pcm, ax=ax_heat, label='Temperature (°C)',
                 fraction=0.025, pad=0.01)
    ax_heat.set_yticks([])
    ax_heat.set_xlabel('Position (mm)', fontsize=8)
    ax_heat.set_xlim(0, L)
    ax_heat.tick_params(labelsize=8)

    # Material labels centred inside the heatmap strip — avoids overlap with
    # any title text that lives above the axes.
    x0 = 0.0
    for lay in layers:
        xmid = (x0 + lay.thickness / 2) / L      # axes-fraction x
        ax_heat.text(
            xmid, 0.5, lay.material.name,
            ha='center', va='center', transform=ax_heat.transAxes,
            fontsize=8, fontweight='bold', color='#111111', zorder=6,
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      alpha=0.60, ec='none'),
        )
        x0 += lay.thickness

    fL = ax_heat.axvline(0, color='cyan', lw=2.0, ls='--', zorder=4)
    fR = ax_heat.axvline(L, color='lime', lw=2.0, ls='--', zorder=4)

    # Initial sim-state title (updated every frame)
    ax_heat.set_title(
        f't = 0.000 s   |   remaining = {L:.3f} mm',
        fontsize=9, fontweight='bold', color='#111111',
    )

    # ── temperature profile ───────────────────────────────────────────────────
    x0 = 0.0
    for lay in layers:
        ax_temp.axvspan(x0, x0 + lay.thickness, alpha=0.07,
                        color=lay.material.color)
        x0 += lay.thickness

    # markevery caps visible markers at ~50 so nodes in the cold (flat)
    # region don't all pile on the same pixel and become invisible.
    _every = max(1, n // 50)
    line_T, = ax_temp.plot(x_centres, solver.temp_profile(),
                           '-o', ms=3.5, markevery=_every,
                           color='#111111', lw=1.5)

    for mat in unique_mats:
        if mat.T_ablation < 1e15:
            ax_temp.axhline(mat.T_ablation, color=mat.color, ls=':', lw=1.2,
                            label=f'{mat.name}  T_ab = {mat.T_ablation:.0f} °C')

    ax_temp.set_xlim(0, L)
    ax_temp.set_ylim(T_init, T_max * 1.08)
    _style_ax(ax_temp, 'Temperature Profile', 'Position (mm)', 'T (°C)')
    ax_temp.legend(fontsize=7.5, loc='upper right', framealpha=0.85)

    # ── recession & mass ablated ──────────────────────────────────────────────
    tlog, recL, recR, mL, mR = [], [], [], [], []
    col_L = unique_mats[0].color
    col_R = unique_mats[-1].color

    lrecL, = ax_rec.plot([], [], color=col_L, lw=1.8, label='Rec. L')
    lrecR, = ax_rec.plot([], [], color=col_R, lw=1.8, label='Rec. R')
    ax_rec.set_xlim(0, t_end)
    ax_rec.set_ylim(0, L)
    ax_rec.set_title('Recession & Mass', fontsize=9, fontweight='semibold', pad=4)
    ax_rec.set_xlabel('Time (s)', fontsize=8)
    ax_rec.set_ylabel('Recession (mm)', fontsize=8)
    ax_rec.tick_params(labelsize=7.5)
    ax_rec.grid(alpha=0.22, lw=0.7, zorder=0)

    ax_mass = ax_rec.twinx()
    lmL, = ax_mass.plot([], [], color=col_L, lw=1.3, ls='--', label='Mass L')
    lmR, = ax_mass.plot([], [], color=col_R, lw=1.3, ls='--', label='Mass R')
    ax_mass.set_ylabel('Mass (g/mm²)', fontsize=7.5)
    ax_mass.tick_params(axis='y', labelsize=7)

    ax_rec.legend(
        [lrecL, lrecR, lmL, lmR],
        ['Rec. L (mm)', 'Rec. R (mm)', 'Mass L (g/mm²)', 'Mass R (g/mm²)'],
        fontsize=6.5, loc='upper left', framealpha=0.85,
    )

    # ── energy budget ─────────────────────────────────────────────────────────
    telog, ein_log, es_log, ea_log, res_log = [], [], [], [], []

    lein, = ax_erg.plot([], [], color='#1a6fbd', lw=1.8, label='E_in')
    lest, = ax_erg.plot([], [], color='#2ca02c', lw=1.8, label='E_stored')
    leab, = ax_erg.plot([], [], color='#d62728', lw=1.8, label='E_ablated')
    ax_erg.set_xlim(0, t_end)
    ax_erg.set_title('Energy Budget', fontsize=9, fontweight='semibold', pad=4)
    ax_erg.set_xlabel('Time (s)', fontsize=8)
    ax_erg.set_ylabel('Energy (J/mm²)', fontsize=8)
    ax_erg.tick_params(labelsize=7.5)
    ax_erg.grid(alpha=0.22, lw=0.7, zorder=0)

    ax_res = ax_erg.twinx()
    lres, = ax_res.plot([], [], color='darkorange', lw=1.2, ls=':',
                        label='|resid|/E_in')
    ax_res.set_ylabel('|resid|/E_in', color='darkorange', fontsize=7.5)
    ax_res.tick_params(axis='y', labelcolor='darkorange', labelsize=7)
    ax_res.set_ylim(0, 2e-9)

    ax_erg.legend(
        [lein, lest, leab, lres],
        ['E_in', 'E_stored', 'E_ablated', '|resid|/E_in'],
        fontsize=6.5, loc='upper left', framealpha=0.85,
    )

    # ── animation update ──────────────────────────────────────────────────────
    n_events_seen = [0]   # mutable container so the closure can mutate it

    def update(_):
        for _ in range(steps_per_frame):
            if solver.done or solver.t >= t_end:
                break
            solver.step()

        prof = solver.temp_profile()

        # Shift boundary-cell nodes to the centre of their *remaining* material
        # so the temperature profile aligns with the recession-front markers.
        # Without this, nodes sit at original cell centres while the cyan/lime
        # recession lines track the actual sub-cell surface position, creating
        # a visible gap (up to dx/2) that is worst at layer transitions.
        x_plot = x_centres.copy()
        if solver.iL <= solver.iR:
            if solver.mode_L == 'ablate':
                x_plot[solver.iL] = solver.rec_left + solver.frac_L * dx / 2.0
            if solver.mode_R == 'ablate':
                x_plot[solver.iR] = (L - solver.rec_right) - solver.frac_R * dx / 2.0

        pcm.set_array(prof.reshape(1, -1))
        line_T.set_xdata(x_plot)
        line_T.set_ydata(prof)
        fL.set_xdata([solver.rec_left,      solver.rec_left])
        fR.set_xdata([L - solver.rec_right, L - solver.rec_right])

        tlog.append(solver.t)
        recL.append(solver.rec_left)
        recR.append(solver.rec_right)
        mL.append(solver.mass_ablated_left)
        mR.append(solver.mass_ablated_right)
        lrecL.set_data(tlog, recL)
        lrecR.set_data(tlog, recR)
        lmL.set_data(tlog, mL)
        lmR.set_data(tlog, mR)
        ax_mass.set_ylim(0, max(max(mL + [1e-9]), max(mR + [1e-9])) * 1.2)

        telog.append(solver.t)
        ein_log.append(solver.E_in)
        es_log.append(solver.E_stored())
        ea_log.append(solver.E_ablated)
        r = solver.residual() / max(solver.E_in, 1e-30)
        res_log.append(abs(r))
        lein.set_data(telog, ein_log)
        lest.set_data(telog, es_log)
        leab.set_data(telog, ea_log)
        lres.set_data(telog, res_log)
        ax_erg.set_ylim(0, max(ein_log) * 1.1 if ein_log else 1)

        for evt in solver.layer_events[n_events_seen[0]:]:
            print(f"  t={evt[0]:.4f}s: {evt[2]}")
        n_events_seen[0] = len(solver.layer_events)

        rem = solver.remaining_thickness()
        if solver.done or solver.t >= t_end:
            ax_heat.set_title(
                f'DONE  —  t = {solver.t:.3f} s   |   remaining = {rem:.3f} mm'
                f'   |   resid/E_in = {r:+.2e}',
                fontsize=9, fontweight='bold', color='darkred',
            )
            ani.event_source.stop()
        else:
            ax_heat.set_title(
                f't = {solver.t:.3f} s   |   remaining = {rem:.3f} mm   |   '
                f'rec_L = {solver.rec_left:.3f}   rec_R = {solver.rec_right:.3f} mm',
                fontsize=9, fontweight='bold', color='#111111',
            )

        return pcm, line_T, lrecL, lrecR, lmL, lmR, lein, lest, leab, lres, fL, fR

    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, blit=False,
        repeat=False, cache_frame_data=False,
    )
    plt.show()
    if gif_path is not None:
        print(f"Saving animation to {gif_path!r} …")
        ani.save(gif_path, writer=PillowWriter(fps=30))

    return ani


def plot_results(results: 'Results', problem: 'Problem | None' = None):
    """
    Static summary plots from a completed Results object.

    Panels
    ------
    1. Recession vs time (both faces)
    2. Ablation rate vs time
    3. Mass ablated vs time
    4. Energy budget vs time
    5. Energy conservation check
    6. Temperature snapshots (if available) or simulation summary
    7. Simulation summary (only when snapshots fill panel 6)
    """
    import matplotlib.pyplot as plt

    t = results.t
    if len(t) == 0:
        print("No data to plot.")
        return

    n_snap  = len(results.snapshots)
    n_rows  = 3 if n_snap == 0 else 4
    BLUE    = "#1379c3"
    RED     = "#cc1414"
    GREEN   = "#067806"

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(12, 3.8 * n_rows),
        constrained_layout=True,
    )
    axes = axes.flatten()

    def _s(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=10, fontweight='semibold', pad=5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, lw=0.7, zorder=0)

    # ── 0: Recession ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, results.recession_left_mm,  lw=2, color=BLUE, label='Left face')
    ax.plot(t, results.recession_right_mm, lw=2, color=RED, ls='--',
            label='Right face')
    _s(ax, 'Surface Recession', 'Time (s)', 'Recession (mm)')
    ax.legend(fontsize=8, framealpha=0.85)

    # ── 1: Ablation rate ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, results.ablation_rate_left,  lw=1.5, color=BLUE,
            label='Left (mm/s)')
    ax.plot(t, results.ablation_rate_right, lw=1.5, color=RED, ls='--',
            label='Right (mm/s)')
    _s(ax, 'Instantaneous Ablation Rate', 'Time (s)', 'Rate (mm/s)')
    ax.legend(fontsize=8, framealpha=0.85)

    # ── 2: Mass ablated ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, np.asarray(results.mass_ablated_left)  * 1e3, lw=2,
            color=BLUE, label='Left')
    ax.plot(t, np.asarray(results.mass_ablated_right) * 1e3, lw=2,
            color=RED, ls='--', label='Right')
    _s(ax, 'Cumulative Mass Ablated', 'Time (s)', 'Mass ablated (mg/mm²)')
    ax.legend(fontsize=8, framealpha=0.85)

    # ── 3: Energy budget ──────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, results.E_in,      lw=2, color='#1a6fbd', label='E_in')
    ax.plot(t, results.E_stored,  lw=2, color=GREEN,     label='E_stored')
    ax.plot(t, results.E_ablated, lw=2, color=RED,       label='E_ablated')
    _s(ax, 'Energy Budget', 'Time (s)', 'Energy (J/mm²)')
    ax.legend(fontsize=8, framealpha=0.85)

    # ── 4: Energy conservation ────────────────────────────────────────────────
    ax = axes[4]
    ein  = np.asarray(results.E_in)
    res  = np.asarray(results.energy_residual)
    frac = np.where(ein > 0, res / ein, 0.0)
    ax.semilogy(t, np.maximum(frac, 1e-20), lw=1.5, color='darkorange')
    _s(ax, 'Energy Conservation Check', 'Time (s)', '|residual| / E_in')

    # ── 5 / 6 / 7: snapshots + summary ───────────────────────────────────────
    def _draw_summary(ax):
        ax.axis('off')
        ax.set_title('Simulation Summary', fontsize=10, fontweight='semibold', pad=5)
        ax.text(
            0.5, 0.5, results.summary(),
            ha='center', va='center', transform=ax.transAxes,
            family='monospace', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.55', fc='#f8f8f8',
                      ec='#c8c8c8', lw=1.0),
        )

    if n_snap == 0:
        # Panel 5 → summary (no snapshot data)
        _draw_summary(axes[5])
    else:
        # Panel 5 → temperature snapshots
        ax = axes[5]
        snaps   = results.snapshots
        cmap    = plt.cm.plasma
        n_cells = len(snaps[0][1])
        if problem is not None:
            total_L = sum(lay.thickness for lay in problem.layers)
            x_snap  = (np.arange(n_cells) + 0.5) * (total_L / n_cells)
            x_label = 'Position (mm)'
        else:
            x_snap  = np.arange(n_cells)
            x_label = 'Cell index'
        step = max(1, len(snaps) // 5)
        for k, (t_snap, T_snap) in enumerate(snaps):
            color = cmap(k / max(len(snaps) - 1, 1))
            label = f't = {t_snap:.2f} s' if k % step == 0 else ''
            ax.plot(x_snap, T_snap, color=color, lw=1, alpha=0.85, label=label)
        _s(ax, 'Temperature Snapshots', x_label, 'T (°C)')
        ax.legend(fontsize=7, framealpha=0.85)

        # Panel 6 → summary
        _draw_summary(axes[6])

        # Panel 7 → unused
        axes[7].axis('off')

    fig.suptitle('Ablation Simulation Results', fontsize=13, fontweight='bold')
    plt.show()
    return fig
