"""
Example problems demonstrating the ablation solver.

Run modes
---------
    python run.py                          # animate the bilayer demo
    python run.py --demo bilayer           # two-material bilayer (default)
    python run.py --demo convection        # aluminium under convective aero-heating
    python run.py --demo arcjet            # PICA under combined convection + radiation
    python run.py --demo conduction        # pure thermal soak (no ablation)
    python run.py --demo tacot             # TACOT slab with convection + radiation
    python run.py --headless               # run headlessly and print energy summary
    python run.py --test                   # headless energy-conservation check, all demos
    python run.py --tacot                  # TACOT benchmark vs. analytical Stefan solution

GIF output
----------
    python run.py --gif                    # save animation to result.gif
    python run.py --gif my_output.gif      # save animation to a custom path
"""
import argparse
import numpy as np

# ── package imports ───────────────────────────────────────────────────────────
from ablation import (
    ALUMINIUM, ALUMINA, COPPER,
    SILICA, PICA, TACOT,
    HeatFlux, Convection, Radiation, Adiabatic, Combined,
    Layer, Problem, Solver,
)
from ablation import viz


# ─────────────────────────────────────────────────────────────────────────────
# Problem definitions
# ─────────────────────────────────────────────────────────────────────────────

def make_bilayer(nodes_per_mm: float = 2.0) -> tuple[Problem, float]:
    prob = Problem.bilayer(
        ALUMINIUM, 25.0,
        COPPER,   25.0,
        left_bc=HeatFlux(q=8.0),     # 8 W/mm² = 8 MW/m²
        right_bc=HeatFlux(q=8.0),
        T_init=20.0,
        nodes_per_mm=nodes_per_mm,
    )
    return prob, 40.0


def make_convection(nodes_per_mm: float = 2.0) -> tuple[Problem, float]:
    prob = Problem.one_sided(
        ALUMINIUM, thickness=30.0,
        left_bc=Convection(h=2e-4, T_env=1500.0),
        T_init=20.0,
        nodes_per_mm=nodes_per_mm,
    )
    return prob, 60.0


def make_arcjet(nodes_per_mm: float = 10.0) -> tuple[Problem, float]:
    aeroheating = Convection(h=5e-4, T_env=1305.0)
    reradiation = Radiation(emissivity=0.85, T_env=26.0)
    front_bc    = Combined([aeroheating, reradiation])

    prob = Problem.one_sided(
        PICA, thickness=50.0,
        left_bc=front_bc,
        T_init=20.0,
        nodes_per_mm=nodes_per_mm,
    )
    return prob, 50.0


def make_conduction(nodes_per_mm: float = 2.0) -> tuple[Problem, float]:
    prob = Problem.one_sided(
        ALUMINIUM, thickness=20.0,
        left_bc=Convection(h=1e-4, T_env=800.0),
        T_init=20.0,
        nodes_per_mm=nodes_per_mm,
    )
    return prob, 300.0


def make_tacot(nodes_per_mm: float = 2.0) -> tuple[Problem, float]:
    T_init      = 26.85   # 300 K
    aeroheating = Convection(h=3.6e-4, T_env=1400.0)
    reradiation = Radiation(emissivity=0.9, T_env=20)
    front_bc    = Combined([aeroheating, reradiation])
    prob = Problem.one_sided(
        TACOT, thickness=50.0,
        left_bc=front_bc,
        T_init=T_init,
        nodes_per_mm=nodes_per_mm,
    )
    return prob, 60.0


DEMOS = {
    'bilayer':    make_bilayer,
    'convection': make_convection,
    'arcjet':     make_arcjet,
    'conduction': make_conduction,
    'tacot':      make_tacot,
}


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_headless(name: str = 'bilayer', verbose: bool = True) -> None:
    """Run a demo to completion and print the energy-conservation summary."""
    make_fn = DEMOS[name]
    prob, t_end = make_fn()
    print(f"\n{'─'*60}")
    print(f"  Demo: {name}")
    print(prob.describe())
    print(f"  dt = {Solver(prob).dt:.3e} s   t_end = {t_end} s")

    solver  = Solver(prob)
    results = solver.run(t_end=t_end, record_every=200, snapshot_every=4000,
                         verbose=verbose, print_every=10000)
    print(results.summary())


def run_test_all() -> None:
    """Run all demos headlessly and assert energy conservation."""
    print("Running energy-conservation checks for all demos …")
    for name in DEMOS:
        make_fn = DEMOS[name]
        prob, t_end = make_fn(nodes_per_mm=1.5)
        solver  = Solver(prob)
        results = solver.run(t_end=min(t_end, 20.0), record_every=500)
        frac    = results.max_energy_residual_fraction
        status  = 'success' if frac < 1e-8 else 'failure'
        print(f"  {status}  {name:15s}  max|resid|/E_in = {frac:.2e}")
        if frac >= 1e-8:
            raise AssertionError(f"Energy conservation failed for demo '{name}': {frac:.2e}")
    print("All demos pass energy conservation check.")


def run_animation(name: str = 'bilayer', gif_path: str | None = None) -> None:
    """Run the live animation for the named demo.

    Parameters
    ----------
    name     : demo key from DEMOS
    gif_path : if provided, save the animation to this path after it finishes
    """
    make_fn = DEMOS[name]
    prob, t_end = make_fn()
    print(prob.describe())
    solver = Solver(prob)
    viz.animate(solver, steps_per_frame=80, t_end=t_end, gif_path=gif_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='run.py',
        description='1-D ablation solver demo runner.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            'Demos: ' + ', '.join(DEMOS),
            '',
            'Examples:',
            '  python run.py                        # animate bilayer demo',
            '  python run.py --demo arcjet --gif    # animate + save result.gif',
            '  python run.py --headless --demo tacot',
            '  python run.py --test',
        ]),
    )
    p.add_argument('--demo', default='bilayer',
                   choices=list(DEMOS),
                   metavar='NAME',
                   help=f'demo to run (default: bilayer). Choices: {", ".join(DEMOS)}')
    p.add_argument('--headless', action='store_true',
                   help='run without animation and print energy summary')
    p.add_argument('--test', action='store_true',
                   help='run energy-conservation checks for all demos and exit')
    p.add_argument('--gif', nargs='?', const='result.gif', default=None,
                   metavar='PATH',
                   help='save animation to a GIF (default path: result.gif)')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    if args.test:
        run_test_all()
    elif args.headless:
        run_headless(args.demo)
    else:
        run_animation(args.demo, gif_path=args.gif)
