"""
ablation — 1-D finite-volume ablation / heat-conduction solver.

Public API
----------
Materials  : ALUMINIUM, ALUMINA, COPPER, SILICA, PICA, TACOT
             materials.get(name)  — look up by string
Boundaries : HeatFlux, Convection, Radiation, Adiabatic, Combined
Geometry   : Layer, Problem
Solver     : Solver
"""

from .materials import (
    Material,
    ALUMINIUM, ALUMINA, COPPER,
    SILICA, PICA, TACOT,
    get as get_material,
)
from .boundaries import (
    BoundaryCondition,
    HeatFlux, Convection, Radiation, Adiabatic, Combined,
)
from .problem import Layer, Problem
from .solver  import Solver
from .results import Results

__all__ = [
    # materials
    'Material',
    'ALUMINIUM', 'ALUMINA', 'COPPER',
    'SILICA', 'PICA', 'TACOT',
    'get_material',
    # boundaries
    'BoundaryCondition',
    'HeatFlux', 'Convection', 'Radiation', 'Adiabatic', 'Combined',
    # geometry / problem
    'Layer', 'Problem',
    # solver + results
    'Solver', 'Results',
]
