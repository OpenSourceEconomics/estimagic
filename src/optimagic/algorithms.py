import inspect

from optimagic.optimizers import (
    bhhh,
    fides,
    ipopt,
    nag_optimizers,
    neldermead,
    nlopt_optimizers,
    pounders,
    pygmo_optimizers,
    scipy_optimizers,
    tao_optimizers,
    tranquilo,
)

MODULES = [
    ipopt,
    fides,
    nag_optimizers,
    nlopt_optimizers,
    pygmo_optimizers,
    scipy_optimizers,
    tao_optimizers,
    bhhh,
    neldermead,
    pounders,
    tranquilo,
]

ALL_ALGORITHMS = {}
AVAILABLE_ALGORITHMS = {}
for module in MODULES:
    func_dict = dict(inspect.getmembers(module, inspect.isfunction))
    for name, func in func_dict.items():
        if hasattr(func, "_algorithm_info"):
            ALL_ALGORITHMS[name] = func
            if func._algorithm_info.is_available:
                AVAILABLE_ALGORITHMS[name] = func


GLOBAL_ALGORITHMS = [
    name for name, func in ALL_ALGORITHMS.items() if func._algorithm_info.is_global
]
