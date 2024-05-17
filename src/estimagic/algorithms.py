import inspect

from estimagic.optimization import (
    bhhh,
    cyipopt_optimizers,
    fides_optimizers,
    nag_optimizers,
    neldermead,
    nlopt_optimizers,
    pounders,
    pygmo_optimizers,
    scipy_optimizers,
    simopt_optimizers,
    tao_optimizers,
    tranquilo,
)

MODULES = [
    cyipopt_optimizers,
    fides_optimizers,
    nag_optimizers,
    nlopt_optimizers,
    pygmo_optimizers,
    scipy_optimizers,
    simopt_optimizers,
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
    name for name, func in ALL_ALGORITHMS.items() if func._algorithm_info.is_global  # type: ignore
]
