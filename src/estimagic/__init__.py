import warnings
from dataclasses import dataclass

from optimagic import OptimizeLogReader as _OptimizeLogReader
from optimagic import OptimizeResult as _OptimizeResult
from optimagic import __version__
from optimagic import check_constraints as _check_constraints
from optimagic import convergence_plot as _convergence_plot
from optimagic import convergence_report as _convergence_report
from optimagic import count_free_params as _count_free_params
from optimagic import criterion_plot as _criterion_plot
from optimagic import first_derivative as _first_derivative
from optimagic import get_benchmark_problems as _get_benchmark_problems
from optimagic import maximize as _maximize
from optimagic import minimize as _minimize
from optimagic import params_plot as _params_plot
from optimagic import profile_plot as _profile_plot
from optimagic import rank_report as _rank_report
from optimagic import run_benchmark as _run_benchmark
from optimagic import second_derivative as _second_derivative
from optimagic import slice_plot as _slice_plot
from optimagic import traceback_report as _traceback_report
from optimagic.decorators import deprecated

from estimagic import utilities
from estimagic.bootstrap import BootstrapResult, bootstrap
from estimagic.estimate_ml import LikelihoodResult, estimate_ml
from estimagic.estimate_msm import MomentsResult, estimate_msm
from estimagic.estimation_table import (
    estimation_table,
    render_html,
    render_latex,
)
from estimagic.lollipop_plot import lollipop_plot
from estimagic.msm_weighting import get_moments_cov

MSG = (
    "estimagic.{name} has been deprecated in version 0.5.0. Use optimagic.{name} "
    "instead. This function will be removed in version 0.6.0."
)

minimize = deprecated(_minimize, MSG.format(name="minimize"))
maximize = deprecated(_maximize, MSG.format(name="maximize"))
first_derivative = deprecated(_first_derivative, MSG.format(name="first_derivative"))
second_derivative = deprecated(_second_derivative, MSG.format(name="second_derivative"))
run_benchmark = deprecated(_run_benchmark, MSG.format(name="run_benchmark"))
get_benchmark_problems = deprecated(
    _get_benchmark_problems, MSG.format(name="get_benchmark_problems")
)
convergence_report = deprecated(
    _convergence_report, MSG.format(name="convergence_report")
)
rank_report = deprecated(_rank_report, MSG.format(name="rank_report"))
traceback_report = deprecated(_traceback_report, MSG.format(name="traceback_report"))
profile_plot = deprecated(_profile_plot, MSG.format(name="profile_plot"))
convergence_plot = deprecated(_convergence_plot, MSG.format(name="convergence_plot"))
slice_plot = deprecated(_slice_plot, MSG.format(name="slice_plot"))
check_constraints = deprecated(_check_constraints, MSG.format(name="check_constraints"))
count_free_params = deprecated(_count_free_params, MSG.format(name="count_free_params"))
criterion_plot = deprecated(_criterion_plot, MSG.format(name="criterion_plot"))
params_plot = deprecated(_params_plot, MSG.format(name="params_plot"))


class OptimizeLogReader(_OptimizeLogReader):
    def __init__(self, path):
        warnings.warn(
            "estimagic.OptimizeLogReader has been deprecated in version 0.5.0. Use "
            "optimagic.OptimizeLogReader instead. This class will be removed in version"
            " 0.6.0.",
            FutureWarning,
        )
        super().__init__(path)


@dataclass
class OptimizeResult(_OptimizeResult):
    def __post_init__(self):
        warnings.warn(
            "estimagic.OptimizeResult has been deprecated in version 0.5.0. Use "
            "optimagic.OptimizeResult instead. This class will be removed in version "
            "0.6.0.",
            FutureWarning,
        )


__all__ = [
    "LikelihoodResult",
    "estimate_ml",
    "estimate_msm",
    "MomentsResult",
    "estimate_msm",
    "BootstrapResult",
    "bootstrap",
    "get_moments_cov",
    "estimation_table",
    "render_html",
    "render_latex",
    "utilities",
    "minimize",
    "maximize",
    "first_derivative",
    "second_derivative",
    "run_benchmark",
    "get_benchmark_problems",
    "profile_plot",
    "convergence_plot",
    "convergence_report",
    "rank_report",
    "traceback_report",
    "lollipop_plot",
    "slice_plot",
    "check_constraints",
    "count_free_params",
    "OptimizeLogReader",
    "OptimizeResult",
    "criterion_plot",
    "params_plot",
    "__version__",
]
