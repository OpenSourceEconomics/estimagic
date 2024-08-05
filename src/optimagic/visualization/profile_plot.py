import numpy as np
import pandas as pd
import plotly.express as px

from optimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)
from optimagic.config import PLOTLY_TEMPLATE


def profile_plot(
    problems,
    results,
    *,
    runtime_measure="n_evaluations",
    normalize_runtime=False,
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
    template=PLOTLY_TEMPLATE,
):
    """Compare optimizers over a problem set.

    This plot answers the question: What percentage of problems can each algorithm
    solve within a certain runtime budget?

    The runtime budget is plotted on the x axis and the share of problems each
    algorithm solved on the y axis.

    Thus, algorithms that are very specialized and perform well on some share of
    problems but are not able to solve more problems with a larger computational budget
    will have steep increases and then flat lines. Algorithms that are robust but slow,
    will have low shares in the beginning but reach very high.

    Note that failing to converge according to the given stopping_criterion and
    precisions is scored as needing an infinite computational budget.

    For details, see the description of performance and data profiles by
    Moré and Wild (2009).

    Args:
        problems (dict): optimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): optimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        runtime_measure (str): "n_evaluations", "n_batches" or "walltime".
            This is the runtime until the desired convergence was reached by an
            algorithm. This is called performance measure by Moré and Wild (2009).
        normalize_runtime (bool): If True the runtime each algorithm needed for each
            problem is scaled by the time the fastest algorithm needed. If True, the
            resulting plot is what Moré and Wild (2009) called data profiles.
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled.
        template (str): The template for the figure. Default is "plotly_white".

    Returns:
        plotly.Figure

    """
    if stopping_criterion is None:
        raise ValueError(
            "You must specify a stopping criterion for the performance plot. "
        )
    if runtime_measure not in ["walltime", "n_evaluations", "n_batches"]:
        raise ValueError(
            "Only 'walltime' or 'n_evaluations' are allowed as "
            f"runtime_measure. You specified {runtime_measure}."
        )

    df, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    solution_times = create_solution_times(
        df,
        runtime_measure=runtime_measure,
        converged_info=converged_info,
    )

    if normalize_runtime:
        solution_times = solution_times.divide(solution_times.min(axis=1), axis=0)
        solution_times[~converged_info] = np.inf

    alphas = _determine_alpha_grid(solution_times)
    for_each_alpha = pd.concat(
        {alpha: solution_times <= alpha for alpha in alphas},
        names=["alpha"],
    )
    performance_profiles = for_each_alpha.groupby("alpha").mean().stack().reset_index()

    fig = px.line(performance_profiles, x="alpha", y=0, color="algorithm")

    xlabels = {
        (
            "n_evaluations",
            True,
        ): "Multiple of Minimal Number of Function Evaluations<br>"
        "Needed to Solve the Problem",
        (
            "walltime",
            True,
        ): "Multiple of Minimal Wall Time<br>Needed to Solve the Problem",
        (
            "n_batches",
            True,
        ): "Multiple of Minimal Number of Batches<br>Needed to Solve the Problem",
        ("n_evaluations", False): "Number of Function Evaluations",
        ("walltime", False): "Wall Time Needed to Solve the Problem",
        ("n_batches", False): "Number of Batches",
    }

    fig.update_layout(
        xaxis_title=xlabels[(runtime_measure, normalize_runtime)],
        yaxis_title="Share of Problems Solved",
        title=None,
        height=300,
        width=500,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        template=template,
    )

    fig.add_hline(y=1)
    return fig


def create_solution_times(df, runtime_measure, converged_info, return_tidy=True):
    """Find the solution time for each algorithm and problem.

    Args:
        df (pandas.DataFrame): contains 'problem', 'algorithm' and 'runtime_measure'
            as columns.
        runtime_measure (str): 'walltime', 'n_batches' or 'n_evaluations'.
        converged_info (pandas.DataFrame): columns are the algorithms, indexes are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.
        return_tidy (bool): If True, the resulting DataFrame will be a tidy DataFrame
            with problem and algorithm as indexes and runtime_measure as column.
            If False, the resulting DataFrame will have problem, algorithm and
            runtime_measure as columns.

    Returns:
        solution_times (pandas.DataFrame): If return_tidy is True, indexes are the
            problems, columns are the algorithms. If return_tidy is False, columns are
            problem, algorithm and runtime_measure. The values are either the number
            of evaluations or the walltime each algorithm needed to achieve the
            desired precision. If the desired precision was not achieved the value is
            set to np.inf.

    """
    solution_times = df.groupby(["problem", "algorithm"])[runtime_measure].max()
    solution_times = solution_times.unstack()
    # We convert the dtype to float to support the use of np.inf
    solution_times = solution_times.astype(float).where(converged_info, other=np.inf)

    if not return_tidy:
        solution_times = solution_times.stack().reset_index()
        solution_times = solution_times.rename(
            columns={solution_times.columns[2]: runtime_measure}
        )

    return solution_times


def _determine_alpha_grid(solution_times):
    switch_points = _find_switch_points(solution_times=solution_times)

    point_to_right = switch_points[-1] * 1.05
    extended_switch_points = np.append(switch_points, point_to_right)
    mid_points = (extended_switch_points[:-1] + extended_switch_points[1:]) / 2
    alphas = sorted(np.append(extended_switch_points, mid_points))
    return alphas


def _find_switch_points(solution_times):
    """Determine the switch points of the performance profiles.

    Args:
        solution_times (pandas.DataFrame): columns are the names of the algorithms,
            the indexes are the problems. Values are performance measures.
            They can be either float, when normalize_runtime was True or int when the
            runtime_measure are not normalized function evaluations or datetime when
            the not normalized walltime is used.

    Returns:
        list: sorted switching points

    """
    switch_points = np.unique(solution_times.values)
    if pd.api.types.is_float_dtype(switch_points):
        switch_points += 1e-10
    switch_points = switch_points[np.isfinite(switch_points)]
    return switch_points
