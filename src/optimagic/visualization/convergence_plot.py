import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from optimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)
from optimagic.config import PLOTLY_TEMPLATE
from optimagic.utilities import propose_alternatives
from optimagic.visualization.plotting_utilities import create_grid_plot, create_ind_dict


def convergence_plot(
    problems,
    results,
    *,
    problem_subset=None,
    algorithm_subset=None,
    n_cols=2,
    distance_measure="criterion",
    monotone=True,
    normalize_distance=True,
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
    combine_plots_in_grid=True,
    template=PLOTLY_TEMPLATE,
    palette=px.colors.qualitative.Plotly,
):
    """Plot convergence of optimizers for a set of problems.

    This creates a grid of plots, showing the convergence of the different
    algorithms on each problem. The faster a line falls, the faster the algorithm
    improved on the problem. The algorithm converged where its line reaches 0
    (if normalize_distance is True) or the horizontal blue line labeled "true solution".

    Each plot shows on the x axis the runtime_measure, which can be walltime, number
    of evaluations or number of batches. Each algorithm's convergence is a line in the
    plot. Convergence can be measured by the criterion value of the particular
    time/evaluation. The convergence can be made monotone (i.e. always taking the bast
    value so far) or normalized such that the distance from the start to the true
    solution is one.

    Args:
        problems (dict): optimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): optimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        problem_subset (list, optional): List of problem names. These must be a subset
            of the keys of the problems dictionary. If provided the convergence plot is
            only created for the problems specified in this list.
        algorithm_subset (list, optional): List of algorithm names. These must be a
            subset of the keys of the optimizer_options passed to run_benchmark. If
            provided only the convergence of the given algorithms are shown.
        n_cols (int): number of columns in the plot of grids. The number
            of rows is determined automatically.
        distance_measure (str): One of "criterion", "parameter_distance".
        monotone (bool): If True the best found criterion value so far is plotted.
            If False the particular criterion evaluation of that time is used.
        normalize_distance (bool): If True the progress is scaled by the total distance
            between the start value and the optimal value, i.e. 1 means the algorithm
            is as far from the solution as the start value and 0 means the algorithm
            has reached the solution value.
        runtime_measure (str): "n_evaluations", "walltime" or "n_batches".
        stopping_criterion (str): "x_and_y", "x_or_y", "x", "y" or None. If None, no
            clipping is done.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled.
        combine_plots_in_grid (bool): decide whether to return a one
            figure containing subplots for each factor pair or a dictionary
            of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".
        palette: The coloring palette for traces. Default is "qualitative.Plotly".

    Returns:
        plotly.Figure: The grid plot or dict of individual plots

    """

    df, _ = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    # handle string provision for single problems / algorithms
    if isinstance(problem_subset, str):
        problem_subset = [problem_subset]
    if isinstance(algorithm_subset, str):
        algorithm_subset = [algorithm_subset]

    _check_only_allowed_subset_provided(problem_subset, df["problem"], "problem")
    _check_only_allowed_subset_provided(algorithm_subset, df["algorithm"], "algorithm")

    if problem_subset is not None:
        df = df[df["problem"].isin(problem_subset)]
    if algorithm_subset is not None:
        df = df[df["algorithm"].isin(algorithm_subset)]

    # plot configuration
    outcome = (
        f"{'monotone_' if monotone else ''}"
        + distance_measure
        + f"{'_normalized' if normalize_distance else ''}"
    )

    remaining_problems = df["problem"].unique()
    n_rows = int(np.ceil(len(remaining_problems) / n_cols))

    # pre - style plots labels
    y_labels = {
        "criterion": "Current Function Value",
        "monotone_criterion": "Best Function Value Found So Far",
        "criterion_normalized": "Share of Function Distance to Optimum<br>"
        "Missing From Current Criterion Value",
        "monotone_criterion_normalized": "Share of Function Distance to Optimum<br>"
        "Missing From Best So Far",
        "parameter_distance": "Distance Between Current and Optimal Parameters",
        "parameter_distance_normalized": "Share of Parameter Distance to Optimum<br>"
        "Missing From Current Parameters",
        "monotone_parameter_distance_normalized": "Share of the Parameter Distance "
        "to Optimum<br> Missing From the Best Parameters So Far",
        "monotone_parameter_distance": "Distance Between the Best Parameters So Far<br>"
        "and the Optimal Parameters",
    }

    x_labels = {
        "n_evaluations": "Number of Function Evaluations",
        "walltime": "Elapsed Time",
        "n_batches": "Number of Batches",
    }

    # container for individual plots
    g_list = []
    # container for titles
    titles = []

    # creating data traces for plotting faceted/individual plots
    # dropping usage of palette for algoritms, but use the built in pallete
    for prob_name in remaining_problems:
        g_ind = []  # container for data for traces in individual plot
        to_plot = df[df["problem"] == prob_name]
        if runtime_measure == "n_batches":
            to_plot = (
                to_plot.groupby(["algorithm", runtime_measure]).min().reset_index()
            )

        for i, alg in enumerate(to_plot["algorithm"].unique()):
            temp = to_plot[to_plot["algorithm"] == alg]
            trace_1 = go.Scatter(
                x=temp[runtime_measure],
                y=temp[outcome],
                mode="lines",
                legendgroup=i,
                name=alg,
                line={"color": palette[i]},
            )
            g_ind.append(trace_1)

        if distance_measure == "criterion" and not normalize_distance:
            f_opt = problems[prob_name]["solution"]["value"]
            trace_2 = go.Scatter(
                y=[f_opt for i in to_plot[runtime_measure]],
                x=to_plot[runtime_measure],
                mode="lines",
                line={"color": palette[i + 1]},
                name="true solution",
                legendgroup=i + 1,
            )
            g_ind.append(trace_2)

        g_list.append(g_ind)
        titles.append(prob_name.replace("_", " ").title())

    xaxis_title = [x_labels[runtime_measure] for ind in range(len(g_list))]
    yaxis_title = [y_labels[outcome] for ind in range(len(g_list))]

    common_dependencies = {
        "ind_list": g_list,
        "names": titles,
        "clean_legend": True,
        "x_title": xaxis_title,
        "y_title": yaxis_title,
    }
    common_layout = {
        "template": template,
        "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    }

    # Plot with subplots
    if combine_plots_in_grid:
        g = create_grid_plot(
            rows=n_rows,
            cols=n_cols,
            **common_dependencies,
            kws={"height": 320 * n_rows, "width": 500 * n_cols, **common_layout},
        )
        out = g

    # Dictionary for individual plots
    else:
        ind_dict = create_ind_dict(
            **common_dependencies,
            kws={"height": 320, "width": 500, "title_x": 0.5, **common_layout},
        )

        out = ind_dict

    return out


def _check_only_allowed_subset_provided(subset, allowed, name):
    """Check if all entries of a proposed subset are in a Series.

    Args:
        subset (iterable or None): If None, no checks are performed. Else a ValueError
            is raised listing all entries that are not in the provided Series.
        allowed (iterable): allowed entries.
        name (str): name of the provided entries to use for the ValueError.

    Raises:
        ValueError

    """
    allowed = set(allowed)
    if subset is not None:
        missing = [entry for entry in subset if entry not in allowed]
        if missing:
            missing_msg = ""
            for entry in missing:
                proposed = propose_alternatives(entry, allowed)
                missing_msg += f"Invalid {name}: {entry}. Did you mean {proposed}?\n"
            raise ValueError(missing_msg)
