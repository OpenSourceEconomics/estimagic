import pandas as pd

from optimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)
from optimagic.visualization.profile_plot import create_solution_times


def convergence_report(
    problems, results, *, stopping_criterion="y", x_precision=1e-4, y_precision=1e-4
):
    """Create a DataFrame with convergence information for a set of problems.

    Args:
        problems (dict): optimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): optimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions. Default is "y".
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        pandas.DataFrame: indexes are the problems, columns are the algorithms and
            the dimensionality of the benchmark problems. For the algorithms column,
            the values are strings that are either "success", "failed", or "error".
            For the dimensionality column, the values denote the number of dimensions
            of the problem.

    """
    _, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    report = _get_success_info(results, converged_info)
    report["dimensionality"] = report.index.map(_get_problem_dimensions(problems))

    return report


def rank_report(
    problems,
    results,
    *,
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
):
    """Create a DataFrame with rank information for a set of problems.

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
            Default is "n_evaluations".
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        pandas.DataFrame: indexes are the problems, columns are the algorithms and the
            dimensionality of the problems. The values are the ranks of the algorithms
            for each problem, where 0 means the algorithm was the fastest, 1 means it
            was the second fastest and so on. If an algorithm did not converge on a
            problem, the value is "failed". If an algorithm did encounter an error
            during optimization, the value is "error".

    """
    histories, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    solution_times = create_solution_times(
        histories, runtime_measure, converged_info, return_tidy=False
    )
    solution_times["rank"] = (
        solution_times.groupby("problem")[runtime_measure].rank(
            method="dense", ascending=True
        )
        - 1
    ).astype("Int64")

    success_info = _get_success_info(results, converged_info)

    df_wide = solution_times.pivot(index="problem", columns="algorithm", values="rank")
    report = df_wide.astype(str)
    report.columns.name = None

    report[~converged_info] = success_info
    report["dimensionality"] = report.index.map(_get_problem_dimensions(problems))

    return report


def traceback_report(problems, results, return_type="dataframe"):
    """Create traceback report for all problems that have not been solved.

    Args:
        results (dict): optimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        return_type (str): either "text", "markdown", "dict" or "dataframe".
            If "text", the traceback report is returned as a string. If "markdown",
            it is a markdown string. If "dict", it is returned as a dictionary.
            If "dataframe", it is a tidy pandas DataFrame, where indexes are the
            algorithm and problem names, the columns are the tracebacks and the
            dimensionality of the problem. Default is "dataframe".

    Returns:
        (list or str or dict or pandas.DataFrame): traceback report. If return_type
            is "text", the report is a list of strings. If "markdown", it is a
            formatted markdown string with algorithms and problem names as headers.
            If return_type is "dict", the report is a dictionary. If return_type is
            "dataframe", it is a tidy pandas DataFrame. In the latter case, indexes
            are the algorithm and problem names, the columns are the tracebacks and
            the dimensionality of the problems. The values are the tracebacks of the
            algorithms for problems where they stopped with an error.

    """

    if return_type == "text":
        report = []
        for result in results.values():
            if isinstance(result["solution"], str):
                report.append(result["solution"])

    elif return_type == "markdown":
        report = "```python"
        for (problem_name, algorithm_name), result in results.items():
            if isinstance(result["solution"], str):
                if f"### {algorithm_name}" not in report:
                    report += f"\n### {algorithm_name} \n"
                report += f"\n#### {problem_name} \n"
                report += f"\n{result['solution']} \n"
        report += "\n```"

    elif return_type == "dict":
        report = {}
        for (problem_name, algorithm_name), result in results.items():
            if isinstance(result["solution"], str):
                report[(problem_name, algorithm_name)] = result["solution"]

    elif return_type == "dataframe":
        tracebacks = {}
        for (problem_name, algorithm_name), result in results.items():
            if isinstance(result["solution"], str):
                tracebacks[algorithm_name] = tracebacks.setdefault(algorithm_name, {})
                tracebacks[algorithm_name][problem_name] = result["solution"]

        report = pd.DataFrame.from_dict(tracebacks, orient="index").stack().to_frame()
        report.index.set_names(["algorithm", "problem"], inplace=True)
        report.columns = ["traceback"]
        report["dimensionality"] = 0

        for problem_name, dim in _get_problem_dimensions(problems).items():
            if problem_name in report.index.get_level_values("problem"):
                report.loc[(slice(None), problem_name), "dimensionality"] = dim

    else:
        raise ValueError(
            f"return_type {return_type} is not supported. Must be one of "
            f"'text', 'markdown', 'dict' or 'dataframe'."
        )

    return report


def _get_success_info(results, converged_info):
    """Create a DataFrame with information on whether an algorithm succeeded or not.

    Args:
        results (dict): optimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        converged_info (pandas.DataFrame): columns are the algorithms, indexes are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    Returns:
        pandas.DataFrame: indexes are the problems, columns are the algorithms.
           values are strings that are either "success", "failed", or "error".

    """
    success_info = converged_info.replace({True: "success", False: "failed"})

    for key, value in results.items():
        if isinstance(value["solution"], str):
            success_info.at[key] = "error"

    return success_info


def _get_problem_dimensions(problems):
    """Get the dimension of each problem.

    Args:
        problems (dict): dictionary of problems. keys are problem names, values are
            dictionaries with the problem information.

    Returns:
        dict: keys are problem names, values are the dimension of the problem.

    """
    return {prob: len(problems[prob]["inputs"]["params"]) for prob in problems}
