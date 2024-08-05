import itertools
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from pybaum import leaf_names, tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.config import PLOTLY_PALETTE, PLOTLY_TEMPLATE
from optimagic.logging.read_log import (
    OptimizeLogReader,
    read_optimization_problem_table,
)
from optimagic.optimization.history_tools import get_history_arrays
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.tree_registry import get_registry


def criterion_plot(
    results,
    names=None,
    max_evaluations=None,
    template=PLOTLY_TEMPLATE,
    palette=PLOTLY_PALETTE,
    stack_multistart=False,
    monotone=False,
    show_exploration=False,
):
    """Plot the criterion history of an optimization.

    Args:
        results (Union[List, Dict][Union[OptimizeResult, pathlib.Path, str]): A (list or
            dict of) optimization results with collected history. If dict, then the
            key is used as the name in a legend.
        names (Union[List[str], str]): Names corresponding to res or entries in res.
        max_evaluations (int): Clip the criterion history after that many entries.
        template (str): The template for the figure. Default is "plotly_white".
        palette (Union[List[str], str]): The coloring palette for traces. Default is
            "qualitative.Plotly".
        stack_multistart (bool): Whether to combine multistart histories into a single
            history. Default is False.
        monotone (bool): If True, the criterion plot becomes monotone in the sense
            that only that at each iteration the current best criterion value is
            displayed. Default is False.
        show_exploration (bool): If True, exploration samples of a multistart
            optimization are visualized. Default is False.

    Returns:
        plotly.graph_objs._figure.Figure: The figure.

    """
    # ==================================================================================
    # Process inputs
    # ==================================================================================

    if not isinstance(names, list) and names is not None:
        names = [names]

    if not isinstance(results, dict):
        if isinstance(results, list):
            names = range(len(results)) if names is None else names
            if len(names) != len(results):
                raise ValueError("len(results) needs to be equal to len(names).")
            results = dict(zip(names, results, strict=False))
        else:
            name = 0 if names is None else names
            if isinstance(name, list):
                if len(name) > 1:
                    raise ValueError("len(results) needs to be equal to len(names).")
                else:
                    name = name[0]
            results = {name: results}

    if not isinstance(palette, list):
        palette = [palette]
    palette = itertools.cycle(palette)

    key = "monotone_criterion" if monotone else "criterion"

    # ==================================================================================
    # Extract plotting data from results objects / data base
    # ==================================================================================

    data = []
    for name, res in results.items():
        if isinstance(res, OptimizeResult):
            _data = _extract_plotting_data_from_results_object(
                res, stack_multistart, show_exploration, plot_name="criterion_plot"
            )
        elif isinstance(res, (str, Path)):
            _data = _extract_plotting_data_from_database(
                res, stack_multistart, show_exploration
            )
        else:
            msg = "results must be (or contain) an OptimizeResult or a path to a log"
            f"file, but is type {type(res)}."
            raise TypeError(msg)

        _data["name"] = name
        data.append(_data)

    # ==================================================================================
    # Create figure
    # ==================================================================================

    fig = go.Figure()

    plot_multistart = (
        len(data) == 1 and data[0]["is_multistart"] and not stack_multistart
    )

    # ==================================================================================
    # Plot multistart paths

    if plot_multistart:
        scatter_kws = {
            "connectgaps": True,
            "showlegend": False,
        }

        for i, local_history in enumerate(data[0]["local_histories"]):
            history = get_history_arrays(local_history, data[0]["direction"])[key]

            if max_evaluations is not None and len(history) > max_evaluations:
                history = history[:max_evaluations]

            trace = go.Scatter(
                x=np.arange(len(history)),
                y=history,
                mode="lines",
                name=str(i),
                line_color="#bab0ac",
                **scatter_kws,
            )
            fig.add_trace(trace)

    # ==================================================================================
    # Plot main optimization objects

    for _data in data:
        if stack_multistart and _data["stacked_local_histories"] is not None:
            _history = _data["stacked_local_histories"]
        else:
            _history = _data["history"]
        history = get_history_arrays(_history, _data["direction"])[key]

        if max_evaluations is not None and len(history) > max_evaluations:
            history = history[:max_evaluations]

        scatter_kws = {
            "connectgaps": True,
            "showlegend": not plot_multistart,
        }

        _color = next(palette)
        if not isinstance(_color, str):
            msg = "highlight_palette needs to be a string or list of strings, but its "
            f"entry is of type {type(_color)}."
            raise TypeError(msg)

        line_kws = {
            "color": _color,
        }

        trace = go.Scatter(
            x=np.arange(len(history)),
            y=history,
            mode="lines",
            name="best result" if plot_multistart else _data["name"],
            line=line_kws,
            **scatter_kws,
        )
        fig.add_trace(trace)

    fig.update_layout(
        template=template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Criterion value",
        legend={"yanchor": "top", "xanchor": "right", "y": 0.95, "x": 0.95},
    )
    return fig


def params_plot(
    result,
    selector=None,
    max_evaluations=None,
    template=PLOTLY_TEMPLATE,
    show_exploration=False,
):
    """Plot the params history of an optimization.

    Args:
        result (Union[OptimizeResult, pathlib.Path, str]): An optimization results with
            collected history. If dict, then the key is used as the name in a legend.
        selector (callable): A callable that takes params and returns a subset
            of params. If provided, only the selected subset of params is plotted.
        max_evaluations (int): Clip the criterion history after that many entries.
        template (str): The template for the figure. Default is "plotly_white".
        show_exploration (bool): If True, exploration samples of a multistart
            optimization are visualized. Default is False.

    Returns:
        plotly.graph_objs._figure.Figure: The figure.

    """
    # ==================================================================================
    # Process inputs
    # ==================================================================================

    if isinstance(result, OptimizeResult):
        data = _extract_plotting_data_from_results_object(
            result,
            stack_multistart=True,
            show_exploration=show_exploration,
            plot_name="params_plot",
        )
        start_params = result.start_params
    elif isinstance(result, (str, Path)):
        data = _extract_plotting_data_from_database(
            result,
            stack_multistart=True,
            show_exploration=show_exploration,
        )
        start_params = data["start_params"]
    else:
        raise TypeError("result must be an OptimizeResult or a path to a log file.")

    if data["stacked_local_histories"] is not None:
        history = data["stacked_local_histories"]["params"]
    else:
        history = data["history"]["params"]

    # ==================================================================================
    # Create figure
    # ==================================================================================

    fig = go.Figure()

    registry = get_registry(extended=True)

    hist_arr = np.array([tree_just_flatten(p, registry=registry) for p in history]).T
    names = leaf_names(start_params, registry=registry)

    if selector is not None:
        flat, treedef = tree_flatten(start_params, registry=registry)
        helper = tree_unflatten(treedef, list(range(len(flat))), registry=registry)
        selected = np.array(tree_just_flatten(selector(helper), registry=registry))
        names = [names[i] for i in selected]
        hist_arr = hist_arr[selected]

    for name, data in zip(names, hist_arr, strict=False):
        if max_evaluations is not None and len(data) > max_evaluations:
            plot_data = data[:max_evaluations]
        else:
            plot_data = data

        trace = go.Scatter(
            x=np.arange(len(plot_data)),
            y=plot_data,
            mode="lines",
            name=name,
        )
        fig.add_trace(trace)

    fig.update_layout(
        template=template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Parameter value",
        legend={"yanchor": "top", "xanchor": "right", "y": 0.95, "x": 0.95},
    )

    return fig


def _extract_plotting_data_from_results_object(
    res, stack_multistart, show_exploration, plot_name
):
    """Extract data for plotting from results object.

    Args:
        res (OptmizeResult): An optimization results object.
        stack_multistart (bool): Whether to combine multistart histories into a single
            history. Default is False.
        show_exploration (bool): If True, exploration samples of a multistart
            optimization are visualized. Default is False.
        plot_name (str): Name of the plotting function that calls this function. Used
            for rasing errors.

    Returns:
        dict:
        - "history": The results history
        - "direction": maximize or minimize
        - "is_multistart": Whether the optimization used multistart
        - "local_histories": All other multistart histories except for 'history'. If not
        available is None. If show_exploration is True, the exploration phase is
        added as the first entry.
        - "stacked_local_histories": If stack_multistart is True the local histories
        are stacked into a single one.

    """
    if res.history is None:
        msg = f"{plot_name} requires an optimize result with history. Enable history "
        "collection by setting collect_history=True when calling maximize or minimize."
        raise ValueError(msg)

    is_multistart = res.multistart_info is not None

    if is_multistart:
        local_histories = [opt.history for opt in res.multistart_info["local_optima"]]
    else:
        local_histories = None

    if stack_multistart and local_histories is not None:
        stacked = _get_stacked_local_histories(local_histories)
        if show_exploration:
            stacked["params"] = (
                res.multistart_info["exploration_sample"][::-1] + stacked["params"]
            )
            stacked["criterion"] = (
                res.multistart_info["exploration_results"].tolist()[::-1]
                + stacked["criterion"]
            )
    else:
        stacked = None

    data = {
        "history": res.history,
        "direction": res.direction,
        "is_multistart": is_multistart,
        "local_histories": local_histories,
        "stacked_local_histories": stacked,
    }
    return data


def _extract_plotting_data_from_database(res, stack_multistart, show_exploration):
    """Extract data for plotting from database.

    Args:
        res (str or pathlib.Path): A path to an optimization database.
        stack_multistart (bool): Whether to combine multistart histories into a single
            history. Default is False.
        show_exploration (bool): If True, exploration samples of a multistart
            optimization are visualized. Default is False.

    Returns:
        dict:
        - "history": The results history
        - "direction": maximize or minimize
        - "is_multistart": Whether the optimization used multistart
        - "local_histories": All other multistart histories except for 'history'. If not
        available is None. If show_exploration is True, the exploration phase is
        added as the first entry.
        - "stacked_local_histories": If stack_multistart is True the local histories
        are stacked into a single one.

    """
    reader = OptimizeLogReader(res)
    _problem_table = read_optimization_problem_table(res)

    direction = _problem_table["direction"].tolist()[-1]

    history, local_histories, exploration = reader.read_multistart_history(direction)

    if stack_multistart and local_histories is not None:
        stacked = _get_stacked_local_histories(local_histories, history)
        if show_exploration:
            stacked["params"] = exploration["params"][::-1] + stacked["params"]
            stacked["criterion"] = exploration["criterion"][::-1] + stacked["criterion"]
    else:
        stacked = None

    data = {
        "history": history,
        "direction": direction,
        "is_multistart": local_histories is not None,
        "local_histories": local_histories,
        "stacked_local_histories": stacked,
        "start_params": reader.read_start_params(),
    }
    return data


def _get_stacked_local_histories(local_histories, history=None):
    """Stack local histories.

    Local histories is a list of dictionaries, each of the same structure. We transform
    this to a dictionary of lists. Finally, when the data is read from the database we
    append the best history at the end.

    """
    # list of dicts to dict of lists
    stacked = {key: [h[key] for h in local_histories] for key in local_histories[0]}
    # flatten inner lists
    stacked = {key: list(itertools.chain(*value)) for key, value in stacked.items()}
    # append additional history is necessary
    if history is not None:
        stacked = {key: value + history[key] for key, value in stacked.items()}
    return stacked
