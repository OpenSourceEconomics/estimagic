from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from pybaum import tree_just_flatten

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import DEFAULT_N_CORES, PLOTLY_TEMPLATE
from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.parameters.bounds import pre_process_bounds
from optimagic.parameters.conversion import get_converter
from optimagic.parameters.tree_registry import get_registry
from optimagic.visualization.plotting_utilities import combine_plots, get_layout_kwargs


def slice_plot(
    func,
    params,
    bounds=None,
    func_kwargs=None,
    selector=None,
    n_cores=DEFAULT_N_CORES,
    n_gridpoints=20,
    plots_per_row=2,
    param_names=None,
    share_y=True,
    expand_yrange=0.02,
    share_x=False,
    color="#497ea7",
    template=PLOTLY_TEMPLATE,
    title=None,
    return_dict=False,
    make_subplot_kwargs=None,
    batch_evaluator="joblib",
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Plot criterion along coordinates at given and random values.

    Generates plots for each parameter and optionally combines them into a figure
    with subplots.

    # TODO: Use soft bounds to create the grid (if available).

    Args:
        criterion (callable): criterion function that takes params and returns a
            scalar value or dictionary with the entry "value".
        params (pytree): A pytree with parameters.
        bounds: Lower and upper bounds on the parameters. The bounds are used to create
            a grid over which slice plots are drawn. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used for
            slice_plots. Each bound type mirrors the structure of params. Check our
            how-to guide on bounds for examples. If params is a flat numpy array, you
            can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        selector (callable): Function that takes params and returns a subset
            of params for which we actually want to generate the plot.
        n_cores (int): Number of cores.
        n_gridpoins (int): Number of gridpoints on which the criterion function is
            evaluated. This is the number per plotted line.
        plots_per_row (int): Number of plots per row.
        param_names (dict or NoneType): Dictionary mapping old parameter names
            to new ones.
        share_y (bool): If True, the individual plots share the scale on the
            yaxis and plots in one row actually share the y axis.
        share_x (bool): If True, set the same range of x axis for all plots and share
            the x axis for all plots in one column.
        expand_y (float): The ration by which to expand the range of the (shared) y
            axis, such that the axis is not cropped at exactly max of Criterion Value.
        color: The line color.
        template (str): The template for the figure. Default is "plotly_white".
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        title (str): The figure title.
        return_dict (bool): If True, return dictionary with individual plots of each
            parameter, else, ombine individual plots into a figure with subplots.
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots (governed by
            'horizontal_spacing' and 'vertical_spacing'). If None, default arguments
            defined in the function are used.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.


    Returns:
        out (dict or plotly.Figure): Returns either dictionary with individual slice
            plots for each parameter or a plotly Figure combining the individual plots.

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )

    bounds = pre_process_bounds(bounds)

    layout_kwargs = None
    if title is not None:
        title_kwargs = {"text": title}
    else:
        title_kwargs = None

    if func_kwargs is not None:
        func = partial(func, **func_kwargs)

    func_eval = func(params)

    converter, internal_params = get_converter(
        params=params,
        constraints=None,
        bounds=bounds,
        func_eval=func_eval,
        primary_key="value",
    )

    n_params = len(internal_params.values)

    selected = np.arange(n_params, dtype=int)
    if selector is not None:
        helper = converter.params_from_internal(selected)
        registry = get_registry(extended=True)
        selected = np.array(
            tree_just_flatten(selector(helper), registry=registry), dtype=int
        )

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")

    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    evaluation_points, metadata = [], []
    for pos in selected:
        lb = internal_params.lower_bounds[pos]
        ub = internal_params.upper_bounds[pos]
        grid = np.linspace(lb, ub, n_gridpoints)
        name = internal_params.names[pos]
        for param_value in grid:
            if param_value != internal_params.values[pos]:
                meta = {
                    "name": name,
                    "Parameter Value": param_value,
                }

                x = internal_params.values.copy()
                x[pos] = param_value
                point = converter.params_from_internal(x)
                evaluation_points.append(point)
                metadata.append(meta)

    batch_evaluator = process_batch_evaluator(batch_evaluator)

    func_values = batch_evaluator(
        func=func,
        arguments=evaluation_points,
        error_handling="continue",
        n_cores=n_cores,
    )

    # add NaNs where an evaluation failed
    func_values = [
        converter.func_to_internal(val) if not isinstance(val, str) else np.nan
        for val in func_values
    ]

    func_values += [converter.func_to_internal(func_eval)] * len(selected)
    for pos in selected:
        meta = {
            "name": internal_params.names[pos],
            "Parameter Value": internal_params.values[pos],
        }
        metadata.append(meta)

    plot_data = pd.DataFrame(metadata)
    plot_data["Function Value"] = func_values

    if param_names is not None:
        plot_data["name"] = plot_data["name"].replace(param_names)

    lb = plot_data["Function Value"].min()
    ub = plot_data["Function Value"].max()
    y_range = ub - lb
    yaxis_ub = ub + y_range * expand_yrange
    yaxis_lb = lb - y_range * expand_yrange
    layout_kwargs = get_layout_kwargs(
        layout_kwargs,
        None,
        title_kwargs,
        template,
        False,
    )

    plots_dict = {}
    for pos in selected:
        par_name = internal_params.names[pos]
        if param_names is not None and par_name in param_names:
            par_name = param_names[par_name]

        df = plot_data[plot_data["name"] == par_name].sort_values("Parameter Value")
        subfig = px.line(
            df,
            y="Function Value",
            x="Parameter Value",
            color_discrete_sequence=[color],
        )
        subfig.add_trace(
            go.Scatter(
                x=[internal_params.values[pos]],
                y=[converter.func_to_internal(func_eval)],
                marker={"color": color},
            )
        )
        subfig.update_layout(**layout_kwargs)
        subfig.update_xaxes(title={"text": par_name})
        subfig.update_yaxes(title={"text": "Function Value"})
        if share_y is True:
            subfig.update_yaxes(range=[yaxis_lb, yaxis_ub])
        plots_dict[par_name] = subfig
    if return_dict:
        out = plots_dict
    else:
        plots = list(plots_dict.values())
        out = combine_plots(
            plots=plots,
            plots_per_row=plots_per_row,
            sharex=share_x,
            sharey=share_y,
            share_yrange_all=share_y,
            share_xrange_all=share_x,
            expand_yrange=expand_yrange,
            make_subplot_kwargs=make_subplot_kwargs,
            showlegend=False,
            template=template,
            clean_legend=True,
            layout_kwargs=layout_kwargs,
            legend_kwargs={},
            title_kwargs=title_kwargs,
        )
    return out
