import itertools
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimagic.config import PLOTLY_TEMPLATE


def combine_plots(
    plots,
    plots_per_row=2,
    sharex=False,
    sharey=True,
    share_yrange_all=True,
    expand_yrange=0.02,
    share_xrange_all=False,
    make_subplot_kwargs=None,
    showlegend=True,
    template=PLOTLY_TEMPLATE,
    clean_legend=True,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
):
    """Combine individual plots into figure with subplots. Uses list of plotly Figures
    to build plotly Figure with subplots.

    Args:
        plots (list): List with individual plots.
        plots_per_row (int): Number of plots per row.
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots. If None,
            default arguments defined in the function are used.
        sharex (bool): Whether to share the properties of x-axis across subplots. In
            the sam column
        sharey (bool): If True, share the properties of y-axis across subplots in the
        share_yrange_all (bool): If True, set the same range of y axis for all plots.
        y_expand (float): The ration by which to expand the range of the (shared) y
            axis, such that the axis is not cropped at exactly max of y variable.
        share_xrange_all (bool): If True, set the same range of x axis for all plots.
        showlegend (bool): If True, show legend.
        template (str): Plotly layout template. Must be one of plotly.io.templates.
        clean_legend (bool): If True, then cleans the legend from duplicates.
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title.

    Returns:
        fig (plotly.Figure): Plotly figure with subplots that combines individual
            slice plots.

    """
    plots = deepcopy(plots)

    make_subplot_kwargs, nrows = get_make_subplot_kwargs(
        sharex, sharey, make_subplot_kwargs, plots_per_row, plots
    )
    fig = make_subplots(**make_subplot_kwargs)
    layout_kwargs = get_layout_kwargs(
        layout_kwargs, legend_kwargs, title_kwargs, template, showlegend
    )
    for i, (row, col) in enumerate(
        itertools.product(np.arange(nrows), np.arange(plots_per_row))
    ):
        try:
            subfig = plots[i]
            fig.update_xaxes(
                title_text=subfig.layout.xaxis.title.text, col=col + 1, row=row + 1
            )
            if sharey:
                if col == 0:
                    fig.update_yaxes(
                        title_text=subfig.layout.yaxis.title.text,
                        col=col + 1,
                        row=row + 1,
                    )
            else:
                fig.update_yaxes(
                    title_text=subfig.layout.yaxis.title.text, col=col + 1, row=row + 1
                )
        except IndexError:
            subfig = go.Figure()
        for d in subfig.data:
            fig.add_trace(
                d,
                col=col + 1,
                row=row + 1,
            )

    fig.update_layout(**layout_kwargs, width=400 * plots_per_row, height=300 * nrows)
    if share_yrange_all:
        lb = []
        ub = []
        for f in plots:
            for d in f.data:
                lb.append(np.min(d["y"]))
                ub.append(np.max(d["y"]))
        ub = np.max(ub)
        lb = np.min(lb)
        y_range = ub - lb
        y_lower = lb - y_range * expand_yrange
        y_upper = ub + y_range * expand_yrange
        fig.update_yaxes(range=[y_lower, y_upper])
    if share_xrange_all:
        lb = []
        ub = []
        for f in plots:
            for d in f.data:
                lb.append(np.min(d["x"]))
                ub.append(np.max(d["x"]))
        x_upper = np.max(ub)
        x_lower = np.min(lb)
        fig.update_xaxes(range=[x_lower, x_upper])
    if clean_legend:
        fig = _clean_legend_duplicates(fig)
    return fig


def create_grid_plot(
    rows,
    cols,
    ind_list,
    names,
    kws,
    x_title=None,
    y_title=None,
    clean_legend=False,
    scientific_notation=False,
    share_xax=False,
    x_min=None,
    x_max=None,
):
    """Create a dictionary for a grid plot from a list of traces.

    Args:
        rows (int): Number of rows in a plot.
        cols (int): Number of cols in a plot.
        ind_list (iterable): The list of traces for each individual plot.
        names (iterable): The list of titles for the each plot.
        kws (dict): The dictionary for the layout.update, unified for each
        individual plot.
        x_title (iterable or None): The list of x-axis labels for each plot. If None,
        then no labels are added.
        y_title (iterable or None): The list of y-axis labels for each plot. If None,
        then no labels are added.
        clean_legend (bool): If True, then cleans the legend from duplicates.
        Default False.
        sci_notation (bool): If True then updates the ticks on x- and y-axis to
        be displayed in a scientific notation. Default False.
        share_xax (bool): If True, then the x-axis domain is the same
        for each individual plot.
        x_min (int or None): The lower bound for share_xax.
        x_max (int or None): The upped bound for share_xax.

    Returns:
        plotly.Figure: The plot with subplots.

    """
    if x_title is None:
        x_title = ["" for ind in range(len(ind_list))]
    if y_title is None:
        y_title = ["" for ind in range(len(ind_list))]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names)
    for ind, (facet_row, facet_col) in enumerate(
        itertools.product(range(1, rows + 1), range(1, cols + 1))
    ):
        if ind + 1 > len(ind_list):
            break  # if there are empty individual plots

        traces = ind_list[ind]
        for trace in range(len(traces)):
            fig.add_trace(traces[trace], row=facet_row, col=facet_col)
            # style axis labels
            fig.update_xaxes(row=facet_row, col=facet_col, title=x_title[ind])
            fig.update_yaxes(row=facet_row, col=facet_col, title=y_title[ind])

    # deleting duplicates in legend
    if clean_legend:
        fig = _clean_legend_duplicates(fig)

    # scientific notations for axis ticks
    if scientific_notation:
        fig.update_yaxes(tickformat=".2e")
        fig.update_xaxes(tickformat=".2e")

    if share_xax:
        fig.update_xaxes(range=[x_min, x_max])

    # setting template theme and size
    fig.update_layout(**kws)

    return fig


def create_ind_dict(
    ind_list,
    names,
    kws,
    x_title=None,
    y_title=None,
    clean_legend=False,
    scientific_notation=False,
    share_xax=False,
    x_min=None,
    x_max=None,
):
    """Create a dictionary for individual plots from a list of traces.

    Args:
        ind_list (iterable): The list of traces for each individual plot.
        names (iterable): The list of titles for the each plot.
        kws (dict): The dictionary for the layout.update, unified for each
        individual plot.
        x_title (iterable or None): The list of x-axis labels for each plot. If None,
        then no labels are added.
        y_title (iterable or None): The list of y-axis labels for each plot. If None,
        then no labels are added.
        clean_legend (bool): If True, then cleans the legend from duplicates.
        Default False.
        sci_notation (bool): If True then updates the ticks on x- and y-axis to
        be displayed in a scientific notation. Default False.
        share_xax (bool): If True, then the x-axis domain is the same
        for each individual plot.
        x_min (int or None): The lower bound for share_xax.
        x_max (int or None): The upped bound for share_xax.

    Returns:
        Dictionary of individual plots.

    """
    fig_dict = {}
    if x_title is None:
        x_title = ["" for ind in range(len(ind_list))]
    if y_title is None:
        y_title = ["" for ind in range(len(ind_list))]

    for ind in range(len(ind_list)):
        fig = go.Figure()
        traces = ind_list[ind]
        for trace in range(len(traces)):
            fig.add_trace(traces[trace])
        # adding title and styling axes and theme
        fig.update_layout(
            title=names[ind], xaxis_title=x_title[ind], yaxis_title=y_title[ind], **kws
        )
        # scientific notations for axis ticks
        if scientific_notation:
            fig.update_yaxes(tickformat=".2e")
            fig.update_xaxes(tickformat=".2e")
        # deleting duplicates in legend
        if clean_legend:
            fig = _clean_legend_duplicates(fig)
        if share_xax:
            fig.update_xaxes(range=[x_min, x_max])
        # adding to dictionary
        key = names[ind].replace(" ", "_").lower()
        fig_dict[key] = fig

    return fig_dict


def _clean_legend_duplicates(fig):
    trace_names = set()

    def disable_legend_if_duplicate(trace):
        if trace.name in trace_names:
            # in this case the legend is a duplicate
            trace.update(showlegend=False)
        else:
            trace_names.add(trace.name)

    fig.for_each_trace(disable_legend_if_duplicate)
    return fig


def get_make_subplot_kwargs(sharex, sharey, kwrgs, plots_per_row, plots):
    """Define and update keywargs for instantiating figure with subplots."""
    nrows = int(np.ceil(len(plots) / plots_per_row))
    default_kwargs = {
        "rows": nrows,
        "cols": plots_per_row,
        "start_cell": "top-left",
        "print_grid": False,
        "shared_yaxes": sharey,
        "shared_xaxes": sharex,
        "horizontal_spacing": 1 / (plots_per_row * 4),
    }

    if nrows > 1:
        default_kwargs["vertical_spacing"] = (1 / (nrows - 1)) / 3

    if not sharey:
        default_kwargs["horizontal_spacing"] = 2 * default_kwargs["horizontal_spacing"]
    if kwrgs:
        default_kwargs.update(kwrgs)
    return default_kwargs, nrows


def get_layout_kwargs(layout_kwargs, legend_kwargs, title_kwargs, template, showlegend):
    """Define and update default kwargs for update_layout.

    Defines some default keyword arguments to update figure layout, such as title and
    legend.

    """
    default_kwargs = {
        "template": template,
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "showlegend": showlegend,
        "legend": {},
        "title": {},
    }
    if title_kwargs:
        default_kwargs["title"] = title_kwargs
    if legend_kwargs:
        default_kwargs["legend"].update(legend_kwargs)
    if layout_kwargs:
        default_kwargs.update(layout_kwargs)
    return default_kwargs
