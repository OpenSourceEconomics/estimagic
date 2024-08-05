import math

import pandas as pd
import plotly.graph_objects as go
from optimagic.config import PLOTLY_PALETTE, PLOTLY_TEMPLATE
from optimagic.visualization.plotting_utilities import create_grid_plot, create_ind_dict


def lollipop_plot(
    data,
    *,
    sharex=True,
    plot_bar=True,
    n_rows=1,
    scatterplot_kws=None,
    barplot_kws=None,
    combine_plots_in_grid=True,
    template=PLOTLY_TEMPLATE,
    palette=PLOTLY_PALETTE,
):
    """Make a lollipop plot.

    Args:
        data (pandas.DataFrame): The datapoints to be plotted. The whole data will be
        plotted. Thus if you want to plot just some variables or rows you need
        to restrict the dataset before passing it.
        sharex (bool): Whether the x-axis is shared across variables, default True.
        plot_bar (bool): Whether thin bars are plotted, default True.
        n_rows (int): Number of rows for a grid if plots are combined
            in a grid, default 1. The number of columns is determined automatically.
        scatterplot_kws (dict): Keyword arguments to plot the dots of the lollipop plot
            via the scatter function.
        barplot_kws (dict): Keyword arguments to plot the lines of the lollipop plot
            via the barplot function.
        combine_plots_in_grid (bool): decide whether to return a one
        figure containing subplots for each factor pair or a dictionary
        of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".
        palette: The coloring palette for traces. Default is "qualitative.Plotly".

    Returns:
        plotly.Figure: The grid plot or dict of individual plots

    """
    data, varnames = _harmonize_data(data)

    scatter_dict = {
        "mode": "markers",
        "marker": {"color": palette[0]},
        "showlegend": False,
    }

    bar_dict = {
        "orientation": "h",
        "width": 0.03,
        "marker": {"color": palette[0]},
        "showlegend": False,
    }

    scatterplot_kws = (
        scatter_dict
        if scatterplot_kws is None
        else scatter_dict.update(
            {k: v for k, v in scatterplot_kws.items() if k not in scatter_dict}
        )
    )
    barplot_kws = (
        bar_dict
        if barplot_kws is None
        else bar_dict.update(
            {k: v for k, v in barplot_kws.items() if k not in bar_dict}
        )
    )

    # container for individual plots
    g_list = []
    # container for titles
    titles = []

    # creating data traces for plotting faceted/individual plots
    for indep_name in varnames:
        g_ind = []
        # dot plot using the scatter function
        to_plot = data[data["indep"] == indep_name]
        trace_1 = go.Scatter(x=to_plot["values"], y=to_plot["__name__"], **scatter_dict)
        g_ind.append(trace_1)

        # bar plot
        if plot_bar:
            trace_2 = go.Bar(x=to_plot["values"], y=to_plot["__name__"], **bar_dict)
        g_ind.append(trace_2)

        g_list.append(g_ind)
        titles.append(indep_name)

    # common x range
    lower_candidate = data[["indep", "values"]].groupby("indep").min().min()
    upper_candidate = data[["indep", "values"]].groupby("indep").max().max()
    padding = (upper_candidate - lower_candidate) / 10
    lower = lower_candidate - padding
    upper = upper_candidate + padding

    common_dependencies = {
        "ind_list": g_list,
        "names": titles,
        "share_xax": sharex,
        "x_min": lower,
        "x_max": upper,
    }
    common_layout = {
        "template": template,
        "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    }

    # Plot with subplots
    if combine_plots_in_grid:
        n_cols = math.ceil(len(varnames) / n_rows)

        g = create_grid_plot(
            rows=n_rows,
            cols=n_cols,
            **common_dependencies,
            kws={"height": 150 * n_rows, "width": 150 * n_cols, **common_layout},
        )
        out = g

    # Dictionary for individual plots
    else:
        ind_dict = create_ind_dict(
            **common_dependencies,
            kws={"height": 150, "width": 150, "title_x": 0.5, **common_layout},
        )
        out = ind_dict

    return out


def _harmonize_data(data):
    if not isinstance(data, list):
        data = [data]

    to_concat = []
    for i, _df in enumerate(data):
        df = _df.copy()
        df.columns = _make_string_index(df.columns)
        df.index = _make_string_index(df.index)
        df["__name__"] = df.index
        df["__hue__"] = i
        to_concat.append(df)

    combined = pd.concat(to_concat)
    # so that it is possibel to facet the strip plot
    new_data = pd.melt(
        combined, id_vars=["__name__", "__hue__"], var_name="indep", value_name="values"
    )

    varnames = new_data["indep"].unique()

    return new_data, varnames


def _make_string_index(ind):
    if isinstance(ind, pd.MultiIndex):
        out = ind.map(lambda tup: "_".join(str(name) for name in tup)).tolist()
    else:
        out = ind.map(str).tolist()
    return out
