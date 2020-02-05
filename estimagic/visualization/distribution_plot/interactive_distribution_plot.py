"""Main module for the interactive distribution plot."""
from bokeh.layouts import Column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Div
from bokeh.models.widgets import RangeSlider
from bokeh.plotting import figure

from estimagic.visualization.distribution_plot.callbacks import add_hover_tool
from estimagic.visualization.distribution_plot.callbacks import add_select_tools
from estimagic.visualization.distribution_plot.callbacks import create_group_widget
from estimagic.visualization.distribution_plot.callbacks import create_view
from estimagic.visualization.distribution_plot.callbacks import value_slider
from estimagic.visualization.distribution_plot.process_inputs import process_inputs


def interactive_distribution_plot(
    doc,
    source,
    value_col,
    id_col,
    group_cols=None,
    subgroup_col=None,
    lower_bound_col=None,
    upper_bound_col=None,
    figure_height=None,
    width=500,
    x_padding=0.1,
    num_bins=50,
):
    """Create an interactive distribution plot from a tidy DataFrame.

    Args:
        doc (bokeh.Document):
            document to which to add the plot
        source (pd.DataFrame or str or pathlib.Path):
            Tidy DataFrame or location of the database file that contains tidy data.
            see: http://vita.had.co.nz/papers/tidy-data.pdf
        value_col (str):
            Name of the column for which to draw the histogram.
            In case of a parameter comparison plot this would be the "value" columns
            of the params DataFrames returned by maximize or minimize.
        id_col (str):
            Name of the column that identifies
            which values belong to the same observation.
            In case of a parameter comparison plot
            this would be the "model_name" column.
        group_cols (list):
            Name of the columns that identify groups that will be plotted together.
            In case of a parameter comparison plot this would be the parameter group
            and parameter name by default.
        subgroup_col (str, optional):
            Name of a column according to whose values individual bricks will be
            color coded.
        lower_bound_col (str, optional):
            Name of the column identifying the lower bound of the whisker.
        upper_bound_col (str, optional):
            Name of the column identifying the upper bound of the whisker.
        figure_height (int, optional):
            height of the figure (i.e. of all plots together, in pixels).
        width (int, optional):
            width of the figure (in pixels).
        x_padding (float, optional):
            the x_range is extended on each side by this factor of the range of the data
        num_bins (int, optional):
            number of bins

    """
    df, group_cols, plot_height = process_inputs(
        source=source,
        value_col=value_col,
        id_col=id_col,
        group_cols=group_cols,
        subgroup_col=subgroup_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        figure_height=figure_height,
        x_padding=x_padding,
        num_bins=num_bins,
    )

    grid = _create_grid(
        df=df, group_cols=group_cols, plot_height=plot_height, width=width
    )
    doc.add_root(grid)

    source, plots = _plot_bricks(
        doc=doc,
        df=df,
        value_col=value_col,
        id_col=id_col,
        group_cols=group_cols,
        subgroup_col=subgroup_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
    )

    return source, plots


def _create_grid(df, group_cols, plot_height, width):
    """Create the empty grid to which the contributions or parameters will be plotted.

    Args:
        doc (bokeh.Document): document to which to add the plots
        group_cols (list):
            Name of the columns that identify groups that will be plotted together.
            In case of a parameter comparison plot this would be the parameter group
            and parameter name by default.
        plot_height (int): height of the plots in pixels
        width (int): width of the plots in pixels

    """
    plots = [
        RangeSlider(start=0, end=1, value=(0, 1), name="placeholder_value_slider"),
        RangeSlider(start=0, end=1, value=(0, 1), name="placeholder_subgroup_widget"),
    ]
    if len(group_cols) == 0:
        fig = figure(
            title="",
            plot_height=plot_height,
            plot_width=width,
            tools="reset,save",
            y_axis_location="left",
            name="all",
        )
        plots.append(fig)
    else:
        gb = df.groupby(group_cols)
        old_group_tup = tuple(None for name in group_cols)
        for tup, df_slice in gb:
            plots = _add_titles_if_group_switch(
                plots=plots,
                group_cols=group_cols,
                old_group_tup=old_group_tup,
                group_tup=tup,
            )
            old_group_tup = tup
            name = " ".join(str(x) for x in tup)
            fig = figure(
                title="{} {}".format(group_cols[-1].title(), str(tup[-1]).title()),
                plot_height=plot_height,
                plot_width=width,
                tools="reset,save",
                y_axis_location="left",
                x_range=(df_slice["xmin"].min(), df_slice["xmax"].max()),
                name=name,
            )
            fig = _style_plot(fig)
            plots.append(fig)
    grid = Column(*plots)
    return grid


def _add_titles_if_group_switch(plots, group_cols, old_group_tup, group_tup):
    for level in range(len(group_cols) - 1):
        old_name = old_group_tup[level]
        new_name = group_tup[level]
        if old_name != new_name:
            title = "{} {}".format(group_cols[level].title(), str(new_name).title())
            text_size = min(7, 28 - 4 * level)
            plots.append(Div(text=title, name=title, default_size=text_size))
    return plots


def _plot_bricks(
    doc,
    df,
    value_col,
    id_col,
    group_cols,
    subgroup_col,
    lower_bound_col,
    upper_bound_col,
):
    """Create the ColumnDataSource and replace the plots and widgets.

    Args:
        doc (bokeh.Document): document to which to add the plot.
        df (pd.DataFrame): Tidy DataFrame.
        value_col (str):
            Name of the column for which to draw the histogram.
            In case of a parameter comparison plot this would be the "value" columns
            of the params DataFrames returned by maximize or minimize.
        id_col (str):
            Name of the column that identifies
            which values belong to the same observation.
            In case of a parameter comparison plot
            this would be the "model_name" column.
        group_cols (list):
            Name of the columns that identify groups that will be plotted together.
            In case of a parameter comparison plot this would be the parameter group
            and parameter name by default.
        subgroup_col (str):
            Name of a column according to whose values individual bricks will be
            color coded.
        lower_bound_col (str):
            Name of the column identifying the lower bound of the whisker.
        upper_bound_col (str):
            Name of the column identifying the upper bound of the whisker.

    """
    all_elements = doc.roots[0].children
    source = ColumnDataSource(df)
    widget = create_group_widget(source=source, subgroup_col=subgroup_col)
    all_elements[1] = widget

    plots = []
    tuples = _create_tuples(df=df, group_cols=group_cols)
    for tup in tuples:
        fig = doc.get_model_by_name(" ".join(str(x) for x in tup))
        fig.renderers = []
        fig.tools = []
        group_index = df[(df[group_cols] == tup).all(axis=1)].index
        view = create_view(
            source=source,
            group_index=group_index,
            subgroup_col=subgroup_col,
            widget=widget,
        )
        fig = _add_renderers(
            fig=fig,
            source=source,
            view=view,
            id_col=id_col,
            group_cols=group_cols,
            lower_bound_col=lower_bound_col,
            upper_bound_col=upper_bound_col,
        )
        plots.append(fig)

    # this has to happen at he end because all plots must be passed to this
    all_elements[0] = value_slider(
        source=source,
        value_col=value_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        plots=plots,
    )

    return source, all_elements


def _create_tuples(df, group_cols):
    if len(group_cols) == 0:
        tuples = ["all"]
    elif len(group_cols) == 1:
        tuples = df[group_cols[0]].unique().tolist()
    else:
        tuples = list(set(zip(*[df[col] for col in group_cols])))
    return tuples


def _add_renderers(
    fig, source, view, id_col, group_cols, lower_bound_col, upper_bound_col
):
    point_glyph = fig.rect(
        source=source,
        view=view,
        x="binned_x",
        width="rect_width",
        y="dodge",
        height=1,
        color="color",
        selection_color="color",
        nonselection_color="color",
        alpha=0.5,
        selection_alpha=0.7,
        nonselection_alpha=0.1,
    )

    if lower_bound_col is not None and upper_bound_col is not None:
        fig.hbar(
            source=source,
            view=view,
            y="dodge",
            left=lower_bound_col,
            right=upper_bound_col,
            height=0.01,
            alpha=0.0,
            selection_alpha=0.7,
            nonselection_alpha=0.0,
            color="color",
            selection_color="color",
            nonselection_color="color",
        )
    fig = add_hover_tool(
        fig=fig, point_glyph=point_glyph, source=source, group_cols=group_cols
    )
    fig = add_select_tools(
        fig=fig, point_glyph=point_glyph, source=source, id_col=id_col
    )
    return fig


def _style_plot(fig):
    fig.xaxis.minor_tick_line_color = None
    fig.xaxis.axis_line_color = None
    fig.xaxis.major_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None
    fig.yaxis.axis_line_color = None
    fig.yaxis.major_tick_line_color = None

    fig.title.vertical_align = "top"
    fig.title.text_alpha = 70
    fig.title.text_font_style = "normal"
    fig.outline_line_color = None
    fig.min_border_top = 20
    fig.min_border_bottom = 20
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.sizing_mode = "scale_width"

    return fig
