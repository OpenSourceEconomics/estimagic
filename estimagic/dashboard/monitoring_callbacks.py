"""Callbacks for the monitoring app."""
from functools import partial

from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.database_utilities import transpose_nested_list


def logscale_callback(attr, old, new, button, doc):
    """Switch between log and linear scale.

    Args:
        attr: Required by bokeh.
        old: Old state of the Button.
        new: New state of the Button.
        button (bokeh.models.Toggle)
        doc (bokeh.Document)

    """
    linear_criterion_plot = doc.get_model_by_name("linear_criterion_plot")
    log_criterion_plot = doc.get_model_by_name("log_criterion_plot")
    if new is True:
        _switch_to_log_scale(button, linear_criterion_plot, log_criterion_plot)
    else:
        _switch_to_linear_scale(button, linear_criterion_plot, log_criterion_plot)


def _switch_to_log_scale(button, linear_criterion_plot, log_criterion_plot):
    """Make the linear criterion plot invisible, the log plot visible and adjust button.

    Args:
        button (bokeh.Toggle)
        linear_criterion_plot (bokeh.Figure)
        log_criterion_plot (bokeh.Figure)

    """
    button.button_type = "primary"
    button.label = "Show criterion plot on a linear scale"
    linear_criterion_plot.visible = False
    log_criterion_plot.visible = True


def _switch_to_linear_scale(button, linear_criterion_plot, log_criterion_plot):
    """Make the log criterion plot invisible, the linear plot visible and adjust button.

    Args:
        button (bokeh.Toggle)
        linear_criterion_plot (bokeh.Figure)
        log_criterion_plot (bokeh.Figure)

    """
    button.button_type = "default"
    button.label = "Show criterion plot on a logarithmic scale"
    log_criterion_plot.visible = False
    linear_criterion_plot.visible = True


def activation_callback(
    attr,
    old,
    new,
    session_data,
    rollover,
    doc,
    database,
    button,
    tables,
    start_params,
    update_frequency,
    update_chunk,
):
    """Start and reset the convergence plots and their updating.

    Args:
        attr: Required by bokeh.
        old: Old state of the Button.
        new: New state of the Button.

        doc (bokeh.Document)
        database (sqlalchemy.MetaData)
        session_data (dict): This app's entry of infos to be passed between and within
            apps. The keys are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        rollover (int): Maximal number of points to show in the plot.
        button (bokeh.models.Toggle)
        tables (list): List of table names to load and convert to ColumnDataSources.
        start_params (pd.DataFrame)
        update_frequency (float): Number of seconds to wait between updates.
        update_chunk (int): Number of values to add at each update.

    """
    callback_dict = session_data["callbacks"]
    criterion_cds = doc.get_model_by_name("criterion_history_cds")
    param_cds = doc.get_model_by_name("params_history_cds")

    if new is True:
        plot_new_data = partial(
            _update_monitoring_tab,
            criterion_cds=criterion_cds,
            param_cds=param_cds,
            database=database,
            session_data=session_data,
            rollover=rollover,
            tables=tables,
            start_params=start_params,
            update_chunk=update_chunk,
        )
        callback_dict["plot_periodic_data"] = doc.add_periodic_callback(
            plot_new_data, period_milliseconds=1000 * update_frequency,
        )

        # change the button color
        button.button_type = "success"
        button.label = "Reset Plot"
    else:
        doc.remove_periodic_callback(callback_dict["plot_periodic_data"])
        _reset_column_data_sources([criterion_cds, param_cds])
        session_data["last_retrieved"] = 0

        # change the button color
        button.button_type = "danger"
        button.label = "Restart Plot"


def _update_monitoring_tab(
    database,
    criterion_cds,
    param_cds,
    session_data,
    tables,
    rollover,
    start_params,
    update_chunk,
):
    """Callback to look up new entries in the database tables and plot them.

    Args:
        database (sqlalchemy.MetaData)
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        tables (list): list of table names to load and convert to ColumnDataSources
        rollover (int): maximal number of points to show in the plot
        start_params (pd.DataFrame)
        update_chunk (int): Number of values to add at each update.
        criterion_cds (bokeh.ColumnDataSource)
        param_cds (bokeh.ColumnDataSource)

    """
    data, new_last = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=session_data["last_retrieved"],
        return_type="dict_of_lists",
        limit=update_chunk,
    )

    # update the criterion plot
    # todo: remove None entries!
    missing = [i for i, val in enumerate(data["value"]) if val is None]
    crit_data = {
        "iteration": [id_ for i, id_ in enumerate(data["rowid"]) if i not in missing],
        "criterion": [val for i, val in enumerate(data["value"]) if i not in missing],
    }
    criterion_cds.stream(crit_data, rollover=rollover)

    # update the parameter plots
    # Note: we need **all** parameter ids to correctly map them to the parameter entries
    # in the database. Only after can we restrict them to the entries we need.
    param_ids = start_params["id"].tolist()
    params_data = _create_params_data_for_update(data, param_ids)
    available_keys = param_cds.data.keys()
    to_stream = {k: v for k, v in params_data.items() if k in available_keys}
    param_cds.stream(to_stream, rollover=rollover)

    # update last retrieved
    session_data["last_retrieved"] = new_last


def _create_params_data_for_update(data, param_ids):
    """Create the dictionary to stream to the param_cds from data and param_ids.

    Args:
        data
        param_ids (list): list of the length of the arrays in data["external_params"]

    Returns:
        params_data (dict): keys are the parameter names and "iteration". The values
            are lists of values that will be added to the ColumnDataSources columns.

    """
    params_data = [arr.tolist() for arr in data["external_params"]]
    params_data = transpose_nested_list(params_data)
    params_data = dict(zip(param_ids, params_data))
    if params_data == {}:
        params_data = {name: [] for name in param_ids}
    params_data["iteration"] = data["rowid"]
    return params_data


def _reset_column_data_sources(cds_list):
    """Empty each ColumnDataSource in a list such that it has no entries.

    Args:
        cds_list (list): list of boheh ColumnDataSources
    """
    for cds in cds_list:
        column_names = cds.data.keys()
        cds.data = {name: [] for name in column_names}
