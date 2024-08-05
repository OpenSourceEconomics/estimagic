import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from optimagic.shared.compat import pd_df_map

suppress_performance_warnings = np.testing.suppress_warnings()
suppress_performance_warnings.filter(category=pd.errors.PerformanceWarning)


@suppress_performance_warnings
def estimation_table(
    models,
    *,
    return_type="dataframe",
    render_options=None,
    show_col_names=True,
    show_col_groups=None,
    show_index_names=False,
    show_inference=True,
    show_stars=True,
    show_footer=True,
    custom_param_names=None,
    custom_col_names=None,
    custom_col_groups=None,
    custom_index_names=None,
    custom_notes=None,
    confidence_intervals=False,
    significance_levels=(0.1, 0.05, 0.01),
    append_notes=True,
    notes_label="Note:",
    stats_options=None,
    number_format=("{0:.3g}", "{0:.5f}", "{0:.4g}"),
    add_trailing_zeros=True,
    escape_special_characters=True,
    siunitx_warning=True,
):
    r"""Generate html or LaTex tables provided (lists of) of models.

    The function can create publication quality tables in various formats from
    statsmodels or estimagic results.

    It allows for extensive customization via optional arguments and almost limitless
    flexibility when using a two-stage approach where the ``return_type`` is set to
    ``"render_inputs"``, the resulting dictionary representation of the table is
    modified and that modified version is then passed to ``render_latex`` or
    ``render_html``.

    The formatting of the numbers in the table is completely configurable via the
    ``number_format`` argument. By default we round to three significant digits (i.e.
    the three leftmost non-zero digits are displayed). This is very different from
    other table packages and motivated by the fact that most estimation tables give
    a wrong feeling of precision by showing too many decimal points.

    Args:
        models (list): list of estimation results. The models can come from
            statmodels or be constructed from the outputs of `estimagic.estimate_ml`
            or `estimagic.estimate_msm`. With a little bit of work it is also possible
            to construct them out of R or other results. If a model is not a
            statsmodels results they must be dictionaries with the following entries:
            "params" (a DataFrame with value column), "info" (a dictionary with summary
            statistics such as "n_obs", "rsquared", ...) and "name" (a string), or a
            DataFrame with value column. If a models is a statsmodels result,
            model.endog_names is used as name and the rest is extracted from
            corresponding statsmodels attributes. The model names do not have to be
            unique but if they are not, models with the same name need to be grouped
            together.
        return_type (str): Can be "dataframe", "latex", "html", "render_inputs" or a
            file path with the extension .tex or .html. If "render_inputs" is passed,
            a dictionary with the entries "body", "footer" and other
            information is returned. The entries can be modified by the user (
            e.g. change formatting, renameof columns or index, ...) and then passed
            to ``render_latex`` or ``render_html``. Default "dataframe".
        render_options (dict): a dictionary with keyword arguments that are passed to
            df.style.to_latex or df.style.to_html, depending on the return_type.
            The default is None.
        show_col_names (bool): If True, the column names are displayed. The default
            column names are the model names if the model names are unique, otherwise
            (1), (2), etc.. Default True.
        show_col_groups (bool): If True, the column groups are displayed. The default
            column groups are the model names if the model names are not unique and
            undefined otherwise. Default None. None means that the column groups are
            displayed if they are defined.
        show_index_names (bool): If True, the index names are displayed. Default False.
            This is mostly relevant when working with estimagic style params DataFrames
            with a MultiIndex.
        show_inference(bool): If True, inference (standard errors or confidence
            intervals) are displayed below parameter values. Default True.
        show_stars (bool): a boolean variable for displaying significance stars.
            Default is True.
        show_footer (bool): a boolean variable for displaying statistics, e.g. R2,
            Obs numbers. Default is True. Which statistics are displayed and how they
            are labeled can be determined via ``stats_options``.
        custom_param_names (dict): Dictionary that is used to rename parameters. The
            keys are the old parameter names or index entries. The values are
            the new names. Default None.
        custom_col_names (dict or list): A list of column names or dict to rename the
            default column names. The default column names are the model names if the
            model names are unique, otherwise (1), (2), etc..
        custom_col_groups (dict or list): A list of column group or dict to rename
            the default column groups. The default column groups are the model names
            if the model names are not unique and undefined otherwise.
        custom_index_names (dict or list): Dictionary or list to set the names of the
            index levels of the parameters. This is mostly relevant when working with
            estimagic style params DataFrames with a MultiIndex and only used if
            "index_names" is set to True in the render_options. Default None.
        custom_notes (list): A list of strings for additional notes. Default is None.
        confidence_intervals (bool): If True, display confidence intervals as inference
            values. Display standard errors otherwise. Default False.
        significance_levels (list): a list of floats for p value's significance cut-off
            values. This is used to generate the significance stars. Default is
            [0.1,0.05,0.01].
        append_notes (bool): A boolean variable for printing p value cutoff explanation
            and additional notes, if applicable. Default is True.
        notes_label (str): A sting to print as the title of the notes section, if
            applicable. Default is 'Notes'
        stats_options (dict): A dictionary that determines which statistics (e.g.
            R-Squared, No. of Observations) are displayed and how they are labeled.
            The keys are the names of the statistics inside the model['info'] dictionary
            or attribute names of a statsmodels results object. The values are the new
            labels to be displayed for those statistics, i.e. the set of the values is
            used as row names in the table.
        number_format (int, str, iterable or callable): A callable, iterable, integer
            or string that is used to apply string formatter(s) to floats in the
            table. Default ("{0:.3g}", "{0:.5f}", "{0:.4g}").
        add_trailing_zeros (bool): If True, format floats such that they have same
            number of digits after the decimal point. Default True.
        siunitx_warning (bool): If True, print warning about LaTex preamble to add for
            proper compilation of  when working with siunitx package. Default True.
        escape_special_characters (bool): If True, replaces special characters
            in parameter and model names with LaTeX or HTML safe sequences.
    Returns:
        res_table (data frame, str or dictionary): depending on the rerturn type,
            data frame with formatted strings, a string for html or latex tables,
            or a dictionary with statistics and parameters dataframes, and strings
            for footers is returned. If the return type is a path, the function saves
            the resulting table at the given path.

    """
    if not isinstance(models, (tuple, list)):
        raise TypeError(f"models must be a list or tuple. Not: {type(models)}")
    models = [_process_model(model) for model in models]
    model_names = _get_model_names(models)
    default_col_names, default_col_groups = _get_default_column_names_and_groups(
        model_names
    )
    column_groups = _customize_col_groups(
        default_col_groups=default_col_groups, custom_col_groups=custom_col_groups
    )
    column_names = _customize_col_names(
        default_col_names=default_col_names, custom_col_names=custom_col_names
    )
    show_col_groups = _update_show_col_groups(show_col_groups, column_groups)
    stats_options = _set_default_stats_options(stats_options)
    body, footer = _get_estimation_table_body_and_footer(
        models,
        column_names,
        column_groups,
        custom_param_names,
        custom_index_names,
        significance_levels,
        stats_options,
        show_col_names,
        show_col_groups,
        show_stars,
        show_inference,
        confidence_intervals,
        number_format,
        add_trailing_zeros,
    )

    render_inputs = {
        "body": body,
        "footer": footer,
        "render_options": render_options,
    }
    if return_type == "render_inputs":
        out = render_inputs
    elif str(return_type).endswith("tex"):
        out = _render_latex(
            **render_inputs,
            show_footer=show_footer,
            append_notes=append_notes,
            notes_label=notes_label,
            significance_levels=significance_levels,
            custom_notes=custom_notes,
            siunitx_warning=siunitx_warning,
            show_index_names=show_index_names,
            show_col_names=show_col_names,
            escape_special_characters=escape_special_characters,
        )
    elif str(return_type).endswith("html"):
        out = render_html(
            **render_inputs,
            show_footer=show_footer,
            append_notes=append_notes,
            notes_label=notes_label,
            custom_notes=custom_notes,
            significance_levels=significance_levels,
            show_index_names=show_index_names,
            show_col_names=show_col_names,
            escape_special_characters=escape_special_characters,
        )

    elif return_type == "dataframe":
        if show_footer:
            footer.index.names = body.index.names
            out = pd.concat([body.reset_index(), footer.reset_index()]).set_index(
                body.index.names
            )
        else:
            out = body
    else:
        raise ValueError(
            f"""Value of return type can be either of
            ['data_frame', 'render_inputs','latex' ,'html']
            or a path ending with '.html' or '.tex'. Not: {return_type}."""
        )

    return_type = Path(return_type)
    if return_type.suffix not in (".html", ".tex"):
        return out
    else:
        return_type.write_text(out)


@suppress_performance_warnings
def render_latex(
    body,
    footer,
    render_options=None,
    show_footer=True,
    append_notes=True,
    notes_label="Note:",
    significance_levels=(0.1, 0.05, 0.01),
    custom_notes=None,
    siunitx_warning=True,
    show_index_names=False,
    show_col_names=True,
    show_col_groups=True,
    escape_special_characters=True,
):
    r"""Return estimation table in LaTeX format as string.

    Args:
        body (pandas.DataFrame): DataFrame with formatted strings of parameter
            values, inferences (standard errors or confidence intervals, if
            applicable) and significance stars (if applicable).
        footer (pandas.DataFrame): DataFrame with formatted strings of summary
            statistics (such as number of observations, r-squared, etc.)
        render_options(dict): A dictionary with custom kwargs to pass to
            pd.Styler.to_latex(), to update the default options. An example keyword
            argument is:
            - siunitx (bool): If True, the table is structured to be compatible
            with siunitx package. Default is set to True internally.
            For the list of all possible arguments, see documentation of
            `pandas.io.formats.style.Styler.to_latex`.
        show_footer (bool): a boolean variable for displaying footer_df. Default True.
        append_notes (bool): A boolean variable for printing p value cutoff explanation
            and additional notes, if applicable. Default is True.
        notes_label (str): A sting to print as the title of the notes section, if
            applicable. Default is 'Notes'
        significance_levels (list or tuple): a list of floats for p value's significance
            cutt-off values. Default is [0.1,0.05,0.01].
        custom_notes (list): A list of strings for additional notes. Default is None.
        siunitx_warning (bool): If True, print warning about LaTex preamble to add for
            proper compilation of  when working with siunitx package. Default True.
        show_index_names (bool): If True, display index names in the table.
        show_col_names (bool): If True, the column names are displayed.
        show_col_groups (bool): If True, the column groups are displayed.
        escape_special_characters (bool): If True, replaces the characters &, %,
            $, #, _, {, }, ~, ^, and \ in parameter and model names with
            LaTeX-safe sequences.

    Returns:
        latex_str (str): The resulting string with Latex tabular code.

    """
    return _render_latex(
        body=body,
        footer=footer,
        render_options=render_options,
        show_footer=show_footer,
        append_notes=append_notes,
        notes_label=notes_label,
        significance_levels=significance_levels,
        custom_notes=custom_notes,
        siunitx_warning=siunitx_warning,
        show_index_names=show_index_names,
        show_col_names=show_col_names,
        show_col_groups=show_col_groups,
        escape_special_characters=escape_special_characters,
    )


def _render_latex(
    body,
    footer,
    render_options=None,
    show_footer=True,
    append_notes=True,
    notes_label="Note:",
    significance_levels=(0.1, 0.05, 0.01),
    custom_notes=None,
    siunitx_warning=True,
    show_index_names=False,
    show_col_names=True,
    show_col_groups=True,
    escape_special_characters=True,
):
    """See docstring of render_latex for more information."""
    if not pd.__version__ >= "1.4.0":
        raise ValueError(
            r"""render_latex or estimation_table with return_type="latex" requires
            pandas 1.4.0 or higher. Update to a newer version of pandas or use
            estimation_table with return_type="render_inputs" and manually render those
            results using the DataFrame.to_latex method.
        """
        )
    if siunitx_warning:
        warn(
            r"""Proper LaTeX compilation requires the package siunitx and adding
                   \sisetup{
                       input-symbols            = (),
                       table-align-text-post    = false,
                       group-digits             = false,
                    }
                    to your main tex file. To turn
                    this warning off set value of siunitx_warning = False"""
        )
    body = body.copy(deep=True)
    try:
        ci_in_body = body.loc[("",)][body.columns[0]].str.contains(";").any()
    except KeyError:
        ci_in_body = False

    if ci_in_body:
        body.loc[("",)] = pd_df_map(body.loc[("",)], "{{{}}}".format).values
    if body.columns.nlevels > 1:
        column_groups = body.columns.get_level_values(0)
    else:
        column_groups = None

    group_to_col_position = _create_group_to_col_position(column_groups)
    n_levels = body.index.nlevels
    n_columns = len(body.columns)

    if escape_special_characters:
        escape_special_characters = "latex"
    else:
        escape_special_characters = None
    body_styler = _get_updated_styler(
        body,
        show_index_names=show_index_names,
        show_col_names=show_col_names,
        show_col_groups=show_col_groups,
        escape_special_characters=escape_special_characters,
    )
    default_options = {
        "multicol_align": "c",
        "hrules": True,
        "siunitx": True,
        "column_format": "l" * n_levels + "S" * n_columns,
        "multirow_align": "t",
    }
    if render_options:
        default_options.update(render_options)
    latex_str = body_styler.to_latex(**default_options)

    if group_to_col_position:
        temp_str = "\n"
        for k in group_to_col_position:
            max_col = max(group_to_col_position[k]) + n_levels + 1
            min_col = min(group_to_col_position[k]) + n_levels + 1
            temp_str += f"\\cmidrule(lr){{{min_col}-{max_col}}}"
            temp_str += "\n"
        latex_str = (
            latex_str.split("\\\\", 1)[0]
            + "\\\\"
            + temp_str
            + latex_str.split("\\\\", 1)[1]
        )
    latex_str = latex_str.split("\\bottomrule")[0]
    if show_footer:
        footer = footer.copy(deep=True)
        footer = footer.apply(_center_align_integers_and_non_numeric_strings, axis=1)
        footer_styler = footer.style
        stats_str = footer_styler.to_latex(**default_options)
        if "\\midrule" in stats_str:
            stats_str = (
                "\\midrule" + stats_str.split("\\midrule")[1].split("\\bottomrule")[0]
            )
        else:
            stats_str = (
                "\\midrule" + stats_str.split("\\toprule")[1].split("\\bottomrule")[0]
            )
        latex_str += stats_str
    notes = _generate_notes_latex(
        append_notes, notes_label, significance_levels, custom_notes, body
    )
    latex_str += notes
    latex_str += "\\bottomrule\n\\end{tabular}\n"
    if latex_str.startswith("\\begin{table}"):
        latex_str += "\n\\end{table}\n"
    return latex_str


def render_html(
    body,
    footer,
    render_options=None,
    show_footer=True,
    append_notes=True,
    notes_label="Note:",
    custom_notes=None,
    significance_levels=(0.1, 0.05, 0.01),
    show_index_names=False,
    show_col_names=True,
    show_col_groups=True,
    escape_special_characters=True,
    **kwargs,  # noqa: ARG001
):
    """Return estimation table in html format as string.

    Args:
        body (pandas.DataFrame): DataFrame with formatted strings of parameter
            values, inferences (standard errors or confidence intervals, if
            applicable) and significance stars (if applicable).
        footer (pandas.DataFrame): DataFrame with formatted strings of summary
            statistics (such as number of observations, r-squared, etc.)
        notes (str): The html string with notes with additional information
            (e.g. mapping from pvalues to significance stars) to append to the footer
            of the estimation table string with LaTex code for the notes section.
        render_options(dict): A dictionary with custom kwargs to pass to pd.to_latex(),
            to update the default options. An example is `{header: False}` that
            disables displaying column names.
        show_footer (bool): a boolean variable for displaying footer_df. Default True.
        append_notes (bool): A boolean variable for printing p value cutoff explanation
            and additional notes, if applicable. Default is True.
        notes_label (str): A sting to print as the title of the notes section, if
            applicable. Default is 'Notes'
        significance_levels (list or tuple): a list of floats for p value's significance
            cutt-off values. Default is [0.1,0.05,0.01].
        show_index_names (bool): If True, display index names in the table.
        show_col_names (bool): If True, the column names are displayed.
        show_col_groups (bool): If True, the column groups are displayed.
        escape_special_characters (bool): If True,  replace the characters &, <, >, ',
            and " in parameter and model names with HTML-safe sequences.

    Returns:
        html_str (str): The resulting string with html tabular code.

    """
    if not pd.__version__ >= "1.4.0":
        raise ValueError(
            r"""render_html or estimation_table with return_type="html" requires
            pandas 1.4.0 or higher. Update to a newer version of pandas or use
            estimation_table with return_type="render_inputs" and manually render those
            results using the DataFrame.to_html method.
        """
        )
    n_levels = body.index.nlevels
    n_columns = len(body.columns)
    html_str = ""
    if escape_special_characters:
        escape_special_characters = "html"
    else:
        escape_special_characters = None
    body_styler = _get_updated_styler(
        body,
        show_index_names=show_index_names,
        show_col_names=show_col_names,
        show_col_groups=show_col_groups,
        escape_special_characters=escape_special_characters,
    )
    default_options = {"exclude_styles": True}
    if render_options:
        default_options.update(render_options)
    html_str = body_styler.to_html(**default_options).split("</tbody>\n</table>")[0]
    if show_footer:
        stats_str = """<tr><td colspan="{}" style="border-bottom: 1px solid black">
            </td></tr>""".format(n_levels + n_columns)
        stats_str += (
            footer.style.to_html(**default_options)
            .split("</thead>\n")[1]
            .split("</tbody>\n</table>")[0]
        )
        stats_str = re.sub(r"(?<=[\d)}{)])}", "", re.sub(r"{(?=[}\d(])", "", stats_str))
        html_str += stats_str
    notes = _generate_notes_html(
        append_notes, notes_label, significance_levels, custom_notes, body
    )
    html_str += notes
    html_str += "</tbody>\n</table>"
    return html_str


def _process_model(model):
    """Check model validity, convert to dictionary.

    Args
        model: Estimation result. See docstring of estimation_table for more info.
    Returns:
        processed_model: A dictionary with keys params, info and name.

    """
    if isinstance(model, dict):
        params = model["params"].copy(deep=True)
        info = model.get("info", {})
        name = model.get("name", "")
    elif isinstance(model, pd.DataFrame):
        params = model.copy(deep=True)
        info = {}
        name = None
    else:
        try:
            params = _extract_params_from_sm(model)
            info = {**_extract_info_from_sm(model)}
            name = info.pop("name")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise TypeError(
                f"""Model can  be of type dict,  pd.DataFrame
                or a statsmodels result. Model {model} is of type {type(model)}."""
            ) from e
    if "pvalue" in params.columns:
        params = params.rename(columns={"pvalue": "p_value"})
    processed_model = {"params": params, "info": info, "name": name}
    return processed_model


def _get_estimation_table_body_and_footer(
    models,
    column_names,
    column_groups,
    custom_param_names,
    custom_index_names,
    significance_levels,
    stats_options,
    show_col_names,
    show_col_groups,
    show_stars,
    show_inference,
    confidence_intervals,
    number_format,
    add_trailing_zeros,
):
    """Create body and footer blocs with significance stars and inference values.

    Applies number formatting to parameters and summary statitistics.
    Concatinates infere values to parameter values if applicable,
    Adds significance stars if applicable.

    Args:
        models (list): List of dictionaries with keys 'params', 'info' and 'name'.
        column_names (list): List of strigs to display as names of the model columns in
            estimation table.
        column_groups (list or NoneType): If defined, list of strings to display as
            names of groups of model columns in estimation table.
        custom_param_names (dict or list): A list of strings to display as parameter
            names or a mapping from original to custom paramter names.
        custom_index_names (dict or list): Dictionary or list to set the names of the
            index levels of the parameters.
        significance_levels (list): a list of floats for p value's significance
            cutt-off values.
        stats_options (dict): A dictionary with displayed statistics names as keys,
            and statistics names to be retrieved from model['info'] as values
        show_col_names (bool): If True, the column names are displayed.
        show_col_groups (bool): If True, the column groups are displayed.
        show_stars (bool): a boolean variable for printing significance stars.
        show_inference(bool): If True, inference (standard errors or confidence
            intervals) below param values.
        confidence_intervals (bool): If True, display confidence intervals as inference
            values.
        number_format (int, str, iterable or callable): A callable, iterable, integer
            or callable that is used to apply string formatter(s) to floats in the
            table.
        add_trailing_zeros (bool): If True, format floats such that they have same
            number of digits after the decimal point.

    Returns:
        body (DataFrame): DataFrame data frame with formatted strings of parameter
            and inference values and significance stars to display in estimation table.
        footer (DataFrame): DataFrame with formatted strings of summary statistics to
            display at the bottom of estimation table.

    """
    body, max_trail = _build_estimation_table_body(
        models,
        column_names,
        column_groups,
        custom_param_names,
        custom_index_names,
        show_col_names,
        show_col_groups,
        show_inference,
        show_stars,
        confidence_intervals,
        significance_levels,
        number_format,
        add_trailing_zeros,
    )
    footer = _build_estimation_table_footer(
        models,
        stats_options,
        significance_levels,
        show_stars,
        number_format,
        add_trailing_zeros,
        max_trail,
    )
    footer.columns = body.columns
    return body, footer


def _build_estimation_table_body(
    models,
    column_names,
    column_groups,
    custom_param_names,
    custom_index_names,
    show_col_names,
    show_col_groups,
    show_inference,
    show_stars,
    confidence_intervals,
    significance_levels,
    number_format,
    add_trailing_zeros,
):
    """Create body bloc significance stars and inference values.

    Applies number formatting to parameters. Concatinates inference values
    to parameter values if applicable. Adds significance stars if applicable.

    Args:
        models (list): List of dictionaries with keys 'params', 'info' and 'name'.
        column_names (list): List of strigs to display as names of the model columns in
            estimation table.
        column_groups (list or NoneType): If defined, list of strings to display as
            names of groups of model columns in estimation table.
        custom_param_names (dict or list): A list of strings to display as parameter
            names or a mapping from original to custom paramter names.
        custom_index_names (dict or list): Dictionary or list to set the names of the
            index levels of the parameters.
        significance_levels (list): a list of floats for p value's significance
            cutt-off values.
        show_col_names (bool): If True, the column names are displayed.
        show_col_groups (bool): If True, the column groups are displayed.
        show_stars (bool): a boolean variable for printing significance stars.
        show_inference(bool): If True, inference (standard errors or confidence
            intervals) below param values.
        confidence_intervals (bool): If True, display confidence intervals as inference
            values.
        number_format (int, str, iterable or callable): A callable, iterable, integer
            or callable that is used to apply string formatter(s) to floats in the
            table.
        add_trailing_zeros (bool): If True, format floats such that they have same
            number of digits after the decimal point.

    Returns:
        body (DataFrame): DataFrame data frame with formatted strings of parameter
            and inference values and significance stars to display in estimation table.
        max_trail (int): Integer that shows the maximum number of digits after a decimal
            point in the parameters DataFrame. Is passed to
            `_build_estimation_table_footer` to get same number of trailing zeros as in
            parameters DataFrame and torender_latex for formatting tables in siunitx
            package.

    """
    dfs, max_trail = _reindex_and_float_format_params(
        models, show_inference, confidence_intervals, number_format, add_trailing_zeros
    )
    to_convert = []
    if show_stars:
        for df, mod in zip(dfs, models, strict=False):
            to_convert.append(
                pd.concat([df, mod["params"].reindex(df.index)["p_value"]], axis=1)
            )
    else:
        to_convert = dfs
    # convert DataFrames to string series with inference and siginificance
    # information.
    to_concat = [
        _convert_frame_to_string_series(
            df,
            significance_levels,
            show_stars,
        )
        for df in to_convert
    ]
    df = pd.concat(to_concat, axis=1)
    df = _process_frame_indices(
        df=df,
        custom_param_names=custom_param_names,
        custom_index_names=custom_index_names,
        show_col_names=show_col_names,
        show_col_groups=show_col_groups,
        column_names=column_names,
        column_groups=column_groups,
    )
    return df, max_trail


def _build_estimation_table_footer(
    models,
    stats_options,
    significance_levels,
    show_stars,
    number_format,
    add_trailing_zeros,
    max_trail,
):
    """Create footer bloc of estimation table.

    Applies number formatting to parameters and summary statitistics.
    Concatinates infere values to parameter values if applicable,
    Adds significance stars if applicable.

    Args:
        models (list): List of dictionaries with keys 'params', 'info' and 'name'.
        stats_options (dict): A dictionary with displayed statistics names as keys,
            and statistics names to be retrieved from model['info'] as values
        significance_levels (list): a list of floats for p value's significance cutt-off
            values.
        number_format (int, str, iterable or callable): A callable, iterable, integer
            or callable that is used to apply string formatter(s) to floats in the
            table.
        add_trailing_zeros (bool): If True, format floats such that they haave same
            number of digits after the decimal point.
        max_trail (int): If add_trailing_zeros is True, add corresponding number of
            trailing zeros to floats in the stats DataFrame to have number of digits
            after a decimal point equal to max_trail for each float.

    Returns:
        footer (DataFrame): DataFrame with formatted strings of summary statistics to
            display at the bottom of estimation table.

    """
    to_concat = [
        _create_statistics_sr(
            mod,
            stats_options,
            significance_levels,
            show_stars,
            number_format,
            add_trailing_zeros,
            max_trail,
        )
        for mod in models
    ]
    stats = pd.concat(to_concat, axis=1)
    return stats


def _reindex_and_float_format_params(
    models, show_inference, confidence_intervals, number_format, add_trailing_zeros
):
    """Reindex all params DataFrames with a common index and apply number formatting."""
    dfs = _get_params_frames_with_common_index(models)
    cols_to_format = _get_cols_to_format(show_inference, confidence_intervals)
    formatted_frames, max_trail = _apply_number_formatting_frames(
        dfs, cols_to_format, number_format, add_trailing_zeros
    )
    return formatted_frames, max_trail


def _get_params_frames_with_common_index(models):
    """Get a list of params frames, reindexed with a common index."""
    dfs = [model["params"] for model in models]
    common_index = _get_common_index(dfs)
    out = [model["params"].reindex(common_index) for model in models]
    return out


def _get_common_index(dfs):
    """Get common index from a list of DataFrames."""
    common_index = []
    for d_ in dfs:
        common_index += [ind for ind in d_.index.to_list() if ind not in common_index]
    return common_index


def _get_cols_to_format(show_inference, confidence_intervals):
    """Get the list of names of columns that need to be formatted.

    By default, formatting is applied to  parameter values. If inference values need to
    displayed, adds confidence intervals or standard erros to the list.

    """
    cols = ["value"]
    if show_inference:
        if confidence_intervals:
            cols += ["ci_lower", "ci_upper"]
        else:
            cols.append("standard_error")
    return cols


def _apply_number_formatting_frames(dfs, columns, number_format, add_trailing_zeros):
    """Apply string formatter to specific columns of a list of DataFrames."""

    raw_formatted = [
        _apply_number_format(df[columns], number_format, format_integers=False)
        for df in dfs
    ]
    max_trail = int(max([_get_digits_after_decimal(df) for df in raw_formatted]))
    if add_trailing_zeros:
        formatted = [
            _apply_number_format(df, max_trail, format_integers=True)
            for df in raw_formatted
        ]
    else:
        formatted = raw_formatted
    return formatted, max_trail


def _update_show_col_groups(show_col_groups, column_groups):
    """Set the value of show_col_groups to False or True given column_groups.

    Updates the default None to True if column_groups is not None. Sets to False
    otherwise.

    """
    if show_col_groups is None:
        if column_groups is not None:
            show_col_groups = True
        else:
            show_col_groups = False
    return show_col_groups


def _set_default_stats_options(stats_options):
    """Define some default summary statistics to display in estimation table."""
    if stats_options is None:
        stats_options = {
            "n_obs": "Observations",
            "rsquared": "R$^2$",
            "rsquared_adj": "Adj. R$^2$",
            "resid_std_err": "Residual Std. Error",
            "fvalue": "F Statistic",
        }
    else:
        if not isinstance(stats_options, dict):
            raise TypeError(
                f"""stats_options can be of types dict or NoneType.
            Not: {type(stats_options)}."""
            )
    return stats_options


def _get_model_names(processed_models):
    """Get names of model names if defined, set based on position otherwise.

    Args:
        processed_models (list): List of estimation results processed to dictionaries.

    Returns:
        names (list): List of model names given either by name attribute of each model
            if defined or the position (counting from 1) of each model in parentheses.

    """
    names = []
    for i, mod in enumerate(processed_models):
        if mod.get("name"):
            names.append(mod["name"])
        else:
            names.append(f"({i + 1})")
    _check_order_of_model_names(names)
    return names


def _check_order_of_model_names(model_names):
    """Check identically named models are adjacent.

    Args:
        model_names (list): List of model names.

    Raises:
        ValueError: if models that share a name are not next to each other.

    """
    group_to_col_index = _create_group_to_col_position(model_names)
    for positions in group_to_col_index.values():
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                "If there are repetitions in model_names, models with the "
                f"same name need to be adjacent. You provided: {model_names}"
            )


def _get_default_column_names_and_groups(model_names):
    """Get column names and groups to display in the estimation table.

    Args:
        model_names (list): List of model names.

    Returns:
        col_names (list): List of estimation column names to display in estimation
            table. Same as model_names if model_names are unique. Given by column
            position (counting from 1) in braces otherwise.
        col_groups (list or NoneType): If defined, list of strings unique values
            of which will define column groups. Not defined if model_names are unique.

    """
    if len(set(model_names)) == len(model_names):
        col_groups = None
        col_names = model_names
    else:
        col_groups = model_names
        col_names = [f"({i + 1})" for i in range(len(model_names))]

    return col_names, col_groups


def _customize_col_groups(default_col_groups, custom_col_groups):
    """Change default (inferred) column group titles using custom column groups.

    Args:
        default_col_groups (list or NoneType): The inferred column groups.
        custom_col_groups (list or dict): Dictionary mapping defautl column group
            titles to custom column group titles, if the defautl column groups are
            defined. Must be a list of the same lenght as models otherwise.

    Returns:
        col_groups (list): Column groups to display in estimation table.

    """
    if custom_col_groups:
        if not default_col_groups:
            if not isinstance(custom_col_groups, list):
                raise ValueError(
                    """With unique model names, multiple models can't be grouped
                under common group name. Provide list of unique group names instead,
                if you wish to add column level."""
                )
            col_groups = custom_col_groups
        else:
            if isinstance(custom_col_groups, list):
                col_groups = custom_col_groups
            elif isinstance(custom_col_groups, dict):
                col_groups = (
                    pd.Series(default_col_groups).replace(custom_col_groups).to_list()
                )
            else:
                raise TypeError(
                    f"""Invalid type for custom_col_groups. Can be either list
                    or dictionary, or NoneType. Not: {type(col_groups)}."""
                )
    else:
        col_groups = default_col_groups
    return col_groups


def _customize_col_names(default_col_names, custom_col_names):
    """Change default (inferred) column names using custom column names.

    Args:
        deafult_col_names (list): The default (inferred) column names.
        custom_col_names (list or dict): Dictionary mapping default column names
            to custom column names, or list to display as the name of each
            model column.

    Returns:
        column_names (list): The column names to display in the estimatino table.

    """
    if not custom_col_names:
        col_names = default_col_names
    elif isinstance(custom_col_names, dict):
        col_names = list(pd.Series(default_col_names).replace(custom_col_names))
    elif isinstance(custom_col_names, list):
        if not len(custom_col_names) == len(default_col_names):
            raise ValueError(
                f"""If provided as a list, custom_col_names should have same length as
                default_col_names. Lenght of custom_col_names {len(custom_col_names)}
                !=length of default_col_names {len(default_col_names)}"""
            )
        elif any(isinstance(i, list) for i in custom_col_names):
            raise ValueError("Custom_col_names cannot be a nested list")
        col_names = custom_col_names
    else:
        raise TypeError(
            f"""Invalid type for custom_col_names.
            Can be either list or dictionary, or NoneType. Not: {col_names}."""
        )
    return col_names


def _create_group_to_col_position(column_groups):
    """Get mapping from column groups to column positions.

    Args:
        column_names (list): The column groups to display in the estimatino table.

    Returns:
        group_to_col_index(dict): The mapping from column group titles to column
            positions.

    """
    if column_groups is not None:
        group_to_col_index = {group: [] for group in list(set(column_groups))}
        for i, group in enumerate(column_groups):
            group_to_col_index[group].append(i)
    else:
        group_to_col_index = None
    return group_to_col_index


def _convert_frame_to_string_series(
    df,
    significance_levels,
    show_stars,
):
    """Return processed value series with significance stars and inference information.

    Args:

        df (DataFrame): params DataFrame of the model
        significance_levels (list): see main docstring
        number_format (int, str, iterable or callable): see main docstring
        show_inference (bool): see main docstring
        confidence_intervals (bool): see main docstring
        show_stars (bool): see main docstring

    Returns:
        sr (pd.Series): string series with values and inferences.

    """
    value_sr = df["value"]
    if show_stars:
        sig_bins = [-1, *sorted(significance_levels)] + [2]
        value_sr += "$^{"
        value_sr += (
            pd.cut(
                df["p_value"],
                bins=sig_bins,
                labels=[
                    "*" * (len(significance_levels) - i)
                    for i in range(len(significance_levels) + 1)
                ],
            )
            .astype("str")
            .replace("nan", "")
            .replace(np.nan, "")
        )
        value_sr += " }$"
    if "ci_lower" in df:
        ci_lower = df["ci_lower"]
        ci_upper = df["ci_upper"]
        inference_sr = "("
        inference_sr += ci_lower
        inference_sr += r";"
        inference_sr += ci_upper
        inference_sr += ")"
        sr = _combine_series(value_sr, inference_sr)
    elif "standard_error" in df:
        standard_error = df["standard_error"]
        inference_sr = "(" + standard_error + ")"
        sr = _combine_series(value_sr, inference_sr)
    else:
        sr = value_sr
    # replace empty braces with empty string
    sr = sr.where(sr.apply(lambda x: bool(re.search(r"\d", x))), "")
    sr.name = ""
    return sr


def _combine_series(value_sr, inference_sr):
    """Merge value and inference series.

    Return string series with parameter values and precision values below respective
    param values.

    Args:
        values_sr (Series): string series of estimated parameter values
        inference_sr (Series): string series of inference values

    Returns:
        series: combined string series of param and inference values

    """
    value_df = value_sr.to_frame(name="")
    original_cols = value_df.columns
    value_df.reset_index(drop=False, inplace=True)
    index_names = [item for item in value_df.columns if item not in original_cols]
    # set the index to even numbers, starting at 0
    value_df.index = value_df.index * 2
    inference_df = inference_sr.to_frame(name="")
    inference_df.reset_index(drop=False, inplace=True)
    # set the index to odd numbers, starting at 1
    inference_df.index = (inference_df.index * 2) + 1
    inference_df[index_names[-1]] = ""
    df = pd.concat([value_df, inference_df]).sort_index()
    df.set_index(index_names, inplace=True, drop=True)
    return df[""]


def _create_statistics_sr(
    model,
    stats_options,
    significance_levels,
    show_stars,
    number_format,
    add_trailing_zeros,
    max_trail,
):
    """Process statistics values, return string series.

    Args:
        model (estimation result): see main docstring
        stats_options (dict): see main docstring
        significance_levels (list): see main docstring
        show_stars (bool): see main docstring
        number_format (int, str, iterable or callable): see main docstring
        add_trailing_zeros (bool): If True, format floats such that they haave same
            number of digits after the decimal point.
        max_trail (int): If add_trailing_zeros is True, add corresponding number of
            trailing zeros to floats in the stats DataFrame to have number of digits
            after a decimal point equal to max_trail for each float.

    Returns:
        series: string series with summary statistics values and additional info
            if applicable.

    """
    stats_values = {}
    stats_options = deepcopy(stats_options)
    if "show_dof" in stats_options:
        show_dof = stats_options.pop("show_dof")
    else:
        show_dof = None
    for k in stats_options:
        stats_values[stats_options[k]] = model["info"].get(k, np.nan)

    raw_formatted = _apply_number_format(
        pd.DataFrame(pd.Series(stats_values)), number_format, format_integers=False
    )
    if add_trailing_zeros:
        formatted = _apply_number_format(
            raw_formatted, max_trail, format_integers=False
        )
    else:
        formatted = raw_formatted
    stats_values = formatted.to_dict()[0]
    if "fvalue" in model["info"] and "F Statistic" in stats_values:
        if show_stars and "f_pvalue" in model["info"]:
            sig_bins = [-1, *sorted(significance_levels)] + [2]
            sig_icon_fstat = "*" * (
                len(significance_levels)
                - np.digitize(model["info"]["f_pvalue"], sig_bins)
                + 1
            )
            stats_values["F Statistic"] = (
                stats_values["F Statistic"] + "$^{" + sig_icon_fstat + "}$"
            )
        if show_dof:
            fstat_str = "{{{}(df={};{})}}"
            stats_values["F Statistic"] = fstat_str.format(
                stats_values["F Statistic"],
                int(model["info"]["df_model"]),
                int(model["info"]["df_resid"]),
            )
    if "resid_std_err" in model["info"] and "Residual Std. Error" in stats_values:
        if show_dof:
            rse_str = "{{{}(df={})}}"
            stats_values["Residual Std. Error"] = rse_str.format(
                stats_values["Residual Std. Error"], int(model["info"]["df_resid"])
            )
    stat_sr = pd.Series(stats_values)
    # the following is to make sure statistics dataframe has as many levels of
    # indices as the parameters dataframe.
    stat_ind = np.empty((len(stat_sr), model["params"].index.nlevels - 1), dtype=str)
    stat_ind = np.concatenate(
        [stat_sr.index.values.reshape(len(stat_sr), 1), stat_ind], axis=1
    ).T
    stat_sr.index = pd.MultiIndex.from_arrays(stat_ind)
    return stat_sr.astype("str").replace("nan", "")


def _process_frame_indices(
    df,
    custom_param_names,
    custom_index_names,
    show_col_names,
    show_col_groups,
    column_names,
    column_groups,
):
    """Process body DataFrame, customize the header.

    Args:
        df (DataFrame): string DataFrame with parameter values and inferences.
        custom_param_names (dict): see main docstring
        custom_index_names (list): see main docstring
        show_col_names (bool): see main docstring
        column_names (list): List of column names to display in estimation table.
        column_groups (list): List of column group titles to display in estimation
            table.

    Returns:
        processed_df (DataFrame): string DataFrame with customized header.

    """
    # The column names of the  df are empty strings.
    # If show_col_names is True, rename columns using column_names.
    # Add column level if show col_groups is True.
    if show_col_names:
        if show_col_groups:
            df.columns = pd.MultiIndex.from_tuples(
                [(i, j) for i, j in zip(column_groups, column_names, strict=False)]
            )
        else:
            df.columns = column_names
    if custom_index_names:
        if isinstance(custom_index_names, list):
            df.index.names = custom_index_names
        elif isinstance(custom_index_names, dict):
            df.rename_axis(index=custom_index_names, inplace=True)
        else:
            raise TypeError(
                f"""Invalid custom_index_names can be of type either list or dict,
                or NoneType. Not: {type(custom_index_names)}."""
            )
    if custom_param_names:
        ind = df.index.to_frame()
        ind = ind.replace(custom_param_names)
        df.index = pd.MultiIndex.from_frame(ind)
    return df


def _generate_notes_latex(
    append_notes, notes_label, significance_levels, custom_notes, df
):
    """Generate the LaTex script of the notes section.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        significance_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with LaTex script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    significance_levels = sorted(significance_levels)
    notes_text = ""
    if append_notes:
        notes_text += "\\midrule\n"
        notes_text += "\\textit{{{}}} & \\multicolumn{{{}}}{{r}}{{".format(
            notes_label, str(n_columns + n_levels - 1)
        )
        # iterate over penultimate significance_lelvels since last item of legend
        # is not followed by a semi column
        for i in range(len(significance_levels) - 1):
            star = "*" * (len(significance_levels) - i)
            notes_text += f"$^{{{star}}}$p$<${significance_levels[i]};"
        notes_text += "$^{*}$p$<$" + str(significance_levels[-1]) + "} \\\\\n"
        if custom_notes:
            amp_n = "&" * n_levels
            if isinstance(custom_notes, list):
                if not all(isinstance(n, str) for n in custom_notes):
                    not_str_notes = [n for n in custom_notes if not isinstance(n, str)]
                    not_str_notes_types = [type(n) for n in not_str_notes]
                    raise ValueError(
                        f"""Each custom note can only be of string type.
                        The following notes:
                        {not_str_notes} are of types {not_str_notes_types}
                        respectively."""
                    )
                for n in custom_notes:
                    notes_text += """
                    {}\\multicolumn{{{}}}{{r}}\\textit{{{}}}\\\\\n""".format(
                        amp_n, n_columns, n
                    )
            elif isinstance(custom_notes, str):
                notes_text += "{}\\multicolumn{{{}}}{{r}}\\textit{{{}}}\\\\\n".format(
                    amp_n, n_columns, custom_notes
                )
            else:
                raise TypeError(
                    f"""Custom notes can be either a string or a list of strings.
                    Not: {type(custom_notes)}."""
                )
    return notes_text


def _generate_notes_html(
    append_notes, notes_label, significance_levels, custom_notes, df
):
    """Generate the html script of the notes section of the estimation table.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        significance_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with html script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    significance_levels = sorted(significance_levels)
    notes_text = """<tr><td colspan="{}" style="border-bottom: 1px solid black">
        </td></tr>""".format(n_columns + n_levels)
    if append_notes:
        notes_text += """
        <tr><td style="text-align: left">{}</td><td colspan="{}"
        style="text-align: right">""".format(notes_label, n_columns + n_levels - 1)
        for i in range(len(significance_levels) - 1):
            stars = "*" * (len(significance_levels) - i)
            notes_text += f"<sup>{stars}</sup>p&lt;{significance_levels[i]}; "
        notes_text += f"""<sup>*</sup>p&lt;{significance_levels[-1]} </td>"""
        if custom_notes:
            if isinstance(custom_notes, list):
                if not all(isinstance(n, str) for n in custom_notes):
                    not_str_notes = [n for n in custom_notes if not isinstance(n, str)]
                    not_str_notes_types = [type(n) for n in not_str_notes]
                    raise ValueError(
                        f"""Each custom note can only be of string type.
                        The following notes:
                        {not_str_notes} are of types {not_str_notes_types}
                        respectively."""
                    )
                notes_text += """
                    <tr><td></td><td colspan="{}"style="text-align: right">{}</td></tr>
                    """.format(n_columns + n_levels - 1, custom_notes[0])
                if len(custom_notes) > 1:
                    for i in range(1, len(custom_notes)):
                        notes_text += """
                        <tr><td></td><td colspan="{}"style="text-align: right">
                        {}</td></tr>
                        """.format(n_columns + n_levels - 1, custom_notes[i])
            elif isinstance(custom_notes, str):
                notes_text += """
                    <tr><td></td><td colspan="{}"style="text-align: right">{}</td></tr>
                    """.format(n_columns + n_levels - 1, custom_notes)
            else:
                raise TypeError(
                    f"""Custom notes can be either a string or a list of strings,
                    not {type(custom_notes)}."""
                )

    return notes_text


def _extract_params_from_sm(model):
    """Convert statsmodels like estimation result to estimagic like params dataframe."""
    to_concat = []
    params_list = ["params", "pvalues", "bse"]
    for col in params_list:
        to_concat.append(getattr(model, col))
    to_concat.append(model.conf_int())
    params_df = pd.concat(to_concat, axis=1)
    params_df.columns = ["value", "p_value", "standard_error", "ci_lower", "ci_upper"]
    return params_df


def _extract_info_from_sm(model):
    """Process statsmodels estimation result to retrieve summary statistics as dict."""
    info = {}
    key_values = [
        "rsquared",
        "rsquared_adj",
        "fvalue",
        "f_pvalue",
        "df_model",
        "df_resid",
    ]
    for kv in key_values:
        info[kv] = getattr(model, kv)
    info["name"] = model.model.endog_names
    info["resid_std_err"] = np.sqrt(model.scale)
    info["n_obs"] = model.df_model + model.df_resid + 1
    return info


def _apply_number_format(df_raw, number_format, format_integers):
    """Apply string format to DataFrame cells.

    Args:
        df_raw (DataFrame): The DataFrame with float values to format.
        number_format (str, list, tuple, callable or int): User defined number format
            to apply to the DataFrame.
        format_integers (bool): Apply number format also to integers

    Returns:
        df_formatted (DataFrame): Formatted DataFrame.

    """
    processed_format = _process_number_format(number_format)
    df_raw = df_raw.copy(deep=True)
    if isinstance(processed_format, (list, tuple)):
        df_formatted = df_raw.copy(deep=True).astype("float")
        for formatter in processed_format[:-1]:
            df_formatted = pd_df_map(df_formatted, formatter.format).astype("float")
        df_formatted = pd_df_map(
            df_formatted.astype("float"), processed_format[-1].format
        )
    elif isinstance(processed_format, str):
        df_formatted = pd_df_map(
            df_raw.astype("str"),
            partial(_format_non_scientific_numbers, format_string=processed_format),
        )
    elif callable(processed_format):
        df_formatted = pd_df_map(df_raw, processed_format)

    # Don't format integers: set to original value
    if not format_integers:
        integer_locs = pd_df_map(df_raw, _is_integer)
        df_formatted[integer_locs] = pd_df_map(
            df_raw[integer_locs].astype(float), "{:.0f}".format
        )
    return df_formatted


def _format_non_scientific_numbers(number_string, format_string):
    """Apply number format if the number string is not in scientific format."""
    if "e" in number_string:
        out = number_string
    else:
        out = format_string.format(float(number_string))
    return out


def _process_number_format(raw_format):
    """Process the user define formatter.

    Reduces cases for number format in apply_number_format.

    """
    if isinstance(raw_format, str):
        processed_format = [raw_format]
    elif isinstance(raw_format, int):
        processed_format = f"{{0:.{raw_format}f}}"
    elif callable(raw_format) or isinstance(raw_format, (list, tuple)):
        processed_format = raw_format
    else:
        raise TypeError(
            f"""Number format can be either of [str, int, tuple, list, callable] types.
           Not: {type(raw_format)}."""
        )
    return processed_format


def _get_digits_after_decimal(df):
    """Get the maximum number of digits after a decimal point in a DataFrame."""
    max_trail = 0
    for c in df.columns:
        try:
            trail_length = (
                (
                    df[c][~df[c].astype("str").str.contains("e")]
                    .astype("str")
                    .str.split(".", expand=True)[1]
                    .astype("str")
                    .replace("None", "")
                )
                .str.len()
                .max()
            )
        except KeyError:
            trail_length = 0
        if trail_length > max_trail:
            max_trail = trail_length
    return max_trail


def _center_align_integers_and_non_numeric_strings(sr):
    """Align integer numbers and strings at the center of model column."""
    sr = deepcopy(sr)
    for i in sr.index:
        if _is_integer(sr[i]):
            sr[i] = f"\\multicolumn{{1}}{{c}}{{{str(int(float(sr[i])))}}}"
        else:
            string_without_stars = sr[i].split("$", 1)[0]
            if not string_without_stars.replace(".", "").isnumeric():
                sr[i] = f"\\multicolumn{{1}}{{c}}{{{sr[i]}}}"
    return sr


def _get_updated_styler(
    df, show_index_names, show_col_names, show_col_groups, escape_special_characters
):
    """Return pandas.Styler object based ont the data and styling options."""
    styler = df.style
    if not show_index_names:
        styler = styler.hide(names=True)
    if not show_col_names:
        styler = styler.hide(axis=1)
    if not show_col_groups:
        styler = styler.hide(axis=1, level=0)
    for ax in [0, 1]:
        styler = styler.format_index(escape=escape_special_characters, axis=ax)
    return styler


def _is_integer(num):
    """Check if number is an integer (including a float with only zeros as digits)"""
    try:
        out = int(float(num)) == float(num)
    except ValueError:
        out = False
    return out
