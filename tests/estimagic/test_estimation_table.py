import io
import textwrap

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from estimagic.config import EXAMPLE_DIR
from estimagic.estimation_table import (
    _apply_number_format,
    _center_align_integers_and_non_numeric_strings,
    _check_order_of_model_names,
    _convert_frame_to_string_series,
    _create_group_to_col_position,
    _create_statistics_sr,
    _customize_col_groups,
    _customize_col_names,
    _get_default_column_names_and_groups,
    _get_digits_after_decimal,
    _get_model_names,
    _get_params_frames_with_common_index,
    _process_frame_indices,
    _process_model,
    estimation_table,
    render_html,
    render_latex,
)
from pandas.testing import assert_frame_equal as afe
from pandas.testing import assert_series_equal as ase


# ======================================================================================
# Helper functions
# ======================================================================================
def _get_models_multiindex():
    df = pd.DataFrame(
        data=np.ones((3, 4)), columns=["value", "ci_lower", "ci_upper", "p_value"]
    )
    df.index = pd.MultiIndex.from_tuples(
        [("p_1", "v_1"), ("p_1", "v_2"), ("p_2", "v_2")]
    )
    info = {"n_obs": 400}
    mod1 = {"params": df, "info": info, "name": "m1"}
    mod2 = {"params": df, "info": info, "name": "m2"}
    models = [mod1, mod2]
    return models


def _get_models_single_index():
    df = pd.DataFrame(
        data=np.ones((3, 4)), columns=["value", "ci_lower", "ci_upper", "p_value"]
    )
    df.index = [f"p{i}" for i in [1, 2, 3]]
    info = {"n_obs": 400}
    mod1 = {"params": df, "info": info, "name": "m1"}
    mod2 = {"params": df, "info": info, "name": "m2"}
    models = [mod1, mod2]
    return models


def _get_models_multiindex_multi_column():
    df = pd.DataFrame(
        data=np.ones((3, 4)), columns=["value", "ci_lower", "ci_upper", "p_value"]
    )
    df.index = pd.MultiIndex.from_tuples(
        [("p_1", "v_1"), ("p_1", "v_2"), ("p_2", "v_2")]
    )
    info = {"n_obs": 400}
    mod1 = {"params": df.iloc[1:], "info": info, "name": "m1"}
    mod2 = {"params": df, "info": info, "name": "m2"}
    mod3 = {"params": df, "info": info, "name": "m2"}
    models = [mod1, mod2, mod3]
    return models


def _read_csv_string(string, index_cols=None):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)


# ======================================================================================
# Tests
# ======================================================================================

# test process_model for different model types

fix_path = EXAMPLE_DIR / "diabetes.csv"

df_ = pd.read_csv(fix_path, index_col=0)
est = sm.OLS(endog=df_["target"], exog=sm.add_constant(df_[df_.columns[0:4]])).fit()
est1 = sm.OLS(endog=df_["target"], exog=sm.add_constant(df_[df_.columns[0:5]])).fit()


def test_estimation_table():
    models = [est]
    res = estimation_table(models, return_type="render_inputs", append_notes=False)
    exp = {}
    body = """
        index,target
        const,152.00$^{*** }$
        ,(2.85)
        Age,37.20$^{ }$
        ,(64.10)
        Sex,-107.00$^{* }$
        ,(62.10)
        BMI,787.00$^{*** }$
        ,(65.40)
        ABP,417.00$^{*** }$
        ,(69.50)
    """
    exp["body"] = _read_csv_string(body).fillna("")
    exp["body"].set_index("index", inplace=True)
    footer_str = """
         ,target
        R$^2$,0.40
        Adj. R$^2$,0.40
        Residual Std. Error,60
        F Statistic,72.90$^{***}$
        Observations,442

    """
    exp["footer"] = _read_csv_string(footer_str).fillna("")
    exp["footer"].set_index(" ", inplace=True)
    exp["footer"].index.names = [None]
    exp["footer"].index = pd.MultiIndex.from_arrays([exp["footer"].index])
    afe(exp["footer"].sort_index(), res["footer"].sort_index())
    afe(exp["body"], res["body"], check_index_type=False)


MODELS = [
    _get_models_multiindex(),
    _get_models_single_index(),
    _get_models_multiindex_multi_column(),
]
PARAMETRIZATION = [("latex", render_latex, models) for models in MODELS]
PARAMETRIZATION += [("html", render_html, models) for models in MODELS]


@pytest.mark.parametrize("return_type, render_func,models", PARAMETRIZATION)
def test_one_and_stage_rendering_are_equal(return_type, render_func, models):
    first_stage = estimation_table(
        models, return_type="render_inputs", confidence_intervals=True
    )
    second_stage = render_func(siunitx_warning=False, **first_stage)
    one_stage = estimation_table(
        models,
        return_type=return_type,
        siunitx_warning=False,
        confidence_intervals=True,
    )
    assert one_stage == second_stage


def test_process_model_stats_model():
    params = pd.DataFrame(
        columns=["value", "p_value", "standard_error", "ci_lower", "ci_upper"],
        index=["const", "Age", "Sex", "BMI", "ABP"],
    )
    params["value"] = [152.133484, 37.241211, -106.577520, 787.179313, 416.673772]
    params["p_value"] = [
        2.048808e-193,
        5.616557e-01,
        8.695658e-02,
        5.345260e-29,
        4.245663e-09,
    ]
    params["standard_error"] = [2.852749, 64.117433, 62.125062, 65.424126, 69.494666]
    params["ci_lower"] = [146.526671, -88.775663, -228.678572, 658.594255, 280.088446]
    params["ci_upper"] = [157.740298, 163.258084, 15.523532, 915.764371, 553.259097]
    info = {}
    info["rsquared"] = 0.40026108237714
    info["rsquared_adj"] = 0.39477148130050055
    info["fvalue"] = 72.91259907398705
    info["f_pvalue"] = 2.700722880950139e-47
    info["df_model"] = 4.0
    info["df_resid"] = 437.0
    info["resid_std_err"] = 59.97560860753488
    info["n_obs"] = 442.0
    res = _process_model(est)
    afe(res["params"], params)
    ase(pd.Series(res["info"]), pd.Series(info))
    assert res["name"] == "target"


# test convert_model_to_series for different arguments
def test_convert_model_to_series_with_ci():
    df = pd.DataFrame(
        np.array(
            [[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3], [1.2, 3.3, 4.33]]
        ).T,
        columns=["value", "p_value", "ci_lower", "ci_upper"],
        index=["a", "b", "c"],
    ).astype("str")
    df["p_value"] = df["p_value"].astype("float")
    significance_levels = [0.1, 0.05, 0.01]
    show_stars = True
    res = _convert_frame_to_string_series(df, significance_levels, show_stars)
    exp = pd.Series(
        [
            "0.6$^{ }$",
            r"(0.6;1.2)",
            "2.3$^{** }$",
            r"(2.3;3.3)",
            "3.3$^{*** }$",
            r"(3.3;4.33)",
        ],
        index=["a", "", "b", "", "c", ""],
        name="",
    )
    exp.index.name = "index"
    ase(exp, res)


def test_convert_model_to_series_with_se():
    df = pd.DataFrame(
        np.array([[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3]]).T,
        columns=["value", "p_value", "standard_error"],
        index=["a", "b", "c"],
    ).astype("str")
    df["p_value"] = df["p_value"].astype("float")
    significance_levels = [0.1, 0.05, 0.01]
    show_stars = True
    res = _convert_frame_to_string_series(df, significance_levels, show_stars)
    exp = pd.Series(
        ["0.6$^{ }$", "(0.6)", "2.3$^{** }$", "(2.3)", "3.3$^{*** }$", "(3.3)"],
        index=["a", "", "b", "", "c", ""],
        name="",
    )
    exp.index.name = "index"
    ase(exp, res)


def test_convert_model_to_series_without_inference():
    df = pd.DataFrame(
        np.array([[0.6, 2.3, 3.3], [0.11, 0.049, 0.009]]).T,
        columns=["value", "p_value"],
        index=["a", "b", "c"],
    ).astype("str")
    df["p_value"] = df["p_value"].astype("float")
    significance_levels = [0.1, 0.05, 0.01]
    show_stars = True
    res = _convert_frame_to_string_series(df, significance_levels, show_stars)
    exp = pd.Series(
        ["0.6$^{ }$", "2.3$^{** }$", "3.3$^{*** }$"], index=["a", "b", "c"], name=""
    )
    ase(exp, res)


# test create stat series
def test_create_statistics_sr():
    df = pd.DataFrame(np.empty((10, 3)), columns=["a", "b", "c"])
    df.index = pd.MultiIndex.from_arrays(np.array([np.arange(10), np.arange(10)]))
    info = {"rsquared": 0.45, "n_obs": 400, "rsquared_adj": 0.0002}
    number_format = ("{0:.3g}", "{0:.5f}", "{0:.4g}")
    add_trailing_zeros = True
    sig_levels = [0.1, 0.2]
    show_stars = False
    model = {"params": df, "info": info, "name": "target"}
    stats_options = {
        "n_obs": "Observations",
        "rsquared": "R2",
        "rsquared_adj": "R2 Adj.",
    }
    res = _create_statistics_sr(
        model,
        stats_options,
        sig_levels,
        show_stars,
        number_format,
        add_trailing_zeros,
        max_trail=4,
    )
    exp = pd.Series(["0.4500", "0.0002", "400"])
    exp.index = pd.MultiIndex.from_arrays(
        np.array([np.array(["R2", "R2 Adj.", "Observations"]), np.array(["", "", ""])])
    )
    ase(exp.sort_index(), res.sort_index())


# test _process_frame_axes for different arguments
def test_process_frame_indices_index():
    df = pd.DataFrame(np.ones((3, 3)), columns=["", "", ""])
    df.index = pd.MultiIndex.from_arrays(
        np.array([["today", "today", "today"], ["var1", "var2", "var3"]])
    )
    df.index.names = ["l1", "l2"]
    par_name_map = {"today": "tomorrow", "var1": "1stvar"}
    index_name_map = ["period", "variable"]
    column_names = list("abc")
    res = _process_frame_indices(
        df,
        custom_param_names=par_name_map,
        custom_index_names=index_name_map,
        column_names=column_names,
        show_col_names=True,
        show_col_groups=False,
        column_groups=None,
    )
    # expected:
    params = """
        period,variable,a,b,c
        tomorrow,1stvar,1,1,1
        tomorrow,var2,1,1,1
        tomorrow,var3,1,1,1
    """
    exp = _read_csv_string(params).fillna("")
    exp.set_index(["period", "variable"], inplace=True)
    afe(res, exp, check_dtype=False)


def test_process_frame_indices_columns():
    df = pd.DataFrame(np.ones((3, 3)), columns=["", "", ""])
    col_names = list("abc")
    col_groups = ["first", "first", "second"]
    res = _process_frame_indices(
        df=df,
        custom_index_names=None,
        custom_param_names=None,
        show_col_groups=True,
        show_col_names=True,
        column_names=col_names,
        column_groups=col_groups,
    )
    arrays = [np.array(col_groups), np.array(col_names)]
    exp = pd.DataFrame(data=np.ones((3, 3)), columns=arrays)
    afe(res, exp, check_dtype=False)


def test_apply_number_format_tuple():
    number_format = ("{0:.2g}", "{0:.2f}", "{0:.2g}")
    raw = pd.DataFrame(data=[1234.2332, 0.0001])
    exp = pd.DataFrame(data=["1.2e+03", "0"])
    res = _apply_number_format(
        df_raw=raw, number_format=number_format, format_integers=False
    )
    afe(exp, res)


def test_apply_number_format_int():
    number_format = 3
    raw = pd.DataFrame(data=["1234.2332", "1.2e+03"])
    exp = pd.DataFrame(data=["1234.233", "1200"])
    res = _apply_number_format(
        df_raw=raw, number_format=number_format, format_integers=False
    )
    afe(exp, res)


def test_apply_number_format_callable():
    def nsf(num, n=3):
        """N-Significant Figures."""
        numstr = ("{0:.%ie}" % (n - 1)).format(num)
        return numstr

    raw = pd.DataFrame(data=[1234.2332, 0.0001])
    exp = pd.DataFrame(data=["1.23e+03", "1.00e-04"])
    res = _apply_number_format(df_raw=raw, number_format=nsf, format_integers=False)
    afe(exp, res)


def test_get_digits_after_decimal():
    df = pd.DataFrame(
        data=[["12.456", "0.00003", "1.23e+05"], ["16", "0.03", "1.2e+05"]]
    ).T
    exp = 5
    res = _get_digits_after_decimal(df)
    assert exp == res


def test_create_group_to_col_position():
    col_groups = [
        "a_name",
        "a_name",
        "a_name",
        "second_name",
        "second_name",
        "third_name",
    ]
    exp = {"a_name": [0, 1, 2], "second_name": [3, 4], "third_name": [5]}
    res = _create_group_to_col_position(col_groups)
    assert exp == res


def test_get_model_names():
    m1 = {"params": None, "info": None, "name": "a_name"}
    m3 = {"params": None, "info": None, "name": None}
    m5 = {"params": None, "info": None, "name": "third_name"}
    models = [m1, m3, m5]
    res = _get_model_names(models)
    exp = ["a_name", "(2)", "third_name"]
    assert res == exp


def test_get_default_column_names_and_groups():
    model_names = ["a_name", "a_name", "(3)", "(4)", "third_name"]
    res_names, res_groups = _get_default_column_names_and_groups(model_names)
    exp_names = [f"({i+1})" for i in range(len(model_names))]
    exp_groups = ["a_name", "a_name", "(3)", "(4)", "third_name"]
    assert res_names == exp_names
    assert res_groups == exp_groups


def test_get_default_column_names_and_groups_undefined_groups():
    model_names = ["a_name", "second_name", "(3)", "(4)", "third_name"]
    res_names, res_groups = _get_default_column_names_and_groups(model_names)
    exp_names = model_names
    assert res_names == exp_names
    assert pd.isna(res_groups)


def test_customize_col_groups():
    default = ["a_name", "a_name", "(3)", "(4)", "third_name"]
    mapping = {"a_name": "first_name", "third_name": "fifth_name"}
    exp = ["first_name", "first_name", "(3)", "(4)", "fifth_name"]
    res = _customize_col_groups(default, mapping)
    assert exp == res


def test_customize_col_names_dict():
    default = list("abcde")
    custom = {"a": "1", "c": "3", "e": "5"}
    res = _customize_col_names(default_col_names=default, custom_col_names=custom)
    exp = ["1", "b", "3", "d", "5"]
    assert exp == res


def test_customize_col_names_list():
    default = list("abcde")
    custom = list("12345")
    res = _customize_col_names(default_col_names=default, custom_col_names=custom)
    exp = ["1", "2", "3", "4", "5"]
    assert exp == res


def test_get_params_frames_with_common_index():
    m1 = {
        "params": pd.DataFrame(np.ones(5), index=list("abcde")),
        "info": None,
        "name": None,
    }
    m2 = {
        "params": pd.DataFrame(np.ones(3), index=list("abc")),
        "info": None,
        "name": None,
    }
    res = _get_params_frames_with_common_index([m1, m2])
    exp = [
        pd.DataFrame(np.ones(5), index=list("abcde")),
        pd.DataFrame(
            np.concatenate([np.ones(3), np.ones(2) * np.nan]), index=list("abcde")
        ),
    ]
    afe(res[0], exp[0])
    afe(res[1], exp[1])


def test_get_params_frames_with_common_index_multiindex():
    mi = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2), ("b", 3)])
    m1 = {"params": pd.DataFrame(np.ones(5), index=mi), "info": None, "name": None}
    m2 = {"params": pd.DataFrame(np.ones(3), index=mi[:3]), "info": None, "name": None}
    res = _get_params_frames_with_common_index([m1, m2])
    exp = [
        pd.DataFrame(np.ones(5), index=mi),
        pd.DataFrame(np.concatenate([np.ones(3), np.ones(2) * np.nan]), index=mi),
    ]
    afe(res[0], exp[0])
    afe(res[1], exp[1])


def test_check_order_of_model_names_raises_error():
    model_names = ["a", "b", "a"]
    with pytest.raises(ValueError):
        _check_order_of_model_names(model_names)


def test_manual_extra_info():
    footer_str = """
         ,target
        R$^2$,0.40
        Adj. R$^2$,0.40
        Residual Std. Error,60.5
        F Statistic,72.90$^{***}$
        Observations,442
        Controls,Yes

    """
    footer = _read_csv_string(footer_str).fillna("")
    footer.set_index(" ", inplace=True)
    footer.index.names = [None]
    footer.index = pd.MultiIndex.from_arrays([footer.index])
    exp = footer.copy(deep=True)
    exp.loc["Controls"] = "\\multicolumn{1}{c}{Yes}"
    exp.loc["Observations"] = "\\multicolumn{1}{c}{442}"
    for i, r in footer.iterrows():
        res = _center_align_integers_and_non_numeric_strings(r)
        ase(exp.loc[i], res)
