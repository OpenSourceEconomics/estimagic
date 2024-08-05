import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import statsmodels.api as sm
from estimagic import bootstrap


def aaae(obj1, obj2, decimal=6):
    arr1 = np.asarray(obj1)
    arr2 = np.asarray(obj2)
    np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)


@pytest.fixture()
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )

    y = np.array([[2.0, 8.0], [2.0, 8.0], [2.5, 7.0], [3.0, 6.0], [3.25, 5.75]])
    out["estimates_arr"] = y
    out["estimates_df"] = pd.DataFrame(y, columns=["x1", "x2"])
    out["estimates_dict"] = {"x1": [2, 2, 2.5, 3, 3.25], "x2": [8, 8, 7, 6, 5.75]}

    return out


@pytest.fixture()
def expected():
    out = {}

    summary = np.array(
        [
            [2.5, 0.576222, 1.5, 3.5, np.nan, np.nan],
            [7.0, 0.956896, 5.5, 9.0, np.nan, np.nan],
        ]
    )

    cov = np.array([[0.332032, -0.528158], [-0.528158, 0.915651]])
    p_values = np.array([0.0, 0.0])
    ci_lower = np.array([1.5, 5.5])
    ci_upper = np.array([3.5, 9.0])

    out["summary"] = pd.DataFrame(
        summary,
        columns=["value", "standard_error", "ci_lower", "ci_upper", "p_value", "stars"],
        index=["x1", "x2"],
    )
    out["ci_lower"] = pd.Series(ci_lower, index=["x1", "x2"])
    out["ci_upper"] = pd.Series(ci_upper, index=["x1", "x2"])
    out["ci_lower_x1"] = pd.Series(ci_lower[0], index=["x1"])
    out["ci_upper_x1"] = pd.Series(ci_upper[0], index=["x1"])
    out["cov"] = pd.DataFrame(cov, columns=["x1", "x2"], index=["x1", "x2"])
    out["se"] = pd.Series(np.sqrt(np.diagonal(cov)), index=["x1", "x2"])
    out["p_values"] = pd.Series(p_values, index=["x1", "x2"])
    out["p_value_x1"] = pd.Series(p_values[0], index=["x1"])

    return out


@pytest.fixture()
def seaborn_example():
    out = {}

    raw = sns.load_dataset("exercise", index_col=0)
    replacements = {"1 min": 1, "15 min": 15, "30 min": 30}
    df = raw.assign(time=raw.time.cat.rename_categories(replacements).astype(int))
    df["constant"] = 1

    lower_ci = pd.Series([90.709236, 0.151193], index=["constant", "time"])
    upper_ci = pd.Series([96.827145, 0.627507], index=["constant", "time"])
    expected = {"lower_ci": lower_ci, "upper_ci": upper_ci}

    out["df"] = df
    out["expected"] = expected

    return out


def _outcome_func(data, shift=0):
    """Compute column means.

    Args:
        data (pd.Series or pd.DataFrame): The data set.
        shift (float): Scalar that is added to the column means.

    Returns:
        pd.Series: Series where the k-th row corresponds to the mean
            of the k-th column of the input data.

    """
    # Return pd.Series when .mean() is applied to a Series
    # Only applying .mean() to a pd.Series would yield a float
    return pd.DataFrame(data).mean(axis=0) + shift


def _outcome_ols(data):
    y = data["pulse"]
    x = data[["constant", "time"]]
    params = sm.OLS(y, x).fit().params

    return params


@pytest.mark.parametrize("shift", [0, 10, -10])
def test_bootstrap_with_outcome_kwargs(shift, setup):
    result = bootstrap(
        outcome=_outcome_func,
        data=setup["df"],
        seed=123,
        outcome_kwargs={"shift": shift},
    )

    expected = pd.Series([2.5, 7.0], index=["x1", "x2"])
    aaae(result.base_outcome, expected + shift)


def test_bootstrap_existing_outcomes(setup):
    result = bootstrap(
        data=setup["df"],
        outcome=_outcome_func,
        n_draws=2,
    )
    assert len(result.outcomes) == 2
    result = bootstrap(
        outcome=_outcome_func,
        data=setup["df"],
        existing_result=result,
        n_draws=1,
    )
    assert len(result.outcomes) == 1


def test_bootstrap_from_outcomes(setup, expected):
    result = bootstrap(outcome=_outcome_func, data=setup["df"], seed=1234)

    _ = result.outcomes
    summary = result.summary()
    ci_lower, ci_upper = result.ci()
    covariance = result.cov()
    standard_errors = result.se()

    with pytest.raises(NotImplementedError):
        assert result._p_values

    aaae(ci_lower, expected["ci_lower"])
    aaae(ci_upper, expected["ci_upper"])
    aaae(covariance, expected["cov"])
    aaae(standard_errors, expected["se"])

    aaae(summary["value"], expected["summary"]["value"])
    aaae(summary["standard_error"], expected["summary"]["standard_error"])
    aaae(summary["ci_lower"], expected["summary"]["ci_lower"])
    aaae(summary["ci_upper"], expected["summary"]["ci_upper"])


def test_bootstrap_from_outcomes_private_methods(setup, expected):
    result = bootstrap(outcome=_outcome_func, data=setup["df"], seed=1234)

    _ = result.outcomes
    ci_lower, ci_upper = result._ci
    covariance = result._cov
    standard_errors = result._se

    with pytest.raises(NotImplementedError):
        assert result._p_values

    aaae(ci_lower, expected["ci_lower"])
    aaae(ci_upper, expected["ci_upper"])
    aaae(covariance, expected["cov"])
    aaae(standard_errors, expected["se"])


def test_bootstrap_from_outcomes_single_outcome(setup, expected):
    result = bootstrap(outcome=_outcome_func, data=setup["df"]["x1"], seed=1234)

    _ = result.outcomes
    ci_lower, ci_upper = result.ci()

    aaae(ci_lower, expected["ci_lower_x1"])
    aaae(ci_upper, expected["ci_upper_x1"])


def test_outcome_not_callable(setup):
    expected_msg = "outcome must be a callable."

    with pytest.raises(TypeError) as error:
        assert bootstrap(data=setup["df"], outcome=setup["estimates_df"])

    assert str(error.value) == expected_msg


@pytest.mark.parametrize("input_type", ["arr", "df", "dict"])
def test_existing_result_wrong_input_type(input_type, setup):
    expected_msg = "existing_result must be None or a BootstrapResult."

    with pytest.raises(ValueError) as error:
        assert bootstrap(
            outcome=_outcome_func,
            data=setup["df"],
            existing_result=setup["estimates_" + input_type],
        )

    assert str(error.value) == expected_msg


@pytest.mark.parametrize("return_type", ["array", "dataframe", "pytree"])
def test_cov_correct_return_type(return_type, setup):
    result = bootstrap(
        outcome=_outcome_func,
        data=setup["df"],
    )
    _ = result.cov(return_type=return_type)


def test_cov_wrong_return_type(setup):
    result = bootstrap(
        outcome=_outcome_func,
        data=setup["df"],
    )

    expected_msg = "return_type must be one of pytree, array, or dataframe, not dict."

    with pytest.raises(ValueError) as error:
        assert result.cov(return_type="dict")

    assert str(error.value) == expected_msg


def test_existing_result(seaborn_example):
    first_result = bootstrap(
        data=seaborn_example["df"], outcome=_outcome_ols, seed=1234
    )

    expected_msg = "existing_result must be None or a BootstrapResult."
    with pytest.raises(ValueError) as error:
        assert bootstrap(
            data=seaborn_example["df"],
            outcome=_outcome_ols,
            existing_result=first_result.outcomes,
        )
    assert str(error.value) == expected_msg

    my_result = bootstrap(
        data=seaborn_example["df"],
        outcome=_outcome_ols,
        existing_result=first_result,
        seed=2,
    )
    lower_ci, upper_ci = my_result.ci(ci_method="t")

    aaae(lower_ci, seaborn_example["expected"]["lower_ci"])
    aaae(upper_ci, seaborn_example["expected"]["upper_ci"])
