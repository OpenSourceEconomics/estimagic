"""Test the auxiliary functions of the pounders algorithm."""

from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.batch_evaluators import joblib_batch_evaluator
from optimagic.optimizers._pounders.pounders_auxiliary import (
    add_geomtery_points_to_make_main_model_fully_linear,
    create_initial_residual_model,
    create_main_from_residual_model,
    evaluate_residual_model,
    find_affine_points,
    fit_residual_model,
    get_feature_matrices_residual_model,
    update_main_model_with_new_accepted_x,
    update_residual_model,
    update_residual_model_with_new_accepted_x,
)
from optimagic.optimizers._pounders.pounders_history import LeastSquaresHistory

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def read_yaml(path):
    with open(rf"{path}") as file:
        data = yaml.full_load(file)

    return data


# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def criterion():
    data = pd.read_csv(FIXTURES_DIR / "pounders_example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.fixture()
def data_create_initial_residual_model():
    test_data = read_yaml(FIXTURES_DIR / "update_initial_residual_model.yaml")
    history = LeastSquaresHistory()
    ResidualModel = namedtuple(
        "ResidualModel", ["intercepts", "linear_terms", "square_terms"]
    )

    history.add_entries(
        np.array(test_data["x_candidate"]),
        np.array(test_data["residuals_candidate"]),
    )
    accepted_index = 0
    delta = 0.1

    inputs_dict = {"history": history, "accepted_index": accepted_index, "delta": delta}

    residual_model_expected = ResidualModel(
        intercepts=test_data["residual_model_expected"]["intercepts"],
        linear_terms=test_data["residual_model_expected"]["linear_terms"],
        square_terms=test_data["residual_model_expected"]["square_terms"],
    )

    return inputs_dict, residual_model_expected


@pytest.fixture()
def data_update_residual_model():
    test_data = read_yaml(FIXTURES_DIR / "update_residual_model.yaml")

    ResidualModel = namedtuple(
        "ResidualModel", ["intercepts", "linear_terms", "square_terms"]
    )

    residual_model = ResidualModel(
        intercepts=None,
        linear_terms=np.array(test_data["linear_terms"]),
        square_terms=np.array(test_data["square_terms"]),
    )
    coefficients_to_add = {
        "linear_terms": np.array(test_data["coefficients_linear_terms"]).T,
        "square_terms": np.array(test_data["coefficients_square_terms"]),
    }

    inputs_dict = {
        "residual_model": residual_model,
        "coefficients_to_add": coefficients_to_add,
        "delta": test_data["delta"],
        "delta_old": test_data["delta_old"],
    }

    expected_dict = {
        "linear_terms": test_data["linear_terms_expected"],
        "square_terms": test_data["square_terms_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture()
def data_update_main_from_residual_model():
    test_data = read_yaml(FIXTURES_DIR / "update_main_from_residual_model.yaml")

    ResidualModel = namedtuple(
        "ResidualModel", ["intercepts", "linear_terms", "square_terms"]
    )
    MainModel = namedtuple("MainModel", ["linear_terms", "square_terms"])

    residual_model = ResidualModel(
        intercepts=np.array(test_data["residuals"]),
        linear_terms=np.array(test_data["linear_terms_residual_model"]),
        square_terms=np.array(test_data["square_terms_residual_model"]),
    )

    main_model_expected = MainModel(
        linear_terms=test_data["linear_terms_main_model_expected"],
        square_terms=test_data["square_terms_main_model_expected"],
    )

    return residual_model, main_model_expected


@pytest.fixture()
def data_update_residual_model_with_new_accepted_x():
    test_data = read_yaml(
        FIXTURES_DIR / "update_residual_model_with_new_accepted_x.yaml"
    )

    ResidualModel = namedtuple(
        "ResidualModel", ["intercepts", "linear_terms", "square_terms"]
    )
    inputs_dict = {}
    residual_model_expected = {}

    residual_model = ResidualModel(
        intercepts=np.array(test_data["residuals"]),
        linear_terms=np.array(test_data["linear_terms"]),
        square_terms=np.array(test_data["square_terms"]),
    )

    inputs_dict["residual_model"] = residual_model
    inputs_dict["x_candidate"] = (
        np.array(test_data["x_candidate_uncentered"]) - np.array(test_data["best_x"])
    ) / test_data["delta"]

    residual_model_expected = ResidualModel(
        intercepts=test_data["residuals_expected"],
        linear_terms=test_data["linear_terms_expected"],
        square_terms=np.array(test_data["square_terms"]),
    )

    return inputs_dict, residual_model_expected


@pytest.fixture()
def data_update_main_model_with_new_accepted_x():
    test_data = read_yaml(FIXTURES_DIR / "update_main_model_with_new_accepted_x.yaml")

    MainModel = namedtuple("MainModel", ["linear_terms", "square_terms"])

    inputs_dict = {}
    expected_dict = {}

    main_model = MainModel(
        linear_terms=np.array(test_data["linear_terms"]),
        square_terms=np.array(test_data["square_terms"]),
    )

    inputs_dict["main_model"] = main_model
    inputs_dict["x_candidate"] = (
        np.array(test_data["x_candidate_uncentered"]) - np.array(test_data["best_x"])
    ) / test_data["delta"]

    expected_dict["linear_terms"] = test_data["linear_terms_expected"]

    return inputs_dict, expected_dict


@pytest.fixture(
    params=[
        "zero_i",
        "zero_ii",
        "zero_iii",
        "zero_iv",
        "nonzero_i",
        "nonzero_ii",
        "nonzero_iii",
    ]
)
def data_find_affine_points(request):
    test_data = read_yaml(FIXTURES_DIR / f"find_affine_points_{request.param}.yaml")

    history = LeastSquaresHistory()
    history_x = np.array(test_data["history_x"])
    history.add_entries(history_x, np.zeros(history_x.shape))

    inputs_dict = {
        "history": history,
        "x_accepted": np.array(test_data["x_accepted"]),
        "model_improving_points": np.array(test_data["model_improving_points"]),
        "project_x_onto_null": test_data["project_x_onto_null"],
        "delta": test_data["delta"],
        "theta1": test_data["theta1"],
        "c": test_data["c"],
        "model_indices": np.array(test_data["model_indices"]),
        "n_modelpoints": test_data["n_modelpoints"],
    }

    expected_dict = {
        "model_improving_points": test_data["model_improving_points_expected"],
        "model_indices": test_data["model_indices_expected"],
        "n_modelpoints": test_data["n_modelpoints_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture(params=["i", "ii"])
def data_add_points_until_main_model_fully_linear(request, criterion):
    test_data = read_yaml(
        FIXTURES_DIR / f"add_points_until_main_model_fully_linear_{request.param}.yaml"
    )

    history = LeastSquaresHistory()
    n = 3
    n_modelpoints = test_data["n_modelpoints"]
    history.add_entries(
        np.array(test_data["history_x"])[: -(n - n_modelpoints)],
        np.array(test_data["history_criterion"])[: -(n - n_modelpoints)],
    )

    MainModel = namedtuple("MainModel", ["linear_terms", "square_terms"])
    main_model = MainModel(
        linear_terms=np.array(test_data["linear_terms"]),
        square_terms=np.array(test_data["square_terms"]),
    )

    index_best_x = test_data["index_best_x"]
    x_accepted = test_data["history_x"][index_best_x]

    inputs_dict = {
        "history": history,
        "main_model": main_model,
        "model_improving_points": np.array(test_data["model_improving_points"]),
        "model_indices": np.array(test_data["model_indices"]),
        "x_accepted": np.array(x_accepted),
        "n_modelpoints": n_modelpoints,
        "delta": test_data["delta"],
        "criterion": criterion,
        "lower_bounds": None,
        "upper_bounds": None,
    }

    expected_dict = {
        "model_indices": test_data["model_indices_expected"],
        "history_x": test_data["history_x_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture()
def data_get_interpolation_matrices_residual_model():
    test_data = read_yaml(
        FIXTURES_DIR / "get_interpolation_matrices_residual_model.yaml"
    )

    history = LeastSquaresHistory()
    history_x = np.array(test_data["history_x"])
    history.add_entries(history_x, np.zeros(history_x.shape))

    n_params = 3
    n_maxinterp = 2 * n_params + 1
    n_modelpoints = 7

    inputs_dict = {
        "history": history,
        "x_accepted": np.array(test_data["x_accepted"]),
        "model_indices": np.array(test_data["model_indices"]),
        "delta": test_data["delta"],
        "c2": 10,
        "theta2": 1e-4,
        "n_maxinterp": n_maxinterp,
    }

    expected_dict = {
        "x_sample_monomial_basis": np.array(
            test_data["x_sample_monomial_basis_expected"]
        )[: n_params + 1, : n_params + 1],
        "monomial_basis": np.array(test_data["monomial_basis_expected"])[
            :n_modelpoints
        ],
        "basis_null_space": test_data["basis_null_space_expected"],
        "lower_triangular": np.array(test_data["lower_triangular_expected"])[
            :, n_params + 1 : n_maxinterp
        ],
        "n_modelpoints": test_data["n_modelpoints_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture(params=["4", "7"])
def data_evaluate_residual_model(request):
    test_data = read_yaml(FIXTURES_DIR / f"interpolate_f_iter_{request.param}.yaml")

    history = LeastSquaresHistory()
    history.add_entries(
        np.array(test_data["history_x"]),
        np.array(test_data["history_criterion"]),
    )

    ResidualModel = namedtuple(
        "ResidualModel", ["intercepts", "linear_terms", "square_terms"]
    )
    residual_model = ResidualModel(
        intercepts=np.array(test_data["residuals"]),
        linear_terms=np.array(test_data["linear_terms_residual_model"]),
        square_terms=np.array(test_data["square_terms_residual_model"]),
    )

    x_accepted = np.array(test_data["x_accepted"])
    model_indices = np.array(test_data["model_indices"])
    n_modelpoints = test_data["n_modelpoints"]
    delta_old = test_data["delta_old"]

    center_info = {"x": x_accepted, "radius": delta_old}
    centered_xs = history.get_centered_xs(
        center_info, index=model_indices[:n_modelpoints]
    )

    center_info = {"residuals": residual_model.intercepts}
    centered_residuals = history.get_centered_residuals(
        center_info, index=model_indices
    )

    inputs_dict = {
        "centered_xs": centered_xs,
        "centered_residuals": centered_residuals,
        "residual_model": residual_model,
    }

    expected_dict = {
        "y_residuals": test_data["f_interpolated_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture()
def data_fit_residual_model():
    test_data = read_yaml(FIXTURES_DIR / "get_coefficients_residual_model.yaml")

    n_params = 3
    n_maxinterp = 2 * n_params + 1
    n_modelpoints = 7

    inputs_dict = {
        "m_mat": np.array(test_data["x_sample_monomial_basis"])[
            : n_params + 1, : n_params + 1
        ],
        "n_mat": np.array(test_data["monomial_basis"])[:n_modelpoints],
        "z_mat": np.array(test_data["basis_null_space"]),
        "n_z_mat": np.array(test_data["lower_triangular"])[
            :, n_params + 1 : n_maxinterp
        ],
        "y_residuals": np.array(test_data["f_interpolated"]),
        "n_modelpoints": test_data["n_modelpoints"],
    }

    expected_coefficients_dict = {
        "linear_terms": np.array(test_data["linear_terms_expected"]).T,
        "square_terms": np.array(test_data["square_terms_expected"]),
    }

    return inputs_dict, expected_coefficients_dict


# ======================================================================================
# Test cases
# ======================================================================================


@pytest.mark.skip(reason="refactoring")
def test_update_initial_residual_model(data_update_initial_residual_model):
    inputs, residual_model_expected = data_update_initial_residual_model

    residual_model_out = create_initial_residual_model(**inputs)

    aaae(residual_model_out["intercepts"], residual_model_expected["intercepts"])
    aaae(residual_model_out["linear_terms"], residual_model_expected["linear_terms"])


def test_update_residual_model(data_update_residual_model):
    inputs, expected = data_update_residual_model

    residual_model_out = update_residual_model(**inputs)

    aaae(
        residual_model_out.linear_terms,
        expected["linear_terms"],
    )
    aaae(
        residual_model_out.square_terms,
        expected["square_terms"],
    )


def test_update_main_from_residual_model(data_update_main_from_residual_model):
    residual_model, main_model_expected = data_update_main_from_residual_model

    main_model_out = create_main_from_residual_model(
        residual_model, multiply_square_terms_with_intercepts=True
    )

    aaae(
        main_model_out.linear_terms,
        main_model_expected.linear_terms,
    )
    aaae(
        main_model_out.square_terms,
        main_model_expected.square_terms,
        decimal=3,
    )


def test_update_residual_model_with_new_accepted_x(
    data_update_residual_model_with_new_accepted_x,
):
    (
        inputs,
        residual_model_expected,
    ) = data_update_residual_model_with_new_accepted_x

    residual_model_out = update_residual_model_with_new_accepted_x(**inputs)

    aaae(residual_model_out.intercepts, residual_model_expected.intercepts)
    aaae(residual_model_out.linear_terms, residual_model_expected.linear_terms)


@pytest.mark.xfail(reason="Known rounding differences between C and Python.")
def test_update_main_model_with_new_accepted_x(
    data_update_main_model_with_new_accepted_x,
):
    (
        inputs,
        main_model_expected,
    ) = data_update_main_model_with_new_accepted_x

    main_model_out = update_main_model_with_new_accepted_x(**inputs)

    aaae(main_model_out.linear_terms, main_model_expected.linear_terms)


def test_find_affine_points(data_find_affine_points):
    inputs, expected = data_find_affine_points

    (
        model_improving_points_out,
        model_indices_out,
        n_modelpoints_out,
        project_x_onto_null_out,
    ) = find_affine_points(**inputs)

    aaae(
        model_improving_points_out,
        expected["model_improving_points"],
    )
    aaae(model_indices_out, expected["model_indices"])
    assert np.allclose(n_modelpoints_out, expected["n_modelpoints"])
    assert np.allclose(project_x_onto_null_out, True)


def test_add_points_until_main_model_fully_linear(
    data_add_points_until_main_model_fully_linear,
):
    inputs, expected = data_add_points_until_main_model_fully_linear
    n = 3

    (
        history_out,
        model_indices_out,
    ) = add_geomtery_points_to_make_main_model_fully_linear(
        **inputs, n_cores=1, batch_evaluator=joblib_batch_evaluator
    )

    aaae(model_indices_out, expected["model_indices"])
    for index_added in range(n - inputs["n_modelpoints"], 0, -1):
        aaae(
            history_out.get_xs(index=-index_added),
            expected["history_x"][-index_added],
        )


def test_get_interpolation_matrices_residual_model(
    data_get_interpolation_matrices_residual_model,
):
    inputs, expected = data_get_interpolation_matrices_residual_model
    (
        x_sample_monomial_basis,
        monomial_basis,
        basis_null_space,
        lower_triangular,
        n_modelpoints,
    ) = get_feature_matrices_residual_model(**inputs)

    aaae(x_sample_monomial_basis, expected["x_sample_monomial_basis"])
    aaae(monomial_basis, expected["monomial_basis"])
    aaae(basis_null_space, expected["basis_null_space"])
    aaae(lower_triangular, expected["lower_triangular"])
    assert np.allclose(n_modelpoints, expected["n_modelpoints"])


def test_evaluate_residual_model(data_evaluate_residual_model):
    inputs, expected = data_evaluate_residual_model
    y_residuals = evaluate_residual_model(**inputs)

    aaae(y_residuals, expected["y_residuals"])


def test_fit_residual_model(data_fit_residual_model):
    inputs, expected_coefficients = data_fit_residual_model

    coefficients_to_add = fit_residual_model(**inputs)

    aaae(
        coefficients_to_add["linear_terms"],
        expected_coefficients["linear_terms"],
    )
    aaae(
        coefficients_to_add["square_terms"],
        expected_coefficients["square_terms"],
    )
