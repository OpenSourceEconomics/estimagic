"""Tests for the logit example."""

import numpy as np
from estimagic.examples.logit import logit_derivative, logit_hessian, logit_loglike
from numpy.testing import assert_array_almost_equal as aaae


def test_logit_loglike(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected_value = logit_object.loglike(x)
    expected_contribs = logit_object.loglikeobs(x)
    calculated = logit_loglike(**logit_inputs)

    assert np.allclose(calculated["value"], expected_value)
    aaae(calculated["contributions"], expected_contribs)


def test_logit_derivative(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected = {
        "value": logit_object.score(x),
        "contributions": logit_object.score_obs(x),
    }

    calculated = logit_derivative(**logit_inputs)

    for key, val in expected.items():
        aaae(calculated[key], val)


def test_logit_hessian(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected = logit_object.hessian(x)
    calculated = logit_hessian(**logit_inputs)
    aaae(calculated, expected)
