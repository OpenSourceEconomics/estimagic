import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.differentiation.derivatives import Evals
from optimagic.differentiation.finite_differences import jacobian
from optimagic.differentiation.generate_steps import Steps


@pytest.fixture()
def jacobian_inputs():
    """Very contrived test case for finite difference formulae with linear function."""
    steps_pos = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])
    steps = Steps(pos=steps_pos, neg=-steps_pos)

    jac1 = (np.arange(1, 13)).reshape(3, 4)
    jac2 = jac1 * 1.1

    evals_pos1 = jac1 @ (np.zeros((4, 4)) + np.eye(4) * 0.1)
    evals_pos2 = jac2 @ (np.zeros((4, 4)) + np.eye(4) * 0.2)
    evals_neg1 = jac1 @ (np.zeros((4, 4)) - np.eye(4) * 0.1)
    evals_neg2 = jac2 @ (np.zeros((4, 4)) - np.eye(4) * 0.2)
    evals = Evals(
        pos=np.array([evals_pos1, evals_pos2]), neg=np.array([evals_neg1, evals_neg2])
    )

    expected_jac = np.array([jac1, jac2])

    f0 = np.zeros(3)

    out = {"evals": evals, "steps": steps, "f0": f0, "expected_jac": expected_jac}
    return out


methods = ["forward", "backward", "central"]


@pytest.mark.parametrize("method", methods)
def test_jacobian_finite_differences(jacobian_inputs, method):
    expected_jac = jacobian_inputs.pop("expected_jac")
    calculated_jac = jacobian(**jacobian_inputs, method=method)
    aaae(calculated_jac, expected_jac)
