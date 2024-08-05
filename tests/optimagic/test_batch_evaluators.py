import itertools
import warnings

import pytest
from optimagic.batch_evaluators import process_batch_evaluator

batch_evaluators = ["joblib"]

n_core_list = [1, 2]

test_cases = list(itertools.product(batch_evaluators, n_core_list))


def double(x):
    return 2 * x


def buggy_func(x):  # noqa: ARG001
    raise AssertionError()


def add_x_and_y(x, y):
    return x + y


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_without_exceptions(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    calculated = batch_evaluator(
        func=double,
        arguments=list(range(10)),
        n_cores=n_cores,
    )

    expected = list(range(0, 20, 2))

    assert calculated == expected


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_unhandled_exceptions(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    with pytest.raises(AssertionError):
        batch_evaluator(
            func=buggy_func,
            arguments=list(range(10)),
            n_cores=n_cores,
            error_handling="raise",
        )


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_handled_exceptions(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        calculated = batch_evaluator(
            func=buggy_func,
            arguments=list(range(10)),
            n_cores=n_cores,
            error_handling="continue",
        )

        for calc in calculated:
            assert isinstance(calc, str)


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_list_unpacking(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    calculated = batch_evaluator(
        func=add_x_and_y,
        arguments=[(1, 2), (3, 4)],
        n_cores=n_cores,
        unpack_symbol="*",
    )
    expected = [3, 7]
    assert calculated == expected


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_dict_unpacking(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    calculated = batch_evaluator(
        func=add_x_and_y,
        arguments=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        n_cores=n_cores,
        unpack_symbol="**",
    )
    expected = [3, 7]
    assert calculated == expected


def test_get_batch_evaluator_invalid_value():
    with pytest.raises(ValueError):
        process_batch_evaluator("bla")


def test_get_batch_evaluator_invalid_type():
    with pytest.raises(TypeError):
        process_batch_evaluator(3)


def test_get_batch_evaluator_with_callable():
    assert callable(process_batch_evaluator(lambda x: x))
