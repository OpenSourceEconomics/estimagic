import pytest
from optimagic.decorators import catch, mark_minimizer, unpack


def test_catch_at_defaults():
    @catch
    def f():
        raise ValueError

    with pytest.warns(UserWarning):
        assert f() is None

    @catch
    def g():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        g()


def test_catch_with_reraise():
    @catch(reraise=True)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()


def test_unpack_decorator_none():
    @unpack(symbol=None)
    def f(x):
        return x

    assert f(3) == 3


def test_unpack_decorator_one_star():
    @unpack(symbol="*")
    def f(x, y):
        return x + y

    assert f((3, 4)) == 7


def test_unpack_decorator_two_stars():
    @unpack(symbol="**")
    def f(x, y):
        return x + y

    assert f({"x": 3, "y": 4}) == 7


def test_mark_minimizer_decorator():
    @mark_minimizer(name="bla")
    def minimize_stupid():
        pass

    assert hasattr(minimize_stupid, "_algorithm_info")
    assert minimize_stupid._algorithm_info.name == "bla"


def test_mark_minimizer_direct_call():
    def minimize_stupid():
        pass

    first = mark_minimizer(minimize_stupid, name="bla")
    second = mark_minimizer(minimize_stupid, name="blubb")

    assert first._algorithm_info.name == "bla"
    assert second._algorithm_info.name == "blubb"


INVALID_TYPES = [
    {"name": None},
    {"name": [1, 2, 3]},
    {"needs_scaling": 25},
    {"disable_history": 28},
    {"is_available": 30},
]


@pytest.mark.parametrize("kwargs", INVALID_TYPES)
def test_mark_minimizer_invalid_argument_types(kwargs):
    kwargs = {"name": "bla", **kwargs}
    with pytest.raises(TypeError):

        @mark_minimizer(**kwargs)
        def minimize_stupid():
            pass


def test_mark_minimizer_invalid_argument_primary_criterion_entry():
    kwargs = {"name": "bla", "primary_criterion_entry": "bla"}
    with pytest.raises(ValueError):

        @mark_minimizer(**kwargs)
        def minimize_stupid():
            pass
