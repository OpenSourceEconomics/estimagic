from typing import get_args, get_type_hints

from optimagic.differentiation.numdiff_options import NumdiffOptions, NumdiffOptionsDict
from optimagic.optimization.multistart_options import (
    MultistartOptions,
    MultistartOptionsDict,
)
from optimagic.parameters.scaling import ScalingOptions, ScalingOptionsDict


def assert_attributes_and_type_hints_are_equal(dataclass, typed_dict):
    """Test that dataclass and typed_dict have same attributes and types.

    This assertion purposefully ignores that all type hints in the typed dict are
    wrapped by typing.NotRequired.

    As there is no easy way to *not* read the NotRequired types in 3.10, we need to
    activate include_extras=True to get the NotRequired types in Python 3.11 and
    above. Once we drop support for Python 3.10, we can remove the
    include_extras=True argument and the removal of the NotRequired types.

    Args:
        dataclass: An instance of a dataclass
        typed_dict: An instance of a typed dict

    """
    types_from_dataclass = get_type_hints(dataclass)
    types_from_typed_dict = get_type_hints(typed_dict, include_extras=True)
    types_from_typed_dict = {
        # Remove typing.NotRequired from the types
        k: get_args(v)[0]
        for k, v in types_from_typed_dict.items()
    }
    assert types_from_dataclass == types_from_typed_dict


def test_scaling_options_and_dict_have_same_attributes():
    assert_attributes_and_type_hints_are_equal(ScalingOptions, ScalingOptionsDict)


def test_multistart_options_and_dict_have_same_attributes():
    assert_attributes_and_type_hints_are_equal(MultistartOptions, MultistartOptionsDict)


def test_numdiff_options_and_dict_have_same_attributes():
    assert_attributes_and_type_hints_are_equal(NumdiffOptions, NumdiffOptionsDict)
