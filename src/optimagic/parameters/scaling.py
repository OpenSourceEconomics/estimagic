from dataclasses import dataclass
from typing import Literal, TypedDict

from typing_extensions import NotRequired

from optimagic.exceptions import InvalidScalingError


@dataclass(frozen=True)
class ScalingOptions:
    """Scaling options in optimization problems.

    Attributes:
        method: The method used for scaling. Can be "start_values" or "bounds". Default
            is "start_values".
        clipping_value: The minimum value to which elements are clipped to avoid
            division by zero. Must be a positive number. Default is 0.1.
        magnitude: A factor by which the scaled parameters are multiplied to adjust
            their magnitude. Must be a positive number. Default is 1.0.

    Raises:
        InvalidScalingError: If scaling options cannot be processed, e.g. because they
            do not have the correct type.

    """

    method: Literal["start_values", "bounds"] = "start_values"
    clipping_value: float = 0.1
    magnitude: float = 1.0

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class ScalingOptionsDict(TypedDict):
    method: NotRequired[Literal["start_values", "bounds"]]
    clipping_value: NotRequired[float]
    magnitude: NotRequired[float]


def pre_process_scaling(
    scaling: bool | ScalingOptions | ScalingOptionsDict | None,
) -> ScalingOptions | None:
    """Convert all valid types of scaling options to optimagic.ScalingOptions.

    This just harmonizes multiple ways of specifying scaling options into a single
    format. It performs runtime type checks, but it does not check whether scaling
    options are consistent with other option choices.

    Args:
        scaling: The user provided scaling options.

    Returns:
        The scaling options in the optimagic format.

    Raises:
        InvalidScalingError: If scaling options cannot be processed, e.g. because they
            do not have the correct type.

    """
    if isinstance(scaling, bool):
        scaling = ScalingOptions() if scaling else None
    elif isinstance(scaling, ScalingOptions) or scaling is None:
        pass
    else:
        try:
            scaling = ScalingOptions(**scaling)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if isinstance(e, InvalidScalingError):
                raise e
            raise InvalidScalingError(
                f"Invalid scaling options of type: {type(scaling)}. Scaling options "
                "must be of type optimagic.ScalingOptions, a dictionary with a subset "
                "of the keys {'method', 'clipping_value', 'magnitude'}, None, or a "
                "boolean."
            ) from e

    return scaling


def _validate_attribute_types_and_values(options: ScalingOptions) -> None:
    if options.method not in ("start_values", "bounds"):
        raise InvalidScalingError(
            f"Invalid scaling method: {options.method}. Valid methods are "
            "'start_values' and 'bounds'."
        )

    if (
        not isinstance(options.clipping_value, int | float)
        or options.clipping_value <= 0
    ):
        raise InvalidScalingError(
            f"Invalid clipping value: {options.clipping_value}. Clipping value "
            "must be a positive number."
        )

    if not isinstance(options.magnitude, int | float) or options.magnitude <= 0:
        raise InvalidScalingError(
            f"Invalid scaling magnitude: {options.magnitude}. Scaling magnitude "
            "must be a positive number."
        )
