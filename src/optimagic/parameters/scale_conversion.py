from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic.parameters.scaling import ScalingOptions
from optimagic.parameters.space_conversion import InternalParams


@dataclass(frozen=True)
class ScaleConverter:
    factor: NDArray[np.float64] | None
    offset: NDArray[np.float64] | None

    def params_to_internal(self, vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale a parameter vector from external scale to internal one."""
        if self.offset is not None:
            vec = vec - self.offset
        if self.factor is not None:
            vec = vec / self.factor
        return vec

    def params_from_internal(self, vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale a parameter vector from internal scale to external one."""
        if self.factor is not None:
            vec = vec * self.factor
        if self.offset is not None:
            vec = vec + self.offset
        return vec

    def derivative_to_internal(
        self, derivative: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Scale a derivative vector from external scale to internal one."""
        if self.factor is not None:
            derivative = derivative * self.factor
        return derivative

    def derivative_from_internal(
        self, derivative: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Scale a derivative vector from internal scale to external one."""
        if self.factor is not None:
            derivative = derivative / self.factor
        return derivative


def get_scale_converter(
    internal_params: InternalParams,
    scaling: ScalingOptions | None,
) -> tuple[ScaleConverter, InternalParams]:
    """Get a converter between scaled and unscaled parameters.

    Args:
        internal_params: NamedTuple of internal and possibly reparametrized but not yet
            scaled parameter values and bounds.
        scaling: Scaling options. If None, no scaling is performed.

    Returns:
        ScaleConverter: Dataclass with methods to convert between scaled and unscaled
            internal parameters and derivatives.
        InternalParams: Dataclass with internal parameter values and bounds.

    """
    # fast path
    if scaling is None:
        return ScaleConverter(factor=None, offset=None), internal_params

    factor, offset = calculate_scaling_factor_and_offset(
        internal_params=internal_params,
        options=scaling,
    )

    converter = ScaleConverter(factor=factor, offset=offset)

    if internal_params.soft_lower_bounds is not None:
        _soft_lower = converter.params_to_internal(internal_params.soft_lower_bounds)
    else:
        _soft_lower = None

    if internal_params.soft_upper_bounds is not None:
        _soft_upper = converter.params_to_internal(internal_params.soft_upper_bounds)
    else:
        _soft_upper = None

    params = InternalParams(
        values=converter.params_to_internal(internal_params.values),
        lower_bounds=converter.params_to_internal(internal_params.lower_bounds),
        upper_bounds=converter.params_to_internal(internal_params.upper_bounds),
        names=internal_params.names,
        soft_lower_bounds=_soft_lower,
        soft_upper_bounds=_soft_upper,
    )

    return converter, params


def calculate_scaling_factor_and_offset(
    internal_params: InternalParams,
    options: ScalingOptions,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    x = internal_params.values
    lower_bounds = internal_params.lower_bounds
    upper_bounds = internal_params.upper_bounds

    if options.method == "start_values":
        raw_factor = np.clip(np.abs(x), options.clipping_value, np.inf)
        scaling_offset = None
    elif options.method == "bounds":
        raw_factor = upper_bounds - lower_bounds
        scaling_offset = lower_bounds
    else:
        raise ValueError(f"Invalid scaling method: {options.method}")

    scaling_factor = raw_factor / options.magnitude

    return scaling_factor, scaling_offset
