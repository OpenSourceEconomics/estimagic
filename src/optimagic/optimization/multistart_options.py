from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.deprecations import replace_and_warn_about_deprecated_multistart_options
from optimagic.exceptions import InvalidMultistartError
from optimagic.typing import PyTree

# ======================================================================================
# Public Options
# ======================================================================================


@dataclass(frozen=True)
class MultistartOptions:
    """Multistart options in optimization problems.

    Attributes:
        n_samples: The number of points at which the objective function is evaluated
            during the exploration phase. If None, n_samples is set to 100 times the
            number of parameters.
        stopping_maxopt: The maximum number of local optimizations to run. Defaults to
            10% of n_samples. This number may not be reached if multistart converges
            earlier.
        sampling_distribution: The distribution from which the exploration sample is
            drawn. Allowed are "uniform" and "triangular". Defaults to "uniform".
        sampling_method: The method used to draw the exploration sample. Allowed are
            "sobol", "random", "halton", and "latin_hypercube". Defaults to "random".
        sample: A sequence of PyTrees or None. If None, a sample is drawn from the
            sampling distribution.
        mixing_weight_method: The method used to determine the mixing weight, i,e, how
            start parameters for local optimizations are calculated. Allowed are
            "tiktak" and "linear", or a custom callable. Defaults to "tiktak".
        mixing_weight_bounds: The lower and upper bounds for the mixing weight.
            Defaults to (0.1, 0.995).
        convergence_max_discoveries: The maximum number of discoveries for convergence.
            Determines after how many re-descoveries of the currently best local
            optima the multistart algorithm stops. Defaults to 2.
        convergence_xtol_rel: The relative tolerance in parameters
            for convergence. Determines the maximum relative distance two parameter
            vecctors can have to be considered equal. Defaults to 0.01.
        n_cores: The number of cores to use for parallelization. Defaults to 1.
        batch_evaluator: The evaluator to use for batch evaluation. Allowed are "joblib"
            and "pathos", or a custom callable.
        batch_size: The batch size for batch evaluation. Must be larger than n_cores
            or None.
        seed: The seed for the random number generator.
        error_handling: The error handling for exploration and optimization errors.
            Allowed are "raise" and "continue".

    Raises:
        InvalidMultistartError: If the multistart options cannot be processed, e.g.
            because they do not have the correct type.

    """

    n_samples: int | None = None
    stopping_maxopt: int | None = None
    sampling_distribution: Literal["uniform", "triangular"] = "uniform"
    sampling_method: Literal["sobol", "random", "halton", "latin_hypercube"] = "random"
    sample: Sequence[PyTree] | None = None
    mixing_weight_method: (
        Literal["tiktak", "linear"] | Callable[[int, int, float, float], float]
    ) = "tiktak"
    mixing_weight_bounds: tuple[float, float] = (0.1, 0.995)
    convergence_xtol_rel: float | None = None
    convergence_max_discoveries: int = 2
    n_cores: int = 1
    # TODO: Add more informative type hint for batch_evaluator
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib"  # type: ignore
    batch_size: int | None = None
    seed: int | np.random.Generator | None = None
    error_handling: Literal["raise", "continue"] | None = None
    # Deprecated attributes
    share_optimization: float | None = None
    convergence_relative_params_tolerance: float | None = None
    optimization_error_handling: Literal["raise", "continue"] | None = None
    exploration_error_handling: Literal["raise", "continue"] | None = None

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class MultistartOptionsDict(TypedDict):
    n_samples: NotRequired[int | None]
    stopping_maxopt: NotRequired[int | None]
    sampling_distribution: NotRequired[Literal["uniform", "triangular"]]
    sampling_method: NotRequired[
        Literal["sobol", "random", "halton", "latin_hypercube"]
    ]
    sample: NotRequired[Sequence[PyTree] | None]
    mixing_weight_method: NotRequired[
        Literal["tiktak", "linear"] | Callable[[int, int, float, float], float]
    ]
    mixing_weight_bounds: NotRequired[tuple[float, float]]
    convergence_xtol_rel: NotRequired[float | None]
    convergence_max_discoveries: NotRequired[int]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[Literal["joblib", "pathos"] | Callable]  # type: ignore
    batch_size: NotRequired[int | None]
    seed: NotRequired[int | np.random.Generator | None]
    error_handling: NotRequired[Literal["raise", "continue"] | None]
    # Deprecated attributes
    share_optimization: NotRequired[float | None]
    convergence_relative_params_tolerance: NotRequired[float | None]
    optimization_error_handling: NotRequired[Literal["raise", "continue"] | None]
    exploration_error_handling: NotRequired[Literal["raise", "continue"] | None]


def pre_process_multistart(
    multistart: bool | MultistartOptions | MultistartOptionsDict | None,
) -> MultistartOptions | None:
    """Convert all valid types of multistart to a optimagic.MultistartOptions.

    This just harmonizes multiple ways of specifying multistart options into a single
    format. It performs runime type checks, but it does not check whether multistart
    options are consistent with other option choices.

    Args:
        multistart: The user provided multistart options.
        n_params: The number of parameters in the optimization problem.

    Returns:
        The multistart options in the optimagic format.

    Raises:
        InvalidMultistartError: If the multistart options cannot be processed, e.g.
            because they do not have the correct type.

    """
    if isinstance(multistart, bool):
        multistart = MultistartOptions() if multistart else None
    elif isinstance(multistart, MultistartOptions) or multistart is None:
        pass
    else:
        try:
            multistart = MultistartOptions(**multistart)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if isinstance(e, InvalidMultistartError):
                raise e
            raise InvalidMultistartError(
                f"Invalid multistart options of type: {type(multistart)}. Multistart "
                "options must be of type optimagic.MultistartOptions, a dictionary "
                "with valid keys, None, or a boolean."
            ) from e

    if multistart is not None:
        multistart = replace_and_warn_about_deprecated_multistart_options(multistart)
        # The replace and warn function cannot be typed due to circular imports, but
        # we know that the return type is MultistartOptions
        multistart = cast(MultistartOptions, multistart)

    return multistart


def _validate_attribute_types_and_values(options: MultistartOptions) -> None:
    if options.n_samples is not None and (
        not isinstance(options.n_samples, int) or options.n_samples < 1
    ):
        raise InvalidMultistartError(
            f"Invalid number of samples: {options.n_samples}. Number of samples "
            "must be a positive integer or None."
        )

    if options.stopping_maxopt is not None and (
        not isinstance(options.stopping_maxopt, int) or options.stopping_maxopt < 0
    ):
        raise InvalidMultistartError(
            f"Invalid number of optimizations: {options.stopping_maxopt}. Number of "
            "optimizations must be a positive integer or None."
        )

    if (
        options.n_samples is not None
        and options.stopping_maxopt is not None
        and options.n_samples < options.stopping_maxopt
    ):
        raise InvalidMultistartError(
            f"Invalid number of samples: {options.n_samples}. Number of samples "
            "must be at least as large as the number of optimizations."
        )

    if options.sampling_distribution not in ("uniform", "triangular"):
        raise InvalidMultistartError(
            f"Invalid sampling distribution: {options.sampling_distribution}. Sampling "
            f"distribution must be one of ('uniform', 'triangular')."
        )

    if options.sampling_method not in ("sobol", "random", "halton", "latin_hypercube"):
        raise InvalidMultistartError(
            f"Invalid sampling method: {options.sampling_method}. Sampling method "
            f"must be one of ('sobol', 'random', 'halton', 'latin_hypercube')."
        )

    if not isinstance(options.sample, Sequence | None):
        raise InvalidMultistartError(
            f"Invalid sample: {options.sample}. Sample must be a sequence of "
            "parameters."
        )

    if not callable(
        options.mixing_weight_method
    ) and options.mixing_weight_method not in ("tiktak", "linear"):
        raise InvalidMultistartError(
            f"Invalid mixing weight method: {options.mixing_weight_method}. Mixing "
            "weight method must be Callable or one of ('tiktak', 'linear')."
        )

    if (
        not isinstance(options.mixing_weight_bounds, tuple)
        or len(options.mixing_weight_bounds) != 2
        or not set(type(x) for x in options.mixing_weight_bounds) <= {int, float}
    ):
        raise InvalidMultistartError(
            f"Invalid mixing weight bounds: {options.mixing_weight_bounds}. Mixing "
            "weight bounds must be a tuple of two numbers."
        )

    if options.convergence_xtol_rel is not None and (
        not isinstance(options.convergence_xtol_rel, int | float)
        or options.convergence_xtol_rel < 0
    ):
        raise InvalidMultistartError(
            "Invalid relative params tolerance:"
            f"{options.convergence_xtol_rel}. Relative params "
            "tolerance must be a number."
        )

    if (
        not isinstance(options.convergence_max_discoveries, int | float)
        or options.convergence_max_discoveries < 1
    ):
        raise InvalidMultistartError(
            f"Invalid max discoveries: {options.convergence_max_discoveries}. Max "
            "discoveries must be a positive integer or infinity."
        )

    if not isinstance(options.n_cores, int) or options.n_cores < 1:
        raise InvalidMultistartError(
            f"Invalid number of cores: {options.n_cores}. Number of cores "
            "must be a positive integer."
        )

    if not callable(options.batch_evaluator) and options.batch_evaluator not in (
        "joblib",
        "pathos",
    ):
        raise InvalidMultistartError(
            f"Invalid batch evaluator: {options.batch_evaluator}. Batch evaluator "
            "must be a Callable or one of 'joblib' or 'pathos'."
        )

    if options.batch_size is not None and (
        not isinstance(options.batch_size, int) or options.batch_size < options.n_cores
    ):
        raise InvalidMultistartError(
            f"Invalid batch size: {options.batch_size}. Batch size "
            "must be a positive integer larger than n_cores, or None."
        )

    if not isinstance(options.seed, int | np.random.Generator | None):
        raise InvalidMultistartError(
            f"Invalid seed: {options.seed}. Seed "
            "must be an integer, a numpy random generator, or None."
        )

    if options.error_handling is not None and options.error_handling not in (
        "raise",
        "continue",
    ):
        raise InvalidMultistartError(
            f"Invalid error handling: {options.error_handling}. Error handling must be "
            "'raise' or 'continue'."
        )


# ======================================================================================
# Internal Options
# ======================================================================================


def _tiktak_weights(
    iteration: int, n_iterations: int, min_weight: float, max_weight: float
) -> float:
    return np.clip(np.sqrt(iteration / n_iterations), min_weight, max_weight)


def _linear_weights(
    iteration: int, n_iterations: int, min_weight: float, max_weight: float
) -> float:
    unscaled = iteration / n_iterations
    span = max_weight - min_weight
    return min_weight + unscaled * span


WEIGHT_FUNCTIONS = {
    "tiktak": _tiktak_weights,
    "linear": _linear_weights,
}


@dataclass(frozen=True)
class InternalMultistartOptions:
    """Multistart options used internally in optimagic.

    Compared to `MultistartOptions`, this data class has stricter types and combines
    some of the attributes. It is generated at runtime using a `MultistartOptions`
    instance and the function `get_internal_multistart_options_from_public`.

    """

    n_samples: int
    weight_func: Callable[[int, int], float]
    convergence_xtol_rel: float
    convergence_max_discoveries: int
    sampling_distribution: Literal["uniform", "triangular"]
    sampling_method: Literal["sobol", "random", "halton", "latin_hypercube"]
    sample: NDArray[np.float64] | None
    seed: int | np.random.Generator | None
    n_cores: int
    # TODO: Add more informative type hint for batch_evaluator
    batch_evaluator: Callable  # type: ignore
    batch_size: int
    error_handling: Literal["raise", "continue"]
    stopping_maxopt: int

    def __post_init__(self) -> None:
        must_be_at_least_1 = [
            "n_samples",
            "stopping_maxopt",
            "n_cores",
            "batch_size",
            "convergence_max_discoveries",
        ]

        for attr in must_be_at_least_1:
            if getattr(self, attr) < 1:
                raise InvalidMultistartError(f"{attr} must be at least 1.")

        if self.batch_size < self.n_cores:
            raise InvalidMultistartError("batch_size must be at least n_cores.")

        if self.convergence_xtol_rel < 0:
            raise InvalidMultistartError("convergence_xtol_rel must be at least 0.")


def get_internal_multistart_options_from_public(
    options: MultistartOptions,
    params: PyTree,
    params_to_internal: Callable[[PyTree], NDArray[np.float64]],
) -> InternalMultistartOptions:
    """Get internal multistart options from public multistart options.

    Args:
        options: The pre-processed multistart options.
        params: The parameters of the optimization problem.
        params_to_internal: A function that converts parameters to internal parameters.

    Returns:
        InternalMultistartOptions: The updated options with runtime defaults.

    """
    x = params_to_internal(params)

    if options.sample is not None:
        sample = np.array([params_to_internal(x) for x in list(options.sample)])
        n_samples = len(options.sample)
    else:
        sample = None
        n_samples = options.n_samples  # type: ignore

    batch_size = options.n_cores if options.batch_size is None else options.batch_size
    batch_evaluator = process_batch_evaluator(options.batch_evaluator)

    if callable(options.mixing_weight_method):
        weight_func = options.mixing_weight_method
    else:
        _weight_method = WEIGHT_FUNCTIONS[options.mixing_weight_method]

    weight_func = partial(
        _weight_method,
        min_weight=options.mixing_weight_bounds[0],
        max_weight=options.mixing_weight_bounds[1],
    )

    if n_samples is None:
        if options.stopping_maxopt is None:
            n_samples = 100 * len(x)
        else:
            n_samples = 10 * options.stopping_maxopt

    if options.share_optimization is None:
        share_optimization = 0.1
    else:
        share_optimization = options.share_optimization

    if options.stopping_maxopt is None:
        stopping_maxopt = max(1, int(share_optimization * n_samples))
    else:
        stopping_maxopt = options.stopping_maxopt

    # Set defaults resulting from deprecated attributes
    if options.error_handling is not None:
        error_handling = options.error_handling
    else:
        error_handling = "continue"

    if options.convergence_xtol_rel is not None:
        convergence_xtol_rel = options.convergence_xtol_rel
    else:
        convergence_xtol_rel = 0.01

    return InternalMultistartOptions(
        # Attributes taken directly from MultistartOptions
        convergence_max_discoveries=options.convergence_max_discoveries,
        n_cores=options.n_cores,
        sampling_distribution=options.sampling_distribution,
        sampling_method=options.sampling_method,
        seed=options.seed,
        # Updated attributes
        sample=sample,
        n_samples=n_samples,
        weight_func=weight_func,
        error_handling=error_handling,
        convergence_xtol_rel=convergence_xtol_rel,
        stopping_maxopt=stopping_maxopt,
        batch_evaluator=batch_evaluator,
        batch_size=batch_size,
    )
