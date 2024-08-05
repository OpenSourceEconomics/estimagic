from functools import partial

import numpy as np

from optimagic.benchmarking.cartis_roberts import CARTIS_ROBERTS_PROBLEMS
from optimagic.benchmarking.more_wild import MORE_WILD_PROBLEMS
from optimagic.benchmarking.noise_distributions import NOISE_DISTRIBUTIONS
from optimagic.utilities import get_rng


def get_benchmark_problems(
    name,
    *,
    additive_noise=False,
    additive_noise_options=None,
    multiplicative_noise=False,
    multiplicative_noise_options=None,
    scaling=False,
    scaling_options=None,
    seed=None,
    exclude=None,
):
    """Get a dictionary of test problems for a benchmark.

    Args:
        name (str): The name of the set of test problems. Currently "more_wild"
            is the only supported one.
        additive_noise (bool): Whether to add additive noise to the problem.
            Default False.
        additive_noise_options (dict or None): Specifies the amount and distribution
            of the addititve noise added to the problem. Has the entries:
            - distribition (str): One of "normal", "gumbel", "uniform", "logistic",
            "laplace". Default "normal".
            - std (float): The standard deviation of the noise. This works for all
            distributions, even if those distributions are normally not specified
            via a standard deviation (e.g. uniform).
            - correlation (float): Number between 0 and 1 that specifies the auto
            correlation of the noise.
        multiplicative_noise (bool): Whether to add multiplicative noise to the problem.
            Default False.
        multiplicative_noise_options (dict or None): Specifies the amount and
            distribition of the multiplicative noise added to the problem. Has entries:
            - distribition (str): One of "normal", "gumbel", "uniform", "logistic",
            "laplace". Default "normal".
            - std (float): The standard deviation of the noise. This works for all
            distributions, even if those distributions are normally not specified
            via a standard deviation (e.g. uniform).
            - correlation (float): Number between 0 and 1 that specifies the auto
            correlation of the noise.
            - clipping_value (float): A non-negative float. Multiplicative noise
            becomes zero if the function value is zero. To avoid this, we do not
            implement multiplicative noise as `f_noisy = f * epsilon` but by
            `f_noisy` = f + (epsilon - 1) * f_clipped` where f_clipped is bounded
            away from zero from both sides by the clipping value.
        scaling (bool): Whether the parameter space of the problem should be rescaled.
        scaling_options (dict): Dict containing the keys "min_scale", and "max_scale".
            If scaling is True, the parameters the optimizer sees are the standard
            parameters multiplied by np.linspace(min_scale, max_scale, len(params)).
            If min_scale and max_scale have very different orders of magnitude, the
            problem becomes harder to solve for many optimizers.
        seed (Union[None, int, numpy.random.Generator]): If seed is None or int the
            numpy.random.default_rng is used seeded with seed. If seed is already a
            Generator instance then that instance is used.
        exclude (str or List): Problems to exclude.


    Returns:
        dict: Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might contain information about the test problem.

    """
    if exclude is None:
        exclude = {}
    elif isinstance(exclude, str):
        exclude = [exclude]
    else:
        exclude = set(exclude)

    rng = get_rng(seed)
    raw_problems = _get_raw_problems(name)

    raw_problems = {k: v for k, v in raw_problems.items() if k not in exclude}

    is_noisy = bool(additive_noise or multiplicative_noise)

    if additive_noise:
        additive_options = _process_noise_options(additive_noise_options, False)
    else:
        additive_options = None

    if multiplicative_noise:
        multiplicative_options = _process_noise_options(
            multiplicative_noise_options, True
        )
    else:
        multiplicative_options = None

    if scaling:
        scaling_options = scaling_options if scaling_options is not None else {}
        scaling_options = {"min_scale": 0.1, "max_scale": 10, **scaling_options}
    else:
        scaling_options = None

    problems = {}
    for prob_name, specification in raw_problems.items():
        inputs = _create_problem_inputs(
            specification,
            additive_options=additive_options,
            multiplicative_options=multiplicative_options,
            scaling_options=scaling_options,
            rng=rng,
        )

        problems[prob_name] = {
            "inputs": inputs,
            "noise_free_fun": specification["fun"],
            "solution": _create_problem_solution(
                specification, scaling_options=scaling_options
            ),
            "noisy": is_noisy,
            "info": specification.get("info", {}),
            "start_criterion": specification["start_criterion"],
        }

    return problems


def _get_raw_problems(name):
    if name == "more_wild":
        raw_problems = MORE_WILD_PROBLEMS
    elif name == "cartis_roberts":
        raw_problems = CARTIS_ROBERTS_PROBLEMS
    elif name == "example":
        subset = {
            "rosenbrock_good_start",
            "helical_valley_good_start",
            "powell_singular_good_start",
            "freudenstein_roth_good_start",
            "bard_good_start",
            "box_3d",
            "brown_dennis_good_start",
            "chebyquad_6",
            "bdqrtic_8",
            "mancino_5_good_start",
        }
        raw_problems = {k: v for k, v in MORE_WILD_PROBLEMS.items() if k in subset}
    elif name == "estimagic":
        subset_mw = {
            "cube_8",
            "chebyquad_6",
            "bdqrtic_8",
            "linear_full_rank_bad_start",
            "chebyquad_7",
            "osborne_two_bad_start",
            "bdqrtic_10",
            "bdqrtic_11",
            "heart_eight_bad_start",
            "mancino_5_bad_start",
            "chebyquad_8",
            "cube_6",
            "cube_5",
            "bdqrtic_12",
            "chebyquad_10",
            "chebyquad_9",
            "chebyquad_11",
            "mancino_8",
            "mancino_10",
            "mancino_12_bad_start",
        }
        subset_cr = {
            "hatfldg",
            "bratu_3d",
            "cbratu_2d",
            "chnrsbne",
            "bratu_2d",
            "vardimne",
            "penalty_1",
            "arglale",
            "arglble",
        }
        subset_add_steps = {
            "rosenbrock_good_start",
            "cube_5",
            "chebyquad_10",
        }
        raw_problems = {}
        for k, v in MORE_WILD_PROBLEMS.items():
            if k in subset_mw:
                raw_problems[k] = v
            if k in subset_add_steps:
                problem = v.copy()
                raw_func = problem["fun"]

                problem["fun"] = partial(_step_func, raw_func=raw_func)
                raw_problems[f"{k}_with_steps"] = problem

        for k, v in CARTIS_ROBERTS_PROBLEMS.items():
            if k in subset_cr:
                raw_problems[k] = v

    else:
        raise NotImplementedError()
    return raw_problems


def _step_func(x, raw_func):
    return raw_func(x.round(3))


def _create_problem_inputs(
    specification, additive_options, multiplicative_options, scaling_options, rng
):
    _x = np.array(specification["start_x"])

    if scaling_options is not None:
        scaling_factor = _get_scaling_factor(_x, scaling_options)
        _x = _x * scaling_factor
    else:
        scaling_factor = None

    _criterion = partial(
        _internal_criterion_template,
        criterion=specification["fun"],
        additive_options=additive_options,
        multiplicative_options=multiplicative_options,
        scaling_factor=scaling_factor,
        rng=rng,
    )

    inputs = {"fun": _criterion, "params": _x}
    return inputs


def _create_problem_solution(specification, scaling_options):
    _solution_x = specification.get("solution_x")
    if _solution_x is None:
        _solution_x = np.array(specification["start_x"]) * np.nan
    elif isinstance(_solution_x, list):
        _solution_x = np.array(_solution_x)
    _params = _solution_x
    if scaling_options is not None:
        _params = _params * _get_scaling_factor(_params, scaling_options)

    _value = specification["solution_criterion"]

    solution = {
        "params": _params,
        "value": _value,
    }
    return solution


def _get_scaling_factor(x, options):
    return np.linspace(options["min_scale"], options["max_scale"], len(x))


def _internal_criterion_template(
    params, criterion, additive_options, multiplicative_options, scaling_factor, rng
):
    if scaling_factor is not None:
        params = params / scaling_factor

    critval = criterion(params)

    noise = _get_combined_noise(
        critval,
        additive_options=additive_options,
        multiplicative_options=multiplicative_options,
        rng=rng,
    )

    noisy_critval = critval + noise

    if isinstance(noisy_critval, np.ndarray):
        out = {
            "root_contributions": noisy_critval,
            "value": noisy_critval @ noisy_critval,
        }
    else:
        out = noisy_critval

    return out


def _get_combined_noise(fval, additive_options, multiplicative_options, rng):
    size = len(np.atleast_1d(fval))
    if multiplicative_options is not None:
        options = multiplicative_options.copy()
        std = options.pop("std")
        clipval = options.pop("clipping_value")
        scaled_std = std * _clip_away_from_zero(fval, clipval)
        multiplicative_noise = _sample_from_distribution(
            **options, std=scaled_std, size=size, rng=rng
        )
    else:
        multiplicative_noise = 0

    if additive_options is not None:
        additive_noise = _sample_from_distribution(
            **additive_options, size=size, rng=rng
        )
    else:
        additive_noise = 0

    return multiplicative_noise + additive_noise


def _sample_from_distribution(distribution, mean, std, size, rng, correlation=0):
    sample = NOISE_DISTRIBUTIONS[distribution](size=size, rng=rng)
    dim = size if isinstance(size, int) else size[1]
    if correlation != 0 and dim > 1:
        chol = np.linalg.cholesky(np.diag(np.ones(dim) - correlation) + correlation)
        sample = (chol @ sample.T).T
        sample = sample / sample.std()
    sample *= std
    sample += mean
    return sample


def _process_noise_options(options, is_multiplicative):
    options = {} if options is None else options

    defaults = {"std": 0.01, "distribution": "normal", "correlation": 0, "mean": 0}
    if is_multiplicative:
        defaults["clipping_value"] = 1

    processed = {
        **defaults,
        **options,
    }

    distribution = processed["distribution"]
    if distribution not in NOISE_DISTRIBUTIONS:
        raise ValueError(
            f"Invalid distribution: {distribution}. "
            "Allowed are {list(NOISE_DISTRIBUTIONS)}"
        )

    std = processed["std"]
    if std < 0:
        raise ValueError(f"std must be non-negative. Not: {std}")

    corr = processed["correlation"]
    if corr < 0:
        raise ValueError(f"corr must be non-negative. Not: {corr}")

    if is_multiplicative:
        clipping_value = processed["clipping_value"]
        if clipping_value < 0:
            raise ValueError(
                f"clipping_value must be non-negative. Not: {clipping_value}"
            )

    return processed


def _clip_away_from_zero(a, clipval):
    is_scalar = np.isscalar(a)
    a = np.atleast_1d(a)

    is_positive = a >= 0

    clipped = np.where(is_positive, np.clip(a, clipval, np.inf), a)
    clipped = np.where(~is_positive, np.clip(clipped, -np.inf, -clipval), clipped)

    if is_scalar:
        clipped = clipped[0]
    return clipped
