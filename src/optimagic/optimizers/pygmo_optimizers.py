"""Implement pygmo optimizers."""

import contextlib
import warnings

import numpy as np

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import IS_PYGMO_INSTALLED
from optimagic.decorators import mark_minimizer
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_XTOL_REL,
    STOPPING_MAXFUN_GLOBAL,
    get_population_size,
)

STOPPING_MAX_ITERATIONS_GENETIC = 250

with contextlib.suppress(ImportError):
    import pygmo as pg


@mark_minimizer(
    name="pygmo_gaco",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_gaco(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    kernel_size=63,
    speed_parameter_q=1.0,
    oracle=0.0,
    accuracy=0.01,
    threshold=1,
    speed_of_std_values_convergence=7,
    stopping_max_n_without_improvements=100000,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
    focus=0.0,
    cache=False,
):
    """Minimize a scalar function using the generalized ant colony algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    if isinstance(speed_of_std_values_convergence, float):
        if not speed_of_std_values_convergence.is_integer():
            raise ValueError(
                "The speed_of_std_values_convergence parameter must be an integer. "
                f"You specified {speed_of_std_values_convergence}."
            )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "ker": kernel_size,
        "q": speed_parameter_q,
        "oracle": oracle,
        "acc": accuracy,
        "threshold": threshold,
        "n_gen_mark": int(speed_of_std_values_convergence),
        "impstop": stopping_max_n_without_improvements,
        "evalstop": stopping_maxfun,
        "focus": focus,
        "memory": cache,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="gaco",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_bee_colony",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_bee_colony(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    seed=None,
    discard_start_params=False,
    max_n_trials=1,
    population_size=None,
):
    """Minimize a scalar function using the artifical bee colony algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=20
    )
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options={
            "limit": max_n_trials,
            "gen": int(stopping_maxiter),
        },
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="bee_colony",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_de",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_de(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    weight_coefficient=0.8,
    crossover_probability=0.9,
    mutation_variant="rand/1/exp",
    convergence_criterion_tolerance=1e-6,
    convergence_relative_params_tolerance=CONVERGENCE_XTOL_REL,
):
    """Minimize a scalar function using the differential evolution algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    # support both integer and string specification of the mutation variant
    mutation_variant_str_to_int = {
        "best/1/exp": 1,
        "rand/1/exp": 2,
        "rand-to-best/1/exp": 3,
        "best/2/exp": 4,
        "rand/2/exp": 5,
        "best/1/bin": 6,
        "rand/1/bin": 7,
        "rand-to-best/1/bin": 8,
        "best/2/bin": 9,
        "rand/2/bin": 10,
    }
    mutation_variant = _convert_str_to_int(
        str_to_int=mutation_variant_str_to_int, value=mutation_variant
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "F": weight_coefficient,
        "CR": crossover_probability,
        "variant": mutation_variant,
        "ftol": convergence_criterion_tolerance,
        "xtol": convergence_relative_params_tolerance,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="de",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_sea",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_sea(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=10_000,  # Each generation will compute the objective once
):
    r"""Minimize a scalar function using the (N+1)-ES simple evolutionary algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options={"gen": int(stopping_maxiter)},
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="sea",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_sga",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_sga(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    crossover_probability=0.9,
    crossover_strategy="exponential",
    eta_c=None,
    mutation_probability=0.02,
    mutation_strategy="polynomial",
    mutation_polynomial_distribution_index=None,
    mutation_gaussian_width=None,
    selection_strategy="tournament",
    selection_truncated_n_best=None,
    selection_tournament_size=None,
):
    """Minimize a scalar function using a simple genetic algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    if eta_c is not None and crossover_strategy != "sbx":
        warnings.warn(
            f"You specified crossover strategy {crossover_strategy} and eta_c. "
            "However, eta_c is ignored because it is only used when the "
            "crossover_strategy is set to sbx."
        )
    eta_c = 1.0 if eta_c is None else eta_c

    if (
        mutation_polynomial_distribution_index is not None
    ) and mutation_strategy != "polynomial":
        warnings.warn(
            "You specified a mutation_polynomial_distribution_index but did not choose "
            "polynomial as your mutation_strategy. Thus, "
            "mutation_polynomial_distribution_index will be ignored."
        )
    if mutation_gaussian_width is not None and mutation_strategy != "gaussian":
        warnings.warn(
            "You specified a mutation_gaussian_width but did not choose gaussion as "
            "your mutation_strategy. Thus, mutation_gaussian_width will be ignored."
        )
    if selection_strategy != "truncated" and selection_truncated_n_best is not None:
        warnings.warn(
            "You specified selection_truncated_n_best but did not specify truncated as "
            "your selection strategy. Therefore, selection_truncated_n_best is ignored."
        )
    if selection_strategy != "tournament" and selection_tournament_size is not None:
        warnings.warn(
            "You specified selection_tournament_size but did not specify tournament as "
            "your selection strategy. Therefore, selection_tournament_size is ignored."
        )

    if mutation_strategy == "gaussian" and mutation_gaussian_width is not None:
        param_m = mutation_gaussian_width
    elif (
        mutation_strategy == "polynomial"
        and mutation_polynomial_distribution_index is not None
    ):
        param_m = mutation_polynomial_distribution_index
    else:
        param_m = 1.0

    if selection_strategy == "truncated" and selection_truncated_n_best is not None:
        param_s = selection_truncated_n_best
    elif selection_strategy == "tournament" and selection_tournament_size is not None:
        param_s = selection_tournament_size
    else:
        param_s = 2

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "cr": crossover_probability,
        "eta_c": eta_c,
        "m": mutation_probability,
        "param_m": param_m,
        "crossover": crossover_strategy,
        "mutation": mutation_strategy,
        "selection": selection_strategy,
        "param_s": param_s,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="sga",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_sade",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_sade(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    jde=True,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    mutation_variant="rand/1/exp",
    keep_adapted_params=False,
    ftol=1e-6,
    xtol=1e-6,
):
    """Minimize a scalar function using Self-adaptive Differential Evolution.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )
    mutation_variant_str_to_int = {
        "best/1/exp": 1,
        "rand/1/exp": 2,
        "rand-to-best/1/exp": 3,
        "best/2/exp": 4,
        "rand/2/exp": 5,
        "best/1/bin": 6,
        "rand/1/bin": 7,
        "rand-to-best/1/bin": 8,
        "best/2/bin": 9,
        "rand/2/bin": 10,
        "rand/3/exp": 11,
        "rand/3/bin": 12,
        "best/3/exp": 13,
        "best/3/bin": 14,
        "rand-to-current/2/exp": 15,
        "rand-to-current/2/bin": 16,
        "rand-to-best-and-current/2/exp": 17,
        "rand-to-best-and-current/2/bin": 18,
    }
    mutation_variant = _convert_str_to_int(
        str_to_int=mutation_variant_str_to_int, value=mutation_variant
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "variant": mutation_variant,
        "variant_adptv": 1 if jde else 2,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="sade",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_cmaes",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_cmaes(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    backward_horizon=None,
    variance_loss_compensation=None,
    learning_rate_rank_one_update=None,
    learning_rate_rank_mu_update=None,
    initial_step_size=0.5,
    ftol=1e-6,
    xtol=1e-6,
    keep_adapted_params=False,
):
    r"""Minimize a scalar function using the Covariance Matrix Evolutionary Strategy.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "cc": _replace_none(var=backward_horizon, none_value=-1.0),
        "cs": _replace_none(var=variance_loss_compensation, none_value=-1.0),
        "c1": _replace_none(var=learning_rate_rank_one_update, none_value=-1.0),
        "cmu": _replace_none(var=learning_rate_rank_mu_update, none_value=-1.0),
        "sigma0": initial_step_size,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
        "force_bounds": True,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="cmaes",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_simulated_annealing",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_simulated_annealing(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    start_temperature=10.0,
    end_temperature=0.01,
    n_temp_adjustments=10,
    n_range_adjustments=10,
    bin_size=10,
    start_range=1.0,
):
    """Minimize a function with the simulated annealing algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "Ts": start_temperature,
        "Tf": end_temperature,
        "n_T_adj": int(n_temp_adjustments),
        "n_range_adj": int(n_range_adjustments),
        "bin_size": bin_size,
        "start_range": start_range,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="simulated_annealing",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_pso",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_pso(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    omega=0.7298,
    force_of_previous_best=2.05,
    force_of_best_in_neighborhood=2.05,
    max_velocity=0.5,
    algo_variant=5,
    neighbor_definition="lbest",
    neighbor_param=None,
    keep_velocities=False,
):
    r"""Minimize a scalar function using Particle Swarm Optimization.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    if neighbor_definition in [1, 3] and neighbor_param is not None:
        warnings.warn(
            "You gave a neighbor parameter but selected a neighbor_definition "
            "that ignores this parameter."
        )

    neighbor_param = _replace_none(neighbor_param, 4)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    neighbor_definition_str_to_int = {
        "gbest": 1,
        "lbest": 2,
        "Von Neumann": 3,
        "Adaptive random": 4,
    }
    algo_variant_str_to_int = {
        "canonical_inertia": 1,
        "social_and_cog_rand": 2,
        "all_components_rand": 3,
        "one_rand": 4,
        "canonical_constriction": 5,
        "fips": 6,
    }

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "omega": omega,
        "eta1": force_of_previous_best,
        "eta2": force_of_best_in_neighborhood,
        "max_vel": max_velocity,
        "variant": _convert_str_to_int(algo_variant_str_to_int, algo_variant),
        "neighb_type": _convert_str_to_int(
            neighbor_definition_str_to_int, neighbor_definition
        ),
        "neighb_param": neighbor_param,
        "memory": keep_velocities,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="pso",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_pso_gen",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_pso_gen(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    omega=0.7298,
    force_of_previous_best=2.05,
    force_of_best_in_neighborhood=2.05,
    max_velocity=0.5,
    algo_variant="canonical_constriction",
    neighbor_definition=2,
    neighbor_param=None,
    keep_velocities=False,
):
    r"""Minimize a scalar function with generational Particle Swarm Optimization.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    if neighbor_definition in [1, 3] and neighbor_param is not None:
        warnings.warn(
            "You gave a neighbor parameter but selected a neighbor_definition "
            "that ignores this parameter."
        )
    neighbor_param = _replace_none(neighbor_param, 4)
    neighbor_str_to_int = {
        "gbest": 1,
        "lbest": 2,
        "Von Neumann": 3,
        "Adaptive random": 4,
    }
    neighbor_param = _convert_str_to_int(neighbor_str_to_int, neighbor_param)

    algo_variant_str_to_int = {
        "canonical_inertia": 1,
        "social_and_cog_rand": 2,
        "all_components_rand": 3,
        "one_rand": 4,
        "canonical_constriction": 5,
        "fips": 6,
    }
    algo_variant = _convert_str_to_int(algo_variant_str_to_int, algo_variant)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "omega": omega,
        "eta1": force_of_previous_best,
        "eta2": force_of_best_in_neighborhood,
        "max_vel": max_velocity,
        "variant": algo_variant,
        "neighb_type": neighbor_definition,
        "neighb_param": neighbor_param,
        "memory": keep_velocities,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="pso_gen",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_mbh",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_mbh(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    inner_algorithm=None,
    # this is 30 instead of 5 in pygmo for our sum of squares test to pass
    stopping_max_inner_runs_without_improvement=30,
    perturbation=0.01,
):
    """Minimize a scalar function using generalized Monotonic Basin Hopping.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    # the min default population size is this large to pass our sum of squares tests.
    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=250
    )

    algo_specific_options = {
        "algo": inner_algorithm,
        "stop": stopping_max_inner_runs_without_improvement,
        "perturb": perturbation,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="mbh",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_xnes",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_xnes(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    learning_rate_mean_update=1.0,
    learning_rate_step_size_update=None,
    learning_rate_cov_matrix_update=None,
    initial_search_share=1.0,
    ftol=1e-6,
    xtol=1e-6,
    keep_adapted_params=False,
):
    r"""Minimize a scalar function using Exponential Evolution Strategies.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "eta_mu": _replace_none(learning_rate_mean_update, -1),
        "eta_sigma": _replace_none(learning_rate_step_size_update, -1),
        "eta_b": _replace_none(learning_rate_cov_matrix_update, -1),
        "sigma0": initial_search_share,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
        "force_bounds": True,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="xnes",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_gwo",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_gwo(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
):
    """Minimize a scalar function using the Grey Wolf Optimizer.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options={"gen": int(stopping_maxiter)},
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="gwo",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_compass_search",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_compass_search(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
    start_range=0.1,
    stop_range=0.01,
    reduction_coeff=0.5,
):
    """Minimize a scalar function using compass search.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    if population_size is not None:
        warnings.warn(
            f"You specified population size {population_size}. "
            "compass_search does not have a population so this argument is ignored."
        )
    else:
        # if discard_start_params is False population_size - 1 must still be positive
        population_size = 100

    algo_specific_options = {
        "max_fevals": stopping_maxfun,
        "start_range": start_range,
        "stop_range": stop_range,
        "reduction_coeff": reduction_coeff,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="compass_search",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_ihs",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_ihs(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    choose_from_memory_probability=0.85,
    min_pitch_adjustment_rate=0.35,
    max_pitch_adjustment_rate=0.99,
    min_distance_bandwidth=1e-5,
    max_distance_bandwidth=1.0,
):
    """Minimize a scalar function using the improved harmony search algorithm.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    if population_size is not None:
        warnings.warn("The population size has no effect on IHS' performance.")

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=1
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "phmcr": choose_from_memory_probability,
        "ppar_min": min_pitch_adjustment_rate,
        "ppar_max": max_pitch_adjustment_rate,
        "bw_min": min_distance_bandwidth,
        "bw_max": max_distance_bandwidth,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="ihs",
        algo_options=algo_options,
    )
    return res


@mark_minimizer(
    name="pygmo_de1220",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_PYGMO_INSTALLED,
)
def pygmo_de1220(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    seed=None,
    discard_start_params=False,
    jde=True,
    stopping_maxiter=STOPPING_MAX_ITERATIONS_GENETIC,
    allowed_variants=None,
    keep_adapted_params=False,
    ftol=1e-6,
    xtol=1e-6,
):
    """Minimize a scalar function using Self-adaptive Differential Evolution, pygmo
    flavor.

    For details see
    :ref: `list_of_pygmo_algorithms`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    variant_str_to_int = {
        "best/1/exp": 1,
        "rand/1/exp": 2,
        "rand-to-best/1/exp": 3,
        "best/2/exp": 4,
        "rand/2/exp": 5,
        "best/1/bin": 6,
        "rand/1/bin": 7,
        "rand-to-best/1/bin": 8,
        "best/2/bin": 9,
        "rand/2/bin": 10,
        "rand/3/exp": 11,
        "rand/3/bin": 12,
        "best/3/exp": 13,
        "best/3/bin": 14,
        "rand-to-current/2/exp": 15,
        "rand-to-current/2/bin": 16,
        "rand-to-best-and-current/2/exp": 17,
        "rand-to-best-and-current/2/bin": 18,
    }
    if allowed_variants is None:
        allowed_variant_codes = [2, 3, 7, 10, 13, 14, 15, 16]
    else:
        allowed_variant_codes = [
            _convert_str_to_int(variant_str_to_int, variant)
            for variant in allowed_variants
        ]

    population_size = get_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "gen": int(stopping_maxiter),
        "variant_adptv": 1 if jde else 2,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
        "allowed_variants": allowed_variant_codes,
    }

    algo_options = _create_algo_options(
        population_size=population_size,
        seed=seed,
        discard_start_params=discard_start_params,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion=criterion,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="de1220",
        algo_options=algo_options,
    )
    return res


# ====================================================================================


def _minimize_pygmo(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    method,
    algo_options,
    derivative=None,  # noqa: ARG001
):
    """Minimize a function with pygmo.

    Args:
        criterion (callable):
        x (np.ndarray): Starting values of the parameters.
        lower_bounds (np.ndarray):
        upper_bounds (np.ndarray):
        method (str): One of the optimizers of the pygmo package.
        algo_options (dict): Options for the optimizer. In addition to
            the algo options that will be passed directly to the pygmo
            algorithms we have the following entries:
            - population_size (int): Population size for genetic algorithms.
            - batch_evaluator (str or callable): An optimagic batch evaluator,
                default joblib batch evaluator.
            - n_cores (int): Number of cores used for parallel evaluation of
                the criterion function. Default 1.
            - seed (int or None): Random seed for drawing the initial
                population.
            - discard_start_params (bool): If True, the start params are not
                guaranteed to be part of the initial population. This saves one
                criterion function evaluation that cannot be done in parallel
                with other evaluations. Default False.

    Returns:
        results (dict): Dictionary with optimization results.

    """
    algo_options = algo_options.copy()
    if not IS_PYGMO_INSTALLED:
        raise NotInstalledError(
            f"The {method} algorithm requires the pygmo package to be installed. "
            "You can install it with 'conda install -c conda-forge pygmo'. Visit "
            "https://esa.github.io/pygmo2/install.html for more detailed installation "
            "instructions."
        )

    population_size = algo_options.pop("population_size", 1)
    batch_evaluator = algo_options.pop("batch_evaluator", "joblib")
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    n_cores = algo_options.pop("n_cores", 1)
    seed = algo_options.pop("seed", None)
    discard_start_params = algo_options.pop("discard_start_params", False)

    bounds = (lower_bounds, upper_bounds)
    prob = _create_problem(
        func=criterion,
        bounds=bounds,
        dim=len(x),
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
    )
    algo = _create_algorithm(method, algo_options, n_cores)
    pop = _create_population(
        prob, population_size, x, seed=seed, discard_start_params=discard_start_params
    )
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved)

    return result


def _create_problem(func, bounds, dim, batch_evaluator, n_cores):
    class Problem:
        def fitness(self, x):
            return [func(x)]

        def get_bounds(self):
            return bounds

        def gradient(self, dv):  # noqa: ARG002
            raise ValueError("No pygmo optimizer should use a gradient.")

        def batch_fitness(self, dvs):
            dv_list = dvs.reshape(-1, dim)
            eval_list = batch_evaluator(
                func=func,
                arguments=dv_list,
                n_cores=n_cores,
                # Error handling is done on a higher level
                error_handling="raise",
            )
            evals = np.array(eval_list)
            return evals

    problem = pg.problem(Problem())
    return problem


def _create_algorithm(method, algo_options, n_cores):
    """Create a pygmo algorithm."""
    pygmo_uda = getattr(pg, method)
    algo = pygmo_uda(**algo_options)
    try:
        algo.set_bfe(pg.bfe())
    except AttributeError:
        if n_cores >= 2:
            warnings.warn(
                f"Your specified algorithm {method} does not support parallelization. "
                "Choose another algorithm such as pygmo_gaco to parallelize."
            )
    out = pg.algorithm(algo)
    return out


def _create_population(problem, population_size, x, seed, discard_start_params):
    """Create a pygmo population object.

    Args:
        problem (pygmo.Problem)
        algo_options (dict)
        x (np.ndarray)

    Todo:
        - constrain random initial values to be in some bounds

    """
    if not discard_start_params:
        population_size = population_size - 1

    pop = pg.population(
        problem,
        size=population_size,
        seed=seed,
        b=pg.bfe(),
    )
    if not discard_start_params:
        pop.push_back(x)
    return pop


def _process_pygmo_results(evolved):
    results = {
        # Harmonized results.
        "solution_x": evolved.champion_x,
        "solution_criterion": evolved.champion_f[0],
        "solution_derivative": None,
        "solution_hessian": None,
        "n_fun_evals": evolved.problem.get_fevals(),
        "n_jac_evals": evolved.problem.get_gevals(),
        "n_iterations": None,
        "success": True,
        "reached_convergence_criterion": "Number of generations reached.",
        "message": None,
    }

    return results


def _create_algo_options(
    population_size,
    seed,
    discard_start_params,
    algo_specific_options,
    n_cores=1,
    batch_evaluator=None,
):
    algo_options = {
        "population_size": _replace_none(population_size, -1),
        "n_cores": n_cores,
        "seed": seed,
        "discard_start_params": discard_start_params,
    }
    if batch_evaluator is not None:
        algo_options["batch_evaluator"] = batch_evaluator

    algo_options.update(**algo_specific_options)
    return algo_options


def _check_that_every_param_is_bounded(lower_bounds, upper_bounds):
    assert np.isfinite(lower_bounds).all(), "The lower bounds must all be finite."
    assert np.isfinite(upper_bounds).all(), "The upper bounds must all be finite."


def _convert_str_to_int(str_to_int, value):
    if value in str_to_int:
        out = str_to_int[value]
    elif value not in str_to_int.values():
        raise ValueError(
            f"You specified {value} as value. "
            f"It must be one of {', '.join(str_to_int.keys())}"
        )
    else:
        out = value
    return out


def _replace_none(var, none_value):
    out = var if var is not None else none_value
    return out
