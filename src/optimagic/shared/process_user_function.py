"""Process user provided functions."""

import inspect
from functools import partial

from optimagic.exceptions import InvalidFunctionError, InvalidKwargsError
from optimagic.utilities import propose_alternatives


def process_func_of_params(func, kwargs, name="your function", skip_checks=False):
    # fast path
    if skip_checks and kwargs in (None, {}):
        return func

    kept, ignored = filter_kwargs(func, kwargs)

    if ignored:
        possibilities = [p for p in inspect.signature(func).parameters if p != "params"]
        proposals = [propose_alternatives(arg, possibilities, 1)[0] for arg in ignored]

        msg = (
            "The following user provided keyword arguments are not compatible with "
            f"{name}:\n\n"
        )
        for arg, prop in zip(ignored, proposals, strict=False):
            msg += f"{arg}: Did you mean {prop}?"

        raise InvalidKwargsError(msg)

    out = partial(func, **kept)

    if not skip_checks:
        unpartialled_args = get_unpartialled_arguments(out)
        no_default_args = get_arguments_without_default(out)

        no_free_argument_left = len(unpartialled_args) < 1

        if no_free_argument_left and kept:
            raise InvalidKwargsError(
                f"Too many keyword arguments for {name}. After applying all keyword "
                "arguments there must be at least one free argument (the params) left."
            )
        elif no_free_argument_left:
            raise InvalidFunctionError(f"{name} must have at least one free argument.")

        required_args = unpartialled_args.intersection(no_default_args)
        too_many_required_arguments = len(required_args) > 1

        # Try to discover if we have a jax calculated jacobian that has a weird
        # signature that would not pass this test:
        skip_because_of_jax = required_args == {"args", "kwargs"}

        if too_many_required_arguments and not skip_because_of_jax:
            raise InvalidKwargsError(
                f"Too few keyword arguments for {name}. After applying all keyword "
                "arguments at most one required argument (the params) should remain. "
                "in your case the following required arguments remain: "
                f"{required_args}."
            )

    return out


def filter_kwargs(func, kwargs):
    valid = get_unpartialled_arguments(func)

    kept = {key: val for key, val in kwargs.items() if key in valid}

    ignored = {key: val for key, val in kwargs.items() if key not in valid}

    return kept, ignored


def get_unpartialled_arguments(func):
    unpartialled = set(inspect.signature(func).parameters)

    if isinstance(func, partial):
        partialed_in = set(func.keywords)
        unpartialled = unpartialled - partialed_in

    return unpartialled


def get_arguments_without_default(func):
    args = dict(inspect.signature(func).parameters)
    no_default = []
    for name, arg in args.items():
        if not hasattr(arg.default, "__len__"):
            if arg.default == inspect.Parameter.empty:
                no_default.append(name)

    no_default = set(no_default)
    return no_default


def get_kwargs_from_args(args, func, offset=0):
    """Convert positional arguments to a dict of keyword arguments.

    Args:
        args (list, tuple): Positional arguments.
        func (callable): Function to be called.
        offset (int, optional): Number of arguments to skip. Defaults to 0.

    Returns:
        dict: Keyword arguments.

    """
    names = list(inspect.signature(func).parameters)[offset:]
    kwargs = {name: arg for name, arg in zip(names, args, strict=False)}
    return kwargs
