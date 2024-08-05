from optimagic.config import IS_TRANQUILO_INSTALLED

if IS_TRANQUILO_INSTALLED:
    from functools import partial

    from tranquilo.tranquilo import _tranquilo

    from optimagic.decorators import mark_minimizer

    tranquilo = mark_minimizer(
        func=partial(_tranquilo, functype="scalar"),
        name="tranquilo",
        primary_criterion_entry="value",
        needs_scaling=True,
        is_available=True,
        is_global=False,
    )

    tranquilo_ls = mark_minimizer(
        func=partial(_tranquilo, functype="least_squares"),
        primary_criterion_entry="root_contributions",
        name="tranquilo_ls",
        needs_scaling=True,
        is_available=True,
        is_global=False,
    )

    __all__ = ["tranquilo", "tranquilo_ls"]
else:
    __all__ = []
