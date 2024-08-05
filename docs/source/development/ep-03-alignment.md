(eepalignment)=

# EP-03: Alignment with SciPy

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Janos Gabler <https://github.com/janosg>`_                      |
+------------+------------------------------------------------------------------+
| Status     | Accepted                                                         |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2024-07-09                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Abstract

This enhancement proposal explains how we will better align optimagic with
`scipy.minimize`. Scipy is the most widely used optimizer library in Python and most of
our new users are switching over from SciPy.

The goal is therefore simple: Make it as easy as possible for SciPy users to use
optimagic. In most cases this means that the only thing that has to be changed is the
import statement for the `minimize` function:

```python
# from scipy.optimize import minimize
from optimagic import minimize
```

## Design goals

- If we can make code written for SciPy run with optimagic, we should do so
- If we cannot make it run, the user should get a helpful error message that explains
  how the code needs to be adjusted.

## Aligning names

| **Old Name**                               | **Proposed Name**         | **Source** |
| ------------------------------------------ | ------------------------- | ---------- |
| `criterion`                                | `fun`                     | scipy      |
| `criterion_kwargs`                         | `fun_kwargs`              |            |
| `params`                                   | `x0`                      |            |
| `derivative`                               | `jac`                     | scipy      |
| `derivative_kwargs`                        | `jac_kwargs`              |            |
| `criterion_and_derivative`                 | `fun_and_jac`             |            |
| `criterion_and_derivative_kwargs`          | `fun_and_jac_kwargs`      |            |
| `stopping_max_criterion_evaluations`       | `stopping_maxfun`         | scipy      |
| `stopping_max_iterations`                  | `stopping_maxiter`        | scipy      |
| `convergence_absolute_criterion_tolerance` | `convergence_ftol_abs`    | NlOpt      |
| `convergence_relative_criterion_tolerance` | `convergence_ftol_rel`    | NlOpt      |
| `convergence_absolute_params_tolerance`    | `convergence_xtol_abs`    | NlOpt      |
| `convergence_relative_params_tolerance`    | `convergence_xtol_rel`    | NlOpt      |
| `convergence_absolute_gradient_tolerance`  | `convergence_gtol_abs`    | NlOpt      |
| `convergence_relative_gradient_tolerance`  | `convergence_gtol_rel`    | NlOpt      |
| `convergence_scaled_gradient_tolerance`    | `convergence_gtol_scaled` |            |

While it seems that many names are taken from NlOpt and not from SciPy, this is a bit
misleading. SciPy does use the words `xtol`, `ftol` and `gtol` just like NlOpt, but it
does not completely harmonize them between algorithms. We therefore chose NlOpt's
version which is understandable for everyone who knows SciPy but more readable than
SciPy's.

## Names we do not want to align

- We do not want to rename `algorithm` to `method` because our algorithm names are
  different from SciPy, so people who switch over from SciPy need to adjust their code
  anyways.
- We do not want to rename `algo_options` to `options` for the same reason.

Instead we can provide aliases for those.

## Additional aliases

To make it even easier for SciPy users to switch to optimagic, we can provide additional
aliases in `minimize` and `maximize` that let them used their SciPy code without changes
or help to adjust it by showing good error messages. The following arguments are
relevant:

- `method`: In SciPy this is used instead of `algorithm` to select the optimization
  algorithm. We opted against simply renaming `algorithm` to `method` because our naming
  scheme of algorithms is (and has to be) different from SciPy. By using `method`
  instead of `algorithm`, users could select SciPy algorithms by their SciPy name. If
  `method` and `algorithm` are both provided, they would get an error.
- `tol`: We do not want to support one `tol` argument for all kinds of different
  convergence criteria but could raise an error for people who use it and point them to
  the relevant parts of our documentation.
- `args`: we can support `args` as an alternative to `fun_kwargs`
- `options`: This is the SciPy counterpart to our `algo_options`. We do not want to
  support this as our option names are different but we can provide a good error message
  with pointers to our documentation if someone uses it.
- `hess` and `hessp`: Currently we don't support closed form hessians. If we support
  them they will be called `hess`. In the meantime, this can raise a
  `NotImplementedError`.
- `callback`: Currently we do not support `callback`s. If we support them they will be
  called `callback` and be as compatible with SciPy as possible. In the meantime we can
  raise a `NotImplementedError`.
- If a user sets `jac=True` we raise and error and explain how to use `fun_and_jac`
  instead.

## Letting algorithms pick their default values

Currently we try to align default values for convergence criteria and other algorithm
options across algorithms and even across optimizer packages. This means that sometimes
algorithms that are used via optimagic produce different results than the same algorithm
used via SciPy or other packages.

Moreover, it is possible that we deviate from algorithm options that the original
authors carefully picked because they maximize performance on a relevant benchmark set.

I therefore propose that in the future we do not try to align algorithm options across
algorithms and packages.

## Implementation

All renamings are done with a careful deprecation cycle. The deprecations become active
in version `0.5.0`. Old names will be removed in version `0.6.0` which should be
scheduled for approximately half a year after the release of `0.5.0`.
