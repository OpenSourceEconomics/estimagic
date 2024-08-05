(scaling)=

# How to scale optimization problems

Real world optimization problems often comprise parameters of vastly different orders of
magnitudes. This is typically not a problem for gradient based optimization algorithms
but can considerably slow down derivative free optimizers. Below we describe three
simple heuristics to improve the scaling of optimization problems and discuss the pros
and cons of each approach.

## What does well scaled mean

In short, an optimization problem is well scaled if a fixed step in any direction yields
a roughly similar sized change in the objective function.

In practice, this can never be achieved perfectly (at least for nonlinear problems).
However, one can easily improve over simply ignoring the problem altogether.

## TL;DR

To activate scaling at the default options, pass `scaling=True` to the `minimize` or
`maximize` function. This uses the start values heuristic explained below. The default
options are discussed in the section {ref}`scaling-default-values`.

```{code-block} python
---
emphasize-lines: 13
---
import numpy as np
import optimagic as om


def fun(x):
    return x @ x


res = om.minimize(
    fun=fun,
    x0=np.arange(5),
    algorithm="scipy_lbfgsb",
    scaling=True,
)
```

## Heuristics to improve scaling

(scaling-start-values-heuristic)=

### Divide by absolute value of start parameters

In many applications, parameters with very large start values will vary over a wide
range and a change in that parameter will only lead to a relatively small change in the
objective function. If this is the case, the scaling of the optimization problem can be
improved by simply dividing all parameter vectors by the start parameters.

**Advantages:**

- Straightforward
- Works with any type of constraints

**Disadvantages:**

- Makes scaling dependent on start values
- Parameters with zero start value need special treatment

**How to specify this scaling:**

```{code-block} python
---
emphasize-lines: 5
---
res = om.minimize(
    fun=fun,
    x0=np.arange(5),
    algorithm="scipy_lbfgsb",
    scaling=om.ScalingOptions(method="start_values", clipping_value=0.1),
)
```

### Divide by bounds

In many optimization problems, one has additional information on bounds of the parameter
space. Some of these bounds are hard (e.g. probabilities or variances are non negative),
others are soft and derived from simple considerations (e.g. if a time discount factor
were smaller than 0.7, we would not observe anyone to pursue a university degree in a
structural model of educational choices; or if an infection probability was higher than
20% for distant contacts, the covid pandemic would have been over after a month). For
parameters that strongly influence the objective function, the bounds stemming from
these considerations are typically tighter than for parameters that have a small effect
on the objective function.

Thus, a natural approach to improve the scaling of the optimization problem is to re-map
all parameters such that the bounds are \[0, 1\] for all parameters. This has the
additional advantage that absolute and relative convergence criteria on parameter
changes become the same.

**Advantages:**

- Straightforward
- Works well in many practical applications
- Scaling is independent of start values
- No problems with division by zero

**Disadvantages:**

- Only works if all parameters have bounds
- This prohibits some kinds of other constraints in optimagic

**How to specify this scaling:**

```{code-block} python
---
emphasize-lines: 5,6
---
res = om.minimize(
    fun=fun,
    x0=np.arange(5),
    algorithm="scipy_lbfgsb",
    bounds=om.Bounds(lower=np.zeros(5), upper=2 * np.arange(5) + 1),
    scaling=om.ScalingOptions(method="bounds", clipping_value=0.0),
)
```

## Influencing the magnitude of parameters

The above approaches align the scale of parameters relative to each other. However, the
overall magnitude is set rather arbitrarily. For example, when dividing by start values,
the magnitude of the scaled parameters is around one. When dividing by bounds, it is
somewhere between zero and one.

For the performance of numerical optimizers, only the relative scales are important.

However, influencing the overall magnitude can be helpful to trick some optimizers into
doing things they do not want to do. For example, when there is a minimal allowed
initial trust region radius, increasing the magnitude of parameters allows to
effectively make the trust region radius smaller.

Setting the magnitude means simply adding one more entry to the scaling options. For
example, if you want to scale by bounds and increase the magnitude by a factor of five:

```{code-block} python
---
emphasize-lines: 6
---
res = om.minimize(
    fun=fun,
    x0=np.arange(5),
    algorithm="scipy_lbfgsb",
    bounds=om.Bounds(lower=np.zeros(5), upper=2 * np.arange(5) + 1),
    scaling=om.ScalingOptions(method="bounds", clipping_value=0.0, magnitude=5),
)
```

## Remarks

### What is the `clipping_value`

In all of the above heuristics, the parameter vector is divided (elementwise) by some
other vector and it is possible that some entries of the divisor are zero or close to
zero.

The clipping value bounds the elements of the divisor away from zero. It should be set
to a strictly non-zero number for the `"start_values"` and `"gradient"` approach. The
`"bounds"` approach avoids division by exact zeros by construction. The
`"clipping_value"` can still be used to avoid extreme upscaling of parameters with very
tight bounds. However, this means that the bounds of the re-scaled problem are not
exactly \[0, 1\] for all parameters.

(scaling-default-values)=

### Default values

Scaling is disabled by default. By passing `scaling=True`, we enable scaling at the
default values. We use the `"start_values"` method with a `"clipping_value"` of 0.1 and
a magnitude of 1.0. This is the default method because it can be used for all
optimization problems and has low computational cost. We strongly recommend you read the
above guidelines and choose the method that is most suitable for your problem.
