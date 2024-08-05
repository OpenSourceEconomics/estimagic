# How supported optimization algorithms are tested

optimagic provides a unified interface that supports a large number of optimization
algorithms from different libraries. Additionally, it allows putting constraints on the
optimization problem. To test the external interface of all supported algorithms, we
consider different criterion (benchmark) functions and test each algorithm with every
type of constraint.

## Benchmark functions for testing

### Trid function

> $f({x}) = \Sigma^{D}_{i=1}(x_{i} - 1)^2 - \Sigma^{D}_{i=2}(x_i x_{i-1})$

### Rotated Hyper Ellipsoid function

> $f({x}) = \Sigma^{D}_{i=1} \Sigma^{i}_{j=1}x_j^2$

### Rosenbrock function

> $\Sigma^{D-1}_{i=1}(100(x_i+1 - x_i^2)^2 + (x_i - 1)^2)$

### Sphere function

> $f({x}) = \Sigma^{D}_{i=1} ix_{i}^2$

## How testcases are implemented

We consider different implementations of each criterion and its gradient. All algorithms
accept criterion functions specified in a dictionary, while a subset also accepts the
criterion specified in scalar form. Likewise, if specified, the gradient of a criterion
can be an np.ndarray or a pandas object. We test for all possible cases. For instance,
for rotated hyper ellipsoid, we implement the following functions:

- rotated_hyper_ellipsoid_scalar_criterion
- rotated_hyper_ellipsoid_dict_criterion: This provides a dictionary wherein the
  `contributions` and `root_contributions` keys present the criterion as a least squares
  problem, relevant when we are testing a least squares algorithm.
- rotated_hyper_ellipsoid_gradient
- rotated_hyper_ellipsoid_pandas_gradient: Computes the gradient of the rotated hyper
  ellipsoid function, as a pandas object.
- rotated_hyper_ellipsoid_criterion_and_gradient

These criterion functions are specified in the `examples` directory. For an overview of
all constraints supported in optimagic, please see [this how-to guide].

We write several test functions, each corresponding to the case of one constraint. Given
the constraint, the test function considers all possible combinations of the algorithm,
whether to maximize or to minimize, criterion function implementation, gradient
implementation for that criterion (if provided), and whether `criterion_and_derivative`
has been provided or not.

Below, we show the calculations behind the true values, for each testcase (one criterion
and one constraint).

### Trid: Solutions for three-dimension case

> $f({x}) = (x_1-1)^2 + (x_2-1)^2 + (x_3-1)^2 - x_2 x_1 - x_3 x_2$

```{eval-rst}
.. dropdown::  No constraints

    .. code-block:: python

        constraints = []

    :math:`x* = (3, 4, 3)`
```

```{eval-rst}
.. dropdown:: Fixed constraints

    .. code-block:: python

        constraints = [{"loc": "x_1", "type": "fixed", "value": 1}]

    :math:`x_{1} = 1 \rightarrow f(x) = (x_2 - 1)^2 + (x_3 - 1)^2 - x_2 - x_3 x_2 \\
    \Rightarrow \frac{\delta f({x})}{\delta x_2} = 2x_2 - 3 - x_3 = 0
    \Rightarrow x_3 = 2x_2 - 3\\
    \Rightarrow \frac{\delta f({x})}{\delta x_3} = 2x_3 - 2 - x_2 = 0
    \Rightarrow x_2 = 2x_3 - 2\\
    \Rightarrow x_2 = \frac{8}{3} , \quad x_3 = \frac{7}{3}\\
    \rightarrow x* = (1,\frac{8}{3}, \frac{7}{3})`
```

```{eval-rst}
.. dropdown::  Probability constraint

    .. code-block:: python

        constraints = [{"loc": ["x_1", "x_2"], "type": "probability"}]

    :math:`x_{1} + x_{2} = 1, \quad 0 \leq x_1 \leq 1, \quad 0 \leq x_2 \leq 1 \\
    \rightarrow f({x}) = 3x_1^2 - 3x_1 - 3x_3 + x_3^2 + x_1 x_3 + 2 \\
    \Rightarrow \frac{\delta f({x})}{\delta x_1} = 6x_1 - 3 + x_3 = 0
    \Rightarrow x_3 = 3 - 6x_1\\
    \Rightarrow \frac{\delta f({x})}{\delta x_3} = 2x_3 - 3 + x_1 = 0
    \Rightarrow x_1 = 3 - 2x_3\\
    \Rightarrow x_1 = \frac{3}{11}, \quad x_3 = \frac{15}{11}\\
    \rightarrow x* = (\frac{3}{11}, \frac{8}{11}, \frac{15}{11})`
```

```{eval-rst}
.. dropdown:: Increasing constraint

    .. code-block:: python

        constraints = [{"loc": ["x_2", "x_3"], "type": "increasing"}]

    :math:`\mathcal{L}({x_i}) = (x_1 - 1)^2 + (x_2 - 1)^2 + (x_3 - 1)^2 - x_1 x_2 -
    x_3 x_2 - \lambda(x_3 - x_2)\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta x_1} = 2(x_1 - 1) - x_2 = 0\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta x_2} = 2(x_2 - 1) - x_1 - x_3 +
    \lambda = 0\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta x_3} = 2(x_3 - 1) - x_2 - \lambda
    = 0\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta \lambda} = - x_3 + x_2 = 0\\
    \Rightarrow x_2 = 2(x_1 - 1) = x_3 = \frac{10}{3}\\
    \Rightarrow 2(x_2 - 1) - x_1 - 2 = 0\\
    \Rightarrow 4(x_1 - 1) - 2 - x_1 - 2 = 0\\
    \Rightarrow 3x_1 - 8 = 0 \Rightarrow x_1 = \frac{8}{3}\\
    \rightarrow x* = (\frac{8}{3}, \frac{10}{3}, \frac{10}{3})`
```

```{eval-rst}
.. dropdown::  Decreasing constraint

    .. code-block:: python

        constraints = [{"loc": ["x_1", "x_2"], "type": "decreasing"}]

    Solution unavailable.
```

```{eval-rst}
.. dropdown::  Equality constraint

    .. code-block:: python

        constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "equality"}]

    :math:`x_{1} = x_{2} = x_{3} = x \\
    \rightarrow f({x}) = x^2 - 6x + 3\\
    \Rightarrow \frac{\delta f({x})}{\delta x} = 2x - 6 = 0\\
    \Rightarrow x = 3\\
    \rightarrow x* = (3,3,3)`
```

```{eval-rst}
.. dropdown::   Pairwise equality constraint

    .. code-block:: python

        constraints = [{"locs": ["x_1", "x_2"], "type": "pairwise_equality"}]

    :math:`x_{1} = x_{2} \\
    \rightarrow f({x}) = 2(x_1 - 1)^2 + (x_3 - 1)^2 - x_1^2 - x_3 x_1\\
    \Rightarrow \frac{\delta f({x})}{\delta x_1} = 2x_1 - x_3 - 4 = 0 \Rightarrow x_3
    = 2x_1 - 4\\
    \Rightarrow \frac{\delta f({x})}{\delta x_3} = 2x_3 - x_1 - 2 = 0 \Rightarrow x_1
    = 2x_3 - 2\\
    \Rightarrow x_1 = \frac{10}{3}, x_3 = \frac{8}{3}\\
    \rightarrow x* = (\frac{10}{3},\frac{10}{3},\frac{8}{3})`
```

```{eval-rst}
.. dropdown::   Covariance constraint

    .. code-block:: python

        constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "covariance"}]

    Solution unavailable.

```

```{eval-rst}
.. dropdown::  sdcorr constraint

    .. code-block:: python

        constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "sdcorr"}]

    Solution unavailable.
```

```{eval-rst}
.. dropdown::  Linear constraint

    .. code-block:: python

        constraints = [{"loc": ["x_1", "x_2"], "type": "linear", "weights": [1, 2], "value": 4}]

    :math:`x_1 + 2x_2 = 4\\
    \mathcal{L}({x_i}) = (x_1 - 1)^2 + (x_2 - 1)^2 + (x_3 - 1)^2 - x_1 x_2 - x_3 x_2
    - \lambda(x_1 +2x_2-4)\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta x_1} = 2(x_1 - 1) - x_2 - \lambda = 0\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta x_2} = 2(x_2 - 1) - x_1 - x_3 -
    2\lambda = 0\\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta x_3} = 2(x_3 - 1) - x_2 = 0 \\
    \Rightarrow \frac{\delta \mathcal{L}}{\delta \lambda} = - x_1 - 2x_2 + 4 = 0\\
    \Rightarrow x_2 = 2(x_3 - 1), \quad x_1 = 4 - 2x_2\\
    \Rightarrow 2(4 - 2x_2 - 1) - x_2 = x_2 - 1 - 2 + x_2 - \frac{x_2}{4} -
    \frac{1}{2}\\
    \rightarrow x* = (\frac{32}{27}, \frac{38}{27}, \frac{46}{27})`





```

### Rotated Hyper Ellipsoid: Solutions for three-dimension case

> $f({x}) = x^2_1 + (x^2_1 + x^2_2) + (x^2_1 + x^2_2 + x^2_3)$
>
> > ```{eval-rst}
> > .. dropdown::   No constraints
> >
> >     .. code-block:: python
> >
> >         constraints = []
> >
> >     :math:`x* = (0, 0, 0)`
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::   Fixed constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": "x_1", "type": "fixed", "value": 1}]
> >
> >     :math:`x_{1} = 1
> >     \rightarrow x* = (1, 0, 0)`
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::   Probability constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_1", "x_2"], "type": "probability"}]
> >
> >     :math:`x_{1} + x_{2} = 1, \quad 0 \leq x_1 \leq 1, \quad 0 \leq x_2 \leq 1 \\
> >     \mathcal{L}({x_i}) = x^2_1 + (x^2_1 + x^2_2) + (x^2_1 + x^2_2 + x^2_3)\\
> >     -\lambda(x_1 +x_2-1)\\
> >     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_1}\\
> >     = 6x_1 - \lambda = 0\\
> >     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_2}\\
> >     = 4x_2 - \lambda = 0\\
> >     \Rightarrow \frac{\delta \mathcal{L}}{\delta x_3}\\
> >     = 2 x_3 = 0\\
> >     \Rightarrow \frac{\delta \mathcal{L}}{\delta \lambda} \\
> >     = -x_1 - x_2 + 1 = 0\\
> >     \rightarrow x* = (\frac{2}{5}, \frac{3}{5}, 0),\\
> >     \quad f({x*}) = \frac{6}{5}`
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::  Increasing  constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_2", "x_3"], "type": "increasing"}]
> >
> >     Not binding :math:`\rightarrow x* = (0, 0, 0)`
> >
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::   Decreasing  constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_1", "x_2"], "type": "decreasing"}]
> >
> >     Not binding :math:`\rightarrow x* = (0, 0, 0)`
> >
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::   Equality  constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "equality"}]
> >
> >     Not binding :math:`\rightarrow x* = (0, 0, 0)`
> >
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::  Pairwise equality  constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"locs": ["x_1", "x_2"], "type": "pairwise_equality"}]
> >
> >     Not binding :math:`\rightarrow x* = (0, 0, 0)`
> >
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::   Covariance constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "covariance"}]
> >
> >     Not binding :math:`\rightarrow x* = (0, 0, 0)`
> >
> >
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::   sdcorr constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "sdcorr"}]
> >
> >     Not binding :math:`\rightarrow x* = (0, 0, 0)`
> >
> > ```
> >
> > ```{eval-rst}
> > .. dropdown::  Linear constraints
> >
> >     .. code-block:: python
> >
> >         constraints = [{"loc": ["x_1", "x_2"], "type": "linear", "weights": [1, 2], "value": 4}]
> >
> >     :math:`x_1 + 2x_2 = 4\\\mathcal{L}({x_i}) = x^2_1 + (x^2_1 + x^2_2) +
> >     (x^2_1 + x^2_2 + x^2_3) -\lambda(x_1 +2x_2-4)\\
> >     \Rightarrow \frac{\delta\mathcal{L}}{\delta x_1} = 6x_1 - \lambda = 0\\
> >     \Rightarrow \frac{\delta \\
> >     \mathcal{L}}{\delta x_2} = 4x_2 - 2\lambda = 0\\
> >     \Rightarrow \frac{\delta \\
> >     \mathcal{L}}{\delta x_3} = 2 x_3 = 0\\
> >     \Rightarrow \frac{\delta \\
> >     \mathcal{L}}{\delta \lambda} = -x_1 - 2x_2 + 4 = 0\\
> >     \rightarrow x* = (\frac{4}{7}, \frac{12}{7}, 0)`
> >
> >
> >
> >
> >
> >
> > ```

### Rosenbrock: Solutions for three-dimension case

> $f({x}) = 100(x_2 - x_1^2) + (x_1 - 1)^2$

Global minima: $x* = (1, 1, 1)$

> ```{eval-rst}
> .. dropdown::  No constraints
>
>     .. code-block:: python
>
>         constraints = []
>
>     :math:`x* = (1, 1, 1)`
>
> ```
>
> ```{eval-rst}
> .. dropdown::  Fixed constraints
>
>     .. code-block:: python
>
>        constraints = [{"loc": "x_1", "type": "fixed", "value": 1}]
>
>     :math:`x_{1} = 1 \rightarrow x* = (1, 1, 1)`
> ```
>
> ```{eval-rst}
> .. dropdown::  Fixed constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_1", "x_2"], "type": "probability"}]
>
>     No solution available.
> ```
>
> ```{eval-rst}
> .. dropdown::  Increasing constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_2", "x_3"], "type": "increasing"}]
>
>     Not binding :math:`\rightarrow x* = (1, 1, 1)`
>
> ```
>
> ```{eval-rst}
> .. dropdown::  Decreasing constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_1", "x_2"], "type": "decreasing"}]
>
>     Not binding :math:`\rightarrow x* = (1, 1, 1)`
> ```
>
> ```{eval-rst}
> .. dropdown::  Equality constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "equality"}]
>
>     Not binding :math:`\rightarrow x* = (1, 1, 1)`
> ```
>
> ```{eval-rst}
> .. dropdown::  Pairwise equality constraints
>
>     .. code-block:: python
>
>         constraints = [{"locs": ["x_1", "x_2"], "type": "pairwise_equality"}]
>
>     Not binding :math:`\rightarrow x* = (1, 1, 1)`
> ```
>
> ```{eval-rst}
> .. dropdown::  Covariance constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "covariance"}]
>
>     Not binding :math:`\rightarrow x* = (1, 1, 1)`
> ```
>
> ```{eval-rst}
> .. dropdown::  sdcorr constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_1", "x_2", "x_3"], "type": "sdcorr"}]
>
>     Not binding :math:`\rightarrow x* = (1, 1, 1)`
> ```
>
> ```{eval-rst}
> .. dropdown::  Linear constraints
>
>     .. code-block:: python
>
>         constraints = [{"loc": ["x_1", "x_2"], "type": "linear", "weights": [1, 2], "value": 4}]
>
>     No solution available.
> ```

[this how-to guide]: ../how_to/how_to_constraints.md
