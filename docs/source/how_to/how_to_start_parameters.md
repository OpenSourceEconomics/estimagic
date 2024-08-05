(params)=

# How to specify `params`

`params` is the first argument of any criterion function in optimagic. It collects all
the parameters to estimate, optimize, or differentiate over. In many optimization
libraries, `params` must be a one-dimensional numpy array. In optimagic, it can be an
arbitrary pytree (think nested dictionary) containing numbers, arrays, pandas.Series,
and/or pandas.DataFrames.

Below, we show a few examples of what is possible in optimagic and discuss the
advantages and drawbacks of each of them.

Again, we use the simple `sphere` function you know from other tutorials as an example.

```{eval-rst}
.. tabbed:: Array

    A frequent choice of ``params`` is a one-dimensional numpy array. This is
    because one-dimensional numpy arrays are all that is supported by most optimizer
    libraries.

    In our opinion, it is rarely a good choice to represent parameters as flat numpy arrays
    and then access individual parameters or sclices by positions. The only exception
    are simple optimization problems with very-fast-to-evaluate criterion functions where
    any overhead must be avoided.

    If you still want to use one-dimensional numpy arrays, here is how:

    .. code-block:: python

        import optimagic as om


        def sphere(params):
            return params @ params


        om.minimize(
            fun=sphere,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
        )

```

```{eval-rst}
.. tabbed:: DataFrame

    Originally, pandas DataFrames were the mandatory format for ``params`` in optimagic.
    They are still highly recommended and have a few special features. For example,
    they allow to bundle information on start parameters and bounds together into one
    data structure.

    Let's look at an example where we do that:

    .. code-block:: python

        def sphere(params):
            return (params["value"] ** 2).sum()


        params = pd.DataFrame(
            data={"value": [1, 2, 3], "lower_bound": [-np.inf, 1.5, 0]},
            index=["a", "b", "c"],
        )

        om.minimize(
            fun=sphere,
            params=params,
            algorithm="scipy_lbfgsb",
        )

    DataFrames have many advantages:

    - It is easy to select single parameters or groups of parameters or work with
      the entire parameter vector. Especially, if you use a well designed MultiIndex.
    - It is very easy to produce publication quality LaTeX tables from them.
    - If you have nested models, you can easily update the parameter vector of a larger
      model with the values from a smaller one (e.g. to get good start parameters).
    - You can bundle information on bounds and values in one place.
    - It is easy to compare two params vectors for equality.


    If you are sure you won't have bounds on your parameter, you can also use a
    pandas.Series instead of a pandas.DataFrame.

    A drawback of DataFrames is that they are not JAX compatible. Another one is that
    they are a bit slower than numpy arrays.


```

```{eval-rst}
.. tabbed:: Dict

    ``params`` can also be a (nested) dictionary containing all of the above and more.

    .. code-block:: python

        def sphere(params):
            return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()


        res = om.minimize(
            fun=sphere,
            params={"a": 0, "b": 1, "c": pd.Series([2, 3, 4])},
            algorithm="scipy_neldermead",
        )

    Dictionarys of arrays are ideal if you want to do vectorized computations with
    groups of parameters. They are also a good choice if you calculate derivatives
    with JAX.

    While optimagic won't stop you, don't go too far! Having parameters in very deeply
    nested dictionaries makes it hard to visualize results and/or even to compare two
    estimation results.

```

```{eval-rst}
.. tabbed:: Scalar

    If you have a one-dimensional optimization problem, the natural way to represent
    your params is a float:

    .. code-block:: python

        def sphere(params):
            return params**2


        om.minimize(
            fun=sphere,
            params=3,
            algorithm="scipy_lbfgsb",
        )
```
