(eppytrees)=

# EP-01: Pytrees

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Janos Gabler <https://github.com/janosg>`_                      |
+------------+------------------------------------------------------------------+
| Status     | Accepted                                                         |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2022-01-28                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Abstract

This EEP explains how we will use pytrees to allow for more flexible specification of
parameters for optimization or differentiation, more convenient ways of writing moment
functions for msm estimation and more. The actual code to work with pytrees is
implemented in [Pybaum], developed by {ghuser}`janosg` and {ghuser}`tobiasraabe`.

## Backwards compatibility

All changes are fully backwards compatible.

## Motivation

Estimagic has many functions that require user written functions as inputs. Examples
are:

- criterion functions and their derivatives for optimization
- functions of which numerical derivatives are taken
- functions that calculate simulated moments
- functions that calculate bootstrap statistics

In all cases, there are some restrictions on possible inputs and outputs of the user
written functions. For example, parameters for numerical optimization need to be
provided as pandas.DataFrame with a `"value"` column. Simulated moments and bootstrap
statistics need to be returned as a pandas.Series, etc.

Pytrees allow to relax many of those restrictions on interfaces of user provided
functions. This is not only more convenient for users, but sometimes also allows to
reduce overhead because the user can choose optimal data structures for their problem.

## Background: What is a pytree

Pytree is a term used in TensorFlow and JAX to refer to a tree-like structure built out
of container-like Python objects with arbitrary levels of nesting.

What is a container can be re-defined for each application. By default, lists, tuples
and dicts are considered containers and everything else is a leaf. Then the following
are examples of pytrees:

```python
[1, "a", np.arange(3)]  # 3 leaves

[1, {"k1": 2, "k2": (3, 4)}, 5]  # 5 leaves

np.arange(5)  # 1 leaf
```

What makes pytrees so powerful are the operations defined for them. The most important
ones are:

- `tree_flatten`: Convert any pytree into a flat list of leaves + metadata
- `tree_unflatten`: The inverse of `tree_flatten`
- `tree_map`: Apply a function to all leaves in a pytree
- `leaf_names`: Generate a list of names for all leaves in a pytree

The above examples of pytrees would look as follows when flattened (with a default
definition of containers):

```python
[1, "a", np.arange(3)]

[1, 2, 3, 4, 5]

[np.arange(5)]
```

By adding numpy arrays to the registry of container like objects, each of the three
examples above would have five leafs. The flattened versions would look as follows:

```python
[1, "a", 0, 1, 2]

[1, 2, 3, 4, 5]

[0, 1, 2, 3, 4]
```

Needless to say, it is possible to register anything as container. For example, we would
add pandas.Series and pandas.DataFrame (with varying definitions, depending on the
application).

## Difference between pytrees in JAX and estimagic

Most JAX functions
[only work with Pytrees of arrays](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions)
and scalars, i.e. pytrees where container types are dicts, lists and tuples and all
leaves are arrays or scalars. We will just call them pytrees of arrays because scalars
are converted to arrays by JAX.

There are two ways to look at such pytrees:

1. As pytree of arrays -> `tree_flatten` produces a list of arrays
1. As pytree of numbers -> `tree_flatten` produces a list of numbers

The only difference between the two perspectives is that for the second one, arrays have
been registered as container types that can be flattened. In JAX the term `ravel`
instead of `flatten` is sometimes used to make clear that the second perspective is
meant.

Estimagic functions work with slightly more general pytrees. On top of arrays, they can
also contain scalars, pandas.Series and pandas.DataFrames.

Again, there are two possible ways to look at such pytrees:

1. As pytree of arrays, numbers, Series and DataFrames -> `tree_flatten` produces a list
   of arrays numbers, Series and DataFrames.
1. As pytree of numbers -> `tree_flatten` produces a list of numbers

Again, the difference between the two is which objects are registered as container types
and the rules to flatten and unflatten them are defined.

While numpy arrays, scalars and pandas.Series have only one natural way of defining the
flattening rules, this becomes more complex for DataFrames due to the way `params`
DataFrames were used in estimagic before.

We define the following rules: If a DataFrame contains a column called `"value"`, we
interpret them as classical estimagic DataFrame and only consider the entries in the
`"value"` column when flattening the DataFrame into a list of numbers. If there is no
column `"value"`, all numeric columns of the DataFrame are considered.

Note that internally, we will sometimes define flattening rules such that only some
other columnn, e.g. only `"lower_bound"` is considered. However we never look at more
than one column of a classical estimagic params DataFrame at a time.

To distinguish between the different pytrees we use the terms JAX-pytree and
estimagic-pytree.

## Optimization with pytrees

In this section we look at possible ways to specify optimizations when parameters and
some outputs of criterion functions can be estimagic-pytrees.

As an example we use a hypothetical criterion function with pytree inputs and outputs to
describe how a user can optimize it. We also give a rough intuition what happens behind
the scenes.

### The criterion function

Consider a criterion function that takes parameters in the following format:

```python
params = {
    "delta": 0.95,
    "utility": pd.DataFrame(
        [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"]
    ),
    "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
}
```

The criterion function returns a dictionary of the form:

```python
{
    "value": 1.1,
    "contributions": {"a": np.array([0.36, 0.25]), "b": 0.49},
    "root_contributions": {"a": np.array([0.6, 0.5]), "b": 0.7},
}
```

### Run an optimization

```python
from estimagic import minimize

minimize(
    criterion=crit,
    params=params,
    algorithm="scipy_lbfgsb",
)
```

The internal optimizer (in this case the lbfgsb algorithm from scipy) will see a wrapped
version of `crit`. That version takes a 1d numpy array as its only argument and returns
a scalar float (the `"value"` entry of the result of `crit`). Numerical derivatives are
also taken on that function.

If instead a derivative based least squares optimizer like `"scipy_ls_dogbox"` had been
used, the internal optimizer would see a modified version of `crit` that takes a 1d
numpy array and returns a 1d numpy array (the flattened version of the
`"root_contributions"` entry of the result of `crit`).

### The optimization output

The following entries of the output of minimize are affected by the change:

- `"solution_params"`: A pytree with the same structure as `params`
- `"solution_criterion"`: The output dictionary of `crit` evaluated solution params
- `solution_derivative`: Maybe we should not even have this entry.

```{note}
We need to discuss if and in which form we want to have a solution
derivative entry. In it's current form it is useless if constraints are used. This gets
worse when we allow for pytrees and translating this into a meaningful shape might be
very difficult.
```

### Add bounds

Bounds on parameters that are inside a DataFrame with `"value"` column can simply be
specified as before. For all others, there are separate `lower_bounds` and
`upper_bounds` arguments in `maximize` and `minimize`.

`lower_bounds` and `upper_bounds` are pytrees of the same structure as `params` or a
subtree that preserves enough structure to match all bounds. For example:

```python
minimize(
    criterion=crit,
    params=params,
    algorithm="scipy_lbfgsb",
    lower_bounds={"delta": 0},
    upper_bounds={"delta": 1},
)
```

This would add bounds for delta, keep the bounds on all `"utility"` parameters, and
leave the `"probs"` parameters unbounded.

### Add a constraint

Currently, parameters to which a constraint is applied are selected via a `"loc"` or
`"query"` entry in the constraints dictionary.

This keeps working as long as params are specified as a single DataFrame containing a
`"value"` column. If a more general pytree is used we need a "selector" entry instead.
The value of that entry is a callable that takes the pytree and returns selected
parameters.

The `selector` function may return the parameters in the form of an estimagic-pytree.
Should order play a role for the constraints (e.g., increasing) the constraint will be
applied to the flattened version of the pytree returned by the `selector` function.
However, in the case that order matters, we advise users to return one-dimensional
arrays (explicit is better than implicit).

As an example, let's add probability constraints for each row of `"probs"`:

```python
constraints = [
    {"selector": lambda params: params["probs"][0], "type": "probability"},
    {"selector": lambda params: params["probs"][1], "type": "probability"},
]

minimize(
    criterion=crit,
    params=params,
    algorithm="scipy_lbfgsb",
    constraints=constraints,
)
```

The required changes to support this are relatively simple. This is because most
functions that deal with constraints already work with a 1d array of parameters and the
`"loc"` and `"query"` entries of constraints are internally translated to positions in
that array very early on.

### Derivatives during optimization

If numerical derivatives are used, they are already taken on a modified function that
maps from 1d numpy array to scalars or 1d numpy arrays. Allowing for estimagic-pytrees
in parameters and criterion outputs will not pose any difficulties here.

Closed form derivatives need to have the following interface: They expect `params` in
the exact same format as the criterion function as first argument. They return a
derivative in the same format as our numerical derivative functions or JAXs autodiff
functions when applied to the criterion function.

## Numerical derivatives with pytrees

### Problem: Higher dimensional extensions of pytrees

The derivative of a function that maps from a 1d array to a 1d array (usually called
Jacobian) is a 2d matrix. If the 1d arrays are replaced by pytrees, we need a two
dimensional extension of the pytrees. Below we will look at how JAX does this and why we
cannot simply copy that solution.

### The JAX solution

Let's look at an example. We first define a function in terms of 1d arrays and then in
terms of pytrees and look at a JAX calculated jacobian in both cases:

```python
def square(x):
    return x**2


x = jnp.array([1, 2, 3, 4, 5, 6.0])

jacobian(square)(x)
```

```bash
DeviceArray([[ 2.,  0.,  0.,  0.,  0.,  0],
             [ 0.,  4.,  0.,  0.,  0.,  0],
             [ 0.,  0.,  6.,  0.,  0.,  0],
             [ 0.,  0.,  0.,  8.,  0.,  0],
             [ 0.,  0.,  0.,  0., 10.,  0],
             [ 0.,  0.,  0.,  0.,  0., 12]], dtype=float32)
```

```python
def tree_square(x):
    out = {
        "c": x["a"] ** 2,
        "d": x["b"].flatten() ** 2,
    }

    return out


tree_x = {"a": jnp.array([1, 2.0]), "b": jnp.array([[3, 4], [5, 6.0]])}


jacobian(tree_square)(tree_x)
```

Instead of showing the entire results, let's just look at the resulting tree structure
and array shapes:

```python
{
    "c": {
        "a": (2, 2),
        "b": (2, 2, 2),
    },
    "d": {
        "a": (4, 2),
        "b": (4, 2, 2),
    },
}
```

The outputs for hessians have even deeper nesting and three dimensional arrays inside
the nested dictionary. Similarly, we would get higher dimensional arrays if one of the
original pytrees had already contained a 2d array.

### Extending the JAX solution to estimagic-pytrees

JAX pytrees can only contain arrays, whereas estimagic-pytrees may contain scalars,
pandas.Series and pandas.DataFrames (with or without `"value"` column). Unfortunately,
this poses non-trivial challenges for numerical derivatives because those data types
have no natural extension in arbtirary dimensions.

Our solution needs to fulfill two requirements:

1\. Compatible with JAX in the sense than whenever a derivative can be calculated with
JAX it can also be calculated with estimagic and the result has the same structure. 2.
Compatible with the rest of estimagic in the sense that any function that can be
optimized can also be differentiated. In the special case of differentiating with
respect to a DataFrame it also needs to be backwards compatible.

A solution that achieves this is to treat Series and DataFrames with `"value"` columns
as 1d arrays and other DataFrames as 2d arrays, then proceed as in JAX and finally try
to preserve as much index and column information as possible.

This leads to very natural results in the typical usecases with flat dicts of Series or
params DataFrames both as inputs and outputs and is backwards compatible with everything
that is supported already.

However, similar to JAX, not everything that is supported will also be a good idea.
Predicting where a pandas Object is preserved and where it will be replaced by an array
might be hard for very nested pytrees. However, these rules are mainly defined to avoid
hard limitations that have to be checked and documented. Users will learn to avoid too
much complexity by avoiding complex pytrees as inputs and outputs at the same time.

To see this in action, let's look at an example. We repeat the example from the JAX
interface above with the following changes:

1. The 1d numpy array in x\["a"\] is replaced by a DataFrame with `"value"` column
1. The "d" entry in the output becomes a Series instead of a 1d numpy array.

```python
def pd_tree_square(x):
    out = {
        "c": x["a"]["value"] ** 2,
        "d": pd.Series(x["b"].flatten() ** 2, index=list("jklm")),
    }

    return out


pd_tree_x = {
    "a": pd.DataFrame(data=[[1], [2]], index=["alpha", "beta"], columns=["value"]),
    "b": np.array([[3, 4], [5, 6]]),
}

pd_tree_square(pd_tree_x)
```

```
{
    'c':
        "alpha"    1
        "beta"     4
        dtype: int64,
    'd':
        "j"        9
        "k"       16
        "l"       25
        "m"       36
        dtype: int64,
}
```

The resulting shapes of the jacobian will be the same as before. For all arrays with
only two dimensions we can preserve some information from the Series and DataFrame
indices. On the higher dimensional ones, this will be lost.

```python
{
    "c": {
        "a": (2, 2),  # df with columns ["alpha", "beta"], index ["alpha", "beta"]
        "b": (2, 2, 2),  # numpy array without label information
    },
    "d": {
        "a": (4, 2),  # columns ["alpha", "beta"], index [0, 1, 2, 3]
        "b": (4, 2, 2),  # numpy array without label information
    },
}
```

To get more intuition for the structure of the result, let's add a few labels to the
very first jacobian:

```{eval-rst}
+--------+----------+----------+----------+----------+----------+----------+----------+
|        |          | a        |          | b        |          |          |          |
+--------+----------+----------+----------+----------+----------+----------+----------+
|        |          | alpha    | beta     | j        | k        | l        | m        |
+--------+----------+----------+----------+----------+----------+----------+----------+
| c      | alpha    | 2        | 0        | 0        | 0        | 0        | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | beta     | 0        | 4        | 0        | 0        | 0        | 0        |
+--------+----------+----------+----------+----------+----------+----------+----------+
| d      | 0        | 0        | 0        | 6        | 0        | 0        | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | 1        | 0        | 0        | 0        | 8        | 0        | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | 2        | 0        | 0        | 0        | 0        | 10       | 0        |
+        +----------+----------+----------+----------+----------+----------+----------+
|        | 3        | 0        | 0        | 0        | 0        | 0        | 12       |
+--------+----------+----------+----------+----------+----------+----------+----------+
```

The indices \["j", "k", "l", "m"\] unfortunately never made it into the result because
they were only applied to elements that already came from a 2d array and thus always
have a 3d Jacobian, i.e. the result entry `["c"][b"]` is a reshaped version of the upper
right 2 by 4 array and the result entry `["d"]["b"]` is a reshaped version of the lower
right 4 by 4 array.

### Implementation

We use the following terminology to describe the implementation:

- input_tree: The pytree containing parameters, i.e. inputs to the function that is
  differentiated.
- output_tree: The pytree that is returned by the function being differentiated
- derivative_tree: The pytree we want to generate, i.e. the pytree that would be
  returned by JAX jacobian.
- flat_derivative: The matrix version of the derivative_tree

To simply reproduce the JAX behavior with pytrees of arrays, we could proceed in the
following steps:

- Create a modified function that maps from 1d array to 1d array
- Calculate flat_derivative by taking numerical derivatives just as before
- Calculate the shapes of all arrays in derivative_tree by concatenating the shapes of
  the cartesian product of flattend output_tree and input_tree
- Calculate the 2d versions of those arrays by taking the product over elements in the
  shape tuple before concatenating.
- Create a list of lists containing all arrays that will be in derivative_tree. The
  values are taken from flat_derivative, using the previously calculated shapes.
- call `tree_unflatten` on the inner lists with the treedef corresponding to input_tree.
- call `tree_unflatten` on the result of that with the treedef corresponding to
  output_tree.

To implement the extension to estimagic pytrees we would probably do exactly the same
but have a bit more preparation and post-processing to do.

## General aspects of pytrees in estimation functions

### Estimation summaries

Currently, estimation summaries are DataFrames. The estimated parameters are in the
`"value"` column. There are other columns with standard errors, p-values, significance
stars and confidence intervals.

This is another form of higher dimensional extension of pytrees, where we need to add
additional columns. There are two ways in which estimation summaries could be presented.
I suggest we offer both. The first is more geared towards generating estimation tables
and serving as actual summary to be looked at in a jupyter notebook. It is also
backwards compatible and should thus be the default. The second is more geared towards
further calculations. There will be utility functions to convert between the two.

Both formats will be explained using the `params` pytree from the optimization example
(reproduced here for convenience):

#### Format 1: Everything becomes a DataFrame

In this approach we do the following conversions:

1. numpy arrays are flattened and converted to DataFrames with one column called
   `"value"`. The index contains the original positions of elements.
1. pandas.Series are converted to DataFrames. The index remains unchanged. The column is
   called `"value"`.
1. scalars become DataFrames with one row with index 0 and one column called `"value"`.
1. DataFrames without `"value"` column are stacked into a DataFrame with just one column
   called `"value"`.
1. DataFrames with `"value"` column are reduced to that column.

After these transformations, all numbers of the original pytree are stored in DataFrames
with `"value"` column. Additional columns with standard errors and the like can then
simply be assigned as before.

For more intuition, let's see how this would look in an example. For simplicity we only
add a column with stars and ommit standard errors, p-values and confidence intervals. We
use the same example as in the optimization section:

```python
params = {
    "delta": 0.95,
    "utility": pd.DataFrame(
        [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"]
    ),
    "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
}
```

```
{
'delta':
          value stars
    0     0.95   ***,
'utility':
          value stars
    a     0.5    **
    b     0.5    **
    c     0.5    **,
'probs':
          value stars
    0 0   0.8   ***
      1   0.2     *
    1 0   0.3    **
      1   0.7   ***,
}
```

#### Format 2: Dictionary of pytrees

The second solution is a dictionary of pytrees the keys are the columns of the current
summary but probably in plural, i.e. "values", "standard_errors", "p-values", ...;

Each value is a pytree with the exact same structure as `params`. If this pytree
contains DataFrames with `"value"` column, only that column is updated. i.e. standard
errors would be accessed via `summary["standard_errors"]["my_df"]["value"]`.

### Representation of covariance matrices

A covariance matrix is a two dimensional extension of a `params` pytree. We could
theoretically handle it exactly the same way as Jacobians. However, this would not be
useful for statistical tests and visualization if it contains more than 2 dimensional
arrays (as the Jacobian example does).

We thus propose to have two possible formats in which covariance matrices can be
returned:

1. The pytree variant described in the above Jacobian example. This will be useful to
   look at sub-matrices of the full covariance matrix as long as the `params` pytree
   only contains one dimensional arrays, Series and DataFrames with `"value"` columns.
1. A DataFrame containing the covariance matrix of the flattened parameter vector. The
   index and columns of the DataFrames can be constructed from the `leaf_names` function
   in `pybaum`. We could also triviall add a function there that constructs an index
   that is easier to work with for selecting elements and let the user choose between
   the two versions.

The function that maps from the flat version (which would be calculated internally) to
the pytree version is the same as we need for numerical derivatives. The inverse of that
function is probably not too difficult to implement and can also be useful for
derivatives.

### params

Everything that can be used as `params` in optimization and differentiation can also be
used as `params` in estimation. The registries used in pytree functions are identical.

## ML specific aspects of pytrees

The output of the log likelihood functions is a dictionary with the entries:

- `"value"`: a scalar float
- `"contributions"`: a 1d numpy array or pandas.Series

Moreover, there can be arbitrary additional entries.

The only change is that `"contributions"` can now be any estimagic pytree.

## MSM specific aspects of pytrees

### Valid formats of empirical and simulated moments

There are three types of moments in MSM estimation:

- `empirical moments`
- The output of `simulate_moments`
- The output of `calculate_moments`, needed to get a moments covariance matrix.

We propose that moments can be stored as any valid estimagic pytree but of course all
three types of moments have to be aligned, i.e. be stored in a tree of the same
structure. We will raise an error if the trees do not have the same structure.

This is a generalization of an interface that has already proven useful in
[respy](https://github.com/OpenSourceEconomics/respy),
[sid](https://github.com/covid-19-impact-lab/sid) and other applications. In the future,
the project specific implementations of flatten and unflatten functions could simply be
deleted.

### Representation of the weighting matrix and moments_cov

The weighting matrix for MSM estimation is represented as a DataFrame in the same way as
the flat representation of the covariance matrices. Of course, the conversion functions
that work for covariance matrices would also work here, but it is highly unlikely that a
different representation of a weighting matrix is ever needed.

Note that the user does not have to construct this weighting matrix manually. They can
generate them using `get_moments_cov` and `get_weighting_matrix`, so they do not need
any knowledge of how the flattening works.

### Pepresentation of sensitivity measures

Sensitivity measures are similar to covariance matrices in the sense that they require a
two dimensional extension of pytrees. The only difference is that for covariance
matrices the two pytrees the same (namely the `params`) and for sensitivity measures
they are different (one is `params`, the other `moments`).

We therefore suggest to use the same solution, i.e. to offer a flat representation in
form of a DataFrame, a pytree representation and functions to convert between the two.

## Compatibility with estimation tables

Estimation tables are constructed from estimation summaries. This continues to work for
summaries where everything has been converted to DataFrames. Users will select
individual DataFrames from a pytree of DataFrames, possibly concatenate or filter them
and pass them to the estimation table function.

## Compatibility with plotting functions

The following functions are affected:

- `plot_univariate_effects`
- `convergence_plot`
- `lollipop_plot`
- `derivative_plot`

Most of them can be adjusted easily to the proposed changes. On all others we will
simply raise errors and provide tutorials to work around the limitations.

## Compatibility with Dashboard

The main challenge for the dashboard is that pytrees have no natural multi-column
extension and thus it becomes harder to specify a group or name column. However, these
features have not been used very much anyways.

We propose to write a better automatic grouping and naming function for pytrees. That
way it is simply not necessary to provide group and name columns and most of the users
will get a better dashboard experience.

Rules of thumb for both should be:

1. Only parameters where the start values have a similar magnitude can be in the same
   group, i.e. displayed in one lineplot.
1. Parameters that are close to each other in the tree (i.e. have a common beginning in
   their leaf_name should be in the same group.
1. The plot title should subsume the commen parts of the tree-structure (i.e. name we
   get from `pybaum.leaf_names`.
1. Most line plots should have approximately 5 lines, none should have more than 8.

## Advanced options for functions that work with pytrees

There are two argument to `tree_flatten` and other pytree functions that determine which
entries in a pytree are considered a leaf and which a container as well as how
containers are flattened. 1. `registry` and 2. `is_leaf`. See the documentation of
`pybaum` for details.

To allow for absolute flexibility, each function that works with pytrees needs to allow
a user to pass in a `registry` and an `is_leaf` argument. If a function works with
multiple pytrees (e.g. in `estimate_msm` the `params` are a pytree and
`emprirical_moments` are a pytree) it needs to allow users to pass in multiple
registries and is_leaf functions (e.g. `params_registry`, `params_is_leaf` and
`moments_registry`, `moments_is_leaf`.

However, we need only as many registries as there are different pytrees. For example
since `simulated_moments` and `empirical_moments` always need to be pytrees with the
same structure, they do not need separate registries and is_leaf functions.

## Pytree related reasons for a switch to result objects

There will be an other EEP that proposes to replace the result dictionaries we currently
use everywhere in estimagic by result objects. While this in not completely related to
pytrees, the switch to pytrees provides a few additional reasons:

1. Since we sometimes provide provide results in several formats (e.g. summaries as dict
   of pytrees and as pytree of DataFrames), the result dictionary would become too large
   and confusing. Having result objects that just calculate specific formats on demand
   can alleviate this.
1. The result object can serve as a simplfied wrapper to pytree functions and pytree
   conversion functions between pytree formats that abstracts from registry, is_leaf and
   treedefs.
1. Results objects allow to define a `__repr__` which becomes really useful as soon as
   parameters are not just one DataFrame but for example, a dict of DataFrames.

## Compatibility with JAX autodiff

While we allow for pytrees of arrays, numbers and DataFrames, JAX only allows pytrees of
arrays and numbers for automatic differentiation.

If you want to use automatic differentiation with estimagic you will thus have to
restrict yourself in the way you specify parameters.

[pybaum]: https://github.com/OpenSourceEconomics/pybaum
