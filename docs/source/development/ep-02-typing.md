(eeptyping)=

# EP-02: Static typing

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Janos Gabler <https://github.com/janosg>`_                      |
+------------+------------------------------------------------------------------+
| Status     | Accepted                                                         |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2024-05-02                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Abstract

This enhancement proposal explains the adoption of static typing in optimagic. The goal
is to reap a number of benefits:

- Users will benefit from IDE tools such as easier discoverability of options and
  autocompletion.
- Developers and users will find code easier to read due to type hints.
- The codebase will become more robust due to static type checking and use of stricter
  types in internal functions.

Achieving these goals requires more than adding type hints. optimagic is currently
mostly [stringly typed](https://wiki.c2.com/?StringlyTyped). For example, optimization
algorithms are selected via strings. Another example are
[constraints](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html),
which are dictionaries with a fixed set of required keys.

This enhancement proposal outlines how we can accommodate the changes needed to reap the
benefits of static typing without breaking users' code in too many places.

## Motivation and resources

- [Writing Python like it's Rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html).
  A very good blogpost that summarizes the drawbacks of "stringly-typed" Python code and
  shows how to incorporate typing philosophies from Rust into Python projects. Read this
  if you don't have time to read the other resources.
- [Robust Python](https://www.oreilly.com/library/view/robust-python/9781098100650/), an
  excellent book that discusses how to design code around types and provides an
  introduction to static type checkers in Python.
- [jax enhancement proposal](https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html)
  for adopting static typing. It has a very good discussion on benefits of static
  typing.
- [Subclassing in Python Redux](https://hynek.me/articles/python-subclassing-redux/)
  explains which types of subclassing are considered harmful and was very helpful for
  designing this proposal.

(design-philosophy)=

## Design Philosophy

The core principles behind this enhancement proposal can be summarized by the following
points. This is an extension to our existing
[styleguide](https://estimagic.org/en/latest/development/styleguide.html) which will be
updated if this proposal is accepted.

- User facing functions should be generous regarding their input type. Example: the
  `algorithm` argument can be a string, `Algorithm` class or `Algorithm` instance. The
  `algo_options` can be an `AlgorithmOptions` object or a dictionary of keyword
  arguments.
- User facing functions should be strict about their output types. A strict output type
  does not just mean that the output type is known (and not a generous Union), but that
  it is a proper type that enables static analysis for available attributes. Example:
  whenever possible, public functions should not return dicts but proper result types
  (e.g. `OptimizeResult`, `NumdiffResult`, ...)
- Internal functions should be strict about input and output types; Typically, a public
  function will check all arguments, convert them to a proper type and then call an
  internal function. Example: `minimize` will convert any valid value for `algorithm`
  into an `Algorithm` instance and then call an internal function with that type.
- Each argument that previously accepted strings or option dictionaries now also accepts
  input types that are more amenable to static analysis and offer better autocomplete.
  Example: `algo_options` could just be a dict of keyword arguments. Now it can also be
  an `AlgorithmOptions` instance that enables autocomplete and static analysis for
  attribute access.
- Fixed field types should only be used if all fields are known. An example where this
  is not the case are collections of benchmark problems, where the set of fields depends
  on the selected benchmark sets and other things. In such situations, dictionaries that
  map strings to BenchmarkProblem objects are a good idea.
- For backwards compatibility and compatibility with SciPy, we allow things we don't
  find ideal (e.g. selecting algorithms via strings). However, the documentation should
  mostly show our prefered way of doing things. Alternatives can be hidden in tabs and
  expandable boxes.
- Whenever possible, use immutable types. Whenever things need to be changeable,
  consider using an immutable type with copy constructors for modified instances.
  Example: instances of `Algorithm` are immutable but using `Algorithm.with_option`
  users can create modified copies.
- The main entry point to optimagic are functions, objects are mostly used for
  configuration and return types. This takes the best of both worlds: we get the safety
  and static analysis that (in Python) can only be achieved using objects but the
  beginner friendliness and freedom provided by functions. Example: Having a `minimize`
  function, it is very easy to add the possibility of running minimizations with
  multiple algorithms in parallel and returning the best value. Having a `.solve` method
  on an algorithm object would require a whole new interface for this.

## Changes for optimization

The following changes apply to all functions that are directly related to optimization,
i.e. `maximize`, `minimize`, `slice_plot`, `criterion_plot`, `params_plot`,
`count_free_params`, `check_constraints` and `OptimizeResult`.

### The objective function

#### Current situation

The objective or criterion function is the function being optimized.

The same criterion function can work for scalar, least-squares and likelihood
optimizers. Moreover, a criterion function can return additional data that is stored in
the log file (if logging is active). All of this is achieved by returning a dictionary
instead of just a scalar float.

For the simplest case, where only scalar optimizers are used, `criterion` returns a
float. Here are two examples of this simple case.

The **first example** represents `params` as a flat numpy array and returns a float.
This would also be compatible with SciPy:

```python
def sphere(params: np.ndarray) -> float:
    return params @ params
```

The **second example** also returns a float but uses a different format for the
parameters:

```python
def dict_sphere(params: dict) -> float:
    return params["a"] ** 2 + params["b"] ** 2
```

If the user wants the criterion function to be compatible with specialized optimizers
for least-squares problems, the criterion function needs to return a dictionary.

```python
def least_squares_sphere(params: np.ndarray) -> dict[str, Any]:
    return {"root_contributions": params}
```

Here the `"root_contributions"` are the least-squares residuals. The dictionary key
tells optimagic how to interpret the output. This is needed because optimagic has no way
of finding out whether a criterion function that returns a vector (or pytree) is a
least-squares function or a likelihood function. Of course all specialized problems can
still be solved with scalar optimizers.

The criterion function can also return a dictionary, if the user wants to store some
information in the log file. This is independent of having a least-squares function or
not. An example is:

```python
def logging_sphere(x: np.ndarray) -> dict[str, Any]:
    return {"value": x @ x, "mean": x.mean(), "std": x.std()}
```

Here `"value"` is the actual scalar criterion value. All other fields are unknown to
optimagic and therefore just logged in the database if logging is active.

The specification of likelihood functions is very analogous to least-squares functions
and therefore omitted here.

**Things we want to keep**

- Allow using the same criterion function for scalar, likelihood and least-squares
  optimizers. This feature makes it easy to try out and compare very different
  algorithms with minimal code changes.
- No restrictions on the type of additional arguments of the criterion function.
- Maintain compatibility with scipy.optimize when the criterion function returns a
  scalar.

**Problems**

- Most users of optimagic find it hard to write criterion functions that return the
  correct dictionary. Therefore, they don't use the logging feature and we often get
  questions about specifying least-squares problems correctly.
- Internally we can make almost no assumptions about the output of a criterion function,
  making the code that processes the criterion output very complex and full of if
  conditions.
- We only know whether the specified criterion function is compatible with the selected
  optimizer after we evaluate it once. This means that users see errors only very late.
- While optional, in least-squares problems it is possible that a user specifies
  `root_contributions`, `contributions` and `value` even though any of them could be
  constructed out of the `root_contributions`. This redundancy of information means that
  we need to check the consistency of all user provided function outputs.

#### Proposal

In the current situation, the dictionary return type solves two different problems that
will now be solved separately.

##### Specifying different problem types

The simplest way of specifying a least-squares function becomes:

```python
import optimagic as om


@om.mark.least_squares
def ls_sphere(params):
    return params
```

Analogously, the simplest way of specifying a likelihood function becomes:

```python
@om.mark.likelihood
def ll_sphere(params):
    return params**2
```

The simplest way of specifying a scalar function stays unchanged, but optionally a
`mark.scalar` decorator can be used:

```python
@om.mark.scalar  # this is optional
def sphere(params):
    return params @ params
```

Except for the decorators, these three functions are specified the same way as in other
python libraries that support specialized optimizers (e.g.
`scipy.optimize.least_squares`). The reason why we need the decorators is that we
support all kinds of optimizers in the same interface.

##### Return additional information

If users additionally want to return information that should be stored in the log file,
they need to use a specific Object as return type.

```python
@dataclass(frozen=True)
class FunctionValue:
    value: float | PyTree
    info: dict[str, Any]
```

An example of a least-squares function that also returns additional info for the log
file would look like this:

```python
from optimagic import FunctionValue


@om.mark.least_squares
def least_squares_sphere(params):
    out = FunctionValue(
        value=params, info={"p_mean": params.mean, "p_std": params.std()}
    )
    return out
```

And analogous for scalar and likelihood functions, where again the `mark.scalar`
decorator is optional.

##### Optionally replace decorators by type hints

The purpose of the decorators is to tell us the output type of the criterion function.
This is necessary because there is no way of distinguishing between likelihood and
least-squares functions from the output alone and because we want to know the function
type before we evaluate the function once.

An alternative that might be more convenient for advanced Python programmers would be to
do this via type hints. In this case, the return types need to be a bit more
fine-grained:

```python
@dataclass(frozen=True)
class ScalarFunctionValue(FunctionValue):
    value: float
    info: dict[str, Any]


@dataclass(frozen=True)
class LeastSquaresFunctionValue(FunctionValue):
    value: PyTree
    info: dict[str, Any]


@dataclass(frozen=True)
class LikelihoodFunctionValue(FunctionValue):
    value: PyTree
    info: dict[str, Any]
```

A least-squares function could then be specified without decorator as follows:

```python
from optimagic import LeastSquaresFunctionValue


def least_squares_sphere(params: np.ndarray) -> LeastSquaresFunctionValue:
    out = LeastSquaresFunctionValue(
        value=params, info={"p_mean": params.mean, "p_std": params.std()}
    )
    return out
```

This approach works nicely in projects that use type hints already. However, it would be
hard for users who have never heard about type hints. Therefore, we should implement it
but not use it in beginner tutorials and always make clear that this is completely
optional.

##### Summary of output types

The output type of the objective function is `float | PyTree[float] | FunctionValue`.

### Bundling bounds

#### Current situation

Currently we have four arguments of `maximize`, `minimize`, and related functions that
let the user specify bounds:

```python
om.minimize(
    # ...
    lower_bounds=params - 1,
    upper_bounds=params + 1,
    soft_lower_bounds=params - 2,
    soft_lower_bounds=params + 2,
    # ...
)
```

Each of them is a pytree that mirrors the structure of `params` or `None`

**Problems**

- Usually, all of these arguments are used together and passing them around individually
  is annoying.
- The names are very long because the word `bounds` is repeated.

#### Proposal

We bundle the bounds together in a `Bounds` type:

```python
bounds = om.Bounds(
    lower=params - 1,
    upper=params + 1,
    soft_lower=params - 2,
    soft_lower=params + 2,
)
om.minimize(
    # ...
    bounds=bounds,
    # ...
)
```

As a bonus feature, the `Bounds` type can do some checks on the bounds at instance
creation time such that users get errors before running an optimization.

Using the old arguments will be deprecated.

Since there is no need to modify instances of `Bounds`, it should be immutable.

To improve the alignment with SciPy, we can also allow users to pass a
`scipy.optimize.Bounds` object as bounds. Internally, this will be converted to our
`Bounds` object.

### Constraints

#### Current situation

Currently, constraints are dictionaries with a set of required keys. The exact
requirements depend on the type of constraints and even on the structure of `params`.

Each constraint needs a way to select the parameters to which the constraint applies.
There are three dictionary keys for this:

- `"loc"`, which works if params are numpy arrays, `pandas.Series` or
  `pandas.DataFrame`.
- `"query"`, which works only if `params` are `pandas.DataFrame`
- `"Selector"`, which works for all valid formats of `params`.

Moreover, each constraint needs to specify its type using the `"type"` key.

Some constraints have additional required keys:

- Linear constraints have `"weights"`, `"lower_bound"`, `"upper_bound"`, and `"value"`.
- Nonlinear constraints have `"func"`, `"lower_bound"`, `"upper_bound"`, and `"value"`.

Details and examples can be found
[here](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html).

**Things we want to keep**

- The constraints interface is very declarative; Constraints purely collect information
  and are completely separate from the implementation.
- All three ways of selecting parameters have their strength and can be very concise and
  readable in specific applications.

**Problems**

- Constraints are hard to document and generally not understood by most users.
- Having multiple ways of selecting parameters (not all compatible with all `params`
  formats) is confusing for users and annoying when processing constraints. We have to
  handle the case where no selection or multiple selections are specified.
- Dicts with required keys are brittle and do not provide autocomplete. This is made
  worse by the fact that each type of constraint requires different sets of keys.

#### Proposal

1. We implement simple dataclasses for each type of constraint.
1. We get rid of `loc` and `query` as parameter selection methods. Instead, we show in
   the documentation how both selection methods can be used inside a `selector`
   function.

Examples of the new syntax are:

```python
constraints = [
    om.constraints.FixedConstraint(selector=lambda x: x[0, 5]),
    om.constraints.IncreasingConstraint(selector=lambda x: x[1:4]),
]

res = om.minimize(
    fun=criterion,
    params=np.array([2.5, 1, 1, 1, 1, -2.5]),
    algorithm="scipy_lbfgsb",
    constraints=constraints,
)
```

Since there is no need to modify instances of constraints, they should be immutable.

All constraints can subclass `Constraint` which will only have the `selector` attribute.
During the deprecation phase, `Constraint` will also have `loc` and `query` attributes.

The current `cov` and `sdcorr` constraints apply to flattened covariance matrices, as
well as standard deviations and flattened correlation matrices. This comes from a time
where optimagic only supported an essentially flat parameter format (`DataFrames` with
`"value"` column). We can exploit the current deprecation cycle to rename the current
`cov` and `sdcorr` constraints to `FlatCovConstraint` and `FlatSdcorrConstraint`. This
prepares the introduction of a more natural `CovConstraint` and `SdcorrConstraint`
later.

(algorithm-selection)=

### Algorithm selection

#### Current situation

`algorithm` is a string or a callable that satisfies the internal algorithm interface.
If the user passes a string, we look up the algorithm implementation in a dictionary
containing all installed algorithms. We implement suggestions for typical typos based on
fuzzy matching of strings.

**Things we want to keep**

- optimagic can be used just like scipy

**Problems**

- There is no autocomplete.
- It is very easy to make typos and they only get caught at runtime.
- Users cannot select algorithms without reading the documentation.

#### Proposal

The following proposal is quite ambitious and split into multiple steps. Thanks to
[@schroedk](https://github.com/schroedk) for helpful discussions on this topic.

##### Step 1: Passing algorithm classes and objects

For compatibility with SciPy we continue to allow algorithm strings. However, the
preferred ways of selecting algorithms are now:

1. Passing an algorithm class
1. Passing a configured algorithm object

Both new ways become possible because of changes to the internal algorithm interface.
See [here](algorithm-interface) for the proposal.

We remove the possibility of passing callables that comply with the old internal
algorithm interface.

In a simple example, algorithm selection via algorithm classes looks as follows:

```python
om.minimize(
    lambda x: x @ x,
    params=np.arange(5),
    algorithm=om.algorithms.scipy_neldermead,
)
```

Passing a configured instance of an algorithm looks as follows:

```python
om.minimize(
    lambda x: x @ x,
    params=np.arange(5),
    algorithm=om.algorithms.scipy_neldermead(adaptive=True),
)
```

##### Step 2: Achieving autocomplete without too much typing

There are many ways in which the above behavior could be achieved with full autocomplete
support. For reasons that will become clear in the next section, we choose to represent
`algorithms` as a dataclass. Alternatives are enums, `__init__` files, NamedTuples, etc.

A prototype for that dataclass looks as follows:

```python
from typing import Type


@dataclass(frozen=True)
class Algorithms:
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    # ...
    # many more
    # ...


algorithms = Algorithms()
```

Currently, all algorithms are collected in a dictionary that is created
programmatically. Representing algorithms in a static data structure instead requires a
lot more typing and therefore code to maintain. This situation will become even worse
with some of the features we propose below. Therefore, we want to automate the creation
of the dataclass.

To this end, we can write a function that automatically creates the code for the
`Algorithms` dataclass. This function can be executed in a local pre-commit hook to make
sure all generated code is up-to-date in every commit. It can also be executed in a
[pytest hook](https://docs.pytest.org/en/7.1.x/how-to/writing_hook_functions.html)
(before the collection phase) to make sure everything is up-to-date when tests run.

Users of optimagic (and their IDEs) will never know that this code was not typed in by a
human, which guarantees that autocomplete and static analysis will work without
problems.

```{note}
We can also use [pytest-hooks](https://docs.pytest.org/en/7.1.x/how-to/writing_hook_functions.html)
to make sure the
```

##### Step 3: Filtered autocomplete

Having the flat `Algorithms` data structure would be enough if every user knew exactly
which algorithm they want to use and just needed help typing in the name. However, this
is very far from realistic. Most users have little knowledge about optimization
algorithms. In the best case, they know a few properties of their problems (e.g. whether
it is differentiable) and their goal (e.g. do they need a local or global solution).

To exemplify what we want to achieve, assume a simplified situation with 4 algorithms.
We only consider whether an algorithm is gradient free or gradient based. Here is the
fictitious list:

- `neldermead`: `gradient_free`
- `bobyqa`: `gradient_free`
- `lbfgs`: `gradient_based`
- `slsqp`: `gradient_based`

We want the following behavior:

The user types `om.algorithms.` and autocomplete shows

|                 |
| --------------- |
| `GradientBased` |
| `GradientFree`  |
| `neldermead`    |
| `bobyqa`        |
| `lbfgs`         |
| `slsqp`         |

A user can either select one of the algorithms (lowercase) directly or filter further by
selecting a category (CamelCase). This would look as follows:

The user types `om.algorithms.GradientFree.` and autocomplete shows

|              |
| ------------ |
| `neldermead` |
| `bobyqa`     |

Once the user arrives at an algorithm, a subclass of `Algorithm` is returned. This class
will be passed to `minimize` or `maximize`. Passing configured instances of `Algorithm`s
will be discussed in [Algorithm Options](algorithm-options).

In practice, we would have a lot more algorithms and a lot more categories. Some
categories might be mutually exclusive, in that case the second category is omitted
after the first one is selected.

We have the following categories:

- `GradientBased` vs. `GradientFree`
- `Local` vs. `Global`
- `Bounded` vs. `Unbounded`
- `Scalar` vs. `LeastSquares` vs. `Likelihood`
- `LinearConstrained` vs. `NonlinearConstrained` vs. `Unconstrained`

Potentially, we could also offer a `.All` attribute that returns a list of all currently
selected algorithms. That way a user could for example loop over all `Bounded` and
`GradientBased` `LeastSquares` algorithms and compare them in a criterion plot.

These categories match nicely with our
[algorithm selection tutorials](https://effective-programming-practices.vercel.app/scientific_computing/optimization_algorithms/objectives_materials.html).

To achieve this behavior, we would have to implement something like this:

```python
@dataclass(frozen=True)
class GradientBasedAlgorithms:
    lbfgs: Type[LBFGS] = LBFGS
    slsqp: Type[SLSQP] = SLSQP

    @property
    def All(self) -> List[om.typing.Algorithm]:
        return [LBFGS, SLSQP]


@dataclass(frozen=True)
class GradientFreeAlgorithms:
    neldermead: Type[NelderMead] = NelderMead
    bobyqa: Type[Bobyqa] = Bobyqa

    @property
    def All(self) -> List[om.typing.Algorithm]:
        return [NelderMead, Bobyqa]


@dataclass(frozen=True)
class Algorithms:
    lbfgs: Type[LBFGS] = LBFGS
    slsqp: Type[SLSQP] = SLSQP
    neldermead: Type[NelderMead] = NelderMead
    bobyqa: Type[Bobyqa] = Bobyqa

    @property
    def GradientBased(self) -> GradientBasedAlgorithms:
        return GradientBasedAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeAlgorithms:
        return GradientFreeAlgorithms()

    @property
    def All(self) -> List[om.typing.Algorithm]:
        return [LBFGS, SLSQP, NelderMead, Bobyqa]
```

If implemented by hand, this would require an enormous amount of typing and introduce a
very high maintenance burden. Whenever a new algorithm was added to optimagic, we would
have to register it in multiple nested dataclasses.

The code generation approach detailed in the previous section can solve this problom.
While it might have been overkill to achieve basic autocomplete, it is justified to
achieve this filtering behavior. How the relevant information for filtering (e.g.
whether an algorithm is gradient based) is collected, will be discussed in
[internal algorithms](algorithm-interface).

```{note}
The use of dataclasses is an implementation detail. This enhancement proposal only
defines the autocomplete behavior we want to achieve. Everything else can be changed
later as we see fit.
```

(algorithm-options)=

### Algorithm options

Algorithm options refer to options that are not handled by optimagic but directly by the
algorithms. Examples are convergence criteria, stopping criteria and advanced
configuration of algorithms. Some of them are supported by many algorithms (e.g.
stopping after a maximum number of function evaluations is reached), some are supported
by certain classes of algorithms (e.g. most genetic algorithms have a population size,
most trustregion algorithms allow to set an initial trustregion radius) and some of them
are completely specific to one algorithm (e.g. ipopt has more than 100 very specific
options, `nag_dfols` supports very specific restarting strategies, ...).

While nothing can be changed about the fact that every algorithm supports different
options (e.g. there is simply no trustregion radius in a genetic algorithm), we go very
far in harmonizing `algo_options` across optimizers:

1. Options that are the same in spirit (e.g. stop after a specific number of iterations)
   get the same name across all optimizers wrapped in optimagic. Most of them even get
   the same default value.
1. Options that have non-descriptive (and often heavily abbreviated) names in their
   original implementation get more readable names, even if they appear only in a single
   algorithm.
1. Options that are specific to a well known optimizer (e.g. `ipopt`) are not renamed

#### Current situation

The user passes `algo_options` as a dictionary of keyword arguments. All options that
are not supported by the selected algorithm are discarded with a warning. The names of
most options are very descriptive (even though a bit too long at times).

We implement basic namespaces by introducing a dot notation. Example:

```python
options = {
    "stopping.max_iterations": 1000,
    "stopping.max_criterion_evaluations": 1500,
    "convergence.relative_criterion_tolerance": 1e-6,
    "convergence.scaled_gradient_tolerance": 1e-6,
    "initial_radius": 0.1,
    "population_size": 100,
}
```

The option dictionary is then used as follows:

```python
minimize(
    # ...
    algorithm="scipy_lbfgsb",
    algo_options=options,
    # ...
)
```

In the example, only the options `stopping.max_criterion_evaluations`,
`stopping.max_iterations` and `convergence.relative_criterion_tolerance` are supported
by `scipy_lbfgsb`. All other options would be ignored.

```{note}
The `.` notation in `stopping.max_iterations` is just syntactic sugar. Internally, the
option is called `stopping_max_iterations` because all options need to be valid
Python variable names.
```

**Things we want to keep**

- The ability to provide global options that are filtered for each optimizer. Mixing the
  options for all optimizers in a single dictionary and discarding options that do not
  apply to the selected optimizer allows to loop very efficiently over very different
  algorithms (without `if` conditions in the user's code). This is very good for quick
  experimentation, e.g. solving the same problem with three different optimizers and
  limiting each optimizer to 100 function evaluations.
- The basic namespaces help to quickly see what is influenced by a specific option. This
  works especially well to distinguish stopping options and convergence criteria from
  other tuning parameters of the algorithms. However, it would be enough to keep them as
  a naming convention if we find it hard to support the `.` notation.
- All options are documented in the optimagic documentation, i.e. we do not link to the
  docs of original packages. Now they will also be discoverable in an IDE.

**Problems**

- There is no autocomplete and the only way to find out which options are supported is
  the documentation.
- A small typo in an option name can easily lead to the option being discarded.
- Option dictionaries can grow very big.
- The fact that option dictionaries are mutable can lead to errors, for example when a
  user wants to try out a grid of values for one tuning parameter while keeping all
  other options constant.

#### Proposal

We want to offer multiple entry points for passing additional options to algorithms.
Users can pick the one that works best for their particular use-case. The current
solution remains valid but not recommended.

##### Configured algorithms

Instead of passing an `Algorithm` class (as described in
[Algorithm Selection](algorithm-selection)) the user can create an instance of their
selected algorithm. When creating the instance, they have autocompletion for all options
supported by the selected algorithm. `Algorithm`s are immutable.

```python
algo = om.algorithms.scipy_lbfgsb(
    stopping_max_iterations=1000,
    stopping_max_criterion_evaluations=1500,
    convergence_relative_criterion_tolerance=1e-6,
)
minimize(
    # ...
    algorithm=algo,
    # ...
)
```

##### Copy constructors on algorithms

Given an instance of an `Algorithm`, a user can easily create a modified copy of that
instance by using the `with_option` method.

```python
# using copy constructors to create variants
base_algo = om.algorithms.fides(stopping_max_iterations=1000)
algorithms = [base_algo.with_option(initial_radius=r) for r in [0.1, 0.2, 0.5]]

for algo in algorithms:
    minimize(
        # ...
        algorithm=algo,
        # ...
    )
```

We can provide additional methods `with_stopping` and `with_convergence` that call
`with_option` internally but provide two additional features:

1. They validate that the option is indeed a stopping/convergence criterion.
1. They allow to omit the `convergence_` or `stopping_` at the beginning of the option
   name and can thus reduce repetition in the option names. This recreates the
   namespaces we currently achieve with the dot notation:

```python
# using copy constructors for better namespaces
algo = (
    om.algorithms.scipy_lbfgsb()
    .with_stopping(
        max_iterations=1000,
        max_criterion_evaluations=1500,
    )
    .with_convergence(
        relative_criterion_tolerance=1e-6,
    )
)

minimize(
    # ...
    algorithm=algo,
    # ...
)
```

##### Global option object

As before, the user can pass a global set of options to `maximize` or `minimize`. We
continue to support option dictionaries but also allow `AlgorithmOption` objects that
enable better autocomplete and immutability. We can construct them using a similar
pre-commit hook approach as discussed in [algorithm selection](algorithm-selection).
Global options override the options that were directly passed to an optimizer. For
consistency, `AlgorithmOptions` can offer the `with_stopping`, `with_convergence` and
`with_option` copy-constructors, so users can modify options safely. Probably, this
approach should be featured less prominently in the documentation as it offers no
guarantees that the specified options are compatible with the selected algorithm.

The previous example continues to work. Examples of the new possibilities are:

```python
options = om.AlgorithmOptions(
    stopping_max_iterations=1000,
    stopping_max_criterion_evaluations=1500,
    convergence_relative_criterion_tolerance=1e-6,
    convergence_scaled_gradient_tolerance=1e-6,
    initial_radius=0.1,
    population_size=100,
)


minimize(
    # ...
    algorithm=om.algorithms.scipy_lbfgsb,
    algo_options=options,
    # ...
)
```

```{note}
In my currently planned implementation, autocomplete will not work reliably for the
copy constructors (`with_option`, `with_stopping` and `with_convergence`). The main
reason is that most editors do not play well with `functools.wraps` or any other means
of dynamic signature creation. For more details, see the discussions about the
[Internal Algorithm Interface](algorithm-interface).
```

### Custom derivatives

Providing custom derivatives to optimagic is slightly complicated because we support
scalar, likelihood and least-squares problems in the same interface. Moreover, we allow
to either provide a `derivative` function or a joint `criterion_and_derivative` function
that allow users to exploit synergies between evaluating the criterion and the
derivative.

#### Current situation

The `derivative` argument can currently be one of three things:

- A `callable`: This is assumed to be the relevant derivative of `criterion`. If a
  scalar optimizer is used, it is the gradient of the criterion value w.r.t. params. If
  a likelihood optimizer is used, it is the jacobian of the likelihood contributions
  w.r.t. params. If a least-squares optimizer is used, it is the jacobian of the
  residuals w.r.t. params.
- A `dict`: The dict must have three keys `"value"`, `"contributions"` and
  `"root_contributions"`. The corresponding values are the three callables described
  above.
- `None`: In this case, a numerical derivative is calculated.

The `criterion_and_derivative` argument exactly mirrors `derivative` but each callable
returns a tuple of the criterion value and the derivative instead.

**Things we want to keep**

- It is good that synergies between `criterion` and `derivative` can be exploited.
- There are three arguments (`criterion`, `derivative`, `criterion_and_derivative`).
  This makes sure that every algorithm can run efficiently when looping over algorithms
  and keeping everything else equal. With SciPy's approach of setting `jac=True` if one
  wants to use a joint criterion and derivative function, a gradient free optimizer
  would have no chance of evaluating just the criterion.
- Scalar, least-squares and likelihood problems are supported in one interface.

**Problems**

- A dict with required keys is brittle
- Autodiff needs to be handled completely outside of optimagic
- The names `criterion`, `derivative` and `criterion_and_derivative` are not aligned
  with scipy and very long.
- Providing derivatives to optimagic is perceived as complicated and confusing.

#### Proposal

```{note}
The following section uses the new names `fun`, `jac` and `fun_and_jac` instead of
`criterion`, `derivative` and `criterion_and_derivative`.
```

To improve the integration with modern automatic differentiation frameworks, `jac` or
`fun_and_jac` can also be a string `"jax"` or a more autocomplete friendly enum
`om.autodiff_backend.JAX`. This can be used to signal that the objective function is jax
compatible and jax should be used to calculate its derivatives. In the long run we can
add PyTorch support and more. Since this is mostly about a signal of compatibility, it
would be enough to set one of the two arguments to `"jax"`, the other one can be left at
`None`. Here is an example:

```python
import jax.numpy as jnp
import optimagic as om


def jax_sphere(x):
    return jnp.dot(x, x)


res = om.minimize(
    fun=jax_sphere,
    params=jnp.arange(5),
    algorithm=om.algorithms.scipy_lbfgsb,
    jac="jax",
)
```

If a custom callable is provided as `jac` or `fun_and_jac`, it needs to be decorated
with `@om.mark.least_squares` or `om.mark.likelihood` if it is not the gradient of a
scalar function values. Using the `om.mark.scalar` decorator is optional. For a simple
least-squares problem this looks as follows:

```python
import numpy as np


@om.mark.least_squares
def ls_sphere(params):
    return params


@om.mark.least_squares
def ls_sphere_jac(params):
    return np.eye(len(params))


res = om.minimize(
    fun=ls_sphere,
    params=np.arange(5),
    algorithm=om.algorithms.scipy_ls_lm,
    jac=ls_sphere_jac,
)
```

Note that here we have a least-squares problem and solve it with a least-squares
optimizer. However, any least-squares problem can also be solved with scalar optimizers.

While optimagic could convert the least-squares derivative to the gradient of the scalar
function value, this is generally inefficient. Therefore, a user can provide multiple
callables of the objective function in such a case, so we can pick the best one for the
chosen optimizer.

```python
@om.mark.scalar
def sphere_grad(params):
    return 2 * params


res = om.minimize(
    fun=ls_sphere,
    params=np.arange(5),
    algorithm=om.algorithms.scipy_lbfgsb,
    jac=[ls_sphere_jac, sphere_grad],
)
```

Since a scalar optimizer was chosen to solve the least-squares problem, optimagic would
pick the `sphere_grad` as derivative. If a leas-squares solver was chosen, we would use
`ls_sphere_jac`.

### Other option dictionaries

#### Current situation

We often allow to switch on some behavior with a bool or a string value and then
configure the behavior with an option dictionary. Examples are:

- `logging` (`str | pathlib.Path | False`) and `log_options` (dict)
- `scaling` (`bool`) and `scaling_options` (dict)
- `error_handling` (`Literal["raise", "continue"]`) and `error_penalty` (dict)
- `multistart` (`bool`) and `multistart_options`

Moreover we have option dictionaries whenever we have nested invocations of optimagic
functions. Examples are:

- `numdiff_options` in `minimize` and `maximize`
- `optimize_options` in `estimate_msm` and `estimate_ml`

**Things we want to keep**

- Complex behavior like logging or multistart can be switched on in extremely simple
  ways, without importing anything and without looking up supported options.
- The interfaces are very declarative and decoupled from our implementation.

**Problems**

- Option dictionaries are brittle and don't support autocomplete.
- It can be confusing if someone provided `scaling_options` or `multistart_options` but
  they take no effect because `scaling` or `multistart` were not set to `True`.

#### Proposal

We want to keep a simple way of enabling complex behavior (with some default options)
but get rid of having two separate arguments (one to switch the behavior on and one to
configure it). This means that we have to be generous regarding input types.

##### Logging

Currently we only implement logging via an sqlite database. All `log_options` are
specific to this type of logging. However, logging is slow and we should support more
types of logging. For this, we can implement a simple `Logger` abstraction. Advanced
users could implement their own logger.

After the changes, `logging` can be any of the following:

- `False` (or anything Falsy): No logging is used.
- A `str` or `pathlib.Path`: Logging is used at default options.
- An instance of `optimagic.Logger`. There will be multiple subclasses, e.g.
  `SqliteLogger` which allow us to switch out the logging backend. Each subclass might
  have different optional arguments.

The `log_options` are deprecated. Using dictionaries instead of `Option` objects will be
supported during a deprecation cycle.

##### Scaling, error handling and multistart

In contrast to logging, scaling, error handling and multistart are deeply baked into
optimagic's minimize function. Therefore, it does not make sense to create abstractions
for these features that would make them replaceable components that can be switched out
for other implementations by advanced users. Most of these features are already
perceived as advanced and allow for a lot of configuration.

We therefore suggest the following argument types:

- `scaling`: `bool | ScalingOptions`
- `error_handling`: `bool | ErrorHandlingOptions`
- `multistart`: `bool | MultistartOptions`

All of the Option objects are simple dataclasses that mirror the current dictionaries.
All `_options` arguments are deprecated.

##### `numdiff_options` and similar

Dictionaries are still supported but we also offer more autocomplete friendly
dataclasses as alternative.

(algorithm-interface)=

### The internal algorithm interface and `Algorithm` objects

#### Current situation

Currently, algorithms are defined as `minimize` functions that are decorated with
`om.mark_minimizer`. The `minimize` function returns a dictionary with a few mandatory
and several optional keys. Algorithms can provide information to optimagic in two ways:

1. The signature of the minimize function signals whether the algorithm needs
   derivatives and whether it supports bounds and nonlinear constraints. Moreover, it
   signals which algorithm specific options are supported. Default values for algorithm
   specific options are also defined in the signature of the minimize function.
1. `@mark_minimizer` collects the following information via keyword arguments:

- Is the algorithm a scalar, least-squares or likelihood optimizer?
- The algorithm name.
- Does the algorithm require well scaled problems?
- Is the algorithm currently installed?
- Is the algorithm global or local?
- Should the history tracking be disabled (e.g. because the algorithm tracks its own
  history)?
- Does the algorithm parallelize criterion evaluations?

A slightly simplified example of the current internal algorithm interface is:

```python
@mark_minimizer(
    name="scipy_neldermead",
    needs_scaling=False,
    primary_criterion_entry="value",
    is_available=IS_SCIPY_AVAILABLE,
    is_global=False,
    disable_history=False,
)
def scipy_neldermead(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=1_000_000,
    stopping_max_criterion_evaluations=1_000_000,
    convergence_absolute_criterion_tolerance=1e-8,
    convergence_absolute_params_tolerance=1e-8,
    adaptive=False,
):
    options = {
        "maxiter": stopping_max_iterations,
        "maxfev": stopping_max_criterion_evaluations,
        # both tolerances seem to have to be fulfilled for Nelder-Mead to converge.
        # if not both are specified it does not converge in our tests.
        "xatol": convergence_absolute_params_tolerance,
        "fatol": convergence_absolute_criterion_tolerance,
        "adaptive": adaptive,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        method="Nelder-Mead",
        options=options,
    )

    return process_scipy_result(res)
```

The first two arguments (`criterion` and `x`) are mandatory. The lack of any arguments
related to derivatives signifies that `scipy_neldermead` is a gradient free algorithm.
The bounds related arguments show that it supports box constraints. The remaining
arguments define the supported stopping criteria and algorithm options as well as their
default values.

The decorator simply attaches information to the function as `_algorithm_info`
attribute. This originated as a hack but was never changed afterwards. The
`AlgorithmInfo` looks as follows:

```python
class AlgoInfo(NamedTuple):
    primary_criterion_entry: str
    name: str
    parallelizes: bool
    needs_scaling: bool
    is_available: bool
    arguments: list  # this is read from the signature
    is_global: bool = False
    disable_history: bool = False
```

**Things we want to keep**

- The internal interface has proven flexible enough for many optimizers we had not
  wrapped when we designed it. It is easy to add more optional arguments to the
  decorator without breaking any existing code.
- The decorator approach completely hides how we represent algorithms internally.
- Since we read a lot of information from function signatures (as opposed to registering
  options somewhere), there is no duplicated information. If we change the approach to
  collecting information, we still need to ensure there is no duplication or possibility
  to provide wrong information to optimagic.

**Problems**

- Type checkers complain about the `._algorithm_info` hack.
- All computations and signature checking are done eagerly for all algorithms at import
  time. This is one of the reasons why imports are slow.
- The first few arguments to the minimize functions follow a naming scheme and any typo
  in those names would lead to situations that are hard to debug (e.g. if `lower_bound`
  was miss-typed as `lower_buond` we would assume that the algorithm does not support
  lower bounds but has a tuning parameter called `lower_buond`).

#### Proposal

We first show the proposed new algorithm interface and discuss the changes later.

```python
@om.mark.minimizer(
    name="scipy_neldermead",
    needs_scaling=False,
    problem_type=om.ProblemType.Scalar,
    is_available=IS_SCIPY_AVAILABLE,
    is_global=False,
    disable_history=False,
    needs_derivatives=False,
    needs_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
)
@dataclass(frozen=True)
class ScipyNelderMead(Algorithm):
    stopping_max_iterations: int = 1_000_000
    stopping_max_criterion_evaluations: int = 1_000_000
    convergence_absolute_criterion_tolerance: float = 1e-8
    convergence_absolute_params_tolerance: float = 1e-8
    adaptive = False

    def __post_init__(self):
        # check everything that cannot be handled by the type system
        assert self.convergence_absolute_criterion_tolerance > 0
        assert self.convergence_absolute_params_tolerance > 0

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[float]
    ) -> InternalOptimizeResult:
        options = {
            "maxiter": self.stopping_max_iterations,
            "maxfev": self.stopping_max_criterion_evaluations,
            "xatol": self.convergence_absolute_params_tolerance,
            "fatol": self.convergence_absolute_criterion_tolerance,
            "adaptive": self.adaptive,
        }

        res = minimize(
            fun=problom.scalar.fun,
            x0=x,
            bounds=_get_scipy_bounds(problom.bounds),
            method="Nelder-Mead",
            options=options,
        )

        return process_scipy_result(res)
```

1. The new internal algorithms are dataclasses, where all algorithm options are
   dataclass fields. This enables us to obtain information about the options via the
   `__dataclass_fields__` attribute without inspecting signatures or imposing naming
   conventions on non-option arguments.
1. The `_solve_internal_problem` method receives an instance of `InternalProblem` and
   `x0` (the start values) as arguments. `InternalProblem` collects the criterion
   function, its derivatives, bounds, etc. This again avoids any potential for typos in
   argument names.
1. The `mark.minimizer` decorator collects all the information that was previously
   collected via optional arguments with naming conventions. This information is
   available while constructing the instance of `InternalProblem`. Thus we can make sure
   that attributes that were not requested (e.g. derivatives if `needs_derivative` is
   `False`) raise an `AttributeError` if used.
1. The minimize function returns an `InternalOptimizeResult` instead of a dictionary.

The copy constructors (`with_option`, `with_convergence`, and `with_stopping`) are
inherited from `optimagic.Algorithm`. This means, that they will have `**kwargs` as
signature and thus do not support autocomplete. However, they can check that all
specified options are actually in the `__dataclass_fields__` and thus provide feedback
before an optimization is run.

All breaking changes of the internal algorithm interface are done without deprecation
cycle.

```{note}
The `_solve_internal_problem` method is private because users should not call it; This
also prepares adding a public `minimize` method that internally calls the
`minimize` function.
```

To make things more concrete, here are prototypes for components related to the
`InternalProblem` and `InternalOptimizeResult`.

```{note}
The names of the internal problem are already aligned with the new names for
the objective function and its derivatives.
```

```python
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Callable, Tuple
import optimagic as om


@dataclass(frozen=True)
class ScalarProblemFunctions:
    fun: Callable[[NDArray[float]], float]
    jac: Callable[[NDArray[float]], NDArray[float]]
    fun_and_jac: Callable[[NDArray[float]], Tuple[float, NDArray[float]]]


@dataclass(frozen=True)
class LeastSquaresProblemFunctions:
    fun: Callable[[NDArray[float]], NDArray[float]]
    jac: Callable[[NDArray[float]], NDArray[float]]
    fun_and_jac: Callable[[NDArray[float]], Tuple[NDArray[float], NDArray[float]]]


@dataclass(frozen=True)
class LikelihoodProblemFunctions:
    fun: Callable[[NDArray[float]], NDArray[float]]
    jac: Callable[[NDArray[float]], NDArray[float]]
    fun_and_jac: Callable[[NDArray[float]], Tuple[NDArray[float], NDArray[float]]]


@dataclass(frozen=True)
class InternalProblem:
    scalar: ScalarProblemFunctions
    least_squares: LeastSquaresProblemFunctions
    likelihood: LikelihoodProblemFunctions
    bounds: om.Bounds | None
    linear_constraints: list[om.LinearConstraint] | None
    nonlinear_constraints: list[om.NonlinearConstraint] | None
```

The `InternalOptimizeResult` formalizes the current dictionary solution:

```python
@dataclass(frozen=True)
class InternalOptimizeResult:
    solution_x: NDArray[float]
    solution_criterion: float
    n_criterion_evaluations: int | None
    n_derivative_evaluations: int | None
    n_iterations: int | None
    success: bool | None
    message: str | None
```

#### Alternative to `mark.minimizer`

Instead of collecting information about the optimizers via the `mark.minimizer`
decorator, we could require the `Algorithm` subclasses to provide that information via
class variables. The presence of all required class variables could be enforced via
`__init_subclass__`.

The two approaches are equivalent in terms of achievable functionality. I see the
following advantages and disadvantages:

**Advantages of decorator approach**

- Easier for beginners as no subtle concepts (such as the difference between instance
  and class variables) are involved
- Very easy way to provide default values for some of the collected variables
- Every user of optimagic is familiar with `mark` decorators
- Autocomplete while filling out the arguments of the mark decorator
- Very clear visual separation of algorithm options and attributes optimagic needs to
  know about.

**Advantages of class variable approach**

- More familiar for people with object oriented background
- Possibly better ways to enforce the presence of the class variables via static
  analysis

I am personally leaning towards the decorator approach but any feedback on this topic is
welcome.

## Numerical differentiation

### Current situation

The following proposal applies to the functions `first_derivative` and
`second_derivative`. Both functions have an interface that has grown over time and both
return a relatively complex result dictionary. There are several arguments that govern
which entries are stored in the result dictionary.

The functions `first_derivative` and `second_derivative` allow params to be arbitrary
pytrees. They work for scalar and vector valued functions and a `key` argument makes
sure that they work for `criterion` functions that return a dict containing `"value"`,
`"contributions"`, and `"root_contributions"`.

In contrast to optimization, all pytree handling (for params and function outputs) is
mixed with the calculation of the numerical derivatives. This can produce more
informative error messages and save some memory. However it increases complexity
extremely because we can make very few assumptions on types. There are many if
conditions to deal with this situation.

The interface is further complicated by supporting Richardson Extrapolation. This
feature was inspired by [numdifftools](https://numdifftools.readthedocs.io/en/latest/)
but has not produced convincing results in benchmarks.

**Things we want to keep**

- `params` and function values can be pytrees
- support for optimagic `criterion` functions (now functions that return
  `FunctionValue`)
- Many optional arguments to influence the details of the numerical differentiation
- Rich output format that helps to get insights on the precision of the numerical
  differentiation
- Ability to optionally pass in a function evaluation at `params` or return a function
  evaluation at `params`

**Problems**

- We can make no assumptions on types inside the function because pytree handling is
  mixed with calculations
- Support for Richardson extrapolation complicates the interface and implementation but
  has not been convincing in benchmarks
- Pytree handling is acatually incomplete (`base_steps`, `min_steps` and `step_ratio`
  are assumed to be flat numpy arrays)
- Many users expect the output of a function for numerical differentiation to be just
  the gradient, jacobian or hessian, not a more complex result object.

### Proposal

#### Separation of calculations and pytree handling

As in numerical optimization, we should implement the core functionality for first and
second derivative for functions that map from 1-Dimensional numpy arrays to
1-Dimensional numpy arrays. All pytree handling or other handling of function outputs
(e.g. functions that return a `FunctionValue`) should be done outside of the core
functions.

#### Deprecate Richardson Extrapolation (and prepare alternatives)

The goal of implementing Richardson Extrapolation was to get more precise estimates of
numerical derivatives when it is hard to find an optimal step size. Example use-cases we
had in mind were:

- Optimization of a function that is piecewise flat, e.g. the likelihood function of a
  naively implemented multinomial probit
- Optimization or standard error estimation of slightly noisy functions, e.g. functions
  of an MSM estimation problem
- Standard error estimation of wiggly functions where the slope and curvature at the
  minimum does not yield reasonable standard errors and confidence intervals

Unfortunately, the computational cost of Richardson extrapolation is too high for any
application during optimization. Moreover, our practical experience with Richardson
Extrapolation was not positive and it seems that Richardson extrapolation is not
designed for our use-cases. It is designed as a sequence acceleration method that
reduces roundoff error while shrinking a step size to zero, whereas in our application
it might often be better to take a larger step size (for example, the success of
derivative free trust-region optimizers suggest less local slope and curvature
information is more useful than actual derivatives for optimization; similarly,
numerical derivatives with larger step sizes could be seen as an estimate of a
[quasi jacobian](https://arxiv.org/abs/1907.13093) and inference based on it might have
good statistical properties).

We therefore propose to remove Richardson extrapolation and open an Issue to work on
alternatives. Examples for alternatives could be:

- [Moré and Wild (2010)](https://www.mcs.anl.gov/papers/P1785.pdf) propose an approach
  to calculate optimal step sizes for finite difference differentiation of noisy
  functions
- We could think about aggregating derivative estimates at multiple step sizes in a way
  that produces worst case standard errors and confidence intervals
- ...

```{note}
Richardson extrapolation was only completed for first derivatives, even though it is
already prepared in the interface for second derivatives.
```

#### Better `NumdiffResult` object

The result dictionary will be replaced by a `NumdiffResult` object. All arguments that
govern which results are stored will be removed. If some of the formerly optional
results require extra computation that we wanted to avoid by making them optional, they
can be properties or methods of the result object.

#### Jax inspired high-level interfaces

Since our `first_derivative` and `second_derivative` functions need to fulfill very
specific requirements for use during optimization, they need to return a complex result
object. However, this can be annoying in simple situations where users just want a
gradient, jacobian or hessian.

To cover these simple situations and provide a high level interface to our numdiff
functions, we can provide a set of jax inspired decorators:

- `@grad`
- `@value_and_grad`
- `@jac` (no distinction between `@jacrev` and `jacfwd` necessary)
- `@value_and_jac`
- `@hessian`
- `@value_and_hessian`

All of these will be very simple wrappers around `first_derivative` and
`second_derivative` with very low implementation and maintenance costs.

## Benchmarking

### `get_benchmark_problems`

#### Current situation

As other functions in optimagic, `get_benchmark_problems` follows a design where
behavior can be switched on by a bool and configured by an options dictionary. The
following arguments are related to this:

- `additive_noise` and `additive_noise_options`
- `multiplicative_noise` and `multiplicative_noise_options`
- `scaling` and `scaling_options`

All of them have the purpose of adding some difficult characteristics to an existing
benchmark set, so we can analyze how well an optimizer can deal with this situation.

The name of the benchmark set is passed in as a string.

The return value of `get_benchmark_problems` is a nested dictionary. The keys in the
outer dictionary are the names of benchmark problems. The inner dictionaries represent
benchmark problems.

**Things we want to keep**

- Benchmark problems are collected in a dict, not in a fixed-field data structure. This
  makes it easy to merge problems from multiple benchmark sets or filter benchmark sets.
  A fixed field data structure would not work here.

**Problems**

- As discussed before, having separate arguments for switching-on behavior and
  configuring it can be dangerous
- Each single benchmark problem should not be represented as a dictionary
- Adding noise or scaling problems should be made more flexible and generic

#### Proposal

##### Add noise to benchmark problems

The four arguments `additive_noise`, `multiplicative_noise`, `additive_noise_options`,
and `multiplicative_noise_options` are combined in one `noise` argument. This `noise`
argument can be `bool | BenchmarkNoise`. If `False`, no noise is added. If `True`,
standard normal noise is added.

We implement several subclasses of `BenchmarkNoise` to cover the current use cases. As
syntactic sugar, we can make `BenchmarkNoise` instances addable (by implementing an
`__add__` method) so multiple sources of noise can be combined.

A rough prototype for `BenchmarkNoise` looks as follows:

```python
FvalType = TypeVar("FvalType", bound=float | NDArray[float])


class BenchmarkNoise(ABC):
    @abstractmethod
    def draw_noise(
        self, fval: FvalType, params: NDArray, size: int, rng: np.random.Generator
    ) -> FvalType:
        pass

    def __add__(self, other: BenchmarkNoise):
        pass
```

Passing `fval` and `params` to `draw_noise` enables use to implement multiplicative
noise (i.e. noise where the standard deviation scales with the function value) and
stochastic or deterministic wiggle (e.g. a sine curve that depends on params).
Therefore, this proposal does not just cover everything that is currently implemented
but also adds new functionality we wanted to implement.

##### Add scaling issues to benchmark problems

The `scaling_options` argument is deprecated. The `scaling` argument can be
`bool | BenchmarkScaler`. We implement `LinspaceBenchmarkScaler` to cover everything
that is implemented right now but more types of scaling can be implemented in the
future. A rough prototype of `BenchmarkScaler` looks as follows:

```python
class BenchmarkScaler(ABC):
    @abstractmethod
    def scale(self, params: NDArray) -> NDArray:
        pass

    @abstractmethod
    def unscale(self, params: NDArray) -> NDArray:
        pass
```

##### Representing benchmark problems

Instead of the fixed-field dictionary we will have a dataclass with corresponding
fields. This would roughly look as follows:

```python
@dataclass
class BenchmarkProblem:
    fun: Callable[[NDArray], FunctionValue]
    start_x: NDArray
    solution_x: NDArray | None
    start_fun: float
    solution_fun: float
```

### `run_benchmark`

#### Current situation

`run_benchmark` takes `benchmark_problems` (covered in the previous section),
`optimize_options` and a few other arguments and returns a nested dictionary
representing benchmark results.

`optimize_options` can be a list of algorithm names, a dict with algorithm names as
values or a nested dict of keyword arguments for `minimize`.

**Things we want to keep**

- Benchmark results are collected in a dict, not in a fixed-field data structure. This
  makes it easy to merge results from multiple benchmark sets or filter benchmark
  results. A fixed field data structure would not work here.

**Problems**

- `optimize_options` are super flexible but error prone and hard to write as there is no
  autocomplete support
- Each single benchmark result should not be represented as a dictionary

#### Proposal

We restrict the typo of `optimize_options` to
`dict[str, Type[Algorithm] | Algorithm | OptimizeOptions]`. Here, `OptimizeOptions` will
be a simple dataclass that we need for `estimate_ml` and `estimate_msm` anyways.

Passing just lists of algorithm names is deprecated. Passing dicts as optimize options
is also deprecated. Most use-cases will be covered by passing dictionaries of configured
Algorithms as optimize options. Actually using the full power of passing
`OptimizeOptions` will be rarely needed.

The return type of `run_benchmark` will be `dict[tuple[str], BenchmarkResult]`

`BenchmarkResult` is a dataclass with fields that mirror the keys of the current
dictionary. It will roughly look as follows:

```python
@dataclass
class BenchmarkResult:
    params_history: list[NDArray]
    fun_history: list[float]
    time_history: list[float]
    batches_history: list[int]
    solution: OptimizeResult
```

## Estimation

The changes to the estimation functions `estimate_ml` and `estimate_msm` will be
minimal:

- `lower_bounds` and `upper_bounds` are replaced by `bounds` (as in optimization)
- `numdiff_options` and `optimize_options` become dataclasses
- `logging` and `log_options` get aligned with our proposal for optimization

In the long run we plan a general overhaul of `MSM` estimation that provides better
access to currently internal objects such as the MSM objective function.

## Type checkers and their configuration

We choose mypy as static type checker and run it as part of our continuous integration.

Once this enhancement proposal is fully implemented, we want to use the following
settings:

```
check_untyped_defs = true
disallow_any_generics = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
```

In addition to CI, we could also run type-checks as part of the pre-commit hooks. An
example where this is done can be found
[here](https://github.com/google/jax/blob/de0fd722f0c4c0c238884f0e64e4ef8da72e4c1d/.pre-commit-config.yaml#L33).

## Runtime type checking

Since most of our users do not use static type checkers we will still need to check the
type of most user inputs so we can give them early feedback when problems arise. Thus we
cannot remove our current error handling just because many of these errors could now be
caught by static analysis.

We can investigate using `jaxtyping`'s pytest hooks to enable runtime typecheckers like
beartype during testing but it is not a priority for now.

## Changes in documentation

All type information in docstrings will be removed.

Whenever there are now multiple ways of doing things, we show the ones that support
autocomplete and static analysis most prominently. We can achieve this via tabs, similar
to how
[pytask](https://pytask-dev.readthedocs.io/en/stable/tutorials/defining_dependencies_products.html#products)
does it.

The general structure of the documentation is not affected by this enhancement proposal.

## Summary of breaking changes

- The internal algorithm interface changes completely without deprecations
- The support for Richardson Extrapolation in `first_derivative` is dropped without
  deprecation; The corresponding arguments `n_steps` and `step_ratio` are removed.
- The return type of `first_derivative` and `second_derivative` changes from dict to
  `NumdiffResult` without deprecations. The arguments `return_func_value` and
  `return_info` are dropped.
- The representation of benchmark problems and benchmark results changes without
  deprecations

## Summary of deprecations

The following deprecations become active in version `0.5.0`. The functionality will be
removed in version `0.6.0` which should be scheduled for approximately half a year after
the realease of `0.5.0`.

- Returning a `dict` in the objective function io deprecated. Return `FunctionValue`
  instead. In addition, likelihood and least-squares problems need to be decorated with
  `om.mark.likelihood` and `om.mark_least_squares`.
- The arguments `lower_bounds`, `upper_bounds`, `soft_lower_bounds` and
  `soft_upper_bounds` are deprecated. Use `bounds` instead. `bounds` can be
  `optimagic.Bounds` or `scipy.optimize.Bounds` objects.
- Specifying constraints with dictionaries is deprecated. Use the corresponding subclass
  of `om.constraints.Constraint` instead. In addition, all selection methods except for
  `selector` are deprecated.
- The `covariance` constraint is renamed to `FlatCovConstraint` and the `sdcorr`
  constraint is renamed to `FlatSdcorrConstraint` to prepare the introduction of more
  natural (non-flattened) covariance and sdcorr constraints.
- The `log_options` argument of `maximize` and `minimize` is deprecated and gets
  subsumed in the `logging` argument.
- The `scaling_options` argument of `maximize` and `minimize` is deprecated and gets
  subsumed in the `scaling` argument.
- The `error_penalty` argument of `maximize` and `minimize` is deprecated and gets
  subsumed in the `error_handling` argument.
- The `multistart_options` argument of `maximize` and `minimize` is deprecated and gets
  subsumed in the `multistart` argument.
- The arguments `additive_noise`, `additive_noise_options`, `multiplicative_noise`, and
  `multiplicative_noise_options` in `get_benchmark_problems` are deprecated and combined
  into `noise`.
- The `scaling_options` argument in `get_benchmark_problems` is deprecated and subsumed
  in the `scaling` argument.
- Passing just a list of algorithm strings as `optimize_options` in `run_benchmark` is
  deprecated.
