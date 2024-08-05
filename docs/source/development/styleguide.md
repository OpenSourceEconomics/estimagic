(style_guide)=

# Styleguide

Your contribution should fulfill the criteria provided below.

## Styleguide for the codebase

- Functions have no side effect. : If you modify a mutable argument, make a copy at the
  beginning of the function.

- Use good names for functions and variables : *"You should name a variable using the
  same care with which you name a first-born child."*, Robert C. Martin, Clean Code: A
  Handbook of Agile Software Craftsmanship.

  A bit more concretely, this means:

  The length of a variable name should be proportional to its scope. In a list
  comprehension or short loop, i might be an acceptable name for the running variable,
  but variables that are used at many different places should have descriptive names.

  The name of variables should reflect the content or meaning of the variable and not
  only the type. Names like `dict_list` would not have been a good name for the
  `constraints`.

  Function names should contain a verb. Moreover, the length of a function name is
  typically inversely proportional to its scope. The public functions like `maximize`
  and `minimize` can have very short names. At a lower level of abstraction you
  typically need more words to describe what a function does.

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

- Fixed field types should only be used if all fields are known. An example where this
  is not the case are collections of benchmark problems, where the set of fields depends
  on the selected benchmark sets and other things. In such situations, dictionaries that
  map strings to BenchmarkProblem objects are a good idea.

- Think about autocomplete! If want to accept a string as argument (e.g. an algorithm
  name) also accept input types that are more amenable to static analysis and offer
  better autocomplete.

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

- Deep modules. : This is a term coined by
  [John Ousterhout](https://www.youtube.com/watch?v=bmSAYlu0NcY). A deep module is a
  module that has just one public function. This function calls the private functions
  (i.e. functions that start with an underscore) defined further down in the module and
  reads almost like a table of contents to the whole module.

- Never import a private function in another module : By following this strictly, you
  can be sure that you can rename or refactor private functions without looking at other
  modules. Of course it is also not a solution to copy paste the function! If you would
  like to import a function that starts with an underscore, rename it.

- All functions have a [Google style](https://tinyurl.com/mxams9k) docstring : The
  docstring describes all arguments and outputs. For arrays, please document how many
  dimensions and what shape they have. Look around in the code to find examples if you
  are in doubt. Example:

  ```python
  def ordered_logit(formula, data):
      """Estimate an ordered probit model with maximum likelihood.

      Args:
          formula (str): A patsy formula.
          data (str): A pandas DataFrame.

      Returns:
          res: optimization result.

      """
      pass
  ```

  In particular each docstring should start with a one liner that describes very
  concisely what the function does. The one liner should be in imperative mode, i.e. not
  "This function does" ..." , but "Do ..." and end with a period.

- Unit tests : If you write a small helper whose interface might change during
  refactoring, it is sufficient if the function that calls it is tested. But all
  functions that are exposed to the user must have unit tests.

- Enable pre-commit hooks by executing `pre-commit install` in a terminal in the root of
  the optimagic repository. This makes sure that your formatting is consistent with what
  we expect.

- Use `pathlib` for all file paths operations. : You can find the pathlib documentation
  [here](https://docs.python.org/3/library/pathlib.html)

- Object serialization. : Pickling and unpickling of DataFrames should be done with
  `pd.read_pickle` and `pd.to_pickle`.

- Don't use global variables unless absolutely necessary : Exceptions are global
  variables from a config file that replace magic numbers. Never use mutable global
  variables!

## Styleguide for the documentation

- General. : The documentation is rendered with
  [Sphinx](https://www.sphinx-doc.org/en/master/) and written in **Markedly Structured
  Text.** How-to guides are usually Jupyter notebooks.

- The documentation follows the [diataxis](https://diataxis.fr) framework.
