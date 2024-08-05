# Changes

This is a record of all past optimagic releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/) and all
releases are available on [Anaconda.org](https://anaconda.org/OpenSourceEconomics/optimagic).

Following the [scientific python guidelines](https://scientific-python.org/specs/spec-0000/)
we drop the official support for Python 3.9.


## 0.5.0

This is a major release with several breaking changes and deprecations. On a high level,
the major changes are:

- Implement EP-02: Static typing
- Implement EP-03: Alignment with SciPy
- Rename the package from `estimagic` to `optimagic` (while keeping the `estimagic`
  namespace for the estimation capabilities).


- {gh}`500` removes the dashboard, the support for simopt optimizers and the
  `derivative_plot` ({ghuser}`janosg`)
- {gh}`504` aligns `maximize` and `minimize` more closely with scipy. All related
  deprecations and breaking changes are listed below. As a result, scipy code that uses
  minimize with the arguments `x0`, `fun`, `jac` and `method` will run without changes
  in optimagic. Similarly, to `OptimizeResult` gets some aliases so it behaves more
  like SciPy's.

### Breaking changes


### Deprecations

- The `criterion` argument of `maximize` and `minimize` is renamed to `fun` (as in
  SciPy).
- The `derivative` argument of `maximize` and `minimize` is renamed to `jac` (as
  in SciPy)
- The `criterion_and_derivative` argument of `maximize` and `minimize` is renamed
  to `fun_and_jac` to align it with the other names.
- The `criterion_kwargs` argument of `maximize` and `minimize` is renamed to
  `fun_kwargs` to align it with the other names.
- The `derivative_kwargs` argument of `maximize` and `minimize` is renamed to
  `jac_kwargs` to align it with the other names.
- The `criterion_and_derivative_kwargs` argument of `maximize` and `minimize` is
  renamed to `fun_and_jac_kwargs` to align it with the other names.
- Algorithm specific convergence and stopping criteria are renamed to align them more
  with NlOpt and SciPy names.
    - `convergence_relative_criterion_tolerance` -> `convergence_ftol_rel`
    - `convergence_absolute_criterion_tolerance` -> `convergence_ftol_abs`
    - `convergence_relative_params_tolerance` -> `convergence_xtol_rel`
    - `convergence_absolute_params_tolerance` -> `convergence_xtol_abs`
    - `convergence_relative_gradient_tolerance` -> `convergence_gtol_rel`
    - `convergence_absolute_gradient_tolerance` -> `convergence_gtol_abs`
    - `convergence_scaled_gradient_tolerance` -> `convergence_gtol_scaled`
    - `stopping_max_criterion_evaluations` -> `stopping_maxfun`
    - `stopping_max_iterations` -> `stopping_maxiter`


## 0.4.7

This release contains minor improvements and bug fixes. It is the last release before
the package will be renamed to optimagic and two large enhancement proposals will be
implemented.

- {gh}`490` adds the attribute `optimize_result` to the `MomentsResult` class
  ({ghuser}`timmens`)
- {gh}`483` fixes a bug in the handling of keyword arguments in `bootstrap`
  ({ghuser}`alanlujan91`)
- {gh}`477` allows to use an identity weighting matrix in MSM estimation
  ({ghuser}`sidd3888`)
- {gh}`473` fixes a bug where bootstrap keyword arguments were ignored
  `get_moments_cov` ({ghuser}`timmens`)
- {gh}`467`, {gh}`478`, {gh}`479` and {gh}`480` improve the documentation
  ({ghuser}`mpetrosian`, {ghuser}`segsell`, and {ghuser}`timmens`)


## 0.4.6

This release drastically improves the optimizer benchmarking capabilities, especially
with noisy functions and parallel optimizers. It makes tranquilo and numba optional
dependencies and is the first version of estimagic to be compatible with Python
3.11.


- {gh}`464` Makes tranquilo and numba optional dependencies ({ghuser}`janosg`)
- {gh}`461` Updates docstrings for procss_benchmark_results ({ghuser}`segsell`)
- {gh}`460` Fixes several bugs in the processing of benchmark results with noisy
  functions ({ghuser}`janosg`)
- {gh}`459` Prepares benchmarking functionality for parallel optimizers
  ({ghuser}`mpetrosian` and {ghuser}`janosg`)
- {gh}`457` Removes some unused files ({ghuser}`segsell`)
- {gh}`455` Improves a local pre-commit hook ({ghuser}`ChristianZimpelmann`)


## 0.4.5

- {gh}`379` Improves the estimation table ({ghuser}`ChristianZimpelmann`)
- {gh}`445` fixes line endings in local pre-commit hook ({ghuser}`ChristianZimpelmann`)
- {gh}`443`, {gh}`444`, {gh}`445`, {gh}`446`, {gh}`448` and {gh}`449` are a major
  refactoring of tranquilo ({ghuser}`timmens` and {ghuser}`janosg`)
- {gh}`441` Adds an aggregated convergence plot for benchmarks ({ghuser}`mpetrosian`)
- {gh}`435` Completes the cartis-roberts benchmark set ({ghuser}`segsell`)

## 0.4.4

- {gh}`437` removes fuzzywuzzy as dependency ({ghuser}`aidatak97`)
- {gh}`432` makes logging compatible with sqlalchemy 2.x ({ghuser}`janosg`)
- {gh}`430` refactors the getter functions in Tranquilo ({ghuser}`janosg`)
- {gh}`427` improves pre-commit setup ({ghuser}`timmens` and {ghuser}`hmgaudecker`)
- {gh}`425` improves handling of notebooks in documentation ({ghuser}`baharcos`)
- {gh}`423` and {gh}`399` add code to calculate poisdeness constants ({ghuser}`segsell`)
- {gh}`420` improve CI infrastructure ({ghuser}`hmgaudecker`, {ghuser}`janosg`)
- {gh}`407` adds global optimizers from scipy ({ghuser}`baharcos`)

## 0.4.3

- {gh}`416` improves documentation and packaging ({ghuser}`janosg`)

## 0.4.2

- {gh}`412` Improves the output of the fides optimizer among other small changes
  ({ghuser}`janosg`)
- {gh}`411` Fixes a bug in multistart optimizations with least squares optimizers.
  See {gh}`410` for details ({ghuser}`janosg`)
- {gh}`404` speeds up the gqtpar subsolver ({ghuser}`mpetrosian` )
- {gh}`400` refactors subsolvers ({ghuser}`mpetrosian`)
- {gh}`398`, {gh}`397`, {gh}`395`, {gh}`390`, {gh}`389`, {gh}`388` continue with the
  implementation of tranquilo ({ghuser}`segsell`, {ghuser}`timmens`,
  {ghuser}`mpetrosian`, {ghuser}`janosg`)
- {gh}`391` speeds up the bntr subsolver ({ghuser}`mpetrosian`)


## 0.4.1

- {gh}`307` Adopts a code of condact and governance model
- {gh}`384` Polish documentation ({ghuser}`janosg` and {ghuser}`mpetrosian`)
- {gh}`374` Moves the documentation to MyST ({ghuser}`baharcos`)
- {gh}`365` Adds copybuttos to documentation ({ghuser}`amageh`)
- {gh}`371` Refactors the pounders algorithm ({ghuser}`segsell`)
- {gh}`369` Fixes CI ({ghuser}`janosg`)
- {gh}`367` Fixes the linux environment ({ghuser}`timmens`)
- {gh}`294` Adds the very first experimental version of tranquilo ({ghuser}`janosg`,
  {ghuser}`timmens`, {ghuser}`segsell`, {ghuser}`mpetrosian`)


## 0.4.0

- {gh}`366` Update  ({ghuser}`segsell`)
- {gh}`362` Polish documentation ({ghuser}`segsell`)

## 0.3.4

- {gh}`364` Use local random number generators ({ghuser}`timmens`)
- {gh}`363` Fix pounders test cases ({ghuser}`segsell`)
- {gh}`361` Update estimation code ({ghuser}`timmens`)
- {gh}`360` Update results object documentation ({ghuser}`timmens`)

## 0.3.3

- {gh}`357` Adds jax support ({ghuser}`janosg`)
- {gh}`359` Improves error handling with violated constaints ({ghuser}`timmens`)
- {gh}`358` Improves cartis roberts set of test functions and improves the
  default latex rendering of MultiIndex tables ({ghuser}`mpetrosian`)

## 0.3.2

- {gh}`355` Improves test coverage of contraints processing ({ghuser}`janosg`)
- {gh}`354` Improves test coverage for bounds processing ({ghuser}`timmens`)
- {gh}`353` Improves history plots ({ghuser}`timmens`)
- {gh}`352` Improves scaling and benchmarking ({ghuser}`janosg`)
- {gh}`351` Improves estimation summaries ({ghuser}`timmens`)
- {gh}`350` Allow empty queries or selectors in constraints ({ghuser}`janosg`)

## 0.3.1

- {gh}`349` fixes multiple small bugs and adds test cases for all of them
  ({ghuser}`mpetrosian`, {ghuser}`janosg` and {ghuser}`timmens`)

## 0.3.0

Fist release with pytree support in optimization, estimation and differentiation
and much better result objects in optimization and estimation.

Breaking changes

- New `OptimizeResult` object is returned by `maximize` and `minimize`. This
  breaks all code that expects the old result dictionary. Usage of the new result is
  explained in the getting started tutorial on optimization.
- New internal optimizer interface that can break optimization with custom optimizers
- The inferface of `process_constraints` changed quite drastically. This breaks
  code that used `process_constraints` to get the number of free parameters or check
  if constraints are valid. There are new high level functions
  `estimagic.check_constraints` and `estimagic.count_free_params` instead.
- Some functions from `estimagic.logging.read_log` are removed and replaced by
  `estimagic.OptimizeLogReader`.
- Convenience functions to create namedtuples are removed from `estimagic.utilities`.
- {gh}`346` Add option to use nonlinear constraints ({ghuser}`timmens`)
- {gh}`345` Moves estimation_table to new latex functionality of pandas
  ({ghuser}`mpetrosian`)
- {gh}`344` Adds pytree support to slice_plot ({ghuser}`janosg`)
- {gh}`343` Improves the result object of estimation functions and makes msm estimation
  pytree compatible ({ghuser}`janosg`)
- {gh}`342` Improves default options of the fides optimizer, allows single constraints
  and polishes the documentation ({ghuser}`janosg`)
- {gh}`340` Enables history collection for optimizers that evaluate the criterion
  function in parallel ({ghuser}`janosg`)
- {gh}`339` Incorporates user feedback and polishes the documentation.
- {gh}`338` Improves log reading functions ({ghuser}`janosg`)
- {gh}`336` Adds pytree support to the dashboard ({ghuser}`roecla`).
- {gh}`335` Introduces an `OptimizeResult` object and functionality for history
  plotting ({ghuser}`janosg`).
- {gh}`333` Uses new history collection feature to speed up benchmarking
  ({ghuser}`segsell`).
- {gh}`330` Is a major rewrite of the estimation code ({ghuser}`timmens`).
- {gh}`328` Improves quadratic surrogate solvers used in pounders and tranquilo
  ({ghuser}`segsell`).
- {gh}`326` Improves documentation of numerical derivatives ({ghuser}`timmens`).
- {gh}`325` Improves the slice_plot ({ghuser}`mpetrosian`)
- {gh}`324` Adds ability to collect optimization histories without logging
  ({ghuser}`janosg`).
- {gh}`311` and {gh}`288` rewrite all plotting code in plotly ({ghuser}`timmens`
  and {ghuser}`aidatak97`).
- {gh}`306` improves quadratic surrogate solvers used in pounders and tranquilo
  ({ghuser}`segsell`).
- {gh}`305` allows pytrees during optimization and rewrites large parts of the
  constraints processing ({ghuser}`janosg`).
- {gh}`303` introduces a new optimizer interface that makes it easier to add optimizers
  and makes it possible to access optimizer specific information outside of the
  intrenal_criterion_and_derivative ({ghuser}`janosg` and {ghuser}`roecla`).

## 0.2.5

- {gh}`302` Drastically improves error handling during optimization ({ghuser}`janosg`).

## 0.2.4

- {gh}`304` Removes the chaospy dependency ({ghuser}`segsell`).

## 0.2.3

- {gh}`295` Fixes a small bug in estimation_table ({ghuser}`mpetrosian`).
- {gh}`286` Adds pytree support for first and second derivative ({ghuser}`timmens`).
- {gh}`285` Allows to use estimation functions with external optimization
  ({ghuser}`janosg`).
- {gh}`283` Adds fast solvers for quadratic trustregion subproblems ({ghuser}`segsell`).
- {gh}`282` Vastly improves estimation tables ({ghuser}`mpetrosian`).
- {gh}`281` Adds some tools to work with pytrees ({ghuser}`janosg`
  and {ghuser}`timmens`).
- {gh}`278` adds Estimagic Enhancement Proposal 1 for the use of Pytrees in Estimagic
  ({ghuser}`janosg`)

## 0.2.2

- {gh}`276` Add parallel Nelder-Mead algorithm by {ghuser}`jacekb95`
- {gh}`267` Update fides by {ghuser}`roecla`
- {gh}`265` Refactor pounders algorithm by {ghuser}`segsell` and {ghuser}`janosg`.
- {gh}`261` Add pure Python pounders algorithm by {ghuser}`segsell`.

## 0.2.1

- {gh}`260` Update MSM and ML notebooks by {ghuser}`timmens`.
- {gh}`259` Several small fixes and improvements by {ghuser}`janosg` and
  {ghuser}`roecla`.

## 0.2.0

Add a lot of new functionality with a few minor breaking changes. We have more
optimizers, better error handling, bootstrap and inference for method of simulated
moments. The breaking changes are:
\- logging is disabled by default during optimization.
\- the log_option "if_exists" was renamed to "if_table_exists"
\- The comparison plot function is removed.
\- first_derivative now returns a dictionary, independent of arguments.
\- structure of the logging database has changed
\- there is an additional boolean flag named `scaling` in minimize and maximize

- {gh}`251` Allows the loading, running and visualization of benchmarks
  ({ghuser}`janosg`, {ghuser}`mpetrosian` and {ghuser}`roecla`)
- {gh}`196` Adds support for multistart optimizations ({ghuser}`asouther4` and
  {ghuser}`janosg`)
- {gh}`248` Adds the fides optimizer ({ghuser}`roecla`)
- {gh}`146` Adds `estimate_ml` functionality ({ghuser}`janosg`, {ghuser}`LuisCald`
  and {ghuser}`s6soverd`).
- {gh}`235` Improves the documentation ({ghuser}`roecla`)
- {gh}`216` Adds the ipopt optimizer ({ghuser}`roecla`)
- {gh}`215` Adds optimizers from the pygmo library ({ghuser}`roecla` and
  {ghuser}`janosg`)
- {gh}`212` Adds optimizers from the nlopt library ({ghuser}`mpetrosian`)
- {gh}`228` Restructures testing and makes changes to log_options.
- {gh}`149` Adds `estimate_msm` functionality ({ghuser}`janosg` and {ghuser}`loikein`)
- {gh}`219` Several enhancements by ({ghuser}`tobiasraabe`)
- {gh}`218` Improve documentation by ({ghuser}`sofyaakimova`) and ({ghuser}`effieHan`)
- {gh}`214` Fix bug with overlapping "fixed" and "linear" constraints ({ghuser}`janosg`)
- {gh}`211` Improve error handling of log reading functions by ({ghuser}`janosg`)
- {gh}`210` Automatically drop empty constraints by ({ghuser}`janosg`)
- {gh}`192` Add option to scale optimization problems by ({ghuser}`janosg`)
- {gh}`202` Refactoring of bootstrap code ({ghuser}`janosg`)
- {gh}`148` Add bootstrap functionality ({ghuser}`RobinMusolff`)
- {gh}`208` Several small improvements ({ghuser}`janosg`)
- {gh}`206` Improve latex and html tables ({ghuser}`mpetrosian`)
- {gh}`205` Add scipy's least squares optimizers (based on {gh}`197` by
  ({ghuser}`yradeva93`)
- {gh}`198` More unit tests for optimizers ({ghuser}`mchandra12`)
- {gh}`200` Plot intermediate outputs of `first_derivative` ({ghuser}`timmens`)

## 0.1.3 - 2021-06-25

- {gh}`195` Illustrate optimizers in documentation ({ghuser}`sofyaakimova`),
  ({ghuser}`effieHan`) and ({ghuser}`janosg`)
- {gh}`201` More stable covariance matrix calculation ({ghuser}`janosg`)
- {gh}`199` Return intermediate outputs of first_derivative ({ghuser}`timmens`)

## 0.1.2 - 2021-02-07

- {gh}`189` Improve documentation and logging ({ghuser}`roecla`)

## 0.1.1 - 2021-01-13

This release greatly expands the set of available optimization algorithms, has a better
and prettier dashboard and improves the documentation.

- {gh}`187` Implement dot notation in algo_options ({ghuser}`roecla`)
- {gh}`183` Improve documentation ({ghuser}`SofiaBadini`)
- {gh}`182` Allow for constraints in likelihood inference ({ghuser}`janosg`)
- {gh}`181` Add DF-OLS optimizer from Numerical Algorithm Group ({ghuser}`roecla`)
- {gh}`180` Add pybobyqa optimizer from Numerical Algorithm Group ({ghuser}`roecla`)
- {gh}`179` Allow base_steps and min_steps to be scalars ({ghuser}`tobiasraabe`)
- {gh}`178` Refactoring of dashboard code ({ghuser}`roecla`)
- {gh}`177` Add stride as a new dashboard argument ({ghuser}`roecla`)
- {gh}`176` Minor fix of plot width in dashboard ({ghuser}`janosg`)
- {gh}`174` Various dashboard improvements ({ghuser}`roecla`)
- {gh}`173` Add new color palettes and use them in dashboard ({ghuser}`janosg`)
- {gh}`172` Add high level log reading functions ({ghuser}`janosg`)

## 0.1.0dev1 - 2020-09-08

This release entails a complete rewrite of the optimization code with many breaking
changes. In particular, some optimizers that were available before are not anymore.
Those will be re-introduced soon. The breaking changes include:

- The database is restructured. The new version simplifies the code,
  makes logging faster and avoids the sql column limit.
- Users can provide closed form derivative and/or criterion_and_derivative where
  the latter one can exploit synergies in the calculation of criterion and derivative.
  This is also compatible with constraints.
- Our own (parallelized) first_derivative function is used to calculate gradients
  during the optimization when no closed form gradients are provided.
- Optimizer options like convergence criteria and optimization results are harmonized
  across optimizers.
- Users can choose from several batch evaluators whenever we parallelize
  (e.g. for parallel optimizations or parallel function evaluations for numerical
  derivatives) or pass in their own batch evaluator function as long as it has a
  compatible interface. The batch evaluator interface also standardizes error handling.
- There is a well defined internal optimizer interface. Users can select the
  pre-implemented optimizers by algorithm="name_of_optimizer" or their own optimizer
  by algorithm=custom_minimize_function
- Optimizers from pygmo and nlopt are no longer supported (will be re-introduced)
- Greatly improved error handling.
- {gh}`169` Add additional dashboard arguments
- {gh}`168` Rename lower and upper to lower_bound and upper_bound
  ({ghuser}`ChristianZimpelmann`)
- {gh}`167` Improve dashboard styling ({ghuser}`roecla`)
- {gh}`166` Re-add POUNDERS from TAO ({ghuser}`tobiasraabe`)
- {gh}`165` Re-add the scipy optimizers with harmonized options ({ghuser}`roecla`)
- {gh}`164` Closed form derivatives for parameter transformations ({ghuser}`timmens`)
- {gh}`163` Complete rewrite of optimization with breaking changes ({ghuser}`janosg`)
- {gh}`162` Improve packaging and relax version constraints ({ghuser}`tobiasraabe`)
- {gh}`160` Generate parameter tables in tex and html ({ghuser}`mpetrosian`)

## 0.0.31 - 2020-06-20

- {gh}`130` Improve wrapping of POUNDERS algorithm ({ghuser}`mo2561057`)
- {gh}`159` Add Richardson Extrapolation to first_derivative ({ghuser}`timmens`)

## 0.0.30 - 2020-04-22

- {gh}`158` allows to specify a gradient in maximize and minimize ({ghuser}`janosg`)

## 0.0.29 - 2020-04-16

- {gh}`154` Version restrictions for pygmo ({ghuser}`janosg`)
- {gh}`153` adds documentation for the CLI ({ghuser}`tobiasraabe`)
- {gh}`152` makes estimagic work with pandas 1.0 ({ghuser}`SofiaBadini`)

## 0.0.28 - 2020-03-17

- {gh}`151` estimagic becomes a noarch package. ({ghuser}`janosg`).
- {gh}`150` adds command line interface to the dashboard ({ghuser}`tobiasraabe`)
