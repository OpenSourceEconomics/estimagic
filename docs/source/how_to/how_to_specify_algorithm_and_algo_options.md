(algorithms)=

# How to specify algorithms and algorithm specific options

## The *algorithm* argument

The `algorithm` argument can either be a string with the name of an algorithm that is
implemented in optimagic, or a function that fulfills the interface laid out in
{ref}`internal_optimizer_interface`.

Which algorithms are available in optimagic depends on the packages a user has
installed. We list all supported algorithms in {ref}`list_of_algorithms`.

## The *algo_options* argument

`algo_options` is a dictionary with options that are passed to the optimization
algorithm.

We align the names of all `algo_options` across algorithms as far as that is possible.

To make it easier to understand which aspect of the optimization is influenced by an
option, we group them with prefixes. For example, the name of all convergence criteria
starts with `"convergence."`. In general, the prefix is separated from the option name
by a dot.

Which options are supported, depends on the algorithm you selected and is documented in
{ref}`list_of_algorithms`.

An example could look like this:

```python
algo_options = {
    "trustregion.threshold_successful": 0.2,
    "trustregion.threshold_very_successful": 0.9,
    "trustregion.shrinking_factor.not_successful": 0.4,
    "trustregion.shrinking_factor.lower_radius": 0.2,
    "trustregion.shrinking_factor.upper_radius": 0.8,
    "convergence.noise_corrected_criterion_tolerance": 1.1,
}
```

To make it easier to switch between algorithms, we simply ignore non-supported options
and issue a warning that explains which options have been ignored.

To find more information on `algo_options` that are supported by many optimizers, see
{ref}`algo_options`.
