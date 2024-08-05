# Installation

## Basic installation

The preferred way to install optimagic is via `conda` or `mamba`. To do so, open a
terminal and type:

```
conda install -c conda-forge optimagic
```

Alternatively, you can install optimagic via pip:

```
pip install estimagic
```

In both cases, you get optimagic and all of its mandatory dependencies.

## Installing optional dependencies

Only `scipy` is a mandatory dependency of optimagic. Other algorithms become available
if you install more packages. We make this optional because you will rarely need all of
them in the same project.

For an overview of all optimizers and the packages you need to install to enable them,
see {ref}`list_of_algorithms`.

To enable all algorithms at once, do the following:

```
conda -c conda-forge install nlopt
```

```
pip install Py-BOBYQA
```

```
pip install DFO-LS
```

```
conda install -c conda-forge petsc4py
```

*Note*: `` `petsc4py` `` is not available on Windows.

```
conda install -c conda-forge cyipopt
```

```
conda install -c conda-forge pygmo
```

```
pip install fides>=0.7.4
```

*Note*: Make sure you have at least `fides` 0.7.4.
