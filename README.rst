=========
estimagic
=========

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. image:: https://readthedocs.org/projects/estimagic/badge/?version=master
    :target: https://estimagic.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status

.. image:: https://dev.azure.com/OpenSourceEconomics/estimagic/_apis/build/status/OpenSourceEconomics.estimagic?branchName=master
    :target: https://dev.azure.com/OpenSourceEconomics/estimagic/_build/latest?definitionId=1&branchName=master


Introduction
============

Estimagic is a Python package that helps to build high-quality and user friendly
implementations of (structural) econometric models.

It is designed with large structural models in mind. However, it is also useful for any
other estimator that numerically minimizes or maximizes a criterion function (Extremum
Estimator). Examples are maximum likelihood estimation, generalized method of moments,
method of simulated moments and indirect inference.

Estimagic is in a very early stage and should not be used for major projects yet.
However, we do encourage interested users to try it out, report bugs and provide
feedback.


Credits
=======

Estimagic is designed and written by Janos Gabler (`janosg
<https://github.com/janosg>`_).

However, it has been a collaborative project right from the start.

In particular we would like to thank:

- Klara Röhrl (`roecla <https://github.com/roecla>`_) for writing most of the dashboard
  code.
- Tobias Raabe (`tobiasraabe <https://github.com/tobiasraabe>`_) for setting up the
  continuous integration and testing.

If you want to find your name here as well, please contact us or browse through our
Issues and submit a Pull Request.


Installation
============

The package can be installed via conda. To do so, type the following commands in a
terminal:

.. code-block:: bash

    $ conda config --add channels conda-forge $ conda install -c janosg estimagic

The first line adds conda-forge to your conda channels. This is necessary for conda to
find all dependencies of estimagic. The second line installs estimagic and its
dependencies.


Roadmap
=======

Currently, estimagic is mainly a collection of optimizers. In the near future we want to
add the following capabilities:

- Standard error estimation for Maximum Likelihood, Method of Simulated Moments and
  Indirect Inference
- Calculation of weighting matrices for minimum distance estimators
- Specialized estimation functions for Maximum Likelihood, Method of Simulated Moments
  and Indirect Inference that will provide a more convenient syntax and defaults that
  are tailored to the specific application.
