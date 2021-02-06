from setuptools import find_packages
from setuptools import setup

setup(
    name="estimagic",
    version="0.1.2",
    description="Tools for the estimation of (structural) econometric models.",
    long_description="""
        Estimagic is a Python package that helps to build high-quality and user
        friendly implementations of (structural) econometric models.

        It is designed with large structural models in mind. However, it is also
        useful for any other estimator that numerically minimizes or maximizes a
        criterion function (Extremum Estimator). Examples are maximum likelihood
        estimation, generalized method of moments, method of simulated moments and
        indirect inference.""",
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
    ],
    keywords=["econometrics", "statistics", "extremum estimation", "optimization"],
    url="https://github.com/OpenSourceEconomics/estimagic",
    author="Janos Gabler",
    author_email="janos.gabler@gmail.com",
    packages=find_packages(exclude=["tests/*"]),
    entry_points={"console_scripts": ["estimagic=estimagic.cli:cli"]},
    zip_safe=False,
    package_data={"estimagic": ["optimization/algo_dict.json"]},
    include_package_data=True,
)
