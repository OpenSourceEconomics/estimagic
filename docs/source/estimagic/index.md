(estimagic)=

# Estimagic

*estimagic* is a subpackage of *optimagic* that helps you to fit nonlinear statistical
models to data and perform inference on the estimated parameters.

As a user, you need to code up the objective function that defines the estimator. This
is either a likelihood (ML) function or a Method of Simulated Moments (MSM) objective
function. Everything else is done by *estimagic*.

Everything else means:

- Optimize your objective function
- Calculate asymptotic or bootstrapped standard errors and confidence intervals
- Create publication quality tables
- Perform sensitivity analysis on MSM models

`````{grid} 1 2 2 2
---
gutter: 3
---
````{grid-item-card}
:text-align: center
:img-top: ../_static/images/light-bulb.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} tutorials/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Tutorials
```

New users of estimagic should read this first.

````



````{grid-item-card}
:text-align: center
:img-top: ../_static/images/books.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} explanation/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Explanations
```

Background information on key topics central to the package.

````

````{grid-item-card}
:text-align: center
:columns: 12
:img-top: ../_static/images/coding.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} reference/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
API Reference
```

Detailed description of the estimagic API.

````



`````

```{toctree}
---
hidden: true
maxdepth: 1
---
tutorials/index
explanation/index
reference/index
```
