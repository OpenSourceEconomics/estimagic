(implementation_of_constraints)=

# How constraints are implemented

Most of the optimizers wrapped in optimagic cannot deal natively with anything but box
constraints. So the problem they can solve is:

$$
\min_{x \in \mathbb{R}^k} f(x) \quad \text{s.t.} \hspace{0.5cm} l \leq x \leq u
$$

However, in most econometric applications, we also need other constraints. For example,
we may require that some parameters sum to a value, form a covariance matrix, or are
probabilities. More abstractly, the problem becomes:

$$
\min_{x \in \mathbb{R}^k} f(x) \quad \text{s.t.} \hspace{0.5cm} l \leq x \leq u
\text{  and  } C(x) = 0
$$

There are two basic ways of converting optimizers, which, natively, can only deal with
box constraints, into constrained optimizers: Reparametrization and penalties. Below, we
explain what both approaches are, why we chose the reparametrization approach over
penalties, and which reparametrizations we are using for each type of constraint.

In this text, we focus on constraints that can be solved by optimagic via bijective and
differentiable transformations. General nonlinear constraints do not fall into this
category. If you want to use nonlinear constraints, you can still do so, but optimagic
will simply pass the constraints to your chosen optimizer. See {ref}`constraints` for
more details.

## Possible approaches

### Reparametrizations

In the reparametrization approach, we need to find an invertible mapping
$g : \mathbb{R}^{k'} \to \mathbb{R}^k$, and two new bounds $l'$ and $u'$ such that:

$$
l' \leq \tilde{x} \leq u' \iff l \leq g(\tilde{x}) \leq u \text {  and  }
C(g(\tilde{x})) = 0
$$

This means that:

$$
\min_{\tilde{x} \in \mathbb{R}^{k'}} f(g(\tilde{x})) \quad \text{s.t.}
\hspace{0.5cm} l' \leq \tilde{x} \leq u'\\
$$

is equivalent to the original minimization problem.

This sounds more complicated than it is. Let's look at the simple example of a two
dimensional parameter vector, where our constraint is that the two parameters have to
sum to 5.

$$
x = (x_1, x_2)

f(x) = x_1^2 + 2 x_2^2

c(x) = x_1 + x_2 - 5

\tilde{x} = x_1

g(\tilde{x}) = (\tilde{x}, 5 - \tilde{x})
$$

Typically, users implement such reparametrizations manually and write functions to
convert between the parameters of interest and their reparametrized version. optimagic
does this for you, for a large number of constraints that are typically used in
econometric applications.

For this approach to be efficient, it is crucial that the reparametrizations preserve
desirable properties of the original problem. In particular, the mapping $g$ should be
differentiable and if possible linear. Moreover, the dimensionality of $\tilde{x}$
should be chosen as small as possible. optimagic only implements constraints that can be
enforced with differentiable transformations and always achieves full dimensionality
reduction.

### Penalties

The penalty approach is conceptually much simpler. Whenever $C(x) \neq 0$, a penalty
term is added to the criterion function. If the penalty term is large enough (e.g. as
large as the criterion function at the start values), this penalty ensures that any x
that does not satisfy the constraints can not be optimal.

While the generality and conceptual simplicity of this approach is attractive, it also
has its drawbacks. Applying penalties in a naive way can introduce kinks,
discontinuities, and even local optima into the penalized criterion.

## What optimagic does

We chose to implement constraints via reparametrizations for the following reasons:

- Reparametrizations ensure that the criterion function is only evaluated at parameters
  that satisfy all constraints. This is not only efficient, but essential if the
  criterion function is only defined for such parameters.
- Reparametrizations can often achieve a substantial dimensionality reduction. In
  particular, fixes and equality constraints are implemented at zero cost, i.e. as
  efficiently as if you directly plugged them into your original problem. This is
  important because fixes and equality constraints often make user code much nicer and
  more flexible.
- It is easier to preserve desirable properties such as convexity and differentiability
  with reparametrizations rather than penalties.

The constraints that can be implemented via reparametrizations are available for all
optimizers. More general constraints are only available with optimizers that can deal
natively with them. This includes all optimizers from the `nlopt` and `ipopt` libraries.

## The non-trivial reparametrizations

Fixed parameters, equality, and pairwise equality constraints can be implemented
trivially with reparametrizations by simply plugging them into the criterion function.
Increasing and decreasing constraints are internally implemented as linear constraints.
The following section explains how the other types of constraints are implemented:

### Covariance and sdcorr constraints

The main difficulty with covariance and sdcorr constraints is to keep the (implied)
covariance matrix valid, i.e. positive semi-definite. In both cases, $\tilde{x}$
contains the non-zero elements of the lower triangular cholesky factor of the (implied)
covariance matrix. For covariance constraints, $g$ is then simply the product of the
cholesky factor with its transpose. For the sdcorr covariance matrix, the product is
further converted to standard deviations and the unique elements of a covariance matrix.

Several papers show that the cholesky reparametrization is a very efficient way to
optimize over covariance matrices. Examples are {cite}`Pinheiro1996` and
{cite}`Groeneveld1994`.

A limitation of this approach is that there can be no additional fixes, box constraints,
or other constraints on any of the involved parameters.

(linear-constraint-implementation)=

### Linear constraints

Assume we have m linear constraints on an n-dimensional parameter vector. Then the set
of all parameter vectors that satisfies the constraints can be written as:

$$
\mathbf{X} \equiv \{\mathbf{x} \in \mathbb{R}^n \mid \mathbf{l} \leq \mathbf{Ax}
\leq \mathbf{u}\}
$$

We are looking for a set $\mathbf{\tilde{X}}$ that only satisfies box constraints and
reparametrizations. The reparametrizations will turn out to be a linear mapping, and
thus have a matrix representation, say M. We are good if the following holds:

$$
x \in \mathbf{X} \iff \exists \mathbf{\tilde{x}} \in \mathbf{\tilde{X}} \text{s.t.}
\mathbf{x} = \mathbf{M\tilde{x}}
$$

Suitable choices of $\mathbf{\tilde{X}}$ and $\mathbf{M}$ are:

$$
\mathbf{\tilde{X}} \equiv \{(\tilde{x}_1, \tilde{x}_2)^T \mid \mathbf{\tilde{x}}_1
\in \mathbb{R}^{k} \text{ and } \mathbf{l} \leq \mathbf{\tilde{x}}_2 \leq \mathbf{l}\}

\mathbf{M} =
    \left[ {\begin{array}{cc}
    \mathbb{I}_n[k] \\
    A \\
    \end{array} } \right]^{-1}
$$

where $k = m - n$ and $\mathbb{I}_n[k]$ are the k rows of the identity matrix that make
all rows of $\mathbf{M}$ linearly independent.

**Proof:**

"$\Rightarrow$":

Let $x\in \mathbf{X}$, then we define $\mathbf{\tilde{x}} = \mathbf{M}^{-1} x$. Claim:
$\mathbf{\tilde{x}}  \in \mathbf{\tilde{X}}$: \\

$$
\mathbf{\tilde{x}}  = \mathbf{M}^{-1} x =   \left[ {\begin{array}{cc}      \mathbb{I}_n[k]x \\      Ax \\     \end{array} } \right]     = (\tilde{x}_1, \tilde{x}_2)^T
$$

where $\tilde{x}_1 \in \mathbb{R}^k$ and
$\mathbf{l} \leq \mathbf{\tilde{x}}_2 \leq \mathbf{u}$ because
$\mathbf{l} \leq \mathbf{Ax} \leq \mathbf{u}$. Thus
$\mathbf{\tilde{x}} \in \mathbf{\tilde{X}}$.

"$\Leftarrow$" (Proof by negation):

Let $x \not\in \mathbf{X}$ and define $\mathbf{\tilde{x}} = \mathbf{M}^{-1} x$. Claim
$\mathbf{\tilde{x}}  \not\in \mathbf{\tilde{X}}$.

By the same argument as above we can show, that, because
$\neg(\mathbf{l} \leq \mathbf{Ax} \leq \mathbf{u})$,
$\mathbf{\tilde{x}}  \not\in \mathbf{\tilde{X}}$.

The rank condition on M makes it clear that there can be at most as many linear
constraints as involved parameters. This includes any box constraints on the involved
parameters.

### Probability constraints

A probability constraint on k parameters means that all parameters lie in $[0, 1]$ and
their sum equals one. While those are all linear constraints, they cannot be implemented
in the way described above, because there are k + 1 constraints for k parameters.

Instead we do the following

$$
\tilde{x} = (\tilde{x}_1, \tilde{x}_2, \ldots, \tilde{x}_{k - 1})\\ g(\tilde{x}) = (\frac{\tilde{x}_1}{1 + \sum_{i=1}^{k-1}\tilde{x}_i}, \frac{\tilde{x}_2}{1 + \sum_{i=1}^{k-1}\tilde{x}_i}, \ldots, \frac{1}{1 + \sum_{i=1}^{k-1}\tilde{x}_i})\\ l' = (0, 0, \ldots, 0)
$$

A limitation of this approach is that there can be no additional fixes, box constraints
or other constraints on any of the involved parameters.

**References**

```{eval-rst}
.. bibliography:: ../refs.bib
    :filter: docname in docnames
```
