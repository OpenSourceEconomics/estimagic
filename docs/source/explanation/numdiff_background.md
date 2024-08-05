# Background and methods

In this section we explain the mathematical background of forward, backward and central
differences. The main ideas in this chapter are taken from {cite}`Dennis1996`. x is used
for the pandas DataFrame with parameters. We index the entries of x as a n-dimensional
vector, where n is the number of variables in params_sr. The forward difference for the
gradient is given by:

$$
\nabla f(x) = \begin{pmatrix}\frac{f(x + e_0 * h_0) - f(x)}{h_0}\\
\frac{f(x + e_1 * h_1) - f(x)}{h_1}\\.\\.\\.\\ \frac{f(x + e_n * h_n)
- f(x)}{h_n} \end{pmatrix}
$$

The backward difference for the gradient is given by:

$$
\nabla f(x) = \begin{pmatrix}\frac{f(x) - f(x - e_0 * h_0)}{h_0}\\ \frac{f(x) -
f(x - e_1 * h_1)}{h_1}\\.\\.\\.\\ \frac{f(x) - f(x - e_n * h_n)}{h_n}
\end{pmatrix}
$$

The central difference for the gradient is given by:

$$
\nabla f(x) =
\begin{pmatrix}\frac{f(x + e_0 * h_0) - f(x - e_0 * h_0)}{h_0}\\
\frac{f(x + e_1 * h_1) - f(x - e_1 * h_1)}{h_1}\\.\\.\\.\\ \frac{f(x + e_n * h_n)
- f(x - e_n * h_n)}{h_n} \end{pmatrix}
$$

For the optimal stepsize h the following rule of thumb is applied:

$$
h_i = (1 + |x[i]|) * \sqrt\epsilon
$$

With the above in mind it is easy to calculate the Jacobian matrix. The calculation of
the finite difference w.r.t. each variable of params_sr yields a vector, which is the
corresponding column of the Jacobian matrix. The optimal stepsize remains the same.

For the Hessian matrix, we repeatedly call the finite differences functions. As we allow
for central finite differences in the second order derivative only, the deductions for
forward and backward, are left to the interested reader:

$$
f_{i,j}(x)
    = &\frac{f_i(x + e_j * h_j) - f_i(x - e_j * h_j)}{h_j} \\
    = &\frac{\frac{f(x + e_j * h_j + e_i * h_i) - f(x + e_j * h_j - e_i * h_i)}{h_i}
       - \frac{
             f(x - e_j * h_j + e_i * h_i) - f(x - e_j * h_j - e_i * h_i)
         }{h_i}}{h_j} \\
    = &\frac{
           f(x + e_j * h_j + e_i * h_i) - f(x + e_j * h_j - e_i * h_i)
       }{h_j * h_i} \\
      &+ \frac{
             - f(x - e_j * h_j + e_i * h_i) + f(x - e_j * h_j - e_i * h_i)
         }{h_j * h_i}
$$

For the optimal stepsize a different rule is used:

$$
h_i = (1 + |x[i]|) * \sqrt[3]\epsilon
$$

Similar deviations lead to the elements of the Hessian matrix calculated by backward and
central differences.

**References:**

```{eval-rst}
.. bibliography:: ../refs.bib
    :filter: docname in docnames
```
