(bootstrap-cis)=

# Bootstrap Confidence Intervals

We use the notation and formulations provided in chapter 10 of {cite}`Hansen2020`.

The first supported confidence interval type is the **"percentile"** confidence
interval, as discussed in section 10.10 of the Hansen textbook. Let
$\{ \hat{\theta}_1^*, ..., \hat{\theta}_B^*\}$ denote the estimates of estimator
$\hat{\theta}$ for the B bootstrap samples. The idea of the percentile confidence
interval is to simply take the empirical quantiles $q_{p}^*$ of this distributions, so
we have

$$
CI^{percentile} = [q_{\alpha/2}^*, q_{1-\alpha/2}^*].
$$

The second supported confidence interval **"normal"** is based on a normal approximation
and discussed in Hansen's section 10.9. Let $s_{boot}$ be the sample standard error of
the distribution of bootstrap estimators, $z_q$ the q-quantile of a standard normal
distribution and $\hat{\theta}$ be the full sample estimate of $\theta$. Then, the
asymptotic normal confidence interval is given by

$$
CI^{normal} = [\hat{\theta} - z_{1- \alpha/2} s_{boot},
\hat{\theta} + z_{1- \alpha/2} s_{boot}].
$$

The bias-corrected **"bc"** bootstrap confidence interval addresses the issue of biased
estimators. This problem is often present when estimating nonlinear models. Econometric
details are discussed in section 10.17 of Hansen. Let

$$
p^* = \frac{1}{B} \sum_{b=1}^B 1(\hat{\theta}_b^* \leq \hat{\theta})
$$

and define $z_0^* = \Phi^{-1} (p^*)$, where $\Phi$ is the standard normal cdf. The bias
correction works via correcting the significance level. Define
$x(\alpha) = \Phi(z_\alpha + 2 z_0^*)$ as the corrected significance level for a target
significant level of $\alpha$. Then, the bias-corrected confidence interval is given by

$$
CI^{bc} = [q_{x(\alpha/2)}^*, q_{x(1-\alpha/2)}^*].
$$

A further refined version of the bias-corrected confidence interval is the
bias-corrected and accelerated interval, short **"bca"**, as discussed in section 10.20
of Hansen. The general idea is to correct for skewness sampling distribution. Downsides
of this confidence interval are that it takes quite a lot of time to compute, since it
features calculating leave-one-out estimates of the original sample. Formally, again,
the significance levels are adjusted. Define

$$
\hat{a}=\frac{\sum_{i=1}^{n}\left(\bar{\theta}-\hat{\theta}_{(-i)}\right)^{3}}
{6\left(\sum_{i=1}^{n}\left(\bar{\theta}-\hat{\theta}_{(-i)}\right)^{2}
\right)^{3 / 2}},
$$

where $\bar{\theta}=\frac{1}{n} \sum_{i=1}^{n} \widehat{\theta}_{(-i)}$. This is an
estimator for the skewness of $\hat{\theta}$. Then, the corrected significance level is
given by

$$
x(\alpha)=\Phi(z_{0}+\frac{z_{\alpha}+z_{0}}{1-a(z_{\alpha}+z_{0})})
$$

and the bias-corrected and accelerated confidence interval is given by

$$
CI^{bca} = [q_{x(\alpha/2)}^*, q_{x(1-\alpha/2)}^*].
$$

The studentized confidence interval, here called **"t"** type confidence interval first
studentizes the bootstrap parameter distribution, i.e. applies the transformation
$\frac{\hat{\theta}_b-\hat{\theta}}{s_{boot}}$, and then builds the confidence interval
based on the estimated quantile function of the studentized data $\hat{G}$:

$$
CI^{t} = \left[\hat{\theta}+\hat{\sigma} \hat{G}^{-1}(\alpha / 2),
\hat{\theta}+\hat{\sigma} \hat{G}^{-1}(1-\alpha / 2)\right]
$$

The final supported confidence interval method is the **"basic"** bootstrap confidence
interval, which is derived in section 3.4 of {cite}`Wassermann2006`, where it is called
the pivotal confidence interval. It is given by

$$
CI^{basic} = \left[\hat{\theta}+\left(\hat{\theta}-\hat{\theta}_{u}^{\star}\right), \hat{\theta}+\left(\hat{\theta}-\hat{\theta}_{l}^{\star}\right)\right],
$$

where $\hat{\theta}_{u}^{\star}$ denotes the $1-\alpha/2$ empirical quantile of the
bootstrap estimate distribution for parameter $\theta$ and $\hat{\theta}_{l}^{\star}$
denotes the $\alpha/2$ quantile.

```{eval-rst}
.. bibliography:: ../../refs.bib
    :filter: docname in docnames
```
