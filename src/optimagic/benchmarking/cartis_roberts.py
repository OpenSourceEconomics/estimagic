"""Define the medium scale CUTEst Benchmark Set.

This benchmark set is contains 60 test cases for nonlinear least squares
solvers. It was used to benchmark all modern model based non-linear
derivative free least squares solvers (e.g. POUNDERS, DFOGN, DFOLS).

The parameter dimensions are of medium scale, varying between 25 and 100.

The benchmark set is based on Table 3 in Cartis and Roberts (2019).
Implementation is based on
- the original SIF files: https://bitbucket.org/optrove/sif/src/master/
- on sources cited in the SIF files or,
- where available, on AMPL implementaions available here:
- https://vanderbei.princeton.edu/ampl/nlmodels/cute/index.html

"""

from functools import partial

import numpy as np

from optimagic.config import IS_NUMBA_INSTALLED
from optimagic.parameters.bounds import Bounds

if IS_NUMBA_INSTALLED:
    from numba import njit
else:

    def njit(func):
        return func


from optimagic.benchmarking.more_wild import (
    brown_almost_linear,
    linear_full_rank,
    linear_rank_one,
    watson,
)


def luksan11(x):
    dim_in = len(x)
    fvec = np.zeros(2 * (dim_in - 1))
    fvec[::2] = 20 * x[:-1] / (1 + x[:-1] ** 2) - 10 * x[1:]
    fvec[1::2] = x[:-1] - 1
    return fvec


def luksan12(x):
    dim_in = len(x)
    n = (dim_in - 2) // 3
    i = np.arange(0, 3 * n, 3)
    fvec = np.zeros(6 * n)
    fvec[::6] = 10 * (x[i] ** 2 - x[i + 1])
    fvec[1::6] = x[i + 2] - 1
    fvec[2::6] = (x[i + 3] - 1) ** 2
    fvec[3::6] = (x[i + 4] - 1) ** 3
    fvec[4::6] = x[i] ** 2 * x[i + 3] + np.sin(x[i + 3] - x[i + 4]) - 10
    fvec[5::6] = x[i + 1] + (x[i + 2] ** 4) * (x[i + 3] ** 2) - 20
    return fvec


def luksan13(x):
    dim_in = len(x)
    n = (dim_in - 2) // 3
    fvec = np.zeros(n * 7)
    i = np.arange(n)
    k = i * 7
    fvec[k] = 10 * (x[3 * i] ** 2 - x[3 * i + 1])
    fvec[k + 1] = 10 * (x[3 * i + 1] ** 2 - x[3 * i + 2])
    fvec[k + 2] = (x[3 * i + 2] - x[3 * i + 3]) ** 2
    fvec[k + 3] = (x[3 * i + 3] - x[3 * i + 4]) ** 2
    fvec[k + 4] = x[3 * i] + x[3 * i + 1] ** 2 + x[3 * i + 2] - 30
    fvec[k + 5] = x[3 * i + 1] - x[3 * i + 2] ** 2 + x[3 * i + 3] - 10
    fvec[k + 6] = x[3 * i + 1] * x[3 * i + 4] - 10

    return fvec


def luksan14(x):
    dim_in = len(x)
    dim_out = 7 * (dim_in - 2) // 3
    fvec = np.zeros(dim_out, dtype=np.float64)

    for i in range(0, dim_in - 2, 3):
        k = (i // 3) * 7
        fvec[k : k + 7] = [
            10 * (x[i] ** 2 - x[i + 1]),
            x[i + 1] + x[i + 2] - 2,
            x[i + 3] - 1,
            x[i + 4] - 1,
            x[i] + 3 * x[i + 1],
            x[i + 2] + x[i + 3] - 2 * x[i + 4],
            10 * (x[i + 1] ** 2 - x[i + 4]),
        ]

    return fvec


def luksan15(x):
    dim_in = len(x)
    dim_out = (dim_in - 2) * 2
    temp = np.zeros((dim_out, 3), dtype=np.float64)
    y = np.tile([35.8, 11.2, 6.2, 4.4], dim_out // 4)

    for p in range(1, 4):
        k = 0
        for i in range(0, dim_in - 2, 2):
            for j in range(1, 5):
                temp[k, p - 1] = (p**2 / j) * np.abs(
                    x[i] * (x[i + 1] ** 2) * (x[i + 2] ** 3) * (x[i + 3] ** 4)
                ) ** (1 / (p * j))

                k += 1

    fvec = y - np.sum(temp, axis=1)

    return fvec


def luksan16(x):
    dim_in = len(x)
    dim_out = (dim_in - 2) * 2
    temp = np.zeros((dim_out, 3), dtype=np.float64)
    y = np.tile([35.8, 11.2, 6.2, 4.4], dim_out // 4)

    for p in range(1, 4):
        k = 0
        for i in range(0, dim_in - 2, 2):
            for j in range(1, 5):
                temp[k, p - 1] = (p**2 / j) * np.exp(
                    (x[i] + 2 * x[i + 1] + 3 * x[i + 2] + 4 * x[i + 3]) * (1 / (p * j))
                )
                k += 1

    fvec = y - np.sum(temp, axis=1)

    return fvec


def luksan17(x):
    dim_in = len(x)
    dim_out = (dim_in - 2) * 2
    temp = np.zeros((dim_out, 4), dtype=np.float64)
    y = np.tile([30.6, 72.2, 124.4, 187.4], dim_out // 4)

    for q in range(1, 5):
        k = 0
        for i in range(-1, dim_in - 4, 2):
            for j in range(1, 5):
                temp[k, q - 1] += -j * q**2 * np.sin(x[i + q]) + j**2 * q * np.cos(
                    x[i + q]
                )
                k += 1

    fvec = y - np.sum(temp, axis=1)

    return fvec


def luksan21(x):
    dim_out = len(x)
    h = 1 / (dim_out + 1)
    fvec = np.zeros(dim_out, dtype=np.float64)

    fvec[0] = 2 * x[0] + 0.5 * h**2 * (x[0] + h + 1) ** 3 - x[1] + 1
    for i in range(1, dim_out - 1):
        fvec[i] = (
            2 * x[i]
            + 0.5 * h**2 * (x[i] + h * (i + 1) + 1) ** 3
            - x[i - 1]
            - x[i + 1]
            + 1
        )
    fvec[-1] = 2 * x[-1] + 0.5 * h**2 * (x[-1] + h * dim_out + 1) ** 3 - x[-2] + 1

    return fvec


def luksan22(x):
    dim_out = 2 * len(x) - 2
    fvec = np.zeros(dim_out)
    fvec[0] = x[0] - 1
    fvec[1:-1:2] = 10 * (x[:-2] ** 2 - x[1:-1])
    fvec[2:-1:2] = 2 * np.exp(-((x[:-2] - x[1:-1]) ** 2)) + np.exp(
        -2 * (x[1:-1] - x[2:]) ** 2
    )
    fvec[-1] = -10 * (x[-2] ** 2)
    return fvec


def morebvne(x):
    dim_in = len(x)
    h = 1 / (dim_in + 1)
    i = np.arange(1, dim_in + 1)
    fvec = np.zeros(dim_in)

    fvec[0] = 2 * x[0] - x[1] + h**2 / 2 * (x[0] + i[0] * h + 1) ** 3
    fvec[1:-1] = (
        2 * x[1:-1] - x[:-2] - x[2:] + h**2 / 2 * (x[1:-1] + i[1:-1] * h + 1) ** 3
    )
    fvec[-1] = 2 * x[-2] - x[-2] + h**2 / 2 * (x[-1] + i[-1] * h + 1) ** 3

    return fvec


@njit
def flosp2(x, a, b, ra=1.0e7):
    n = 5
    xvec = np.ones((3, n, n), dtype=np.float64)
    xvec[0] = x[: n**2].reshape(n, n)
    xvec[1] = x[n**2 : 2 * n**2].reshape(n, n)
    xvec[2, 1:-1, 1:-1] = x[2 * n**2 :].reshape(n - 2, n - 2)

    h = 1 / 2
    ax = 1.0
    axx = ax**2
    theta = 0.5 * np.pi
    pi1 = -0.5 * ax * ra * np.cos(theta)
    pi2 = 0.5 * ax * ra * np.sin(theta)

    fvec = np.empty(59, dtype=np.float64)

    temp = np.empty((n - 2, n - 2, n - 2), dtype=np.float64)
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            temp[0, i - 1, j - 1] = (
                xvec[0, i, j] * -2 * (1 / h) ** 2
                + xvec[0, i + 1, j] * (1 / h) ** 2
                + xvec[0, i - 1, j] * (1 / h) ** 2
                + xvec[0, i, j] * -2 * axx * (1 / h) ** 2
                + xvec[0, i, j + 1] * axx * (1 / h) ** 2
                + xvec[0, i, j - 1] * ax * (1 / h) ** 2
                + xvec[1, i + 1, j] * -pi1 / (2 * h)
                + xvec[1, i - 1, j] * pi1 / (2 * h)
                + xvec[1, i, j + 1] * -pi2 / (2 * h)
                + xvec[1, i, j - 1] * pi2 / (2 * h)
            )

            temp[1, i - 1, j - 1] = (
                xvec[2, i, j] * -2 * (1 / h) ** 2
                + xvec[2, i + 1, j] * (1 / h) ** 2
                + xvec[2, i - 1, j] * (1 / h) ** 2
                + xvec[2, i, j] * -2 * axx * (1 / h) ** 2
                + xvec[2, i, j + 1] * axx * (1 / h) ** 2
                + xvec[2, i, j - 1] * axx * (1 / h) ** 2
                + xvec[0, i, j] * axx * 0.25
            )

            temp[2, i - 1, j - 1] = (
                xvec[1, i, j] * -2 * (1 / h) ** 2
                + xvec[1, i + 1, j] * (1 / h) ** 2
                + xvec[1, i - 1, j] * (1 / h) ** 2
                + xvec[1, i, j] * -2 * axx * (1 / h) ** 2
                + xvec[1, i, j + 1] * axx * (1 / h) ** 2
                + xvec[1, i, j - 1] * axx * (1 / h) ** 2
                - 0.25
                * ax
                * (1 / h) ** 2
                * (xvec[2, i, j + 1] - xvec[2, i, j - 1])
                * (xvec[1, i + 1, j] - xvec[1, i - 1, j])
                + 0.25
                * ax
                * (1 / h) ** 2
                * (xvec[2, i + 1, j] - xvec[2, i - 1, j])
                * (xvec[1, i, j + 1] - xvec[1, i, j - 1])
            )
    fvec[:27] = temp.flatten()

    temp = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        temp[k, -1] = a[2]
        temp[k, 0] = b[2]
        temp[0, k] = 0
    temp[-1, -1] = 0

    for k in range(n):
        temp[k, -1] += (
            xvec[1, k, -1] * 2 * a[0] * (1 / h)
            + xvec[1, k, -2] * -2 * a[0] * (1 / h)
            + xvec[1, k, -1] * a[1]
        )
        temp[k, 0] += (
            xvec[1, k, 1] * 2 * b[0] * (1 / h)
            + xvec[1, k, 0] * -2 * b[0] * (1 / h)
            + xvec[1, k, 0] * b[1]
        )
        temp[-1, k] += xvec[1, -1, k] * 2 * (1 / (ax * h)) + xvec[1, -2, k] * -2 * (
            1 / (ax * h)
        )
        temp[0, k] += xvec[1, 1, k] * 2 * (1 / (ax * h)) + xvec[1, 0, k] * -2 * (
            1 / (ax * h)
        )

    fvec[27:32] = temp[0]
    fvec[32:37] = temp[-1]
    fvec[37:40] = temp[1:-1, 0]
    fvec[40:43] = temp[1:-1, -1]

    temp = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        temp[k, -1] += xvec[2, k, -1] * -2 * (1 / h) + xvec[2, k, -2] * 2 * (1 / h)
        temp[k, 0] += xvec[2, k, 1] * 2 * (1 / h) + xvec[2, k, 0] * -2 * (1 / h)
        temp[-1, k] += xvec[2, -1, k] * -2 * (1 / (ax * h)) + xvec[2, -2, k] * 2 * (
            1 / (ax * h)
        )
        temp[0, k] += xvec[2, 1, k] * 2 * (1 / (ax * h)) + xvec[2, 0, k] * -2 * (
            1 / (ax * h)
        )

    fvec[43:48] = temp[0]
    fvec[48:53] = temp[-1]
    fvec[53:56] = temp[1:-1, 0]
    fvec[56:] = temp[1:-1, -1]

    return fvec


def oscigrne(x):
    dim_in = len(x)
    rho = 500

    fvec = np.zeros(dim_in)
    fvec[0] = 0.5 * x[0] - 0.5 - 4 * rho * (x[1] - 2.0 * x[0] ** 2 + 1.0) * x[0]
    fvec[1:-1] = (
        2 * rho * (x[1:-1] - 2.0 * x[:-2] ** 2 + 1.0)
        - 4 * rho * (x[2:] - 2.0 * x[:-2] ** 2 + 1.0) * x[2:]
    )
    fvec[-1] = 2 * rho * (x[-1] - 2.0 * x[-2] ** 2 + 1.0)

    return fvec


def spmsqrt(x):
    m = (len(x) + 2) // 3
    xmat = np.diag(x[2:-1:3], -1) + np.diag(x[::3], 0) + np.diag(x[1:-2:3], 1)

    b = np.zeros((m, m), dtype=np.float64)
    b[0, 0] = np.sin(1)
    b[0, 1] = np.sin(4)
    k = 2
    for i in range(1, m - 1):
        k += 1
        b[i, i - 1] = np.sin(k**2)
        k += 1
        b[i, i] = np.sin(k**2)
        k += 1
        b[i, i + 1] = np.sin(k**2)
    k += 1
    b[-1, -2] = np.sin(k**2)
    k += 1
    b[-1, -1] = np.sin(k**2)

    fmat = np.zeros((m, m), dtype=np.float64)
    fmat[0, 0] = xmat[0, 0] ** 2 + xmat[0, 1] * xmat[1, 0]
    fmat[0, 1] = xmat[0, 0] * xmat[0, 1] + xmat[0, 1] * xmat[1, 1]
    fmat[0, 2] = xmat[0, 1] * xmat[1, 2]

    fmat[1, 0] = xmat[1, 0] * xmat[0, 0] + xmat[1, 1] * xmat[1, 0]
    fmat[1, 1] = xmat[1, 0] * xmat[0, 1] + xmat[1, 1] ** 2 + xmat[1, 2] * xmat[2, 1]
    fmat[1, 2] = xmat[1, 1] * xmat[1, 2] + xmat[1, 2] * xmat[2, 2]
    fmat[1, 3] = xmat[1, 2] * xmat[2, 3]

    for i in range(2, m - 2):
        fmat[i, i - 2] = xmat[i, i - 1] * xmat[i - 1, i - 2]
        fmat[i, i - 1] = (
            xmat[i, i - 1] * xmat[i - 1, i - 1] + xmat[i, i] * xmat[i, i - 1]
        )
        fmat[i, i] = (
            xmat[i, i - 1] * xmat[i - 1, i]
            + xmat[i, i] ** 2
            + xmat[i, i + 1] * xmat[i + 1, i]
        )
        fmat[i, i + 1] = (
            xmat[i, i] * xmat[i, i + 1] + xmat[i, i + 1] * xmat[i + 1, i + 1]
        )
        fmat[i, i + 2] = xmat[i, i + 1] * xmat[i + 1, i + 2]

    fmat[-2, -4] = xmat[-2, -3] * xmat[-3, -4]
    fmat[-2, -3] = xmat[-2, -3] * xmat[-3, -3] + xmat[-2, -2] * xmat[-2, -3]
    fmat[-2, -2] = (
        xmat[-2, -3] * xmat[-3, -2] + xmat[-2, -2] ** 2 + xmat[-2, -1] * xmat[-1, -2]
    )
    fmat[-2, -1] = xmat[-2, -2] * xmat[-2, -1] + xmat[-2, -1] * xmat[-1, -1]

    fmat[-1, -3] = xmat[-1, -2] * xmat[-2, -3]
    fmat[-1, -2] = xmat[-1, -2] * xmat[-2, -2] + xmat[-1, -1] * xmat[-1, -2]
    fmat[-1, -1] = xmat[-1, -2] * xmat[-2, -1] + xmat[-1, -1] ** 2

    fmat[0, 0] -= b[0, 0] ** 2 + b[0, 1] * b[1, 0]
    for i in range(1, m - 1):
        fmat[i, i] -= (
            b[i, i] ** 2 + b[i - 1, i] * b[i, i - 1] + b[i + 1, i] * b[i, i + 1]
        )
    fmat[-1, -1] -= b[-1, -1] ** 2 + b[-2, -1] * b[-1, -2]
    for i in range(m - 1):
        fmat[i + 1, i] -= b[i + 1, i] * b[i, i] + b[i + 1, i + 1] * b[i + 1, i]
    for i in range(1, m):
        fmat[i - 1, i] -= b[i - 1, i] * b[i, i] + b[i - 1, i - 1] * b[i - 1, i]
    for i in range(1, m - 1):
        fmat[i + 1, i - 1] -= b[i + 1, i] * b[i, i - 1]
    for i in range(1, m - 1):
        fmat[i - 1, i + 1] -= b[i - 1, i] * b[i, i + 1]

    return fmat.flatten()


def semicon2(x):
    n = len(x) // 1
    ln = 9 * n // 10

    lambda_ = 0.2
    a = -0.00009
    b = 0.00001
    ua = 0.0
    ub = 700.0
    ca = 1e12
    cb = 1e13
    beta = 40.0

    h = (b - a) / (n + 1)
    lb = lambda_ * beta
    lua = lambda_ * ua
    lub = lambda_ * ub

    xvec = np.zeros(n + 2, dtype=np.float64)
    xvec[0] = lua
    xvec[1:-1] = x
    xvec[-1] = lub

    fvec = np.zeros(n, dtype=np.float64)
    for i in range(1, ln + 1):
        fvec[i - 1] = (
            xvec[i - 1]
            - 2 * xvec[i]
            + xvec[i + 1]
            + lambda_ * (h**2) * ca * np.exp(-lb * (xvec[i] - lua))
            - lambda_ * (h**2) * cb * np.exp(lb * (xvec[i] - lub))
            - lambda_ * (h**2) * ca
        )
    for i in range(ln + 1, n + 1):
        fvec[i - 1] = (
            xvec[i - 1]
            - 2 * xvec[i]
            + xvec[i + 1]
            - lambda_ * (h**2) * cb * np.exp(lb * (xvec[i] - lub))
            + lambda_ * (h**2) * ca * np.exp(-lb * (xvec[i] - lua))
            + lambda_ * (h**2) * cb
        )

    return fvec


def qr3d(x, m=5):
    q = x[: m**2].reshape(m, m)
    r = np.zeros((m, m), dtype=np.float64)
    r[np.triu_indices_from(r)] = x[m**2 :]

    a = (
        np.diag((1 - np.arange(2, m + 1)) / m, -1)
        + np.diag(2 * np.arange(1, m + 1) / m, 0)
        + np.diag((1 - np.arange(1, m)) / m, 1)
    )
    a[0, 1] = 0
    a[-1, -2] = (1 - m) / m
    a[-1, -1] = 2 * m

    omat = np.zeros((m, m), dtype=np.float64)  # triu
    fmat = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for j in range(i, m):
            for k in range(m):
                omat[i, j] += q[i, k] * q[j, k]

    for i in range(m):
        for j in range(m):
            for k in range(j + 1):
                fmat[i, j] += q[i, k] * r[k, j]

    for i in range(m):
        omat[i, i] -= 1
    fmat[0, 0] -= a[0, 0]
    fmat[0, 1] -= a[0, 1]
    for i in range(1, m - 1):
        fmat[i, i - 1] -= a[i, i - 1]
        fmat[i, i] -= a[i, i]
        fmat[i, i + 1] -= a[i, i + 1]
    fmat[-1, -2] -= a[-1, -2]
    fmat[-1, -1] -= a[-1, -1]

    return np.concatenate((omat[np.triu_indices_from(omat)].flatten(), fmat.flatten()))


def qr3dbd(x, m=5):
    q = x[: m**2].reshape(m, m)
    r = np.zeros((m, m), dtype=np.float64)
    r[0, :-2] = x[m**2 : -9]
    r[1, 1:-1] = x[-9:-6]
    r[2, 2:] = x[-6:-3]
    r[3, 3:] = x[-3:-1]
    r[4, 4] = x[-1]

    a = (
        np.diag((1 - np.arange(2, m + 1)) / m, -1)
        + np.diag(2 * np.arange(1, m + 1) / m, 0)
        + np.diag((1 - np.arange(1, m)) / m, 1)
    )
    a[0, 1] = 0
    a[-1, -2] = (1 - m) / m
    a[-1, -1] = 2 * m

    omat = np.zeros((m, m), dtype=np.float64)  # triu
    fmat = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for j in range(i, m):
            for k in range(m):
                omat[i, j] += q[i, k] * q[j, k]

    for i in range(m):
        fmat[i, 0] += q[i, 0] * r[0, 0]
        fmat[i, 1] += q[i, 0] * r[0, 1] + q[i, 1] * r[1, 1]
        for j in range(2, m):
            for k in range(j - 2, j + 1):
                fmat[i, j] += q[i, k] * r[k, j]

    for i in range(m):
        omat[i, i] -= 1
    fmat[0, 0] -= a[0, 0]
    fmat[0, 1] -= a[0, 1]
    for i in range(1, m - 1):
        fmat[i, i - 1] -= a[i, i - 1]
        fmat[i, i] -= a[i, i]
        fmat[i, i + 1] -= a[i, i + 1]
    fmat[-1, -2] -= a[-1, -2]
    fmat[-1, -1] -= a[-1, -1]

    return np.concatenate((omat[np.triu_indices_from(omat)].flatten(), fmat.flatten()))


def eigen(x, param):
    dim_in = int(np.sqrt(len(x) + 0.25))
    dvec = x[:dim_in]
    qmat = x[dim_in:].reshape(dim_in, dim_in)
    emat = qmat @ np.diag(dvec) @ qmat - param
    omat = qmat @ qmat - np.eye(dim_in)
    return np.concatenate((emat.flatten(), omat.flatten()))


def powell_singular(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in)
    fvec[::4] = x[::4] + 10 * x[1::4]
    fvec[1::4] = 5 * (x[2::4] - x[3::4])
    fvec[2::4] = (x[1::4] - 2 * x[2::4]) ** 2
    fvec[3::4] = 10 * (x[0::4] - x[3::4]) ** 2
    return fvec


@njit
def hydcar(
    x_in,
    n,
    m,
    k,
):
    x = x_in[: (n * m)].reshape((n, m))
    t = x_in[(n * m) : 4 * n]
    v = x_in[4 * n :]

    avec = np.array([9.647, 9.953, 9.466], dtype=np.float64)
    bvec = np.array([-2998, -3448.10, -3347.25], dtype=np.float64)
    cvec = np.array([230.66, 235.88, 215.31], dtype=np.float64)
    alp = np.array([37.6, 48.2, 45.4], dtype=np.float64)
    be = np.array([8425, 9395, 10466], dtype=np.float64)
    bep = np.array([24.2, 35.6, 31.9], dtype=np.float64)
    fl = np.array([30, 30, 40], dtype=np.float64)
    tf = 100.0
    b = 40.0
    d = 60.0
    q = 2500000.0

    out = np.empty(n * 5 - 1, dtype=np.float64)
    fvec1 = np.zeros(m, dtype=np.float64)
    fvec3 = np.zeros(m, dtype=np.float64)
    fvec2 = np.zeros((n - 2, m), dtype=np.float64)
    fvec7 = np.zeros(n, dtype=np.float64)
    fvec8 = 0
    fvec9 = np.zeros(n - 2, dtype=np.float64)

    for j in range(m):
        fvec1[j] += x[0, j] * b
        fvec3[j] += -x[n - 1, j]

    for j in range(m):
        fvec1[j] += -1 * x[1, j] * (v[0] + b)
        fvec1[j] += v[0] * x[0, j] * np.exp(avec[j] + (bvec[j] / (t[0] + cvec[j])))
        fvec3[j] += x[n - 2, j] * np.exp(avec[j] + (bvec[j] / (t[n - 2] + cvec[j])))

        fvec8 += (
            (
                v[0]
                * x[0, j]
                * np.exp(avec[j] + (bvec[j] / (t[0] + cvec[j])))
                * (be[j] + bep[j] * t[0])
            )
            + b * x[0, j] * (alp[j] * t[0])
            - x[1, j] * (b + v[0]) * (alp[j] * t[1])
        )

        for i in range(1, n - 1):
            fvec2[i - 1, j] += (
                v[i - 1]
                * x[i - 1, j]
                * (-1)
                * np.exp(avec[j] + (bvec[j] / (t[i - 1] + cvec[j])))
            )
            fvec2[i - 1, j] += (
                v[i] * x[i, j] * np.exp(avec[j] + (bvec[j] / (t[i] + cvec[j])))
            )

            fvec9[i - 1] += (
                v[i]
                * x[i, j]
                * np.exp(avec[j] + (bvec[j] / (t[i] + cvec[j])))
                * (be[j] + bep[j] * t[i])
            )
            fvec9[i - 1] += (
                v[i - 1]
                * x[i - 1, j]
                * (-1)
                * np.exp(avec[j] + (bvec[j] / (t[i - 1] + cvec[j])))
                * (be[j] + bep[j] * t[i - 1])
            )

        for i in range(n):
            fvec7[i] += x[i, j] * np.exp(avec[j] + (bvec[j] / (t[i] + cvec[j])))

    for j in range(m):
        for i in range(1, k):
            fvec2[i - 1, j] += -1 * x[i + 1, j] * (v[i] + b)
            fvec2[i - 1, j] += x[i, j] * (v[i - 1] + b)

        fvec2[k - 1, j] += -1 * x[k + 1, j] * (v[k] - d)
        fvec2[k - 1, j] += x[k, j] * (v[k - 1] + b)

        for i in range(k + 1, n - 1):
            fvec2[i - 1, j] += -1 * x[i + 1, j] * (v[i] - d)
            fvec2[i - 1, j] += x[i, j] * (v[i - 1] - d)

    for j in range(m):
        for i in range(1, k):
            fvec9[i - 1] += 1 * x[i, j] * (v[i - 1] + b) * (alp[j] * t[i])
            fvec9[i - 1] += (-1) * x[i + 1, j] * (v[i] + b) * (alp[j] * t[i + 1])

        fvec9[k - 1] += 1 * x[k, j] * (v[k - 1] + b) * (alp[j] * t[i])
        fvec9[k - 1] += (-1) * x[k + 1, j] * (v[k] - d) * (alp[j] * t[k + 1])

        for i in range(k + 1, n - 1):
            fvec9[i - 1] += 1 * x[i, j] * (v[i - 1] - d) * (alp[j] * t[i])
            fvec9[i - 1] += (-1) * x[i + 1, j] * (v[i] - d) * (alp[j] * t[i + 1])

    smallhf = 0
    for j in range(m):
        fvec2[k - 1, j] -= fl[j]
        smallhf += (tf * alp[j]) * fl[j]
    fvec7 -= 1
    fvec8 -= q
    fvec9[k - 1] -= smallhf

    out[:m] = fvec1 * 1e-2
    out[m : 2 * m] = fvec3
    out[2 * m : (n - 2) * m + 2 * m] = fvec2.flatten() * 1e-2
    out[(n - 2) * m + 2 * m : (n - 2) * m + 2 * m + n] = fvec7
    out[(n - 2) * m + 2 * m + n] = fvec8 * 1e-5
    out[-(n - 2) :] = fvec9 * 1e-5

    return out


def methane(x):
    fvec = np.zeros(31, dtype=np.float64)
    fvec[0] = 0.01 * (
        0.000826446280991736
        * x[24]
        * x[1]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[0]))
        - x[4] * (693.37 + x[24])
        + 693.37 * x[1]
    )
    fvec[1] = (
        0.000869565217391304 * np.exp(18.5751 - 3632.649 / (239.2 + x[18])) * x[19]
        - x[22]
    )
    fvec[2] = 0.01 * (
        -0.000826446280991736
        * x[24]
        * x[1]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[0]))
        - x[7] * (693.37 + x[25])
        + x[4] * (693.37 + x[24])
        + 0.000833333333333333
        * x[25]
        * x[4]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[3]))
    )
    fvec[3] = -4.5125 + 0.01 * (
        -0.000833333333333333
        * x[25]
        * x[4]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[3]))
        - x[10] * (-442.13 + x[26])
        + x[7] * (693.37 + x[25])
        + 0.000840336134453782
        * x[26]
        * x[7]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[6]))
    )
    fvec[4] = 0.01 * (
        -0.000840336134453782
        * x[26]
        * x[7]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[6]))
        - x[13] * (-442.13 + x[27])
        + x[10] * (-442.13 + x[26])
        + 0.000847457627118644
        * x[27]
        * x[10]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[9]))
    )
    fvec[5] = 0.01 * (
        -0.000847457627118644
        * x[27]
        * x[10]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[9]))
        - x[16] * (-442.13 + x[28])
        + x[13] * (-442.13 + x[27])
        + 0.000854700854700855
        * x[28]
        * x[13]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[12]))
    )
    fvec[6] = 0.01 * (
        -0.000854700854700855
        * x[28]
        * x[13]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[12]))
        - x[19] * (-442.13 + x[29])
        + x[16] * (-442.13 + x[28])
        + 0.000862068965517241
        * x[29]
        * x[16]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[15]))
    )
    fvec[7] = 0.01 * (
        -0.000862068965517241
        * x[29]
        * x[16]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[15]))
        - x[22] * (-442.13 + x[30])
        + x[19] * (-442.13 + x[29])
        + 0.000869565217391304
        * x[30]
        * x[19]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[18]))
    )
    fvec[8] = 0.01 * (
        0.000826446280991736 * x[24] * x[2] * np.exp(18.3443 - 3841.2203 / (228 + x[0]))
        - x[5] * (693.37 + x[24])
        + 693.37 * x[2]
    )
    fvec[9] = (
        0.000869565217391304 * np.exp(18.3443 - 3841.2203 / (228 + x[18])) * x[20]
        - x[23]
    )
    fvec[10] = 0.01 * (
        -0.000826446280991736
        * x[24]
        * x[2]
        * np.exp(18.3443 - 3841.2203 / (228 + x[0]))
        - x[8] * (693.37 + x[25])
        + x[5] * (693.37 + x[24])
        + 0.000833333333333333
        * x[25]
        * x[5]
        * np.exp(18.3443 - 3841.2203 / (228 + x[3]))
    )
    fvec[11] = -6.8425 + 0.01 * (
        -0.000833333333333333
        * x[25]
        * x[5]
        * np.exp(18.3443 - 3841.2203 / (228 + x[3]))
        - x[11] * (-442.13 + x[26])
        + x[8] * (693.37 + x[25])
        + 0.000840336134453782
        * x[26]
        * x[8]
        * np.exp(18.3443 - 3841.2203 / (228 + x[6]))
    )
    fvec[12] = 0.01 * (
        -0.000840336134453782
        * x[26]
        * x[8]
        * np.exp(18.3443 - 3841.2203 / (228 + x[6]))
        - x[14] * (-442.13 + x[27])
        + x[11] * (-442.13 + x[26])
        + 0.000847457627118644
        * x[27]
        * x[11]
        * np.exp(18.3443 - 3841.2203 / (228 + x[9]))
    )
    fvec[13] = 0.01 * (
        -0.000847457627118644
        * x[27]
        * x[11]
        * np.exp(18.3443 - 3841.2203 / (228 + x[9]))
        - x[17] * (-442.13 + x[28])
        + x[14] * (-442.13 + x[27])
        + 0.000854700854700855
        * x[28]
        * x[14]
        * np.exp(18.3443 - 3841.2203 / (228 + x[12]))
    )
    fvec[14] = 0.01 * (
        -0.000854700854700855
        * x[28]
        * x[14]
        * np.exp(18.3443 - 3841.2203 / (228 + x[12]))
        - x[20] * (-442.13 + x[29])
        + x[17] * (-442.13 + x[28])
        + 0.000862068965517241
        * x[29]
        * x[17]
        * np.exp(18.3443 - 3841.2203 / (228 + x[15]))
    )
    fvec[15] = 0.01 * (
        -0.000862068965517241
        * x[29]
        * x[17]
        * np.exp(18.3443 - 3841.2203 / (228 + x[15]))
        - x[23] * (-442.13 + x[30])
        + x[20] * (-442.13 + x[29])
        + 0.000869565217391304
        * x[30]
        * x[20]
        * np.exp(18.3443 - 3841.2203 / (228 + x[18]))
    )
    fvec[16] = (
        -1
        + 0.000826446280991736 * np.exp(18.5751 - 3632.649 / (239.2 + x[0])) * x[1]
        + 0.000826446280991736 * np.exp(18.3443 - 3841.2203 / (228 + x[0])) * x[2]
    )
    fvec[17] = (
        -1
        + 0.000833333333333333 * np.exp(18.5751 - 3632.649 / (239.2 + x[3])) * x[4]
        + 0.000833333333333333 * np.exp(18.3443 - 3841.2203 / (228 + x[3])) * x[5]
    )
    fvec[18] = (
        -1
        + 0.000840336134453782 * np.exp(18.5751 - 3632.649 / (239.2 + x[6])) * x[7]
        + 0.000840336134453782 * np.exp(18.3443 - 3841.2203 / (228 + x[6])) * x[8]
    )
    fvec[19] = (
        -1
        + 0.000847457627118644 * np.exp(18.5751 - 3632.649 / (239.2 + x[9])) * x[10]
        + 0.000847457627118644 * np.exp(18.3443 - 3841.2203 / (228 + x[9])) * x[11]
    )
    fvec[20] = (
        -1
        + 0.000854700854700855 * np.exp(18.5751 - 3632.649 / (239.2 + x[12])) * x[13]
        + 0.000854700854700855 * np.exp(18.3443 - 3841.2203 / (228 + x[12])) * x[14]
    )
    fvec[21] = (
        -1
        + 0.000862068965517241 * np.exp(18.5751 - 3632.649 / (239.2 + x[15])) * x[16]
        + 0.000862068965517241 * np.exp(18.3443 - 3841.2203 / (228 + x[15])) * x[17]
    )
    fvec[22] = (
        -1
        + 0.000869565217391304 * np.exp(18.5751 - 3632.649 / (239.2 + x[18])) * x[19]
        + 0.000869565217391304 * np.exp(18.3443 - 3841.2203 / (228 + x[18])) * x[20]
    )
    fvec[23] = (
        -1
        + 0.00087719298245614 * np.exp(18.5751 - 3632.649 / (239.2 + x[21])) * x[22]
        + 0.00087719298245614 * np.exp(18.3443 - 3841.2203 / (228 + x[21])) * x[23]
    )
    fvec[24] = -83.862 + 1e-5 * (
        0.000826446280991736
        * x[24]
        * x[1]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[0]))
        * (9566.67 + 0.0422 * x[0] * x[0] - 1.59 * x[0])
        + 693.37 * (0.0422 * x[0] * x[0] + 15.97 * x[0]) * x[1]
        - x[4] * (693.37 + x[24]) * (0.0422 * x[3] * x[3] + 15.97 * x[3])
        + 0.000826446280991736
        * x[24]
        * x[2]
        * np.exp(18.3443 - 3841.2203 / (228 + x[0]))
        * (10834.67 + 8.74 * x[0])
        + 12549.997 * x[2] * x[0]
        - 18.1 * x[5] * (693.37 + x[24]) * x[3]
    )
    fvec[25] = 1e-5 * (
        0.000833333333333333
        * x[25]
        * x[4]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[3]))
        * (9566.67 + 0.0422 * x[3] * x[3] - 1.59 * x[3])
        + x[4] * (693.37 + x[24]) * (0.0422 * x[3] * x[3] + 15.97 * x[3])
        - 0.000826446280991736
        * x[24]
        * x[1]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[0]))
        * (9566.67 + 0.0422 * x[0] * x[0] - 1.59 * x[0])
        - x[7] * (693.37 + x[25]) * (0.0422 * x[6] * x[6] + 15.97 * x[6])
        + 0.000833333333333333
        * x[25]
        * x[5]
        * np.exp(18.3443 - 3841.2203 / (228 + x[3]))
        * (10834.67 + 8.74 * x[3])
        + 18.1 * x[5] * (693.37 + x[24]) * x[3]
        - 0.000826446280991736
        * x[24]
        * x[2]
        * np.exp(18.3443 - 3841.2203 / (228 + x[0]))
        * (10834.67 + 8.74 * x[0])
        - 18.1 * x[8] * (693.37 + x[25]) * x[6]
    )
    fvec[26] = -18.9447111025 + 1e-5 * (
        0.000840336134453782
        * x[26]
        * x[7]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[6]))
        * (9566.67 + 0.0422 * x[6] * x[6] - 1.59 * x[6])
        + x[7] * (693.37 + x[25]) * (0.0422 * x[6] * x[6] + 15.97 * x[6])
        - 0.000833333333333333
        * x[25]
        * x[4]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[3]))
        * (9566.67 + 0.0422 * x[3] * x[3] - 1.59 * x[3])
        - x[10] * (-442.13 + x[26]) * (0.0422 * x[9] * x[9] + 15.97 * x[9])
        + 0.000840336134453782
        * x[26]
        * x[8]
        * np.exp(18.3443 - 3841.2203 / (228 + x[6]))
        * (10834.67 + 8.74 * x[6])
        + 18.1 * x[8] * (693.37 + x[25]) * x[6]
        - 0.000833333333333333
        * x[25]
        * x[5]
        * np.exp(18.3443 - 3841.2203 / (228 + x[3]))
        * (10834.67 + 8.74 * x[3])
        - 18.1 * x[11] * (-442.13 + x[26]) * x[9]
    )
    fvec[27] = 1e-5 * (
        0.000847457627118644
        * x[27]
        * x[10]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[9]))
        * (9566.67 + 0.0422 * x[9] * x[9] - 1.59 * x[9])
        + x[10] * (-442.13 + x[26]) * (0.0422 * x[9] * x[9] + 15.97 * x[9])
        - 0.000840336134453782
        * x[26]
        * x[7]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[6]))
        * (9566.67 + 0.0422 * x[6] * x[6] - 1.59 * x[6])
        - x[13] * (-442.13 + x[27]) * (0.0422 * x[12] * x[12] + 15.97 * x[12])
        + 0.000847457627118644
        * x[27]
        * x[11]
        * np.exp(18.3443 - 3841.2203 / (228 + x[9]))
        * (10834.67 + 8.74 * x[9])
        + 18.1 * x[11] * (-442.13 + x[26]) * x[9]
        - 0.000840336134453782
        * x[26]
        * x[8]
        * np.exp(18.3443 - 3841.2203 / (228 + x[6]))
        * (10834.67 + 8.74 * x[6])
        - 18.1 * x[14] * (-442.13 + x[27]) * x[12]
    )
    fvec[28] = 1e-5 * (
        0.000854700854700855
        * x[28]
        * x[13]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[12]))
        * (9566.67 + 0.0422 * x[12] * x[12] - 1.59 * x[12])
        + x[13] * (-442.13 + x[27]) * (0.0422 * x[12] * x[12] + 15.97 * x[12])
        - 0.000847457627118644
        * x[27]
        * x[10]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[9]))
        * (9566.67 + 0.0422 * x[9] * x[9] - 1.59 * x[9])
        - x[16] * (-442.13 + x[28]) * (0.0422 * x[15] * x[15] + 15.97 * x[15])
        + 0.000854700854700855
        * x[28]
        * x[14]
        * np.exp(18.3443 - 3841.2203 / (228 + x[12]))
        * (10834.67 + 8.74 * x[12])
        + 18.1 * x[14] * (-442.13 + x[27]) * x[12]
        - 0.000847457627118644
        * x[27]
        * x[11]
        * np.exp(18.3443 - 3841.2203 / (228 + x[9]))
        * (10834.67 + 8.74 * x[9])
        - 18.1 * x[17] * (-442.13 + x[28]) * x[15]
    )
    fvec[29] = 1e-5 * (
        0.000862068965517241
        * x[29]
        * x[16]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[15]))
        * (9566.67 + 0.0422 * x[15] * x[15] - 1.59 * x[15])
        + x[16] * (-442.13 + x[28]) * (0.0422 * x[15] * x[15] + 15.97 * x[15])
        - 0.000854700854700855
        * x[28]
        * x[13]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[12]))
        * (9566.67 + 0.0422 * x[12] * x[12] - 1.59 * x[12])
        - x[19] * (-442.13 + x[29]) * (0.0422 * x[18] * x[18] + 15.97 * x[18])
        + 0.000862068965517241
        * x[29]
        * x[17]
        * np.exp(18.3443 - 3841.2203 / (228 + x[15]))
        * (10834.67 + 8.74 * x[15])
        + 18.1 * x[17] * (-442.13 + x[28]) * x[15]
        - 0.000854700854700855
        * x[28]
        * x[14]
        * np.exp(18.3443 - 3841.2203 / (228 + x[12]))
        * (10834.67 + 8.74 * x[12])
        - 18.1 * x[20] * (-442.13 + x[29]) * x[18]
    )
    fvec[30] = 1e-5 * (
        0.000869565217391304
        * x[30]
        * x[19]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[18]))
        * (9566.67 + 0.0422 * x[18] * x[18] - 1.59 * x[18])
        + x[19] * (-442.13 + x[29]) * (0.0422 * x[18] * x[18] + 15.97 * x[18])
        - 0.000862068965517241
        * x[29]
        * x[16]
        * np.exp(18.5751 - 3632.649 / (239.2 + x[15]))
        * (9566.67 + 0.0422 * x[15] * x[15] - 1.59 * x[15])
        - x[22] * (-442.13 + x[30]) * (0.0422 * x[21] * x[21] + 15.97 * x[21])
        + 0.000869565217391304
        * x[30]
        * x[20]
        * np.exp(18.3443 - 3841.2203 / (228 + x[18]))
        * (10834.67 + 8.74 * x[18])
        + 18.1 * x[20] * (-442.13 + x[29]) * x[18]
        - 0.000862068965517241
        * x[29]
        * x[17]
        * np.exp(18.3443 - 3841.2203 / (228 + x[15]))
        * (10834.67 + 8.74 * x[15])
        - 18.1 * x[23] * (-442.13 + x[30]) * x[21]
    )
    return fvec


def argtrig(x):
    dim_in = len(x)
    fvec = (
        dim_in
        - np.sum(np.cos(x))
        + np.arange(1, dim_in + 1) * (1 - np.cos(x) - np.sin(x))
    )
    return fvec


def artif(x):
    dim_in = len(x)
    xvec = np.zeros(dim_in + 2, dtype=np.float64)
    xvec[1:-1] = x
    fvec = np.zeros(dim_in, dtype=np.float64)
    for i in range(dim_in):
        fvec[i] = -0.05 * (xvec[i + 1] + xvec[i + 2] + xvec[i]) + np.arctan(
            np.sin(np.mod(i + 1, 100) * xvec[i + 1])
        )
    return fvec


def arwhdne(x):
    dim_in = len(x)
    fvec = np.zeros(2 * (dim_in - 1))
    fvec[: dim_in - 1] = x[:-1] ** 2 + x[-1] ** 2
    fvec[dim_in - 1 :] = 4 * x[:-1] - 3
    return fvec


@njit
def bdvalues(x):
    dim_in = len(x)
    h = 1 / (dim_in + 1)
    xvec = np.zeros(dim_in + 2, dtype=np.float64)
    for i in range(dim_in):
        xvec[i + 1] = x[i]
    fvec = np.zeros(dim_in, dtype=np.float64)
    for i in range(2, dim_in + 2):
        fvec[i - 2] = (
            -xvec[i - 2]
            + 2 * xvec[i - 1]
            - xvec[i]
            + 0.5 * h**2 * (xvec[i - 1] + i * h + 1) ** 3
        )
    return fvec


def bratu_2d(x, alpha):
    x = x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x)))))
    p = x.shape[0] + 2
    h = 1 / (p - 1)
    c = h**2 * alpha
    xvec = np.zeros((x.shape[0] + 2, x.shape[1] + 2), dtype=np.float64)
    xvec[1 : x.shape[0] + 1, 1 : x.shape[1] + 1] = x
    fvec = np.zeros(x.shape)
    for i in range(2, p):
        for j in range(2, p):
            fvec[i - 2, j - 2] = (
                4 * xvec[i - 1, j - 1]
                - xvec[i, j - 1]
                - xvec[i - 2, j - 1]
                - xvec[i - 1, j]
                - xvec[i - 1, j - 2]
                - c * np.exp(xvec[i - 1, j - 1])
            )
    return fvec.flatten()


def bratu_3d(x, alpha):
    n = int(np.cbrt(len(x)))
    x = x.reshape((n, n, n))
    p = x.shape[0] + 2
    h = 1 / (p - 1)
    c = h**2 * alpha
    xvec = np.zeros((x.shape[0] + 2, x.shape[1] + 2, x.shape[2] + 2), dtype=np.float64)
    xvec[1 : x.shape[0] + 1, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x
    fvec = np.zeros(x.shape, dtype=np.float64)
    for i in range(2, p):
        for j in range(2, p):
            for k in range(2, p):
                fvec[i - 2, j - 2, k - 2] = (
                    6 * xvec[i - 1, j - 1, k - 1]
                    - xvec[i, j - 1, k - 1]
                    - xvec[i - 2, j - 1, k - 1]
                    - xvec[i - 1, j, k - 1]
                    - xvec[i - 1, j - 2, k - 1]
                    - xvec[i - 1, j - 1, k]
                    - xvec[i - 1, j - 1, k - 2]
                    - c * np.exp(xvec[i, j, k])
                )
    return fvec.flatten()


def broydn_3d(x):
    kappa_1 = 2
    kappa_2 = 1
    fvec = np.zeros_like(x)
    fvec[0] = -2 * x[1] + kappa_2 + (3 - kappa_1 * x[0]) * x[0]
    fvec[1 : len(x) - 1] = (
        -x[:-2] - 2 * x[2:] + kappa_2 + (3 - kappa_1 * x[1:-1]) * x[1:-1]
    )
    fvec[-1] = -x[-2] + kappa_2 + (3 - kappa_1 * x[-1]) * x[-1]
    return fvec


def broydn_bd(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in, dtype=np.float64)
    for i in range(1, 1 + dim_in):
        ji = []
        lb = max(1, i - 5)
        ub = min(dim_in, i + 1)
        for j in range(lb, ub + 1):
            if j != i:
                ji.append(j)
        fvec[i - 1] = x[i - 1] * (2 + 5 * x[i - 1] ** 2) - np.sum(
            x[np.array(ji) - 1] * (1 + x[np.array(ji) - 1])
        )
    return fvec


def cbratu_2d(x):
    n = int(np.sqrt(len(x) / 2))
    x = x.reshape((2, n, n))
    xvec = np.zeros((x.shape[0], x.shape[1] + 2, x.shape[2] + 2), dtype=np.float64)
    xvec[0, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x[0, :, :]
    xvec[1, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x[1, :, :]
    p = x.shape[1] + 2
    h = 1 / (p - 1)
    alpha = 5
    c = h**2 * alpha
    fvec = np.zeros(x.shape, dtype=np.float64)
    for i in range(2, p):
        for j in range(2, p):
            fvec[0, i - 2, j - 2] = (
                4 * xvec[0, i - 1, j - 1]
                - xvec[0, i, j - 1]
                - xvec[0, i - 2, j - 1]
                - xvec[0, i - 1, j]
                - xvec[0, i - 1, j - 2]
                - c * np.exp(xvec[0, i - 1, j - 1]) * np.cos(xvec[0, i - 1, j - 1])
            )
            fvec[1, i - 2, j - 2] = (
                4 * xvec[1, i - 1, j - 1]
                - xvec[1, i, j - 1]
                - xvec[1, i - 2, j - 1]
                - xvec[1, i - 1, j]
                - xvec[1, i - 1, j - 2]
                - c * np.exp(xvec[1, i - 1, j - 1]) * np.sin(xvec[1, i - 1, j - 1])
            )
    return fvec.flatten()


def chandheq(x):
    dim_in = len(x)
    constant = 1
    w = np.ones(dim_in, dtype=np.int64) / dim_in
    h = np.ones(dim_in, dtype=np.int64)
    fvec = np.zeros(dim_in, dtype=np.float64)
    for i in range(dim_in):
        fvec[i] = (-0.5 * constant * w * x[i] / (x[i] + x) * h[i] * h + h[i] - 1).sum()
    return fvec


@njit
def chemrcta(x):
    dim_in = int(len(x) / 2)
    x = x.reshape((2, dim_in))
    fvec = np.zeros(2 * dim_in, dtype=np.float64)

    # define some auxiliary params
    pem = 1.0
    peh = 5.0
    d = 0.135
    b = 0.5
    beta = 2.0
    gamma = 25.0
    h = 1 / (dim_in - 1)
    cu1 = -h * pem
    cui1 = 1 / (h**2 * pem) + 1 / h
    cui = -1 / h - 2 / (h**2 * pem)
    ct1 = -h * peh
    cti1 = 1 / (h**2 * peh) + 1 / h
    cti = -beta - 1 / h - 2 / (h**2 * peh)

    fvec[0] = cu1 * x[0, 1] - x[0, 0] + h * pem
    fvec[1] = ct1 * x[1, 1] - x[1, 0] + h * peh
    for i in range(2, dim_in):
        fvec[i] = (
            -d * x[0, i - 1] * np.exp(gamma - gamma / x[1, i - 1])
            + (cui1) * x[0, i - 2]
            + cui * x[0, i - 1]
            + x[0, i] / (h**2 * pem)
        )
        fvec[dim_in - 2 + i] = (
            b * d * x[0, i - 1] * np.exp(gamma - gamma / x[1, i - 1])
            + beta * x[1, i - 1]
            + cti1 * x[1, i - 2]
            + cti * x[1, i - 1]
            + x[1, i] / (h**2 * peh)
        )
    fvec[-2] = x[0, -1] - x[0, -2]
    fvec[-1] = x[1, -1] - x[1, -2]
    return fvec


@njit
def chemrctb(x):
    dim_in = int(len(x))
    fvec = np.zeros(dim_in, dtype=np.float64)

    # define some auxiliary params
    pe = 5.0
    d = 0.135
    b = 0.5
    gamma = 25.0
    h = 1 / (dim_in - 1)
    ct1 = -h * pe
    cti1 = 1 / (h**2 * pe) + 1 / h
    cti = -1 / h - 2 / (h**2 * pe)

    fvec[0] = ct1 * x[1] - x[0] + h * pe
    for i in range(2, dim_in):
        fvec[i - 1] = (
            d * (b + 1 - x[i - 1]) * np.exp(gamma - gamma / x[i - 1])
            + cti1 * x[i - 2]
            + cti * x[i - 1]
            + x[i] / (h**2 * pe)
        )
    fvec[-1] = x[-1] - x[-2]
    return fvec


def chnrsbne(x):
    alfa = np.array(
        [
            1.25,
            1.40,
            2.40,
            1.40,
            1.75,
            1.20,
            2.25,
            1.20,
            1.00,
            1.10,
            1.50,
            1.60,
            1.25,
            1.25,
            1.20,
            1.20,
            1.40,
            0.50,
            0.50,
            1.25,
            1.80,
            0.75,
            1.25,
            1.40,
            1.60,
            2.00,
            1.00,
            1.60,
            1.25,
            2.75,
            1.25,
            1.25,
            1.25,
            3.00,
            1.50,
            2.00,
            1.25,
            1.40,
            1.80,
            1.50,
            2.20,
            1.40,
            1.50,
            1.25,
            2.00,
            1.50,
            1.25,
            1.40,
            0.60,
            1.50,
        ]
    )
    dim_in = len(x)
    fvec = np.zeros(2 * (dim_in - 1))
    fvec[: dim_in - 1] = 4 * alfa[1:] * (x[:-1] - x[1:] ** 2)
    fvec[dim_in - 1 :] = x[1:] - 1
    return fvec


@njit
def drcavty(x, r):
    m = int(np.sqrt(len(x)))
    x = x.reshape((m, m))
    h = 1 / (m + 2)
    xvec = np.zeros((m + 4, m + 4), dtype=np.float64)
    xvec[2 : m + 2, 2 : m + 2] = x
    xvec[-2, :] = -h / 2
    xvec[-1, :] = h / 2
    fvec = np.zeros(x.shape, dtype=np.float64)
    for i in range(m):
        for j in range(m):
            fvec[i, j] = (
                20 * xvec[i + 2, j + 2]
                - 8 * xvec[i + 1, j + 2]
                - 8 * xvec[i + 3, j + 2]
                - 8 * xvec[i + 2, j + 1]
                - 8 * xvec[i + 2, j + 3]
                + 2 * xvec[i + 1, j + 3]
                + 2 * xvec[i + 3, j + 2]
                + 2 * xvec[i + 1, j + 1]
                + 2 * xvec[i + 3, j + 3]
                + xvec[i, j + 2]
                + xvec[i + 4, j + 2]
                + xvec[i + 2, j]
                + xvec[i + 2, j + 4]
                + (r / 4)
                * (xvec[i + 2, j + 3] - xvec[i + 2, j + 1])
                * (
                    xvec[i, j + 2]
                    + xvec[i + 1, j + 1]
                    + xvec[i + 1, j + 3]
                    - 4 * xvec[i + 1, j + 2]
                    - 4 * xvec[i + 3, j + 2]
                    - xvec[i + 3, j + 2]
                    - xvec[i + 3, j + 3]
                    - xvec[i + 4, j + 2]
                )
                - (r / 4)
                * (xvec[i + 3, j + 2] - xvec[i + 1, j + 2])
                * (
                    xvec[i + 2, j]
                    + xvec[i + 1, j + 1]
                    + xvec[i + 3, j + 1]
                    - 4 * xvec[i + 2, j + 1]
                    - 4 * xvec[i + 2, j + 3]
                    - xvec[i + 1, j + 3]
                    - xvec[i + 3, j + 3]
                    - xvec[i + 2, j + 4]
                )
            )

    return fvec.flatten()


def freurone(x):
    dim_in = len(x)
    fvec = np.zeros((2, dim_in - 1), dtype=np.float64)
    for i in range(dim_in - 1):
        fvec[0, i] = (5.0 - x[i + 1]) * x[i + 1] ** 2 + x[i] - 2 * x[i + 1] - 13.0
        fvec[1, i] = (1.0 + x[i + 1]) * x[i + 1] ** 2 + x[i] - 14 * x[i + 1] - 29.0
    return fvec.flatten()


def hatfldg(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in, dtype=np.float64)
    for i in range(1, dim_in - 1):
        fvec[i - 1] = x[i] * (x[i - 1] - x[i + 1]) + x[i] - x[12] + 1
    fvec[-2] = x[0] - x[12] + 1 - x[0] * x[1]
    fvec[-1] = x[-1] - x[12] + 1 + x[-2] * x[-1]
    return fvec


def integreq(x):
    dim_in = len(x)
    h = 1 / (dim_in + 1)
    t = np.arange(1, dim_in + 1) * h
    xvec = np.zeros(dim_in + 2, dtype=np.float64)
    xvec[1:-1] = x
    fvec = np.zeros_like(x)
    for i in range(1, dim_in):
        fvec[i - 1] = (
            xvec[i]
            + h
            * (
                (1 - t[i - 1]) * (t[:i] * (xvec[1 : i + 1] + t[:i] + 1) ** 3).sum()
                + t[i - 1] * ((1 - t[i:]) * (xvec[i + 1 : -1] + t[i:] + 1) ** 3).sum()
            )
            / 2
        )
    fvec[-1] = (
        xvec[-2]
        + h
        * (
            (1 - t[-1]) * (t * (xvec[1:-1] + t + 1) ** 3).sum()
            + t[-1] * ((1 - t[-1]) * (xvec[-2] + t[-1] + 1) ** 3)
        )
        / 2
    )
    return fvec


def msqrta(x):
    dim_in = int(np.sqrt(len(x)))
    xmat = x.reshape((dim_in, dim_in))
    bmat = 5 * xmat
    amat = np.zeros((dim_in, dim_in), dtype=np.float64)
    for i in range(1, dim_in + 1):
        for j in range(1, dim_in + 1):
            amat[i - 1, j - 1] = (bmat[i - 1, :] * bmat[:, j - 1]).sum()
    fmat = np.zeros((dim_in, dim_in))
    for i in range(1, dim_in + 1):
        for j in range(1, dim_in + 1):
            fmat[i - 1, j - 1] = (xmat[i - 1, :] * xmat[:, j - 1]).sum() - amat[
                i - 1, j - 1
            ]
    return fmat.flatten()


def penalty_1(x, a=1e-5):
    fvec = np.sqrt(a) * (x - 2)
    fvec = np.concatenate([fvec, [x @ x - 1 / 4]])
    return fvec


def penalty_2(x, a=1e-10):
    dim_in = len(x)
    y = np.exp(np.arange(1, 2 * dim_in + 1) / 10) + np.exp(np.arange(2 * dim_in) / 10)
    fvec = np.zeros(2 * dim_in)
    fvec[0] = x[0] - 0.2
    fvec[1:dim_in] = np.sqrt(a) * (
        np.exp(x[1:] / 10) + np.exp(x[:-1] / 10) - y[1:dim_in]
    )
    fvec[dim_in:-1] = np.sqrt(a) * (np.exp(x[1:] / 10) - np.exp(-1 / 10))
    fvec[-1] = (np.arange(1, dim_in + 1)[::-1] * x**2).sum() - 1
    return fvec


def vardimne(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in + 2)
    fvec[:-2] = x - 1
    fvec[-2] = (np.arange(1, dim_in + 1) * (x - 1)).sum()
    fvec[-1] = ((np.arange(1, dim_in + 1) * (x - 1)).sum()) ** 2
    return fvec


def yatpsq_1(x, dim_in):
    xvec = x[: dim_in**2]
    xvec = xvec.reshape((dim_in, dim_in))
    yvec = x[dim_in**2 : dim_in**2 + dim_in]
    zvec = x[dim_in**2 + dim_in : dim_in**2 + 2 * dim_in]
    fvec = np.zeros((dim_in, dim_in), dtype=np.float64)
    for i in range(dim_in):
        for j in range(dim_in):
            fvec[i, j] = (
                xvec[i, j] ** 3
                - 10 * xvec[i, j] ** 2
                - (yvec[i] + zvec[j])
                * (xvec[i, j] * np.cos(xvec[i, j]) - np.sin(xvec[i, j]))
            )
    fvec = fvec.flatten()
    temp = (np.sin(xvec) / xvec).sum(axis=0) - 1
    fvec = np.concatenate((fvec, temp))
    temp = (np.sin(xvec) / xvec).sum(axis=1) - 1
    fvec = np.concatenate((fvec, temp))
    return fvec


def yatpsq_2(x, dim_in):
    xvec = x[: dim_in**2]
    xvec = xvec.reshape((dim_in, dim_in))
    yvec = x[dim_in**2 : dim_in**2 + dim_in]
    zvec = x[dim_in**2 + dim_in : dim_in**2 + 2 * dim_in]
    fvec = np.zeros((dim_in, dim_in), dtype=np.float64)
    for i in range(dim_in):
        for j in range(dim_in):
            fvec[i, j] = xvec[i, j] - (yvec[i] + zvec[j]) * (1 + np.cos(xvec[i, j])) - 1
    fvec = fvec.flatten()
    temp = (np.sin(xvec) + xvec).sum(axis=0) - 1
    fvec = np.concatenate((fvec, temp))
    temp = (np.sin(xvec) + xvec).sum(axis=1) - 1
    fvec = np.concatenate((fvec, temp))
    return fvec


def get_start_points_msqrta(dim_in, flag=1):
    bmat = np.zeros((dim_in, dim_in))
    for i in range(1, dim_in + 1):
        for j in range(1, dim_in + 1):
            bmat[i - 1, j - 1] = np.sin(((i - 1) * dim_in + j) ** 2)
    if flag == 2:
        bmat[2, 0] = 0
    xmat = 0.2 * bmat
    return xmat.flatten().tolist()


def get_start_points_bdvalues(n, a=1):
    h = 1 / (n + 1)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (i + 1) * h * ((i + 1) * h - 1)
    return (x * a).tolist()


def get_start_points_spmsqrt(m):
    b = np.zeros((m, m))
    b[0, 0] = np.sin(1)
    b[0, 1] = np.sin(4)
    k = 2
    for i in range(1, m - 1):
        k += 1
        b[i, i - 1] = np.sin(k**2)
        k += 1
        b[i, i] = np.sin(k**2)
        k += 1
        b[i, i + 1] = np.sin(k**2)
    k += 1
    b[-1, -2] = np.sin(k**2)
    k += 1
    b[-1, -1] = np.sin(k**2)

    x = np.zeros((m, m))
    x[:, :2] = 0.2 * b[:, :2]
    x[1:-1, :-2] = 0.2 * b[1:-1, :-2]
    x[1:-1, 1:-1] = 0.2 * b[1:-1, 1:-1]
    x[1:-1, 2:] = 0.2 * b[1:-1, 2:]
    x[-1, -2:] = 0.2 * b[-1, -2:]

    x_out = x[x != 0]

    return x_out.tolist()


def get_start_points_qr3d(m):
    r = np.diag(2 * np.arange(1, m + 1) / m, 0) + np.diag((1 - np.arange(1, m)) / m, 1)
    r[0, 1] = 0
    r[-1, -1] = 2 * m
    return np.concatenate([np.eye(m).flatten(), r[np.triu_indices_from(r)]]).tolist()


def get_start_points_qr3dbd(m):
    r = np.diag(2 * np.arange(1, m + 1) / m, 0) + np.diag((1 - np.arange(1, m)) / m, 1)
    r[0, 1] = 0
    r[-1, -1] = 2 * m
    return np.concatenate(
        [np.eye(m).flatten(), r[0, :-2], r[1, 1:-1], r[2, 2:], r[3, 3:], [r[4, 4]]]
    ).tolist()


def get_start_points_hydcar20():
    x = [
        0.0,
        0.3,
        0.1,
        0.0,
        0.3,
        0.9,
        0.01,
        0.3,
        0.9,
        0.02,
        0.4,
        0.8,
        0.05,
        0.4,
        0.8,
        0.07,
        0.45,
        0.8,
        0.09,
        0.5,
        0.7,
        0.1,
        0.5,
        0.7,
        0.15,
        0.5,
        0.6,
        0.2,
        0.5,
        0.6,
        0.25,
        0.6,
        0.5,
        0.3,
        0.6,
        0.5,
        0.35,
        0.6,
        0.5,
        0.4,
        0.6,
        0.4,
        0.4,
        0.7,
        0.4,
        0.42,
        0.7,
        0.3,
        0.45,
        0.75,
        0.3,
        0.45,
        0.75,
        0.2,
        0.5,
        0.8,
        0.1,
        0.5,
        0.8,
        0.0,
    ]
    return x + [100] * 20 + [300] * 19


def get_start_points_hydcar6():
    x = [
        0.0,
        0.2,
        0.9,
        0.0,
        0.2,
        0.8,
        0.05,
        0.3,
        0.8,
        0.1,
        0.3,
        0.6,
        0.3,
        0.5,
        0.3,
        0.6,
        0.6,
        0.0,
    ]
    return x + [100] * 6 + [300] * 5


def get_start_points_methanb8():
    return [
        107.47,
        0.09203,
        0.908,
        102.4,
        0.1819,
        0.8181,
        97.44,
        0.284,
        0.716,
        96.3,
        0.3051,
        0.6949,
        93.99,
        0.3566,
        0.6434,
        89.72,
        0.468,
        0.532,
        83.71,
        0.6579,
        0.3421,
        78.31,
        0.8763,
        0.1237,
        886.37,
        910.01,
        922.52,
        926.46,
        935.56,
        952.83,
        975.73,
    ]


def get_start_points_methanl8():
    return [
        120,
        0.09203,
        0.908,
        110,
        0.1819,
        0.8181,
        100,
        0.284,
        0.716,
        88,
        0.3051,
        0.6949,
        86,
        0.3566,
        0.6434,
        84,
        0.468,
        0.532,
        80,
        0.6579,
        0.3421,
        76,
        0.8763,
        0.1237,
        886.37,
        910.01,
        922.52,
        926.46,
        935.56,
        952.83,
        975.73,
    ]


solution_x_bdvalues = [
    -0.00501717,
    -0.00998312,
    -0.01489709,
    -0.01975833,
    -0.02456605,
    -0.02931945,
    -0.03401771,
    -0.03866001,
    -0.0432455,
    -0.04777331,
    -0.05224255,
    -0.05665232,
    -0.0610017,
    -0.06528975,
    -0.06951549,
    -0.07367795,
    -0.07777612,
    -0.08180898,
    -0.08577546,
    -0.08967451,
    -0.09350501,
    -0.09726585,
    -0.10095589,
    -0.10457394,
    -0.10811881,
    -0.11158927,
    -0.11498406,
    -0.1183019,
    -0.12154147,
    -0.12470143,
    -0.1277804,
    -0.13077697,
    -0.13368969,
    -0.1365171,
    -0.13925766,
    -0.14190984,
    -0.14447205,
    -0.14694265,
    -0.14931997,
    -0.15160232,
    -0.15378794,
    -0.15587503,
    -0.15786175,
    -0.15974621,
    -0.16152647,
    -0.16320056,
    -0.16476642,
    -0.16622197,
    -0.16756507,
    -0.1687935,
    -0.16990502,
    -0.17089728,
    -0.17176792,
    -0.17251447,
    -0.17313443,
    -0.1736252,
    -0.17398413,
    -0.17420848,
    -0.17429545,
    -0.17424214,
    -0.17404559,
    -0.17370274,
    -0.17321044,
    -0.17256546,
    -0.17176447,
    -0.17080403,
    -0.16968062,
    -0.16839059,
    -0.16693019,
    -0.16529558,
    -0.16348276,
    -0.16148763,
    -0.15930595,
    -0.15693338,
    -0.15436539,
    -0.15159735,
    -0.14862447,
    -0.14544178,
    -0.14204417,
    -0.13842638,
    -0.13458293,
    -0.13050819,
    -0.12619633,
    -0.12164132,
    -0.11683693,
    -0.1117767,
    -0.10645396,
    -0.10086179,
    -0.09499304,
    -0.0888403,
    -0.08239586,
    -0.07565179,
    -0.06859981,
    -0.06123136,
    -0.05353755,
    -0.04550917,
    -0.03713662,
    -0.02840998,
    -0.01931889,
    -0.00985262,
]

solution_x_bratu_2d = [
    0.07234633,
    0.11814877,
    0.1459185,
    0.15914495,
    0.15914495,
    0.1459185,
    0.11814877,
    0.07234633,
    0.11814877,
    0.19875438,
    0.24923944,
    0.27361473,
    0.27361473,
    0.24923944,
    0.19875438,
    0.11814877,
    0.1459185,
    0.24923944,
    0.31530971,
    0.34753593,
    0.34753593,
    0.31530971,
    0.24923944,
    0.1459185,
    0.15914495,
    0.27361473,
    0.34753593,
    0.3837784,
    0.3837784,
    0.34753593,
    0.27361473,
    0.15914495,
    0.15914495,
    0.27361473,
    0.34753593,
    0.3837784,
    0.3837784,
    0.34753593,
    0.27361473,
    0.15914495,
    0.1459185,
    0.24923944,
    0.31530971,
    0.34753593,
    0.34753593,
    0.31530971,
    0.24923944,
    0.1459185,
    0.11814877,
    0.19875438,
    0.24923944,
    0.27361473,
    0.27361473,
    0.24923944,
    0.19875438,
    0.11814877,
    0.07234633,
    0.11814877,
    0.1459185,
    0.15914495,
    0.15914495,
    0.1459185,
    0.11814877,
    0.07234633,
]

solution_x_bratu_2d_t = [
    0.1933024,
    0.33566336,
    0.43355494,
    0.48428111,
    0.48428111,
    0.43355494,
    0.33566336,
    0.1933024,
    0.33566336,
    0.59839893,
    0.78485783,
    0.88316504,
    0.88316504,
    0.78485783,
    0.59839893,
    0.33566336,
    0.43355494,
    0.78485783,
    1.04056365,
    1.17766089,
    1.17766089,
    1.04056365,
    0.78485783,
    0.43355494,
    0.48428111,
    0.88316504,
    1.17766089,
    1.33720634,
    1.33720634,
    1.17766089,
    0.88316504,
    0.48428111,
    0.48428111,
    0.88316504,
    1.17766089,
    1.33720634,
    1.33720634,
    1.17766089,
    0.88316504,
    0.48428111,
    0.43355494,
    0.78485783,
    1.04056365,
    1.17766089,
    1.17766089,
    1.04056365,
    0.78485783,
    0.43355494,
    0.33566336,
    0.59839893,
    0.78485783,
    0.88316504,
    0.88316504,
    0.78485783,
    0.59839893,
    0.33566336,
    0.1933024,
    0.33566336,
    0.43355494,
    0.48428111,
    0.48428111,
    0.43355494,
    0.33566336,
    0.1933024,
]


solution_x_bratu_3d = [
    0.24431369,
    0.27785366,
    0.19682155,
    0.27785366,
    0.32761664,
    0.23878408,
    0.19682155,
    0.23878408,
    0.18908409,
    0.27785366,
    0.32761664,
    0.23878408,
    0.32761664,
    0.39611483,
    0.29367471,
    0.23878408,
    0.29367471,
    0.2314289,
    0.19682155,
    0.23878408,
    0.18908409,
    0.23878408,
    0.29367471,
    0.2314289,
    0.18908409,
    0.2314289,
    0.18663237,
]

solution_x_broydn_3d = [
    -0.57076119,
    -0.68191013,
    -0.70248602,
    -0.70626058,
    -0.70695185,
    -0.70707842,
    -0.70710159,
    -0.70710583,
    -0.70710661,
    -0.70710675,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710677,
    -0.70710675,
    -0.70710669,
    -0.70710654,
    -0.70710612,
    -0.70710498,
    -0.70710185,
    -0.70709332,
    -0.70707001,
    -0.70700634,
    -0.70683248,
    -0.70635771,
    -0.70506153,
    -0.7015252,
    -0.69189463,
    -0.66579752,
    -0.59603531,
    -0.4164123,
]


solution_x_cbratu_2d = [
    0.16692195,
    0.2529246,
    0.2796211,
    0.2529246,
    0.16692195,
    0.2529246,
    0.39198662,
    0.43607163,
    0.39198662,
    0.2529246,
    0.2796211,
    0.43607163,
    0.48598608,
    0.43607163,
    0.2796211,
    0.2529246,
    0.39198662,
    0.43607163,
    0.39198662,
    0.2529246,
    0.16692195,
    0.2529246,
    0.2796211,
    0.2529246,
    0.16692195,
]

solution_x_cbratu_2d = solution_x_cbratu_2d + [0] * 25

solution_x_broydn_bd = [
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -1.00000000e-10,
    -2.00000000e-10,
    -2.00000000e-10,
    -1.00000000e-10,
    4.00000000e-10,
    1.40000000e-09,
    3.00000000e-09,
    4.60000000e-09,
    4.70000000e-09,
    -1.00000000e-10,
    -1.43000000e-08,
    -4.23000000e-08,
    -8.25000000e-08,
    -1.17600000e-07,
    -1.00700000e-07,
    5.54000000e-08,
    4.68300000e-07,
    1.22420000e-06,
    2.22540000e-06,
    2.92120000e-06,
    1.96990000e-06,
    -2.95480000e-06,
    -1.47187000e-05,
    -3.48246000e-05,
    -5.90960000e-05,
    -7.05915000e-05,
    -3.15647000e-05,
    1.19032300e-04,
    4.48647900e-04,
    9.73368600e-04,
    1.53772420e-03,
    1.63273940e-03,
    2.14717600e-04,
    -4.30681910e-03,
    -1.36127680e-02,
    -2.81043041e-02,
    -4.39233903e-02,
    -4.73306566e-02,
    -8.45337580e-03,
    1.04321937e-01,
    2.74938066e-01,
    4.54655029e-01,
    6.22031184e-01,
    7.74293819e-01,
    9.11375485e-01,
    1.03226579e00,
    1.13635201e00,
    1.22498498e00,
    1.30019836e00,
    1.36374913e00,
    1.41711415e00,
    1.46168952e00,
    1.49882961e00,
    1.52972625e00,
    1.55537824e00,
    1.57663224e00,
    1.59421664e00,
    1.60875287e00,
    1.62076049e00,
    1.63067143e00,
    1.63884538e00,
    1.64557503e00,
    1.65102930e00,
    1.65461538e00,
    1.64858082e00,
    1.55247986e00,
]

solution_x_chemrctb = [
    0.05141945,
    0.05203209,
    0.05267567,
    0.05335175,
    0.05406197,
    0.05480806,
    0.05559182,
    0.05641517,
    0.05728009,
    0.05818869,
    0.05914317,
    0.06014585,
    0.06119916,
    0.06230566,
    0.06346804,
    0.06468911,
    0.06597184,
    0.06731935,
    0.0687349,
    0.07022193,
    0.07178406,
    0.07342507,
    0.07514894,
    0.07695987,
    0.07886224,
    0.08086068,
    0.08296004,
    0.08516541,
    0.08748215,
    0.08991588,
    0.09247251,
    0.09515826,
    0.09797963,
    0.10094348,
    0.104057,
    0.10732776,
    0.11076369,
    0.11437313,
    0.11816486,
    0.12214807,
    0.12633243,
    0.13072811,
    0.13534578,
    0.14019665,
    0.14529249,
    0.15064569,
    0.15626923,
    0.16217677,
    0.16838265,
    0.17490194,
    0.18175047,
    0.18894487,
    0.19650261,
    0.20444203,
    0.21278242,
    0.22154402,
    0.23074811,
    0.24041703,
    0.25057426,
    0.26124447,
    0.27245356,
    0.28422875,
    0.29659862,
    0.30959322,
    0.32324409,
    0.33758438,
    0.35264891,
    0.36847425,
    0.38509884,
    0.40256304,
    0.42090924,
    0.440182,
    0.46042812,
    0.48169675,
    0.50403953,
    0.52751072,
    0.55216731,
    0.57806915,
    0.60527915,
    0.63386338,
    0.66389124,
    0.69543563,
    0.72857315,
    0.76338427,
    0.79995348,
    0.83836951,
    0.87872535,
    0.921118,
    0.96564698,
    1.01240975,
    1.06148865,
    1.11291774,
    1.16660739,
    1.22219286,
    1.27878417,
    1.33468684,
    1.38740197,
    1.43443791,
    1.47507982,
    1.51238643,
]


solution_x_drcavty3 = [
    6.90580000e-06,
    -3.04054000e-05,
    -1.34595400e-04,
    -2.98301400e-04,
    -3.97564800e-04,
    -2.82615200e-04,
    -1.00791500e-04,
    1.18693000e-05,
    4.83418000e-05,
    3.86272000e-05,
    -3.61169000e-05,
    -1.56090300e-04,
    -3.44522400e-04,
    -5.22159200e-04,
    -5.02848100e-04,
    -1.96532500e-04,
    4.01814000e-05,
    1.66926300e-04,
    1.64254200e-04,
    9.75942000e-05,
    -1.53179900e-04,
    -3.26999400e-04,
    -5.35655500e-04,
    -5.17594800e-04,
    -2.45473400e-04,
    2.11398200e-04,
    3.85544900e-04,
    4.70161600e-04,
    3.19836200e-04,
    1.48115900e-04,
    -3.44263800e-04,
    -3.05706200e-04,
    -6.07866500e-04,
    -1.40639000e-04,
    3.54345200e-04,
    1.17906180e-03,
    1.27587890e-03,
    6.46781700e-04,
    2.97807400e-04,
    9.77706000e-05,
    -3.80139500e-04,
    5.85784900e-04,
    -4.34699000e-04,
    1.15040270e-03,
    2.93253490e-03,
    5.19921130e-03,
    3.26982700e-03,
    -1.15543100e-03,
    -3.31632400e-04,
    -9.65743000e-05,
    8.88011200e-04,
    4.55121760e-03,
    1.59257740e-03,
    4.02608170e-03,
    5.27395750e-03,
    -2.05009960e-03,
    -7.90681200e-04,
    1.29072190e-03,
    3.92764700e-04,
    -7.23810000e-05,
    1.10527329e-02,
    1.14289463e-02,
    1.01554380e-03,
    -4.10803130e-03,
    -1.39518580e-03,
    1.43680550e-03,
    -2.32410100e-04,
    3.02444440e-03,
    1.54672000e-04,
    -3.88632200e-04,
    4.87177720e-03,
    -1.17441400e-03,
    6.05647400e-04,
    -6.18932200e-04,
    -1.81334350e-03,
    5.15906690e-03,
    1.41277700e-04,
    6.31930020e-03,
    8.67670500e-04,
    1.30191470e-03,
    2.96133460e-03,
    3.64054300e-03,
    2.00721890e-03,
    5.74324870e-03,
    2.01317600e-04,
    5.60508670e-03,
    1.15676060e-03,
    8.20725550e-03,
    -9.88774500e-04,
    1.46054681e-02,
    4.93810300e-04,
    3.65006800e-04,
    6.47333900e-04,
    7.25182800e-04,
    1.71821900e-04,
    2.96466900e-04,
    -7.95212300e-04,
    1.80194150e-03,
    8.79835000e-04,
    1.17217338e-02,
]

solution_x_drcavty2 = [
    -8.30500000e-07,
    1.79025100e-04,
    4.69755400e-04,
    6.91706100e-04,
    7.63680500e-04,
    6.99211100e-04,
    5.59898000e-04,
    4.13496000e-04,
    2.89295400e-04,
    1.58674600e-04,
    1.44396300e-04,
    6.45348200e-04,
    1.19393250e-03,
    1.48581000e-03,
    1.49174680e-03,
    1.30666740e-03,
    1.04594130e-03,
    7.90114600e-04,
    5.54089300e-04,
    2.86541200e-04,
    4.86092200e-04,
    1.31996230e-03,
    1.90360630e-03,
    2.06459340e-03,
    1.95310610e-03,
    1.71807210e-03,
    1.41349750e-03,
    1.06405470e-03,
    6.90019300e-04,
    2.91278200e-04,
    1.00536400e-03,
    1.98412810e-03,
    2.39879170e-03,
    2.47713040e-03,
    2.38995890e-03,
    2.11023730e-03,
    1.58275540e-03,
    9.59023300e-04,
    3.65828300e-04,
    -2.52386000e-05,
    1.70639510e-03,
    2.47798200e-03,
    2.96272640e-03,
    3.23424450e-03,
    2.92194380e-03,
    1.83925430e-03,
    5.17510800e-04,
    -2.72294500e-04,
    -8.42981300e-04,
    -6.44882600e-04,
    3.47563010e-03,
    3.68554070e-03,
    4.85243520e-03,
    4.30556650e-03,
    2.59563830e-03,
    1.23414300e-04,
    -1.12148630e-03,
    -1.16433340e-03,
    -1.76218150e-03,
    -5.08449600e-04,
    1.02089465e-02,
    9.19876750e-03,
    7.38832940e-03,
    8.91347900e-04,
    -2.02918160e-03,
    -4.35306900e-04,
    1.74552680e-03,
    -3.82299000e-04,
    -1.89595900e-03,
    1.44318390e-03,
    8.39182720e-03,
    2.10036430e-03,
    -2.07708990e-03,
    -7.15986500e-04,
    -1.22269490e-03,
    2.85020860e-03,
    1.86361079e-02,
    -1.85665600e-04,
    5.77159200e-04,
    1.16139361e-02,
    7.12641800e-03,
    2.28174230e-03,
    -4.41730960e-03,
    1.19527564e-02,
    2.02136034e-02,
    1.78591365e-02,
    1.06707580e-01,
    3.00444810e-03,
    1.98001460e-02,
    1.37005246e-01,
    2.73846000e-03,
    7.47556450e-03,
    1.07964128e-02,
    1.81864591e-02,
    -1.19626975e-02,
    -5.17858661e-02,
    -2.97147410e-02,
    -5.84116800e-03,
    1.62672675e-01,
    9.02415668e-01,
]

solution_x_freurone = [
    12.26912153,
    -0.83186186,
    -1.50692279,
    -1.53467102,
    -1.53579843,
    -1.53584421,
    -1.53584607,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584617,
    -1.53584665,
    -1.53585846,
    -1.53614941,
    -1.54330584,
]


solution_x_integreq = [
    -0.0049257,
    -0.00980164,
    -0.01462709,
    -0.01940127,
    -0.02412341,
    -0.0287927,
    -0.03340834,
    -0.03796949,
    -0.04247531,
    -0.04692493,
    -0.05131748,
    -0.05565204,
    -0.0599277,
    -0.06414352,
    -0.06829854,
    -0.07239178,
    -0.07642224,
    -0.08038888,
    -0.08429068,
    -0.08812655,
    -0.09189541,
    -0.09559613,
    -0.09922759,
    -0.1027886,
    -0.10627797,
    -0.10969449,
    -0.11303691,
    -0.11630394,
    -0.11949428,
    -0.1226066,
    -0.12563951,
    -0.12859164,
    -0.13146152,
    -0.13424771,
    -0.1369487,
    -0.13956294,
    -0.14208886,
    -0.14452484,
    -0.14686924,
    -0.14912034,
    -0.15127642,
    -0.15333569,
    -0.15529633,
    -0.15715646,
    -0.15891416,
    -0.16056747,
    -0.16211435,
    -0.16355275,
    -0.16488053,
    -0.16609551,
    -0.16719544,
    -0.16817804,
    -0.16904092,
    -0.16978168,
    -0.17039781,
    -0.17088676,
    -0.17124589,
    -0.17147249,
    -0.1715638,
    -0.17151695,
    -0.171329,
    -0.17099693,
    -0.17051763,
    -0.16988789,
    -0.16910442,
    -0.16816384,
    -0.16706265,
    -0.16579725,
    -0.16436394,
    -0.16275891,
    -0.16097822,
    -0.15901783,
    -0.15687354,
    -0.15454105,
    -0.15201592,
    -0.14929355,
    -0.14636922,
    -0.14323804,
    -0.13989497,
    -0.13633479,
    -0.13255212,
    -0.1285414,
    -0.12429688,
    -0.11981262,
    -0.11508247,
    -0.11010007,
    -0.10485884,
    -0.09935198,
    -0.09357242,
    -0.08751287,
    -0.08116576,
    -0.07452324,
    -0.06757719,
    -0.06031917,
    -0.05274045,
    -0.04483194,
    -0.03658422,
    -0.02798751,
    -0.01903165,
    -0.01008278,
]

solution_x_msqrta = [
    8.0e-10,
    -5.0e-10,
    -2.0e-10,
    -1.0e-10,
    1.0e-10,
    -5.0e-10,
    -1.0e-10,
    -4.0e-10,
    -2.0e-10,
    2.0e-10,
    2.1e-09,
    -1.0e-09,
    -7.0e-10,
    -4.0e-10,
    7.0e-10,
    -1.4e-09,
    -3.0e-10,
    -8.0e-10,
    4.0e-10,
    6.0e-10,
    2.2e-09,
    -1.6e-09,
    8.0e-10,
    -5.0e-10,
    -1.2e-09,
    -5.0e-10,
    -7.0e-10,
    -1.7e-09,
    -2.1e-09,
    2.0e-10,
    -1.2e-09,
    -3.0e-10,
    1.4e-09,
    -1.0e-10,
    -1.7e-09,
    1.4e-09,
    1.0e-10,
    -9.0e-10,
    -2.1e-09,
    -5.0e-10,
    -2.7e-09,
    6.0e-10,
    -1.0e-09,
    2.0e-10,
    1.2e-09,
    -4.0e-10,
    2.1e-09,
    8.0e-10,
    6.0e-10,
    4.0e-10,
    -2.9e-09,
    1.0e-09,
    -4.0e-10,
    3.0e-10,
    5.0e-10,
    6.0e-10,
    1.5e-09,
    9.0e-10,
    4.0e-10,
    -1.0e-10,
    2.9e-09,
    -1.4e-09,
    5.0e-10,
    -4.0e-10,
    -7.0e-10,
    -7.0e-10,
    -1.2e-09,
    -1.4e-09,
    -1.2e-09,
    2.0e-10,
    -1.3e-09,
    9.0e-10,
    -4.0e-10,
    3.0e-10,
    6.0e-10,
    2.0e-10,
    5.0e-10,
    1.1e-09,
    1.0e-09,
    -1.0e-10,
    2.5e-09,
    -1.1e-09,
    1.3e-09,
    -3.0e-10,
    -1.6e-09,
    3.0e-10,
    -1.6e-09,
    -1.4e-09,
    -1.7e-09,
    -3.0e-10,
    -3.1e-09,
    7.0e-10,
    4.0e-10,
    3.0e-10,
    -6.0e-10,
    1.4e-09,
    1.3e-09,
    2.0e-10,
    -8.0e-10,
    -5.0e-10,
]


solution_x_msqrtb = [
    1.2e-09,
    -3.0e-10,
    2.0e-10,
    -4.0e-10,
    -3.0e-10,
    6.0e-10,
    -2.0e-10,
    -8.0e-10,
    -6.0e-10,
    -0.0e00,
    8.0e-10,
    -1.6e-09,
    -1.3e-09,
    -5.0e-10,
    -0.0e00,
    -2.0e-09,
    3.0e-10,
    -9.0e-10,
    -1.0e-10,
    1.0e-09,
    2.3e-09,
    -0.0e00,
    1.3e-09,
    -6.0e-10,
    -1.3e-09,
    1.1e-09,
    -1.0e-09,
    -1.2e-09,
    -1.9e-09,
    0.0e00,
    -1.1e-09,
    1.9e-09,
    1.9e-09,
    7.0e-10,
    -5.0e-10,
    1.0e-09,
    -8.0e-10,
    1.3e-09,
    -4.0e-10,
    -6.0e-10,
    -3.1e-09,
    -1.0e-10,
    -2.2e-09,
    7.0e-10,
    2.2e-09,
    -7.0e-10,
    1.8e-09,
    1.3e-09,
    3.1e-09,
    -4.0e-10,
    -1.6e-09,
    1.7e-09,
    1.3e-09,
    8.0e-10,
    -0.0e00,
    1.0e-09,
    -3.0e-10,
    1.5e-09,
    3.0e-10,
    -7.0e-10,
    4.2e-09,
    -1.0e-09,
    5.0e-10,
    -1.4e-09,
    -9.0e-10,
    2.7e-09,
    -7.0e-10,
    -2.7e-09,
    -2.0e-09,
    -3.0e-10,
    -8.0e-10,
    5.0e-10,
    -4.0e-10,
    2.0e-10,
    1.2e-09,
    2.2e-09,
    7.0e-10,
    3.0e-10,
    1.3e-09,
    -1.1e-09,
    2.7e-09,
    4.0e-10,
    2.6e-09,
    -5.0e-10,
    -2.8e-09,
    -6.0e-10,
    -2.2e-09,
    -9.0e-10,
    -3.6e-09,
    8.0e-10,
    -2.4e-09,
    3.1e-09,
    2.9e-09,
    1.4e-09,
    -8.0e-10,
    1.1e-09,
    -1.1e-09,
    2.5e-09,
    -4.0e-10,
    -9.0e-10,
]

solution_x_penalty2 = [
    1.00248452e-01,
    -1.60000000e-09,
    -1.40000000e-09,
    -1.50000000e-09,
    -1.00000000e-09,
    -9.00000000e-10,
    -8.00000000e-10,
    -6.00000000e-10,
    -1.20000000e-09,
    -8.00000000e-10,
    -7.00000000e-10,
    -6.00000000e-10,
    -4.00000000e-10,
    -4.00000000e-10,
    -2.00000000e-10,
    -2.00000000e-10,
    -3.00000000e-10,
    -3.00000000e-10,
    -1.00000000e-10,
    -0.00000000e00,
    0.00000000e00,
    1.00000000e-10,
    1.00000000e-10,
    0.00000000e00,
    0.00000000e00,
    1.00000000e-10,
    1.00000000e-10,
    1.00000000e-10,
    -1.00000000e-10,
    -4.00000000e-10,
    -4.00000000e-10,
    -1.00000000e-09,
    -7.00000000e-10,
    -6.00000000e-10,
    -8.00000000e-10,
    -8.00000000e-10,
    -1.20000000e-09,
    -9.00000000e-10,
    -6.00000000e-10,
    -4.00000000e-10,
    -1.00000000e-10,
    2.00000000e-10,
    5.00000000e-10,
    1.00000000e-09,
    5.40000000e-09,
    2.20000000e-09,
    7.20000000e-09,
    8.50000000e-09,
    9.60000000e-09,
    1.00000000e-08,
    1.14000000e-08,
    5.80000000e-09,
    1.62000000e-08,
    1.82000000e-08,
    2.15000000e-08,
    2.35000000e-08,
    2.57000000e-08,
    2.90000000e-08,
    3.19000000e-08,
    3.09000000e-08,
    4.16000000e-08,
    4.66000000e-08,
    5.31000000e-08,
    6.20000000e-08,
    6.85000000e-08,
    7.99000000e-08,
    9.37000000e-08,
    1.08300000e-07,
    1.24300000e-07,
    1.46700000e-07,
    1.64900000e-07,
    1.87800000e-07,
    2.08200000e-07,
    2.40000000e-07,
    2.78900000e-07,
    3.24200000e-07,
    3.77700000e-07,
    4.22400000e-07,
    4.89200000e-07,
    5.68200000e-07,
    6.64900000e-07,
    7.72800000e-07,
    8.93500000e-07,
    1.05180000e-06,
    1.24180000e-06,
    1.46270000e-06,
    1.73540000e-06,
    2.06050000e-06,
    2.45940000e-06,
    2.96690000e-06,
    3.61230000e-06,
    4.44100000e-06,
    5.50230000e-06,
    6.96620000e-06,
    8.98120000e-06,
    1.18847000e-05,
    1.64570000e-05,
    2.42465000e-05,
    4.02062000e-05,
    4.21655000e-05,
]


solution_x_watson = [
    -0.00000000e00,
    1.00000000e00,
    -9.80000000e-09,
    3.33333416e-01,
    3.52440000e-06,
    1.33262416e-01,
    5.10786700e-04,
    5.27159393e-02,
    -4.88557280e-03,
    7.04208489e-02,
    -1.76310368e-01,
    3.59652497e-01,
    -3.34930081e-01,
    -1.06954704e-01,
    7.10806973e-01,
    -7.44769987e-01,
    1.39770112e-01,
    2.23466491e-01,
    -2.30955205e-02,
    -4.13852010e-03,
    -2.40655722e-01,
    2.11825083e-01,
    -7.89005230e-02,
    2.52472539e-02,
    5.33065585e-02,
    1.99652115e-01,
    -5.38278039e-01,
    2.27228847e-01,
    2.79127878e-01,
    -2.82103374e-01,
    7.20994063e-02,
]


solution_x_yatpsq_1 = [7.06817436] * 100 + [4011.71977601] * 10 + [-4045.83698215] * 10

solution_x_yatpsq_2 = [0.0500104219] * 100 + [31.74567612] * 10 + [-32.22096803] * 10

solution_x_arglble = [
    1.47780425,
    1.95560851,
    3.92938553,
    2.91121702,
    8.01627094,
    6.85877107,
    15.21786782,
    4.82243403,
    6.00860058,
    15.03254187,
    23.28986922,
    12.71754213,
    -26.58525249,
    29.43573564,
    15.37597436,
    8.64486806,
    -29.5319652,
    11.01720116,
    -42.30702246,
    29.06508375,
    -6.28404714,
    45.57973843,
    -20.97480585,
    24.43508427,
    -31.03091276,
    -54.17050498,
    -28.61801666,
    57.87147127,
    51.17010631,
    29.75194872,
    -20.71188224,
    16.28973612,
    150.20741347,
    -60.0639304,
    -83.05993998,
    21.03440232,
    91.61798856,
    -85.61404492,
    18.37256156,
    57.1301675,
    37.08398118,
    -13.56809428,
    121.49145482,
    90.15947687,
    12.46203868,
    -42.9496117,
    -29.66369732,
    47.87016853,
    142.29920239,
    -63.06182553,
    3.443138,
    -109.34100996,
    -194.46227627,
    -58.23603332,
    74.84482245,
    114.74294254,
    -32.27461444,
    101.34021261,
    48.55025892,
    58.50389743,
    66.92509996,
    -42.42376449,
    -96.64369487,
    31.57947224,
    -83.36761354,
    299.41482695,
    -67.29433552,
    -121.1278608,
    -35.15915764,
    -167.11987996,
    -88.92036545,
    41.06880464,
    242.22658666,
    182.23597711,
    72.43687845,
    -172.22808985,
    156.51103949,
    35.74512311,
    -129.889251,
    113.260335,
    61.76069205,
    73.16796235,
    -226.09539309,
    -28.13618857,
    -140.01767085,
    241.98290964,
    -188.31315279,
    179.31895373,
    -102.19596978,
    23.92407737,
    -20.6453831,
    -86.8992234,
    -260.60293104,
    -60.32739463,
    -187.46576175,
    94.74033707,
    -26.40609501,
    283.59840478,
    -81.00626161,
    -6.23327543,
]


solution_x_morebvne = [
    -0.00480171244711894,
    -0.009553656441466189,
    -0.014255063178091082,
    -0.018905148312847776,
    -0.02350311156607665,
    -0.028048136314090373,
    -0.03253938916802462,
    -0.03697601953959521,
    -0.04135715919328434,
    -0.04568192178445848,
    -0.04994940238289979,
    -0.054158676981210775,
    -0.058308801987528955,
    -0.06239881370196436,
    -0.06642772777614718,
    -0.07039453865524663,
    -0.07429821900179416,
    -0.07813771910061508,
    -0.08191196624414197,
    -0.0856198640973513,
    -0.08926029204153081,
    -0.09283210449604987,
    -0.09633413021726782,
    -0.09976517157367622,
    -0.10312400379632972,
    -0.10640937420357734,
    -0.10962000139906009,
    -0.11275457444189355,
    -0.11581175198790325,
    -0.11879016140072804,
    -0.12168839783155079,
    -0.12450502326615709,
    -0.12723856553796054,
    -0.12988751730556838,
    -0.13245033499339215,
    -0.13492543769373558,
    -0.13731120602871583,
    -0.13960598097029298,
    -0.14180806261659795,
    -0.1439157089226587,
    -0.14592713438352983,
    -0.14784050866773057,
    -0.1496539551987893,
    -0.15136554968258126,
    -0.1529733185780274,
    -0.15447523750859646,
    -0.15586922961191965,
    -0.15715316382468714,
    -0.15832485309984537,
    -0.1593820525529578,
    -0.16032245753442315,
    -0.1611437016240685,
    -0.16184335454444554,
    -0.16241891998896002,
    -0.16286783336074998,
    -0.16318745941800578,
    -0.16337508982118337,
    -0.16342794057730933,
    -0.1633431493763049,
    -0.16311777281396858,
    -0.16274878349595126,
    -0.16223306701673054,
    -0.1615674188072447,
    -0.16074854084447576,
    -0.15977303821587704,
    -0.15863741553111918,
    -0.15733807317317933,
    -0.15587130338031904,
    -0.15423328614998352,
    -0.15242008495510762,
    -0.15042764226272823,
    -0.14825177484417715,
    -0.145888168865457,
    -0.14333237474568616,
    -0.14057980177072898,
    -0.13762571244830293,
    -0.13446521658997057,
    -0.13109326510447508,
    -0.12750464348585883,
    -0.12369396497871003,
    -0.11965566340170586,
    -0.11538398560935402,
    -0.11087298357047293,
    -0.10611650604048345,
    -0.1011081898030038,
    -0.09584145045453528,
    -0.09030947270418657,
    -0.08450520015839938,
    -0.07842132455849246,
    -0.07205027443652223,
    -0.06538420315244962,
    -0.058414976272886505,
    -0.05113415824875308,
    -0.04353299834598826,
    -0.03560241577999431,
    -0.027332984000740478,
    -0.018714914071368417,
    -0.009738037078703543,
    -0.00039178550924665913,
    0.009334826481040273,
]

solution_x_oscigrne = [
    -0.999903551150572,
    1.000114247321587,
    0.9998642692032601,
    1.0001606899137698,
    0.9998089690015318,
    1.00022598434592,
    0.999731099815014,
    1.0003177559633685,
    0.9996214137443762,
    1.000446688078944,
    0.9994668378325214,
    1.0006277224039304,
    0.999248858409288,
    1.000881705203141,
    0.9989411926230166,
    1.0012376192059906,
    0.998506407461758,
    1.0017355588402022,
    0.9978909678746266,
    1.0024305943941474,
    0.9970179188459485,
    1.0033975851656824,
    0.9957759910544586,
    1.0047367560724156,
    0.9940033482196368,
    1.0065792892561265,
    0.991463576275275,
    1.0090910440981682,
    0.9878113950447263,
    1.0124704597271386,
    0.9825476906558169,
    1.016933474160337,
    0.97497223558056,
    1.0226745117037057,
    0.9641665140655163,
    1.0297915464178238,
    0.949087499582646,
    1.0381739331985491,
    0.9289092394703534,
    1.0473881814350956,
    0.9037032879438561,
    1.0566576332592028,
    0.8751912025461293,
    1.0650493325263044,
    0.8467346025073008,
    1.071836505625808,
    0.8219637719978666,
    1.0767742560792348,
    0.8030052123113574,
    1.0800668731949534,
    0.7899438210393462,
    1.0821271313195384,
    0.781606186399683,
    1.0833627968298138,
    0.7765461227304188,
    1.0840845613674477,
    0.7735701654509769,
    1.0844995317623756,
    0.7718524469999704,
    1.0847359223290596,
    0.7708717537908896,
    1.084869871714956,
    0.7703153481248035,
    1.0849455446733298,
    0.770000790359205,
    1.084988222260974,
    0.7698233176390366,
    1.0850122686562047,
    0.7697233024137095,
    1.0850258120070317,
    0.7696669756983013,
    1.0850334441314584,
    0.7696352680150108,
    1.0850377667549869,
    0.7696174327289053,
    1.0850402906360344,
    0.769607435938671,
    1.0850420199210213,
    0.7696019492096186,
    1.0850440403901882,
    0.7695993333871978,
    1.085048727912368,
    0.7695994609478275,
    1.0850634025349974,
    0.769604955537232,
    1.0851124682328854,
    0.7696264320469576,
    1.0852784437585632,
    0.7697008512420949,
    1.0858411700848751,
    0.7699541437010324,
    1.0877518530579322,
    0.7708145141755096,
    1.0942649130343773,
    0.7737450447320231,
    1.1167550385396223,
    0.7838340360674543,
    1.197692695917145,
    0.8197587877734469,
    0.34400894026358253,
]

solution_x_spmsqrt = [
    0.8414709848078964,
    -0.7568024953079281,
    0.41211848524175654,
    -0.2879033166650651,
    -0.13235175009777303,
    -0.991778853443116,
    -0.9537526527594719,
    0.9200260381967906,
    -0.6298879942744537,
    -0.5063656411097588,
    0.9988152247235795,
    -0.4910215938984694,
    -0.6019998676776046,
    0.9395300555699313,
    -0.9300948780045254,
    -0.9992080341070627,
    -0.026521020285755953,
    -0.40406521945636065,
    0.2793865543595699,
    -0.8509193596391765,
    0.923470012926003,
    0.1935029667421232,
    0.9364725475338365,
    -0.8859527784925296,
    0.176016272833866,
    -0.5291338443628917,
    0.14993681711330134,
    -0.9851359060614224,
    -0.8115681644677004,
    0.9978032744219705,
    -0.32153677367579575,
    -0.15853338004399595,
    0.9055399984980432,
    -0.10589758762554138,
    -0.21933702833760824,
    0.9956757929363228,
    -0.6701396839379524,
    -0.9055272090161384,
    0.45213333953209767,
    -0.8012247906768953,
    -0.24539810131000517,
    -0.9999908622413068,
    0.9851203677373821,
    0.7025150575473956,
    0.97049168633502,
    -0.9905826083622151,
    -0.4442747122315391,
    -0.9365254011824229,
    0.7333337958292518,
    -0.6501275235748957,
    -0.236456371968843,
    0.7902854647755708,
    0.4042582281073567,
    0.5663064119145462,
    0.346394965535536,
    0.6369471771360007,
    0.5590140193623636,
    0.6017832141649304,
    0.11508425966985522,
    -0.2620839590180966,
    0.9766556656643753,
    -0.9660321335212897,
    -0.9201559227267819,
    -0.5946419876082146,
    0.4278557468834321,
    0.9835224135737828,
    0.3296208750563675,
    -0.4117614029834671,
    -0.9965019983464922,
    -0.7736233386803075,
    0.9509241545016164,
    0.3635926207547267,
    0.7570979728966365,
    -0.20259269090077123,
    0.9997657290235363,
    0.9835006076878136,
    -0.7274941973722288,
    0.9535984876805766,
    0.9745271031531163,
    -0.5444763096196569,
    0.9767074399435044,
    0.8369692092360629,
    0.49052257006311906,
    -0.017099129324754637,
    -0.6155654443683672,
    0.6372259975359011,
    -0.7853724073864938,
    0.025888206258587287,
    -0.8648845336882347,
    0.8272184413093554,
    -0.23598771211618058,
    0.5221681399851388,
    -0.1941831363419754,
    0.963594168151382,
    0.714349265095485,
    -0.9904998832255181,
    0.06994035488705914,
    -0.1506818641899401,
    -0.6954738915705097,
    -0.3056143888882522,
]


solution_x_semicon2 = [
    -6.349124087002204e-18,
    -2.720672286121828e-18,
    -2.0494594670589808e-18,
    2.2161886201338454e-18,
    1.2316133510181184e-18,
    -3.2705004691851026e-18,
    -3.807751676148402e-18,
    3.4276320135398005e-18,
    2.2326202305653343e-18,
    -7.019626524587994e-18,
    -5.858568597552852e-18,
    -9.211616078255237e-18,
    -7.71770563976305e-18,
    -7.876848461287058e-18,
    -1.6000476267872514e-18,
    -1.734489564328033e-18,
    1.7775061650713925e-19,
    -5.073461303752388e-18,
    -7.328302750507605e-18,
    -3.1238828312289746e-18,
    5.980268474857619e-19,
    -1.6975325642760375e-18,
    4.937452418560979e-18,
    1.3511147727766607e-17,
    5.749609112964981e-17,
    1.801648085519756e-16,
    5.786887187854167e-16,
    1.878507150702686e-15,
    6.136128040698149e-15,
    2.0030487111145893e-14,
    6.534292422120726e-14,
    2.1313997714716273e-13,
    6.952358252095008e-13,
    2.2677726547784577e-12,
    7.39723184163866e-12,
    2.412907024826009e-11,
    7.870673668386567e-11,
    2.567338541527492e-10,
    8.374412732822885e-10,
    2.731653274793591e-09,
    8.910391477241349e-09,
    2.906484367660483e-08,
    9.480673285799459e-08,
    3.092504307341268e-07,
    1.008744689447865e-06,
    3.2904220583605265e-06,
    1.073297189268266e-05,
    3.500918292749676e-05,
    0.00011418868707255388,
    0.00037238836057455765,
    0.001213800218967423,
    0.003949812221062431,
    0.012784144445260801,
    0.040678780990535776,
    0.12303542035650665,
    0.32818258157808705,
    0.7151930120566522,
    1.2976206270033757,
    2.0761013706015463,
    3.050641312078585,
    4.221240463432072,
    5.587898824666944,
    7.1506163957832,
    8.90939317678084,
    10.864229167659865,
    13.015124368420274,
    15.362078779062067,
    17.905092399585243,
    20.644165229989806,
    23.57929727027575,
    26.71048852044308,
    30.037738980491792,
    33.56104865042189,
    37.28041753023337,
    41.19584561992624,
    45.30733291950049,
    49.61487942895613,
    54.118485148293146,
    58.81815007751155,
    63.71387421661134,
    68.80565756559251,
    74.09350012445506,
    79.577401893199,
    85.25736287182431,
    91.13338306033101,
    97.2054624587191,
    103.47360106698856,
    109.93779888513943,
    116.59805591317166,
    123.45437215108528,
    130.5067475988803,
    135.59853094786146,
    138.72972219802878,
    139.9003970309516,
    139.9942331482723,
    139.99967247706724,
    139.9999814190815,
    139.99999894593952,
    139.99999994020595,
    139.9999999966189,
]

solution_x_qr3d = [
    0.8944271909999159,
    0.39036002917941326,
    0.18505699313910443,
    0.095507370926703,
    0.0652019862276467,
    -0.4472135954999579,
    0.7807200583588265,
    0.3701139862782089,
    0.19101474185340606,
    0.13040397245529342,
    0,
    -0.4879500364742666,
    0.7402279725564178,
    0.3820294837068121,
    0.26080794491058684,
    0,
    0,
    -0.5299359348983446,
    0.7003873867958222,
    0.47814789900274257,
    0,
    0,
    0,
    -0.5638286020497468,
    0.8258918255501917,
    0.447213595499958,
    -0.35777087639996635,
    0.08944271909999157,
    0,
    0,
    0.8197560612767679,
    -0.7416840554408852,
    0.1951800145897066,
    0,
    1.132212330751066,
    -1.1439886848599183,
    0.31796156093900735,
    1.4188709070303882,
    -6.058518452574962,
    7.972029516100272,
]


solution_x_qr3dbd = [
    0.8944271909999159,
    0.3903600291794133,
    0.1850569931391044,
    0.09550737092670301,
    0.06520198622764671,
    -0.4472135954999579,
    0.7807200583588265,
    0.3701139862782088,
    0.19101474185340603,
    0.1304039724552934,
    0,
    -0.48795003647426655,
    0.7402279725564178,
    0.38202948370681217,
    0.26080794491058684,
    0,
    0,
    -0.5299359348983446,
    0.7003873867958221,
    0.4781478990027425,
    0,
    0,
    0,
    -0.5638286020497468,
    0.8258918255501917,
    0.447213595499958,
    -0.3577708763999664,
    0.08944271909999163,
    0.819756061276768,
    -0.7416840554408851,
    0.19518001458970657,
    1.1322123307510663,
    -1.1439886848599186,
    0.3179615609390067,
    1.4188709070303884,
    -6.058518452574962,
    7.972029516100272,
]


solution_x_eigenb = [
    2.1880343004416827,
    2.433369037415534,
    2.4128467655642227,
    2.4143168471412966,
    2.4142066676219205,
    2.414206682597956,
    2.4143168514760567,
    2.4128467763690584,
    2.433369040912926,
    2.1880343004587717,
    0.963618832502572,
    -0.19358332415060756,
    -0.020571030789440843,
    -0.0042978748491313435,
    -0.0011189543814576976,
    -0.0003256602342870186,
    -0.00010143977363046447,
    -3.309302901033653e-05,
    -1.1092507161708885e-05,
    -4.10236648587512e-06,
    -9.758954956834986e-05,
    0.9208420627147608,
    -0.19113674746578155,
    -0.019750356974082468,
    -0.004087291838641473,
    -0.0010576174015411869,
    -0.00030654777315460436,
    -9.524470464093356e-05,
    -3.082138926676854e-05,
    -1.1092450516891782e-05,
    -2.1507171255294457e-05,
    3.236210199985498e-05,
    0.9240994322190695,
    -0.19135572211934757,
    -0.019818514674832036,
    -0.004104765871306482,
    -0.0010626629889412406,
    -0.0003082323428087962,
    -9.524506106163913e-05,
    -3.30929923080829e-05,
    -5.86920342958478e-06,
    8.343154992559033e-06,
    -1.7090950629164782e-06,
    0.9238630466026214,
    -0.19134060715511186,
    -0.019813760575551793,
    -0.004103453696629484,
    -0.0010626616637818649,
    -0.0003065476688168684,
    -0.0001014390007803748,
    -1.7629336348662287e-06,
    2.414698832828228e-06,
    -4.1012794468935585e-07,
    1.7073395190219743e-07,
    0.9238806428059877,
    -0.19134186729883576,
    -0.019813762593004474,
    -0.004104766449316578,
    -0.001057618179089577,
    -0.00032566103285368623,
    -5.617848676115081e-07,
    7.498487330424776e-07,
    -1.1366246620226694e-07,
    4.233248357697225e-08,
    -1.8184553525805125e-08,
    0.9238806405230912,
    -0.19134060742073378,
    -0.019818514337183922,
    -0.004087292358165013,
    -0.001118954298914211,
    -1.8565094044360727e-07,
    2.461558460853512e-07,
    -2.9698560165568933e-08,
    2.5691549193155624e-08,
    4.188439547971367e-08,
    1.7091126942261566e-07,
    0.9238630462038411,
    -0.19135572096178424,
    -0.019750357283584624,
    -0.004297874115889032,
    -6.452367583021426e-08,
    7.874956290802419e-08,
    -2.175478191008932e-08,
    -2.9611984365300305e-08,
    -1.125976562800656e-07,
    -4.101727958716196e-07,
    -1.7068431405865914e-06,
    0.9240994307748498,
    -0.19113674668992328,
    -0.020571030343105964,
    -1.2720088883591045e-08,
    5.7045828447717434e-08,
    7.924431455583827e-08,
    2.4584027025267894e-07,
    7.494896735776724e-07,
    2.41326518080458e-06,
    8.34418638100877e-06,
    3.236121345356557e-05,
    0.9208420621116469,
    -0.19358332383274754,
    -9.410619198170363e-09,
    -1.324699653531768e-08,
    -6.476508265084364e-08,
    -1.8558263456164044e-07,
    -5.618026820278632e-07,
    -1.7630007182357942e-06,
    -5.868582984643352e-06,
    -2.1506490807524302e-05,
    -9.758904294524118e-05,
    0.9636188326077207,
]


solution_x_luksan12 = [
    -2.6260067987163516,
    6.896065319147662,
    1.527892692614362,
    1.5497440503992346,
    2.2933547979396463,
    1.6306208450523716,
    1.5819176730889628,
    2.32812470715209,
    1.6291935486837537,
    1.5831335363873629,
    2.3316182375355337,
    1.6290901468343557,
    1.5831778731783577,
    2.331746894500055,
    1.6290866339965575,
    1.5831791924745884,
    2.3317508225088055,
    1.6290863289641941,
    1.5831794312419505,
    2.3317513366797806,
    1.6290862573533014,
    1.5831793883878833,
    2.3317513422592953,
    1.629086261632123,
    1.5831793726762426,
    2.3317512664646154,
    1.6290862864991247,
    1.5831794068996972,
    2.331751192772075,
    1.629086292750654,
    1.5831793960520542,
    2.331751254872966,
    1.6290863143562366,
    1.5831794727104964,
    2.331751220286205,
    1.6290863278083665,
    1.5831794601591884,
    2.3317513895695736,
    1.6290862215089341,
    1.583179580563746,
    2.3317517718530487,
    1.629086273471926,
    1.5831794517394973,
    2.3317514742466026,
    1.629086326246853,
    1.5831793647918584,
    2.3317510961040324,
    1.6290862893642488,
    1.583179357650647,
    2.3317509213902934,
    1.629086379389047,
    1.5831793466774813,
    2.331751016773598,
    1.6290862487394557,
    1.5831795060981861,
    2.331751350596469,
    1.629086344775534,
    1.583179237092884,
    2.331750704884059,
    1.6290861760511046,
    1.583180031048655,
    2.3317524781223806,
    1.6290862227638085,
    1.583179320520067,
    2.3317511883963484,
    1.62908615550431,
    1.5831794393651841,
    2.33175167156419,
    1.6290861496589093,
    1.583179491480598,
    2.3317514438297446,
    1.6290867286777644,
    1.5831787218028068,
    2.331749775035503,
    1.629086341296889,
    1.5831793785655173,
    2.331751379257171,
    1.6290862955034657,
    1.5831794896813611,
    2.331751501905547,
    1.6290857775369425,
    1.583179220493569,
    2.331751008922027,
    1.6290872482674832,
    1.5831776309272405,
    2.331747114910226,
    1.6291104318632341,
    1.5831327312118264,
    2.3316489547138564,
    1.629748307506696,
    1.5818970811223332,
    2.3289347056739356,
    1.6468612471308757,
    1.5492871064990756,
    2.2555720960782186,
    -1.2840500262457888,
    2.551876632332705,
    0.9810837136085581,
]

solution_x_luksan13 = [
    1.3086052241942845,
    1.6216137111993358,
    2.2886144747619968,
    1.34232518154836,
    1.6619515759503343,
    2.3751337553163605,
    1.343855296385397,
    1.6630483896975998,
    2.377485954810846,
    1.3438990216611229,
    1.6630791019372382,
    2.3775518416632186,
    1.3439002468996393,
    1.6630799506592178,
    2.3775536420160925,
    1.3439002707105718,
    1.6630799637087308,
    2.377553663793456,
    1.343900274442869,
    1.6630799703922405,
    2.377553677510941,
    1.343900273735809,
    1.663079968764641,
    2.3775536820493994,
    1.3439002754790357,
    1.6630799760584611,
    2.3775537070702106,
    1.3439002833097773,
    1.6630799799354958,
    2.377553710130499,
    1.343900279016161,
    1.6630799791188497,
    2.377553709151317,
    1.3439002771699002,
    1.6630799775994893,
    2.3775537046342916,
    1.3439002881835356,
    1.6630799956367304,
    2.377553757775651,
    1.3439002886986675,
    1.663079991183404,
    2.377553741510781,
    1.3439002855650906,
    1.6630799931177906,
    2.3775537587908215,
    1.3439002747672082,
    1.6630799713724573,
    2.377553681610918,
    1.3439002854677142,
    1.6630799870142512,
    2.377553735404843,
    1.3439002800895123,
    1.6630799851565248,
    2.3775537353686165,
    1.3439002784725658,
    1.663079982758718,
    2.377553724300102,
    1.3439002839408525,
    1.663079984436528,
    2.377553726731763,
    1.3439002742018746,
    1.6630799686940994,
    2.377553673295027,
    1.343900285055898,
    1.663079986851812,
    2.3775537287002404,
    1.3439002797888209,
    1.6630799728327539,
    2.377553682977633,
    1.3439002761687657,
    1.6630799673959926,
    2.3775536708099745,
    1.3439002798986437,
    1.6630799763732456,
    2.3775537001976423,
    1.3439002780805065,
    1.6630799801142993,
    2.3775537125819097,
    1.3439002916123894,
    1.6630800028052062,
    2.3775537813761924,
    1.3439003043419797,
    1.6630800585945529,
    2.3775539515552806,
    1.3439012394354732,
    1.6630827445655045,
    2.3775621184439015,
    1.3439344570959444,
    1.6631783889211809,
    2.37785290158486,
    1.3451213454947373,
    1.666597519847485,
    2.38828410134482,
    1.3886122940189465,
    1.7939836213544151,
    2.885811280705054,
    4.723092522164855,
    5.389923139518635,
]

solution_x_luksan14 = [
    -0.692015438465357,
    0.4736192819939741,
    1.2412079054782652,
    -0.4783782506182583,
    0.23882842116612296,
    1.094079486551747,
    -0.2641170714882838,
    0.08143516139066553,
    1.0679283525636625,
    -0.1494417718117012,
    0.033925047353142675,
    1.0809287152115696,
    -0.13657207013818243,
    0.02960520381904979,
    1.0827798919700173,
    -0.1362649003784561,
    0.029450043690314048,
    1.0828484754308758,
    -0.13625660575465595,
    0.029445194398704543,
    1.0828506279698982,
    -0.13625635947650283,
    0.029445045430998882,
    1.082850692128227,
    -0.13625634849900412,
    0.029445040594224484,
    1.0828506890538272,
    -0.13625634095713987,
    0.029445038872369666,
    1.0828506892763552,
    -0.1362563401019553,
    0.029445038661562405,
    1.0828506865668028,
    -0.1362563352007162,
    0.029445038297226032,
    1.0828506897645718,
    -0.1362563399662581,
    0.029445038930055314,
    1.0828506855121347,
    -0.13625633491870875,
    0.029445037517807717,
    1.0828506970059786,
    -0.13625635176495168,
    0.029445039882406293,
    1.0828506971123373,
    -0.13625635316124712,
    0.029445040472916795,
    1.0828506840532126,
    -0.13625633315729457,
    0.029445037711023858,
    1.0828506931348947,
    -0.13625634492000505,
    0.029445039530404147,
    1.082850698256033,
    -0.1362563552104686,
    0.029445040416000965,
    1.0828506949203678,
    -0.13625634915587218,
    0.02944504055043263,
    1.0828506838415628,
    -0.13625633328649378,
    0.029445037473531936,
    1.0828506996527274,
    -0.1362563547825522,
    0.029445040998217188,
    1.0828506836226952,
    -0.13625633328336106,
    0.029445037480123,
    1.0828506844275172,
    -0.13625633160755746,
    0.029445037363800216,
    1.0828506910381746,
    -0.13625634142989135,
    0.02944503900512941,
    1.0828506970833573,
    -0.13625635262199126,
    0.029445040274926663,
    1.0828506858747455,
    -0.13625633628442352,
    0.029445037869996665,
    1.0828506843744086,
    -0.13625633006537194,
    0.029445038276720837,
    1.082850772934407,
    -0.13625645049722282,
    0.029445066824156065,
    1.0828531756767654,
    -0.13625960631817075,
    0.029445905929758334,
    1.082930785041219,
    -0.13636157993795717,
    0.029472948037119334,
    1.0853841906187118,
    -0.13957177103726676,
    0.03033477911863786,
    0.9997884952922044,
    0.03012334667228848,
    0.03001755818766747,
]


solution_x_luksan15 = [
    -2.8128120376543353,
    1.2947106505184869,
    -2.81440005752448,
    0.5152023646747894,
    -2.0089506145318516,
    1.0515415361025104,
    -3.4386919581246973,
    0.5351040934480955,
    -1.6697901903837928,
    1.1273918094584376,
    -3.0058465049859246,
    0.5987020506877548,
    -2.1273657980099876,
    0.9192000873982219,
    -3.4535594653315687,
    0.5623710862956811,
    -1.9386092963220432,
    0.9821839081455959,
    -3.5093911936842477,
    0.5501687807342822,
    -2.1778372107639,
    0.9063872377904626,
    -3.424790911399904,
    0.5665652947300606,
    -2.279179522364104,
    0.8685030719682756,
    -3.145617106466628,
    0.6099286765580767,
    -1.9284952155742618,
    0.9691898205061202,
    -3.016417857801295,
    0.6212423706928658,
    -1.9216427599092167,
    0.9730403697373208,
    -3.285655214260605,
    0.5820210274753043,
    -2.1458132202272386,
    0.9058785725661358,
    -2.958575186535638,
    0.6348101011039402,
    -1.9602973787877798,
    0.952916227779147,
    -3.2178165098362,
    0.5944426358988449,
    -1.72911123316871,
    1.0594397548002128,
    -3.1978737322906863,
    0.5844520814478443,
    -1.7679802490196521,
    1.0524244273162242,
    -3.146733476316577,
    0.5902404606592925,
    -2.128977747267094,
    0.914704760106901,
    -3.2345320433357836,
    0.5920343641247361,
    -2.2610082037405834,
    0.8670334110534724,
    -3.4060367708426935,
    0.576246511012118,
    -2.355877477880136,
    0.8412164300993267,
    -3.081758650591148,
    0.6241626014976729,
    -1.6525049806156873,
    1.081267498611019,
    -2.963721310470893,
    0.6194531536445858,
    -1.506685332675987,
    1.1746494074889837,
    -2.9257857154408438,
    0.6141113169233168,
    -2.1021910871806577,
    0.9219352856117872,
    -3.3401105799593105,
    0.5774971684592886,
    -1.7547078621368493,
    1.0532221932141288,
    -3.3352090067149023,
    0.5658942245431572,
    -2.353376910071777,
    0.854027290091373,
    -3.2416948400866707,
    0.5965555845011213,
    -1.9356880509818681,
    0.9699374813403614,
    -3.292568634235838,
    0.580974296002741,
    -1.6861988438404256,
    1.0857879072493273,
    -3.1295933491980374,
    0.5904364080984017,
    -1.9200181272009214,
    0.9895807082139652,
    -3.2489040016986324,
    0.582149002451738,
    -0.5601900925614515,
    2.4870578822567175,
    -3.5588729390698672,
    -0.4666353059476335,
]

solution_x_luksan16 = [
    10.23347504140921,
    20.59876890029168,
    -35.3558957906405,
    14.159724748767388,
    -6.192790944245422,
    6.904260897808911,
    -12.426375656338006,
    7.916405145745939,
    9.883332945956017,
    -7.763552251913891,
    9.829215962712826,
    -5.460412966230336,
    -2.0247375676154316,
    2.292011784517189,
    -8.854644657757955,
    6.501718109308879,
    -10.00818055426963,
    6.969493641831156,
    5.794385035460475,
    -4.827934342605883,
    -0.3589290720778674,
    1.7351238328795198,
    -20.46786870905146,
    15.073627999686598,
    -22.38633414526394,
    14.87045990266218,
    -9.108444080860473,
    5.493242761973028,
    -4.436790715837233,
    3.35863879246109,
    -12.778497913669044,
    9.51430783434134,
    17.974890839174535,
    -14.543141451864756,
    -0.878903552061843,
    3.937581796604095,
    -37.95441551229194,
    27.217302740157244,
    1.9384003997032608,
    -5.0732916754344854,
    -9.834240436833431,
    9.928282181783743,
    24.901369369122225,
    -20.6810518922668,
    5.609710258509786,
    0.40845702633880704,
    7.517485147363342,
    -6.744213821972444,
    6.283831933809041,
    -2.719582209866709,
    1.1727545558714283,
    -0.5901766790421645,
    -10.83801088444533,
    8.63096398021895,
    4.262706411344401,
    -4.302452961158821,
    -1.1028476495022923,
    2.413241731236865,
    -4.9611988982883455,
    3.2905463368286787,
    -11.011449594878954,
    8.354169868673353,
    17.534906240840428,
    -14.074846099952074,
    1.5433478836011245,
    1.9967416934301132,
    -5.0398763810168425,
    2.8962555845029465,
    -2.503087792307807,
    2.18971326359533,
    -5.425121985743007,
    4.100312921945939,
    -7.90674647105912,
    5.736740005101778,
    -10.955014606035165,
    7.825133686094932,
    13.983943593069979,
    -11.161214770039159,
    4.399884927901469,
    -0.7147360927864606,
    16.135713247282684,
    -12.343832004737795,
    -6.450092935863761,
    7.476113508806722,
    -22.97438911325564,
    15.605814430808019,
    24.62552916846743,
    -20.02790069712666,
    32.528132112236335,
    -20.037974911481278,
    -1.5215783873530375,
    3.528694334566693,
    7.356145013974544,
    -6.400505214585759,
    -0.5849689371220534,
    2.3004991729965987,
    -7.749666453881994,
    5.308798604551641,
    -12.801456405933285,
    9.38466573197068,
]

solution_x_luksan17 = [
    -0.8437781517444214,
    5.211601301412691,
    -0.8255686986099557,
    -1.1224808830025863,
    -0.9115016416553087,
    -1.0449996789508758,
    -0.8370067076638087,
    -1.1138351780732039,
    -0.8829775814636974,
    -1.0662539739396077,
    -1.0011441498985942,
    -0.9971962219303508,
    -0.9921060603499269,
    -1.0107562326286939,
    -0.9812584988122448,
    -1.0142511875210711,
    -0.9884985874862595,
    -1.0098353738374723,
    -0.9872157109718606,
    -1.0112133037704427,
    -0.9864952703702744,
    -1.0113724963264314,
    -0.9870842717327896,
    -1.011035340568662,
    -0.9869391605540793,
    -1.0111661235119436,
    -0.9868977564988685,
    -1.0111670258624144,
    -0.9869449215955661,
    -1.0111419398459263,
    -0.9869301696083717,
    -1.0111537677001405,
    -0.9869283405594838,
    -1.0111528331131907,
    -0.986932002795563,
    -1.0111510466790732,
    -0.9869305922818844,
    -1.0111520809826104,
    -0.9869305749770071,
    -1.011151923374323,
    -0.9869308492975264,
    -1.0111518042563021,
    -0.9869307203053385,
    -1.0111518914759727,
    -0.9869307309805547,
    -1.0111518713966636,
    -0.9869307504505271,
    -1.011151864732727,
    -0.9869307391816526,
    -1.0111518711394447,
    -0.986930743051711,
    -1.0111518679059937,
    -0.9869307405857761,
    -1.011151871272733,
    -0.9869307406810994,
    -1.0111518690271912,
    -0.9869307442357795,
    -1.0111518679286133,
    -0.9869307403926516,
    -1.0111518706700777,
    -0.9869307426486206,
    -1.011151868044014,
    -0.9869307395007838,
    -1.0111518726198288,
    -0.9869307376627915,
    -1.0111518716425005,
    -0.9869307459735903,
    -1.0111518650959377,
    -0.9869307441838526,
    -1.011151868558413,
    -0.9869307370598098,
    -1.011151873493444,
    -0.9869307388844694,
    -1.011151870953494,
    -0.9869307414667764,
    -1.0111518699114064,
    -0.9869307413827505,
    -1.0111518693727195,
    -0.9869307428780796,
    -1.0111518683864686,
    -0.986930742414142,
    -1.0111518688900905,
    -0.9869307423425782,
    -1.0111518687612688,
    -0.9869307388492422,
    -1.0111518725840953,
    -0.9869307379963231,
    -1.0111518716476386,
    -0.9869307390986436,
    -1.0111518722894763,
    -0.9869307349748873,
    -1.011151874864258,
    -0.986930740183915,
    -1.0111518696551045,
    -0.9869307419219207,
    -1.011151869862698,
    -0.9869307393375487,
    -1.0111518711716971,
    -0.98693074172968,
    -1.0111518690786507,
]

solution_x_luksan21 = [
    -6.072657993990953,
    -11.151678306331249,
    -15.281680185264895,
    -18.553572785910436,
    -21.08878192552139,
    -23.018420220928494,
    -24.46705052798465,
    -25.543512880064885,
    -26.337637149910563,
    -26.920674289315414,
    -27.347575624380077,
    -27.65988732407303,
    -27.88857759276986,
    -28.05649817926434,
    -28.18038720009252,
    -28.272439283160423,
    -28.341505869674798,
    -28.39399571405599,
    -28.434542523545463,
    -28.466491445203996,
    -28.492253969229775,
    -28.51356599746666,
    -28.53167622506297,
    -28.54748213206502,
    -28.561629796383773,
    -28.574583635599762,
    -28.58667947877867,
    -28.59815838663125,
    -28.609194634285124,
    -28.61991238997938,
    -28.630402327046635,
    -28.640728722950577,
    -28.650938105649047,
    -28.661061977374047,
    -28.671123050731868,
    -28.68113826085573,
    -28.69112022311497,
    -28.701078262087883,
    -28.71101815506324,
    -28.72094477397387,
    -28.730862903992644,
    -28.740776507637953,
    -28.75068864243215,
    -28.760599670061964,
    -28.77051000987731,
    -28.780418360465934,
    -28.790323772732222,
    -28.80022409430249,
    -28.810119012418326,
    -28.82001016509087,
    -28.829900708980315,
    -28.839793590776342,
    -28.849689118959287,
    -28.859586440413942,
    -28.86948336412596,
    -28.879378174667828,
    -28.88926990461213,
    -28.8991581591508,
    -28.909043599093003,
    -28.918925344790882,
    -28.92880159055073,
    -28.938669244304233,
    -28.94852535832674,
    -28.95836634260835,
    -28.96818485329044,
    -28.97796943084888,
    -28.98770372929353,
    -28.997369201223027,
    -29.006941800115108,
    -29.016389375603524,
    -29.025665682937056,
    -29.03470507627482,
    -29.04341491397447,
    -29.051663075623473,
    -29.059265584091445,
    -29.065967527124695,
    -29.07141754001239,
    -29.07512741421742,
    -29.07641564980266,
    -29.074334399240477,
    -29.067568644255143,
    -29.054295689963208,
    -29.0319862391214,
    -28.997130480357225,
    -28.944864342494807,
    -28.86845848063996,
    -28.758620076438536,
    -28.602552441434927,
    -28.38270523359173,
    -28.075144666563702,
    -27.647479062397462,
    -27.056331868057303,
    -26.24447489795629,
    -25.137976027248023,
    -23.64410279764132,
    -21.651318090974016,
    -19.03331370504794,
    -15.65922569153753,
    -11.41085934966044,
    -6.2035977373817195,
]


solution_x_luksan22 = [
    0.960000892597201,
    0.9188402401000288,
    0.8346999741225976,
    0.6824138623323005,
    0.44429684165068956,
    0.16679643843737746,
    -0.023004295169824662,
    0.015327433907712216,
    -0.009931088912543472,
    0.006684223516831371,
    -0.004411575260245885,
    0.0029571424982889867,
    -0.001966221180203038,
    0.001316579307305194,
    -0.0008787716047566526,
    0.0005864520202429219,
    -0.00039319743428295955,
    0.000262756179445633,
    -0.0001760817809970661,
    0.00011752859355313591,
    -7.938290213698181e-05,
    5.4582276287576205e-05,
    -3.676956244840184e-05,
    2.2892784894905886e-05,
    -1.4844959572369287e-05,
    8.978467841981641e-06,
    -7.419590428825042e-06,
    5.0134136153247085e-06,
    -4.8296331309899045e-06,
    4.4148810133672814e-06,
    -5.451937781671956e-06,
    5.28902886393822e-06,
    -5.486579973708986e-06,
    4.589797533333856e-06,
    -3.686156736885388e-06,
    1.3813327052080713e-06,
    -2.875286216624316e-07,
    -1.966618190117126e-07,
    -5.984127105225435e-08,
    -7.936593253722943e-07,
    2.1703743349261694e-07,
    -8.894851154156264e-07,
    3.573122556113253e-07,
    8.966129390508038e-08,
    -1.3677877479650928e-06,
    6.43379960693636e-07,
    -6.357331910140075e-07,
    -1.7198971213860295e-07,
    7.720236116499906e-07,
    -1.5538185090359862e-06,
    8.378707374504217e-07,
    -2.73224811890867e-07,
    1.0658698182255397e-07,
    -7.054426255706567e-07,
    5.652296338005435e-08,
    2.3167064968509738e-07,
    -7.1526763080569e-07,
    1.1597873712533682e-08,
    -2.863395818025362e-07,
    -8.633020826805924e-07,
    4.2697411766659695e-07,
    -1.1541837705801389e-06,
    4.7551953352402166e-07,
    3.5199581610952056e-07,
    -2.5742042022428385e-06,
    2.499335480925905e-06,
    -4.079663388392796e-06,
    3.723178696281719e-06,
    -2.919751243874922e-06,
    1.4829783071179273e-06,
    -1.1191190311723908e-06,
    5.588506976522976e-07,
    -4.825768343696162e-07,
    9.863313432267673e-07,
    -1.3577296762331155e-06,
    1.0976659429101464e-06,
    -1.3266509165639996e-06,
    8.804177825565544e-07,
    -1.59655791383051e-06,
    2.2532672723180185e-07,
    -1.5600238860251166e-06,
    1.7351280778268136e-06,
    -1.618517725020847e-06,
    1.287719767348254e-06,
    -1.2077855625401572e-06,
    -1.226034002734674e-06,
    1.2900045894163097e-06,
    -1.883605391514802e-06,
    -1.279989121101194e-06,
    1.7054561610327153e-06,
    -4.501630088393006e-06,
    3.38756209930697e-06,
    -5.221884694101911e-06,
    6.8241273128299335e-06,
    -7.215782067624106e-06,
    4.201540035034105e-06,
    -3.7683204911021953e-06,
    2.681881990832518e-07,
    7.907138005741486e-07,
    3.0687261099791043,
]


solution_x_hydcar20 = [
    2.6759127057952494e-07,
    0.0020075973345538593,
    0.9979921350741772,
    1.1074382518158015e-06,
    0.003926319294260064,
    0.9960725732674882,
    4.473878508666662e-06,
    0.007432799593801248,
    0.9925627265276902,
    1.791855353550396e-05,
    0.013805819144958344,
    0.9861762623015062,
    7.125759668355649e-05,
    0.025273323829558065,
    0.9746554185737586,
    0.0002803534439901862,
    0.04553620774789967,
    0.9541834388081104,
    0.001082782951168712,
    0.08018717893154169,
    0.9187300381172898,
    0.0040498954829321235,
    0.13606692468745604,
    0.859883179829612,
    0.014346338806928278,
    0.21713770052872539,
    0.7685159606643465,
    0.04646668013286012,
    0.31340489138177297,
    0.640128428485367,
    0.05166573387892863,
    0.42190983200015447,
    0.526424434120917,
    0.05782376960889372,
    0.5508311256800345,
    0.39134510471107226,
    0.0640697514942442,
    0.6757845188863061,
    0.26014572961945,
    0.07002653957502589,
    0.7737845233677303,
    0.15618893705724401,
    0.07687228280882921,
    0.836427017007432,
    0.08670070018373897,
    0.08838584185008973,
    0.8661393596329539,
    0.04547479851695662,
    0.11319983773418042,
    0.8640793740698467,
    0.022720788195972993,
    0.16949917403437423,
    0.8198575022045703,
    0.010643323761055593,
    0.2889793132934084,
    0.7066247193552677,
    0.004395967351324038,
    0.4999998216058201,
    0.4986616017769641,
    0.0013385766172157153,
    138.21584043438,
    138.13772749931923,
    137.99511335767673,
    137.73620245773773,
    137.27057972483576,
    136.44580451915726,
    135.01759125081972,
    132.61817941469988,
    128.72481163454674,
    122.65561152670969,
    119.34453517365907,
    115.7706674624033,
    112.60284049804272,
    110.23710719573587,
    108.6126130079421,
    107.35503374607997,
    105.88999629459819,
    103.3849389558914,
    98.83590533616024,
    92.07928040428027,
    290.69252437645963,
    290.65561780338373,
    290.59025193849226,
    290.4776378323827,
    290.2922675253222,
    290.0081841660331,
    289.615549643397,
    289.14599494303985,
    288.7724567817456,
    277.8716922412405,
    280.6638306878377,
    284.24018735028017,
    287.7729960147694,
    290.52761603695006,
    292.2929262974615,
    293.2381872872007,
    293.71591921265576,
    294.59496871170194,
    298.03881543410006,
]

solution_x_hydcar6 = [
    0.005639001019299217,
    0.13023671278902904,
    0.8641242861916716,
    0.020668138451243257,
    0.22352172363375894,
    0.7558101379149978,
    0.06627260583120169,
    0.3309386117749258,
    0.6027887823938726,
    0.11820814762835632,
    0.4418686949212894,
    0.43992315745035443,
    0.24917284487890107,
    0.500513105207918,
    0.2503140499131811,
    0.49624066598713407,
    0.4131755248073142,
    0.09058380920555179,
    132.63808144202352,
    127.84951651511513,
    120.39649297520073,
    113.59593052273564,
    104.00971299398577,
    93.2001658166645,
    288.8739911256674,
    288.5263717741471,
    275.6104698527219,
    280.00166220797604,
    289.1545657231987,
]

solution_x_methane = [
    107.7653795798063,
    0.09225723236421929,
    0.9077427676357807,
    102.68498908598441,
    0.1821995285312063,
    0.8178004714687936,
    97.71772476055263,
    0.28421889854366617,
    0.7157811014563338,
    96.57726115013135,
    0.30530731675985745,
    0.6946926832401424,
    94.26309926727772,
    0.3566490103973165,
    0.6433509896026832,
    89.98899748591788,
    0.46779111102816195,
    0.5322088889718375,
    83.97342066984532,
    0.6573895966863115,
    0.34261040331368847,
    78.32157508655418,
    0.8759450903481356,
    0.12405490965186436,
    886.7137742582912,
    910.3656929177117,
    922.1591291059583,
    926.0766775727482,
    935.1735260591255,
    952.4236258294623,
    975.0192103423753,
]


CARTIS_ROBERTS_PROBLEMS = {
    "arglale": {
        # arglale is the same as linear_full_rank with specific settings
        "fun": partial(linear_full_rank, dim_out=400),
        "start_x": [1] * 100,
        "solution_x": [-0.99999952] * 100,
        "start_criterion": 700,
        "solution_criterion": 300,
    },
    "arglble": {
        # arglble is the same as linear_rank_one with specific settings
        "fun": partial(linear_rank_one, dim_out=400),
        "start_x": [1] * 100,
        "solution_x": solution_x_arglble,
        "start_criterion": 5.460944e14,
        "solution_criterion": 99.62547,
    },
    "argtrig": {
        "fun": argtrig,
        "start_x": [1 / 100] * 100,
        "solution_x": [0] * 100,
        "start_criterion": 32.99641,
        "solution_criterion": 0,
    },
    "artif": {
        "fun": artif,
        "start_x": [1] * 100,
        "solution_x": None,
        "start_criterion": 36.59115,
        "solution_criterion": 0,
    },
    "arwhdne": {
        "fun": arwhdne,
        "start_x": [1] * 100,
        "solution_x": [0.706011] * 99 + [0],
        "start_criterion": 495,
        "solution_criterion": 27.66203,
    },
    "bdvalues": {
        "fun": bdvalues,
        "start_x": get_start_points_bdvalues(100, 1000),
        "solution_x": solution_x_bdvalues,
        "start_criterion": 1.943417e7,
        "solution_criterion": 0,
    },
    "bratu_2d": {
        "fun": partial(bratu_2d, alpha=4),
        "start_x": [0] * 64,
        "solution_x": solution_x_bratu_2d,
        "start_criterion": 0.1560738,
        "solution_criterion": 0,
    },
    "bratu_2d_t": {
        "fun": partial(bratu_2d, alpha=6.80812),
        "start_x": [0] * 64,
        "solution_x": solution_x_bratu_2d_t,
        "start_criterion": 0.4521311,
        "solution_criterion": 1.8534736e-05,
    },
    "bratu_3d": {
        "fun": partial(bratu_3d, alpha=6.80812),
        "start_x": [0] * 27,
        "solution_x": solution_x_bratu_3d,
        "start_criterion": 4.888529,
        "solution_criterion": 0,
    },
    "brownale": {
        "fun": brown_almost_linear,
        "start_x": [0.5] * 100,
        "solution_x": [1] * 100,
        "start_criterion": 2.524757e5,
        "solution_criterion": 0,
    },
    "broydn_3d": {
        "fun": broydn_3d,
        "start_x": [-1] * 100,
        "solution_x": solution_x_broydn_3d,
        "start_criterion": 111,
        "solution_criterion": 0,
    },
    "cbratu_2d": {
        "fun": cbratu_2d,
        "start_x": [0] * (2 * 5 * 5),
        "solution_x": solution_x_cbratu_2d,
        "start_criterion": 0.4822531,
        "solution_criterion": 0,
    },
    "broydn_bd": {
        "fun": broydn_bd,
        "start_x": [1] * 100,
        "solution_x": solution_x_broydn_bd,
        "start_criterion": 2404,
        "solution_criterion": 0,
    },
    "chandheq": {
        "fun": chandheq,
        "start_x": (np.arange(1, 101) / 100).tolist(),
        "solution_x": None,
        "start_criterion": 6.923365,
        "solution_criterion": 0,
    },
    "chemrcta": {
        "fun": chemrcta,
        "start_x": [1] * 100,
        "solution_x": None,
        "start_criterion": 3.0935,
        "solution_criterion": 0,
        "bounds": Bounds(lower=np.concatenate([np.zeros(50), 1e-6 * np.ones(50)])),
    },
    "chemrctb": {
        "fun": chemrctb,
        "start_x": [1] * 100,
        "solution_x": solution_x_chemrctb,
        "start_criterion": 1.446513,
        "solution_criterion": 1.404424e-3,
        "bounds": Bounds(lower=1e-6 * np.ones(100)),
    },
    "chnrsbne": {
        "fun": chnrsbne,
        "start_x": [-1] * 50,
        "solution_x": [1] * 50,
        "start_criterion": 7635.84,
        "solution_criterion": 0,
    },
    "drcavty1": {
        "fun": partial(drcavty, r=500),
        "start_x": [0] * 100,
        "solution_x": None,
        "start_criterion": 0.4513889,
        "solution_criterion": 0,
    },
    "drcavty2": {
        "fun": partial(drcavty, r=1000),
        "start_x": [0] * 100,
        "solution_x": solution_x_drcavty2,
        "start_criterion": 0.4513889,
        "solution_criterion": 3.988378e-4,
    },
    "drcavty3": {
        "fun": partial(drcavty, r=4500),
        "start_x": [0] * 100,
        "solution_x": solution_x_drcavty3,
        "start_criterion": 0.4513889,
        "solution_criterion": 0,
    },
    "eigena": {
        "fun": partial(eigen, param=np.diag(np.arange(1, 11))),
        "start_x": [1] * 10 + np.eye(10).flatten().tolist(),
        "solution_x": [*np.arange(1, 11).tolist(), 1] + ([0] * 10 + [1]) * 9,
        "start_criterion": 285,
        "solution_criterion": 0,
        "bounds": Bounds(lower=np.zeros(110)),
    },
    "eigenb": {
        "fun": partial(
            eigen, param=np.diag(2 * np.ones(10)) + np.diag(-np.ones(9), k=1)
        ),
        "start_x": [1] * 10 + np.eye(10).flatten().tolist(),
        "solution_x": solution_x_eigenb,
        "start_criterion": 19,
        "solution_criterion": 1.55654284,
        # we suspect a typo in Cartis and Roberts (2019);
        # according to table 3 in their paper, the minimum is at 0.
    },
    "flosp2hh": {
        "fun": partial(
            flosp2,
            a=np.array([1, 0, -1], dtype=np.int64),
            b=np.array([1, 0, -1], dtype=np.int64),
            ra=1e7,
        ),
        "start_x": [0] * 59,
        "solution_x": None,  # multiple argmins
        "start_criterion": 519,
        "solution_criterion": 1 / 3,
    },
    "flosp2hl": {
        "fun": partial(
            flosp2,
            a=np.array([1, 0, -1], dtype=np.float64),
            b=np.array([1, 0, -1], dtype=np.float64),
            ra=1e3,
        ),
        "start_x": [0] * 59,
        "solution_x": None,  # multiple argmins
        "start_criterion": 519,
        "solution_criterion": 1 / 3,
    },
    "flosp2hm": {
        "fun": partial(
            flosp2,
            a=np.array([1, 0, -1], dtype=np.float64),
            b=np.array([1, 0, -1], dtype=np.float64),
            ra=1e5,
        ),
        "start_x": [0] * 59,
        "solution_x": None,  # multiple argmins
        "start_criterion": 519,
        "solution_criterion": 1 / 3,
    },
    "flosp2th": {
        "fun": partial(
            flosp2,
            a=np.array([0, 1, 0], dtype=np.float64),
            b=np.array([0, 1, 1], dtype=np.float64),
            ra=1e7,
        ),
        "start_x": [0] * 59,
        "solution_x": None,  # multiple argmins
        "start_criterion": 516,
        "solution_criterion": 0,
    },
    "flosp2tl": {
        "fun": partial(
            flosp2,
            a=np.array([0, 1, 0], dtype=np.float64),
            b=np.array([0, 1, 1], dtype=np.float64),
            ra=1e3,
        ),
        "start_x": [0] * 59,
        "solution_x": None,  # multiple argmins
        "start_criterion": 516,
        "solution_criterion": 0,
    },
    "flosp2tm": {
        "fun": partial(
            flosp2,
            a=np.array([0, 1, 0], dtype=np.float64),
            b=np.array([0, 1, 1], dtype=np.float64),
            ra=1e5,
        ),
        "start_x": [0] * 59,
        "solution_x": None,  # multiple argmins
        "start_criterion": 516,
        "solution_criterion": 0,
    },
    "freurone": {
        "fun": freurone,
        "start_x": [0.5, -2] + [0] * 98,
        "solution_x": solution_x_freurone,
        "start_criterion": 9.95565e4,
        "solution_criterion": 1.196458e4,
    },
    "hatfldg": {
        "fun": hatfldg,
        "start_x": [1] * 25,
        "solution_x": [0] * 11 + [-1, 1] + [0] * 12,
        "start_criterion": 27,
        "solution_criterion": 0,
    },
    "hydcar20": {
        "fun": partial(hydcar, n=20, m=3, k=9),
        "start_x": get_start_points_hydcar20(),
        "solution_x": solution_x_hydcar20,
        "start_criterion": 1341.663,
        "solution_criterion": 0,
    },
    "hydcar6": {
        "fun": partial(hydcar, n=6, m=3, k=2),
        "start_x": get_start_points_hydcar6(),
        "solution_x": solution_x_hydcar6,
        "start_criterion": 704.1073,
        "solution_criterion": 0,
    },
    "integreq": {
        "fun": integreq,
        "start_x": (np.arange(1, 101) / 101 * (np.arange(1, 101) / 101 - 1)).tolist(),
        "solution_x": solution_x_integreq,
        "start_criterion": 0.5730503,
        "solution_criterion": 0,
    },
    "luksan11": {
        "fun": luksan11,
        "start_x": [-0.8] * 100,
        "solution_x": [1] * 100,
        "start_criterion": 626.0640,
        "solution_criterion": 0,
    },
    "luksan12": {
        "fun": luksan12,
        "start_x": [-1] * 98,
        "solution_x": None,
        "start_criterion": 3.2160e4,
        "solution_criterion": None,
        # we found a lower minimum than Cartis and Roberts (2019) at 1651.837;
        # according to table 3 in their paper, the minimum is at 4292.197.
        # We suspect, however, that the true optimum is even lower.
        # That is why we disable this test function for the time being.
    },
    "luksan13": {
        "fun": luksan13,
        "start_x": [-1] * 98,
        "solution_x": solution_x_luksan13,
        "start_criterion": 6.4352e4,
        "solution_criterion": 24949.67040503685711883,
        # we found a lower minimum than Cartis and Roberts (2019);
        # according to table 3 in their paper, the minimum is at 25188.86
    },
    "luksan14": {
        "fun": luksan14,
        "start_x": [-1] * 98,
        "solution_x": solution_x_luksan14,
        "start_criterion": 2.6880e4,
        "solution_criterion": 123.9235,
    },
    "luksan15": {
        "fun": luksan15,
        "start_x": [-0.8, 1.2, -1.2, 0.8] * 25,
        "solution_x": solution_x_luksan15,
        "start_criterion": 2.701585e4,
        "solution_criterion": 3.569697,
    },
    "luksan16": {
        "fun": luksan16,
        "start_x": [-0.8, 1.2, -1.2, 0.8] * 25,
        "solution_x": solution_x_luksan16,
        "start_criterion": 1.306848e4,
        "solution_criterion": 3.569697,
    },
    "luksan17": {
        "fun": luksan17,
        "start_x": [-0.8, 1.2, -1.2, 0.8] * 25,
        "solution_x": None,  # multiple argmins
        "start_criterion": 1.687370e6,
        "solution_criterion": 0.4931613,
    },
    "luksan21": {
        "fun": luksan21,
        "start_x": [ih * (ih - 1) for ih in np.arange(1, 101) * (1 / 101)],
        "solution_x": solution_x_luksan21,
        "start_criterion": 99.98751,
        "solution_criterion": 0,
    },
    "luksan22": {
        "fun": luksan22,
        "start_x": [-1.2 if i % 2 == 0 else 1 for i in range(100)],
        "solution_x": solution_x_luksan22,
        "start_criterion": 2.487686e4,
        "solution_criterion": 872.9230,
    },
    "methanb8": {
        "fun": methane,
        "start_x": get_start_points_methanb8(),
        "solution_x": solution_x_methane,
        "start_criterion": 1.043105,
        "solution_criterion": 0,
    },
    "methanl8": {
        "fun": methane,
        "start_x": get_start_points_methanl8(),
        "solution_x": solution_x_methane,
        "start_criterion": 4345.100,
        "solution_criterion": 0,
    },
    "morebvne": {
        "fun": morebvne,
        "start_x": [t * (t - 1) for t in np.arange(1, 101) * (1 / 101)],
        "solution_x": solution_x_morebvne,
        "start_criterion": 3.633100e-4,
        "solution_criterion": 0,
    },
    "msqrta": {
        "fun": msqrta,
        "start_x": get_start_points_msqrta(10),
        "solution_x": solution_x_msqrta,
        "start_criterion": 212.7162,
        "solution_criterion": 0,
    },
    "msqrtb": {
        "fun": msqrta,
        "start_x": get_start_points_msqrta(10, flag=2),
        "solution_x": solution_x_msqrtb,
        "start_criterion": 205.0753,
        "solution_criterion": 0,
    },
    "oscigrne": {
        "fun": oscigrne,
        "start_x": [-2] + [1] * 99,
        "solution_x": solution_x_oscigrne,
        "start_criterion": 6.120720e8,
        "solution_criterion": 0,
    },
    "penalty_1": {
        "fun": penalty_1,
        "start_x": list(range(1, 101)),
        "solution_x": None,
        "start_criterion": 1.144806e11,
        "solution_criterion": 9.025000e-9,
    },
    "penalty_2": {
        "fun": penalty_2,
        "start_x": [0.5] * 100,
        "solution_x": solution_x_penalty2,
        "start_criterion": 1.591383e6,
        "solution_criterion": 0.9809377,
    },
    "powellse": {
        "fun": powell_singular,
        "start_x": [3.0, -1.0, 0.0, 1] * 25,
        "solution_x": [0] * 100,
        "start_criterion": 41875,
        "solution_criterion": 0,
    },
    "qr3d": {
        "fun": partial(qr3d, m=5),
        "start_x": get_start_points_qr3d(5),
        "solution_x": solution_x_qr3d,
        "start_criterion": 1.2,
        "solution_criterion": 0,
        "bounds": Bounds(
            lower=[-np.inf] * 25
            + [0 if i == j else -np.inf for i in range(5) for j in range(5)]
        ),
    },
    "qr3dbd": {
        "fun": partial(qr3dbd, m=5),
        "start_x": get_start_points_qr3dbd(5),
        "solution_x": solution_x_qr3dbd,
        "start_criterion": 1.2,
        "solution_criterion": 0,
        "bounds": Bounds(
            lower=[-np.inf] * 25
            + [0 if i == j else -np.inf for i in range(5) for j in range(5)]
        ),
    },
    "spmsqrt": {
        "fun": spmsqrt,
        "start_x": get_start_points_spmsqrt(34),
        "solution_x": solution_x_spmsqrt,
        "start_criterion": 74.33542,
        "solution_criterion": 0,
    },
    "semicn2u": {
        "fun": semicon2,
        "start_x": [0] * 100,
        "solution_x": solution_x_semicon2,
        "start_criterion": 2.025037e4,
        "solution_criterion": 0,
    },
    "semicon2": {
        "fun": semicon2,
        "start_x": [0] * 100,
        "solution_x": solution_x_semicon2,
        "start_criterion": 2.025037e4,
        "solution_criterion": 0,
        "bounds": Bounds(lower=-5 * np.ones(100), upper=0.2 * 700 * np.ones(100)),
    },
    "vardimne": {
        "fun": vardimne,
        "start_x": [1 - i / 100 for i in range(1, 101)],
        "solution_x": [1] * 100,
        "start_criterion": 1.310584e14,
        "solution_criterion": 0,
    },
    "watsonne": {
        "fun": watson,
        "start_x": [0] * 31,
        "solution_x": solution_x_watson,
        "start_criterion": 30,
        "solution_criterion": 0,
    },
    "yatpsq_1": {
        "fun": partial(yatpsq_1, dim_in=10),
        "start_x": [6] * 100 + [0] * 20,
        "solution_x": solution_x_yatpsq_1,
        "start_criterion": 2.073643e6,
        "solution_criterion": 0,
    },
    "yatpsq_2": {
        "fun": partial(yatpsq_2, dim_in=10),
        "start_x": [10] * 100 + [0] * 20,
        "solution_x": solution_x_yatpsq_2,
        "start_criterion": 1.831687e5,
        "solution_criterion": 0,
    },
}
