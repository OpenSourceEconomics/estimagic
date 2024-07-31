"""Test various solvers for quadratic trust-region subproblems."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimizers._pounders._conjugate_gradient import (
    minimize_trust_cg,
)
from optimagic.optimizers._pounders._steihaug_toint import (
    minimize_trust_stcg,
)
from optimagic.optimizers._pounders._trsbox import minimize_trust_trsbox
from optimagic.optimizers._pounders.bntr import (
    bntr,
)
from optimagic.optimizers._pounders.gqtpar import (
    gqtpar,
)
from optimagic.optimizers._pounders.pounders_auxiliary import MainModel

# ======================================================================================
# Subsolver BNTR
# ======================================================================================

TEST_CASES_BNTR = [
    (
        np.array([0.0002877431832243, 0.00763968126032, 0.01217268029151]),
        np.array(
            [
                [
                    4.0080360351800763e00,
                    1.6579091056425378e02,
                    1.7322297746691254e02,
                ],
                [
                    1.6579091056425378e02,
                    1.6088016292793940e04,
                    1.1041403355728811e04,
                ],
                [
                    1.7322297746691254e02,
                    1.1041403355728811e04,
                    9.2992625728417297e03,
                ],
            ]
        ),
        -np.ones(3),
        np.ones(3),
        np.array([0.000122403, 3.92712e-06, -8.2519e-06]),
    ),
    (
        np.array([7.898833044695e-06, 254.9676549378, 0.0002864050095122]),
        np.array(
            [
                [3.97435226e00, 1.29126446e02, 1.90424789e02],
                [1.29126446e02, 1.08362658e04, 9.05024598e03],
                [1.90424789e02, 9.05024598e03, 1.06395102e04],
            ]
        ),
        np.array([-1.0, 0, -1.0]),
        np.ones(3),
        np.array([-4.89762e-06, 0.0, 6.0738e-08]),
    ),
    (
        np.array([0.000208896, 0.040137, 0.0237668]),
        np.array(
            [
                [
                    8.6267971128257614e-01,
                    3.3589357331133463e01,
                    3.8550834275262481e01,
                ],
                [
                    3.3589357331133463e01,
                    4.0625660472990171e03,
                    2.7006581320776222e03,
                ],
                [
                    3.8550834275262481e01,
                    2.7006581320776222e03,
                    2.3157072223295277e03,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.000404701, -8.56315e-06, -7.01394e-06]),
    ),
    (
        np.array([1053.998577258, -1768.195151975, 1091.754813306]),
        np.array(
            [
                [
                    5.1009001863913858e02,
                    -2.9142602235646069e02,
                    2.4000221805201900e02,
                ],
                [
                    -2.9142602235646069e02,
                    1.3922341317778117e04,
                    5.7863734667132694e03,
                ],
                [
                    2.4000221805201900e02,
                    5.7863734667132694e03,
                    1.5911148658889811e03,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([-1, 0.52169, -1]),
    ),
    (
        np.array([-191889.2320478, -1002015.908232, -573072.9226335]),
        np.array(
            [
                [
                    1.1012704153339069e07,
                    4.9533363163771488e07,
                    2.9628266883962810e07,
                ],
                [
                    4.9533363163771488e07,
                    2.2267942225630835e08,
                    1.3303758212303287e08,
                ],
                [
                    2.9628266883962810e07,
                    1.3303758212303287e08,
                    7.9554367206848219e07,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([-1, 0.148669, 0.131015]),
    ),
    (
        np.array([1076.73, -4802.74, 828.249]),
        np.array(
            [
                [
                    4.8212187042743824e02,
                    -9.8489480047918653e02,
                    1.1822837156689332e03,
                ],
                [
                    -9.8489480047918653e02,
                    7.7891876734093257e03,
                    2.1566788126264223e03,
                ],
                [
                    1.1822837156689332e03,
                    2.1566788126264223e03,
                    1.9148005132287210e03,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([1.0, 1, -1]),
    ),
    (
        np.array([39307.4, 43176.2, 19136.1]),
        np.array(
            [
                [
                    2.1888915578112096e05,
                    1.9734665605071097e05,
                    1.0865582588513123e05,
                ],
                [
                    1.9734665605071097e05,
                    1.5802957082548781e05,
                    9.3932751210457645e04,
                ],
                [
                    1.0865582588513123e05,
                    9.3932751210457645e04,
                    6.9919507495186845e04,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.835475, -1, -0.228586]),
    ),
    (
        np.array([15924.6, -7936.89, 4559.77]),
        np.array(
            [
                [
                    1.4823363165787258e05,
                    -9.3991198881618606e04,
                    -6.7423849020288171e03,
                ],
                [
                    -9.3991198881618606e04,
                    1.0299013233992350e05,
                    2.7454282523562739e04,
                ],
                [
                    -6.7423849020288171e03,
                    2.7454282523562739e04,
                    -8.7825122820168282e04,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.15422, 0.484382, -1.0]),
    ),
    (
        np.array([-223.491, -2375.1, -3508.53]),
        np.array(
            [
                [
                    1.8762040451468388e03,
                    4.5209129063298806e03,
                    3.7587689627124179e04,
                ],
                [
                    4.5209129063298806e03,
                    2.6540113149319626e06,
                    1.3806874591227937e06,
                ],
                [
                    3.7587689627124179e04,
                    1.3806874591227937e06,
                    1.4430203128871324e06,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.700966, 0.0157984, -0.0309433]),
    ),
    (
        np.array([-0.00566046, -0.26497, -0.24923]),
        np.array(
            [
                [
                    9.0152048402068141e-01,
                    3.9069240493708740e01,
                    4.0976585309530130e01,
                ],
                [
                    3.9069240493708740e01,
                    4.0339538281863297e03,
                    2.7447144903267226e03,
                ],
                [
                    4.0976585309530130e01,
                    2.7447144903267226e03,
                    2.3178455554478642e03,
                ],
            ],
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.0141205, 0.000131845, -0.000298234]),
    ),
    (
        np.array([16459.6, 42312.7, 33953.9]),
        np.array(
            [
                [
                    3.4897766687256113e07,
                    1.7536007046689782e08,
                    1.0424382825704373e08,
                ],
                [
                    1.7536007046689782e08,
                    8.8481756045390594e08,
                    5.2619306030723321e08,
                ],
                [
                    1.0424382825704373e08,
                    5.2619306030723321e08,
                    3.1297679051347983e08,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([-0.131066, 0.180817, -0.260453]),
    ),
    (
        np.array([17660.3, 18827.2, 28759.5]),
        np.array(
            [
                [
                    9.7041306729050993e04,
                    1.0613110916937439e05,
                    1.5558443292460032e05,
                ],
                [
                    1.0613110916937439e05,
                    1.0840421118778562e05,
                    1.5388850550829183e05,
                ],
                [
                    1.5558443292460032e05,
                    1.5388850550829183e05,
                    2.1840298326937514e05,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([1, 0.266874, -1]),
    ),
    (
        np.array([16678, 65723.7, -153755]),
        np.array(
            [
                [
                    2.8786103367161286e04,
                    1.0278873046014908e05,
                    -2.4232333719251846e05,
                ],
                [
                    1.0278873046014908e05,
                    7.9423330424583505e05,
                    -4.3975347261092327e04,
                ],
                [
                    -2.4232333719251846e05,
                    -4.3975347261092327e04,
                    3.5707186446013493e06,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([1, -0.206169, 0.108385]),
    ),
    (
        np.array([26602.2, -118867, 7457.08]),
        np.array(
            [
                [
                    1.3510413991352668e05,
                    -4.4190620422288636e05,
                    1.6183211956800147e04,
                ],
                [
                    -4.4190620422288636e05,
                    6.7224673907168563e06,
                    1.5956835170839101e05,
                ],
                [
                    1.6183211956800147e04,
                    1.5956835170839101e05,
                    6.7613560286023448e03,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.0743402, 0.0463054, -1.0]),
    ),
    (
        np.array([-1726.71, -394.745, -340.876]),
        np.array(
            [
                [
                    3.2235026082366367e03,
                    3.5903801754879023e03,
                    1.4504956347170955e03,
                ],
                [
                    3.5903801754879023e03,
                    1.0326690788609463e04,
                    4.9152962632434155e03,
                ],
                [
                    1.4504956347170955e03,
                    4.9152962632434155e03,
                    2.7645273367617360e03,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([0.925468, -0.722815, 0.922884]),
    ),
    (
        np.array([-1460.95, -48078.5, -61349.4]),
        np.array(
            [
                [
                    -2.1558862194927831e04,
                    2.9346854336376925e05,
                    3.6945385626803833e05,
                ],
                [
                    2.9346854336376925e05,
                    7.6788393809145853e07,
                    5.7299202312126122e07,
                ],
                [
                    3.6945385626803833e05,
                    5.7299202312126122e07,
                    5.0198599698606022e07,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([1, 0.00933713, -0.0167956]),
    ),
    (
        np.array([-7292.55, -299376, -269052]),
        np.array(
            [
                [
                    3.6778621108518197e05,
                    1.5160538979173467e07,
                    1.3518289246498797e07,
                ],
                [
                    1.5160538979173467e07,
                    6.1341858259608674e08,
                    5.4813989289859617e08,
                ],
                [
                    1.3518289246498797e07,
                    5.4813989289859617e08,
                    4.9252782230468601e08,
                ],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.ones(3),
        np.array([-1, 0.0341927, -0.0100605]),
    ),
]


@pytest.mark.slow()
@pytest.mark.parametrize(
    "linear_terms, square_terms, lower_bounds, upper_bounds, x_expected",
    TEST_CASES_BNTR,
)
def test_bounded_newton_trustregion(
    linear_terms,
    square_terms,
    lower_bounds,
    upper_bounds,
    x_expected,
):
    main_model = MainModel(linear_terms=linear_terms, square_terms=square_terms)

    options = {
        "conjugate_gradient_method": "cg",
        "maxiter": 50,
        "maxiter_gradient_descent": 5,
        "gtol_abs": 1e-8,
        "gtol_rel": 1e-8,
        "gtol_scaled": 0,
        "gtol_abs_conjugate_gradient": 1e-8,
        "gtol_rel_conjugate_gradient": 1e-6,
    }

    result = bntr(
        main_model,
        lower_bounds,
        upper_bounds,
        x_candidate=np.zeros_like(x_expected),
        **options,
    )
    aaae(result["x"], x_expected, decimal=5)


# ======================================================================================
# Subsolver GQTPAR
# ======================================================================================

TEST_CASES_GQTPAR = [
    (
        np.array([-0.0005429824695352, -0.1032556117176, -0.06816855282091]),
        np.array(
            [
                [2.05714077e-02, 7.58182390e-01, 9.00050279e-01],
                [7.58182390e-01, 6.25867992e01, 4.20096648e01],
                [9.00050279e-01, 4.20096648e01, 4.03810858e01],
            ]
        ),
        np.array(
            [
                -0.9994584757179,
                -0.007713730538474,
                0.03198833730482,
            ]
        ),
        -0.001340933981148,
    )
]


@pytest.mark.slow()
@pytest.mark.parametrize(
    "linear_terms, square_terms, x_expected, criterion_expected", TEST_CASES_GQTPAR
)
def test_gqtpar_quadratic(linear_terms, square_terms, x_expected, criterion_expected):
    main_model = MainModel(linear_terms=linear_terms, square_terms=square_terms)

    result = gqtpar(main_model, x_candidate=np.zeros_like(x_expected))

    aaae(result["x"], x_expected)
    aaae(result["criterion"], criterion_expected)


# ======================================================================================
# Conjugate Gradient Algorithms
# ======================================================================================

TEST_CASES_CG = [
    (
        np.array([79579.8, 35973.7]),
        np.array(
            [
                [2.2267942225630835e08, 1.3303758212303287e08],
                [1.3303758212303287e08, 7.9554367206848219e07],
            ]
        ),
        0.2393319731158,
        -np.array([0.0958339, -0.159809]),
    ),
    (
        np.array([0.00028774, 0.00763968, 0.01217268]),
        np.array(
            [
                [4.00803604e00, 1.65790911e02, 1.73222977e02],
                [1.65790911e02, 1.60880163e04, 1.10414034e04],
                [1.73222977e02, 1.10414034e04, 9.29926257e03],
            ]
        ),
        9.5367431640625e-05,
        np.array([9.50204689e-05, 3.56030822e-06, -7.30627902e-06]),
    ),
    (
        np.array([0.00028774, 0.00763968, 0.01217268]),
        np.array(
            [
                [4.00803604e00, 1.65790911e02, 1.73222977e02],
                [1.65790911e02, 1.60880163e04, 1.10414034e04],
                [1.73222977e02, 1.10414034e04, 9.29926257e03],
            ]
        ),
        9.5367431640625e-05,
        np.array([9.50204689e-05, 3.56030822e-06, -7.30627902e-06]),
    ),
    (
        -np.array([-6.76002e-06, -6.56323e-08, 2.00988e-07]),
        np.array(
            [
                [
                    4.0080360351800763e00,
                    1.6579091056425378e02,
                    1.7322297746691254e02,
                ],
                [
                    1.6579091056425378e02,
                    1.6088016292793940e04,
                    1.1041403355728811e04,
                ],
                [
                    1.7322297746691254e02,
                    1.1041403355728811e04,
                    9.2992625728417297e03,
                ],
            ]
        ),
        0.0003814697265625,
        np.array([-2.7382e-05, -3.66814e-07, 9.45617e-07]),
    ),
    (
        -np.array([-4.69447, -0.619271, 0.837666]),
        np.array(
            [
                [
                    6.9147751896043360e01,
                    2.6192110911280561e03,
                    2.8094172839794960e03,
                ],
                [
                    2.6192110911280561e03,
                    2.4907533417816096e05,
                    1.6917615514201863e05,
                ],
                [
                    2.8094172839794960e03,
                    1.6917615514201863e05,
                    1.4352314212505225e05,
                ],
            ]
        ),
        0.0657627701334,
        np.array([-0.0656472, -0.00168561, 0.00351321]),
    ),
    (
        -np.array([-2.45646e-05, -4.1711e-07, 9.2032e-07]),
        np.array(
            [
                [
                    8.6267971128257614e-01,
                    3.3589357331133463e01,
                    3.8550834275262481e01,
                ],
                [
                    3.3589357331133463e01,
                    4.0625660472990171e03,
                    2.7006581320776222e03,
                ],
                [
                    3.8550834275262481e01,
                    2.7006581320776222e03,
                    2.3157072223295277e03,
                ],
            ]
        ),
        0.0003814697265625,
        np.array([-0.000310185, -3.86464e-06, 9.67128e-06]),
    ),
    (
        -np.array([-4.29172e-08, -1.8127e-06, -1.38313e-06]),
        np.array(
            [
                [
                    1.7207808265135328e06,
                    7.2130304472968280e07,
                    5.5202182930777229e07,
                ],
                [
                    7.2130304472968280e07,
                    3.0516230749633555e09,
                    2.3274035648401971e09,
                ],
                [
                    5.5202182930777229e07,
                    2.3274035648401971e09,
                    1.7782503817136776e09,
                ],
            ]
        ),
        0.390625,
        np.array([-7.44084e-15, -6.07092e-16, 2.47754e-16]),
    ),
    (
        -np.array([79525.7, 3.04463e06, 2.42641e06]),
        np.array(
            [
                [
                    6.3624954351893254e06,
                    2.5406887701711509e08,
                    1.9610463258207005e08,
                ],
                [
                    2.5406887701711509e08,
                    1.0261536342839724e10,
                    7.8819891642426796e09,
                ],
                [
                    1.9610463258207005e08,
                    7.8819891642426796e09,
                    6.0688426371444454e09,
                ],
            ]
        ),
        0.0001192842356654,
        np.array([2.43607e-06, 9.32646e-05, 7.4327e-05]),
    ),
]

TEST_CASES_TRSBOX = [
    (
        np.array([1.0, 0.0, 1.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        2.0,
        np.array([-1.0, 0.0, -0.5]),
    ),
    (
        np.array([1.0, 0.0, 1.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        5.0 / 12.0,
        np.array([-1.0 / 3.0, 0.0, -0.25]),
    ),
    (
        np.array([1.0, 0.0, 1.0]),
        np.array([[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        5.0 / 12.0,
        np.array([-1.0 / 3.0, 0.0, -0.25]),
    ),
    (
        np.array([0.0, 0.0, 1.0]),
        np.array([[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        0.5,
        np.array([0.0, 0.0, -0.5]),
    ),
]


@pytest.mark.slow()
@pytest.mark.parametrize(
    "gradient, hessian, trustregion_radius, x_expected", TEST_CASES_CG
)
def test_trustregion_conjugate_gradient(
    gradient, hessian, trustregion_radius, x_expected
):
    x_out = minimize_trust_cg(
        gradient, hessian, trustregion_radius, gtol_abs=1e-8, gtol_rel=1e-6
    )
    aaae(x_out, x_expected)


@pytest.mark.slow()
@pytest.mark.parametrize(
    "gradient, hessian, trustregion_radius, x_expected", TEST_CASES_CG
)
def test_trustregion_steihaug_toint(gradient, hessian, trustregion_radius, x_expected):
    x_out = minimize_trust_stcg(gradient, hessian, trustregion_radius)
    aaae(x_out, x_expected)


@pytest.mark.slow()
@pytest.mark.parametrize(
    "linear_terms, square_terms, trustregion_radius, x_expected",
    TEST_CASES_CG + TEST_CASES_TRSBOX,
)
def test_trustregion_trsbox(linear_terms, square_terms, trustregion_radius, x_expected):
    lower_bounds = -1e20 * np.ones_like(linear_terms)
    upper_bounds = 1e20 * np.ones_like(linear_terms)

    x_out = minimize_trust_trsbox(
        linear_terms,
        square_terms,
        trustregion_radius,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    aaae(x_out, x_expected, decimal=4)
