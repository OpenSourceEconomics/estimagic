import io
import textwrap
from collections import namedtuple

import numpy as np
import pandas as pd

ProcessedModel = namedtuple("ProcessedModel", "params info name")


def _get_test_inputs_multiindex():
    df = pd.DataFrame(
        data=np.ones((3, 4)), columns=["value", "ci_lower", "ci_upper", "p_value"]
    )
    df.index = pd.MultiIndex.from_tuples(
        [("p_1", "v_1"), ("p_1", "v_2"), ("p_2", "v_2")]
    )
    info = {"n_obs": 400}
    mod1 = ProcessedModel(params=df, info=info, name="m1")
    mod2 = ProcessedModel(params=df, info=info, name="m2")
    models = [mod1, mod2]
    return models


def _get_test_inputs_single_index():
    df = pd.DataFrame(
        data=np.ones((3, 4)), columns=["value", "ci_lower", "ci_upper", "p_value"]
    )
    df.index = [f"p{i}" for i in [1, 2, 3]]
    info = {"n_obs": 400}
    mod1 = ProcessedModel(params=df, info=info, name="m1")
    mod2 = ProcessedModel(params=df, info=info, name="m2")
    models = [mod1, mod2]
    return models


def _get_test_inputs_multiindex_multi_column():
    df = pd.DataFrame(
        data=np.ones((3, 4)), columns=["value", "ci_lower", "ci_upper", "p_value"]
    )
    df.index = pd.MultiIndex.from_tuples(
        [("p_1", "v_1"), ("p_1", "v_2"), ("p_2", "v_2")]
    )
    info = {"n_obs": 400}
    mod1 = ProcessedModel(params=df.iloc[1:], info=info, name="m1")
    mod2 = ProcessedModel(params=df, info=info, name="m2")
    mod3 = ProcessedModel(params=df, info=info, name="m2")
    models = [mod1, mod2, mod3]
    return models


def _read_csv_string(string, index_cols=None):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)
