"""Compatibility module.

Contains wrapper functions to handle compatibility issues between different versions of
external libraries.

"""

from optimagic.config import IS_PANDAS_VERSION_NEWER_OR_EQUAL_TO_2_1_0


def pd_df_map(df, func, na_action=None, **kwargs):
    """Apply a function to a Dataframe elementwise.

    pandas has depricated the .applymap() function with version 2.1.0. This function
    calls either .map() (if pandas version is greater or equal to 2.1.0) or .applymap()
    (if pandas version is smaller than 2.1.0).

    Args:
        df (pd.DataFrame): A pandas DataFrame.
        func (callable): Python function, returns a single value from a single value.
        na_action (str): If 'ignore', propagate NaN values, without passing them to
            func. If None, pass NaN values to func. Default is None.
        **kwargs: Additional keyword arguments to pass as keywords arguments to func.

    Returns:
        pd.DataFrame: Transformed DataFrame.

    """
    if IS_PANDAS_VERSION_NEWER_OR_EQUAL_TO_2_1_0:
        out = df.map(func, na_action=na_action, **kwargs)
    else:
        out = df.applymap(func, na_action=na_action, **kwargs)
    return out
