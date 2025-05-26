# TECHNIC/transform.py

import pandas as pd
import functools
from typing import Callable, Union
import sys

class TSFM:
    """
    Time-series Feature (Transformation) Manager.

    Applies a transform (by name or function) to a pandas Series, then
    shifts the transformed series by `lag` periods.

    Parameters:
    - feature: Union[str, pd.Series]
      Name of the feature (column) or the actual pandas Series to transform.
    - transform_fn: Union[str, Callable[[pd.Series], pd.Series]]
      Either the name of a transform function defined in this module or the function itself.
    - lag: int
      Number of periods to lag the transformed series (default=0).
    - exp_sign: int
      Expected sign of the transformation (1 for positive, -1 for negative, 0 for none).
    """

    def __init__(
        self,
        feature: Union[str, pd.Series],
        transform_fn: Union[str, Callable[[pd.Series], pd.Series]],
        lag: int = 0,
        exp_sign: int = 0
    ):
        # Feature assignment
        if isinstance(feature, pd.Series):
            self.feature = feature
            self.feature_name = feature.name or "feature"
        elif isinstance(feature, str):
            self.feature = None
            self.feature_name = feature
        else:
            raise TypeError("`feature` must be a column name string or a pandas Series")

        # Transform function or name
        if isinstance(transform_fn, str):
            self.transform_fn = None
            self.transform_fn_name = transform_fn
        elif isinstance(transform_fn, functools.partial):
            self.transform_fn = transform_fn
            # partial objects store original function in .func
            self.transform_fn_name = transform_fn.func.__name__
        elif callable(transform_fn):
            self.transform_fn = transform_fn
            self.transform_fn_name = transform_fn.__name__
        else:
            raise TypeError("`transform_fn` must be a function, functools.partial, or the name of one as a string")

        self.lag = lag
        self.exp_sign = exp_sign

    def apply(self) -> pd.Series:
        """
        Apply the transform function (resolved by name if needed) and shift by lag.
        Returns a Series named by the `name` property.
        """
        if self.feature is None:
            raise ValueError("No feature series provided. Assign `feature` before calling apply()")
        # Resolve string-specified transform_fn
        if self.transform_fn is None:
            fn = sys.modules[__name__].__dict__.get(self.transform_fn_name)
            if fn is None or not callable(fn):
                raise ValueError(f"Transform function '{self.transform_fn_name}' not found in module")
            self.transform_fn = fn
        # Apply transformation
        transformed = self.transform_fn(self.feature)
        result = transformed.shift(self.lag) if self.lag > 0 else transformed
        result.name = self.name
        return result

    @property
    def name(self) -> str:
        """
        Construct the feature name as FeatureName_FunctionName[_Llag].
        """
        if self.transform_fn is None:
            fn_name = self.transform_fn_name
        elif isinstance(self.transform_fn, functools.partial):
            fn_name = self.transform_fn.func.__name__
            # include any 'periods' keyword if present and >1
            keywords = getattr(self.transform_fn, 'keywords', {}) or {}
            if 'periods' in keywords and keywords['periods'] > 1:
                fn_name = f"{fn_name}{keywords['periods']}"
        else:
            fn_name = getattr(self.transform_fn, '__name__', self.transform_fn_name)
        lag_part = f"_L{self.lag}" if self.lag > 0 else ""
        return f"{self.feature_name}_{fn_name}{lag_part}"
    

    def __repr__(self) -> str:
        """Use the `name` property as the representation, prefixed with 'TSFM:'."""
        return f"TSFM:{self.name}"


# Core transform functions

def LV(series: pd.Series) -> pd.Series:
    """Identity: returns the original series."""
    return series


def DF(series: pd.Series, periods: int = 1) -> pd.Series:
    """Difference over lag periods: series - series.shift(lag)."""
    return series - series.shift(periods)


def GR(series: pd.Series, periods: int = 1) -> pd.Series:
    """Growth rate over lag periods: (series / series.shift(periods)) - 1."""
    return series / series.shift(periods) - 1


def ABSGR(series: pd.Series, periods: int = 1) -> pd.Series:
    """Absolute growth rate over lag periods."""
    return (series / series.shift(periods) - 1).abs()

# Rolling window transforms

def ROLLAVG(series: pd.Series, periods: int = 4) -> pd.Series:
    """Rolling average over specified periods."""
    return series.rolling(periods).mean()


def DIV_ROLLAVG(series: pd.Series, periods: int = 4) -> pd.Series:
    """Difference from rolling average: series - rolling average."""
    return series - ROLLAVG(series, periods)

# Alias functions for common lags

def DF2(series: pd.Series) -> pd.Series:
    """2-period difference."""
    return DF(series, periods=2)


def DF3(series: pd.Series) -> pd.Series:
    """3-period difference."""
    return DF(series, periods=3)


def GR2(series: pd.Series) -> pd.Series:
    """2-period growth rate."""
    return GR(series, periods=2)


def GR3(series: pd.Series) -> pd.Series:
    """3-period growth rate."""
    return GR(series, periods=3)