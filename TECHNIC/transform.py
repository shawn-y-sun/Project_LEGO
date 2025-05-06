# TECHNIC/transform.py

import pandas as pd
from typing import Callable, Union

class TSFM:
    """
    Time-series Feature (Transformation) Manager.

    Applies a user-supplied transform function to a pandas Series, then
    shifts the transformed series by `max_lag` periods.
    
    Parameters:
    - feature: Union[str, pd.Series]
      Name of the feature (if str) or the actual pandas Series to transform.
    - transform_fn: Callable[[pd.Series], pd.Series]
      A function to apply to the series.
    - max_lag: int
      Number of periods to lag the transformed series (default=2).
    - exp_sign: int
      Expected sign of the transformation (1 for positive, -1 for negative, 0 for none).
    """

    def __init__(
        self,
        feature: Union[str, pd.Series],
        transform_fn: Callable[[pd.Series], pd.Series],
        max_lag: int = 0,
        exp_sign: int = 0
    ):
        # Store feature and derive name
        if isinstance(feature, pd.Series):
            self.feature = feature
            self.feature_name = feature.name or "feature"
        elif isinstance(feature, str):
            self.feature = None
            self.feature_name = feature
        else:
            raise TypeError("`feature` must be a column name string or a pandas Series")

        self.transform_fn = transform_fn
        self.max_lag = max_lag
        self.exp_sign = exp_sign

    def apply(self) -> pd.Series:
        """
        Apply the transform function and shift by max_lag.
        The returned Series is named by the `name` property.
        """
        if self.feature is None:
            raise ValueError(
                "No feature series provided. Provide a pandas Series to use apply_transform()."
            )
        transformed = self.transform_fn(self.feature)
        result = transformed.shift(self.max_lag) if self.max_lag > 0 else transformed
        result.name = self.name
        return result

    @property
    def name(self) -> str:
        """
        Generate the transformed variable name:
        featureName_transformFnName_Lmax_lag (or without _L if max_lag==0).
        """
        fn_name = getattr(self.transform_fn, "__name__", "transform")
        lag_part = f"_L{self.max_lag}" if self.max_lag > 0 else ""
        return f"{self.feature_name}_{fn_name}{lag_part}"



# Core transform functions

def LV(series: pd.Series) -> pd.Series:
    """Identity: returns the original series."""
    return series


def DF(series: pd.Series, lag: int = 1) -> pd.Series:
    """Difference over lag periods: series - series.shift(lag)."""
    return series - series.shift(lag)


def GR(series: pd.Series, lag: int = 1) -> pd.Series:
    """Growth rate over lag periods: (series / series.shift(lag)) - 1."""
    return series / series.shift(lag) - 1


def ABSGR(series: pd.Series, lag: int = 1) -> pd.Series:
    """Absolute growth rate over lag periods."""
    return (series / series.shift(lag) - 1).abs()

# Rolling window transforms

def ROLLAVG(series: pd.Series, window: int = 4) -> pd.Series:
    """Rolling average over specified window."""
    return series.rolling(window).mean()


def DIV_ROLLAVG(series: pd.Series, window: int = 4) -> pd.Series:
    """Difference from rolling average: series - rolling average."""
    return series - ROLLAVG(series, window)

# Alias functions for common lags

def DF2(series: pd.Series) -> pd.Series:
    """2-period difference."""
    return DF(series, lag=2)


def DF3(series: pd.Series) -> pd.Series:
    """3-period difference."""
    return DF(series, lag=3)


def GR2(series: pd.Series) -> pd.Series:
    """2-period growth rate."""
    return GR(series, lag=2)


def GR3(series: pd.Series) -> pd.Series:
    """3-period growth rate."""
    return GR(series, lag=3)