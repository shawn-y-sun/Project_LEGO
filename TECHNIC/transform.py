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



def LV(series: pd.Series) -> pd.Series:
    """
    Level: identity transform.
    
    Returns the original series unchanged.
    """
    return series


def MMGR(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Month‑over‑Month Growth Rate.
    
    Computes (series / series.shift(lag)) - 1.
    Input must be a monthly time series.
    """
    return series / series.shift(lag) - 1


def MMGR2(series: pd.Series) -> pd.Series:
    """
    Month‑over‑Month Growth Rate at lag 2.
    
    Equivalent to MMGR(series, lag=2).
    """
    return MMGR(series, lag=2)


def MMGR3(series: pd.Series) -> pd.Series:
    """
    Month‑over‑Month Growth Rate at lag 3.
    
    Equivalent to MMGR(series, lag=3).
    """
    return MMGR(series, lag=3)


def MM(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Month‑over‑Month Difference.
    
    Computes series - series.shift(lag).
    Input must be a monthly time series.
    """
    return series - series.shift(lag)


def MM2(series: pd.Series) -> pd.Series:
    """
    Month‑over‑Month Difference at lag 2.
    
    Equivalent to MM(series, lag=2).
    """
    return MM(series, lag=2)


def MM3(series: pd.Series) -> pd.Series:
    """
    Month‑over‑Month Difference at lag 3.
    
    Equivalent to MM(series, lag=3).
    """
    return MM(series, lag=3)


def MMGR_abs(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Absolute Month‑over‑Month Growth Rate.
    
    Computes abs(series / series.shift(lag) - 1).
    """
    return (series / series.shift(lag) - 1).abs()

def QQGR(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Quarter‑over‑Quarter Growth Rate.
    
    Computes (series / series.shift(lag)) - 1.
    Input must be a quarterly time series.
    """
    return series / series.shift(lag) - 1


def QQ(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Quarter‑over‑Quarter Difference.
    
    Computes series - series.shift(lag).
    Input must be a quarterly time series.
    """
    return series - series.shift(lag)


def YYGR(series: pd.Series, lag: int = 4) -> pd.Series:
    """
    Year‑over‑Year Growth Rate.
    
    Computes (series / series.shift(lag)) - 1.
    Input must be a quarterly time series.
    """
    return series / series.shift(lag) - 1


def R4QMA(series: pd.Series, window: int = 4) -> pd.Series:
    """
    Four‑Quarter Rolling Average.
    
    Computes the rolling mean over the past `window` quarters.
    Input must be a quarterly time series.
    """
    return series.rolling(window).mean()


def R4QDiv(series: pd.Series, window: int = 4) -> pd.Series:
    """
    Divergence from Four‑Quarter Rolling Average.
    
    Computes series - R4QMA(series).
    Input must be a quarterly time series.
    """
    return series - R4QMA(series, window)