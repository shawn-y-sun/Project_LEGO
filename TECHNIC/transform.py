# TECHNIC/transform.py
import pandas as pd
from typing import Callable

class TSFM:
    """
    Time‐series Feature (Transformation) Manager.

    Applies a user‐supplied transform function to a pandas Series, then
    generates up to `max_lag` lagged versions of the transformed series.
    Optionally tracks an expected sign for downstream filtering.

    Parameters:
      transform_fn: Callable[[pd.Series], pd.Series]
        A function that takes a pandas Series and returns a transformed Series.
      max_lag: int
        Maximum number of lag features to generate (default=2).
      exp_sign: int
        Expected sign of the transformation (1 for positive, -1 for negative, 0 for none; default=0).
    """

    def __init__(
        self,
        transform_fn: Callable[[pd.Series], pd.Series],
        max_lag: int = 2,
        exp_sign: int = 0
    ):
        self.transform_fn = transform_fn
        self.max_lag = max_lag
        self.exp_sign = exp_sign
    

    @property
    def suffix(self) -> str:
        """
        Suffix for naming transformed variables:
        [transform_fn name]_L[max_lag], or just [transform_fn name] if max_lag==0.
        """
        name = getattr(self.transform_fn, "__name__", "transform")
        return f"{name}_L{self.max_lag}" if self.max_lag > 0 else name


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