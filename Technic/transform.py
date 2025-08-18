# =============================================================================
# module: transform.py
# Purpose: Feature transformations as subclasses of Feature
# Dependencies: pandas, typing, .feature.Feature, importlib, functools
# =============================================================================

import pandas as pd
import functools
import importlib
from typing import Callable, Union, Optional, Dict, Any
from .feature import Feature

class TSFM(Feature):
    """
    Transformation feature subclass of Feature.

    Applies a specified function (callable or name string) to an input variable series,
    with optional lag. The exp_sign parameter indicates the expected coefficient sign
    for economic validation during model filtering.

    Parameters
    ----------
    var : str or pandas.Series
        Name or Series of the variable to transform.
    transform_fn : str or Callable[[pandas.Series], pandas.Series]
        Function name (string) to look up in this module or a callable function.
    lag : int, default 0
        Number of periods to lag the series before transformation.
    exp_sign : int, default 0
        Expected coefficient sign for economic validation:
        - 1: expect positive coefficient (positive relationship with target)
        - -1: expect negative coefficient (negative relationship with target)
        - 0: no expectation (no sign constraint)
        Used in exhaustive search filtering to ensure economically sensible models.
    alias : str, optional
        Custom name for the output feature.
    freq : str, optional
        Data frequency ('M' for monthly, 'Q' for quarterly).
        If None, frequency will be inferred from the data index during apply().
        
    Example
    -------
    # GDP growth should have positive relationship with target
    gdp_growth = TSFM('GDP', 'GR', exp_sign=1, freq='Q')
    
    # Interest rates might have negative relationship with target
    interest_rate = TSFM('RATE', 'LV', exp_sign=-1, freq='M')
    
    # No expectation for some variable, frequency auto-detected
    control_var = TSFM('CONTROL', 'DF', exp_sign=0)
    """
    def __init__(
        self,
        var: Union[str, pd.Series],
        transform_fn: Union[str, Callable[[pd.Series], pd.Series]],
        lag: int = 0,
        exp_sign: int = 0,
        alias: Optional[str] = None,
        freq: Optional[str] = None
    ):
        super().__init__(var=var, alias=alias)
        # Resolve transform function if given by name
        if isinstance(transform_fn, str):
            module = importlib.import_module(__name__)
            if hasattr(module, transform_fn):
                self.transform_fn = getattr(module, transform_fn)
            else:
                raise ValueError(f"Unknown transform function '{transform_fn}' in {__name__}.")
        else:
            self.transform_fn = transform_fn
        self.lag = lag
        self.exp_sign = exp_sign
        self.freq = freq  # User-specified frequency (M or Q), None if not specified

    @property
    def name(self) -> str:
        """
        Generate the output feature name.

        Uses alias if provided; otherwise combines:
          • var
          • frequency prefix (MM/QQ) + function name (with a period‐suffix if periods>1)
          • lag indicator (L# if lag>0)

        Examples:
          x_QQDF2        ← Quarterly DF with periods=2, no lag
          x_MMGR3_L1     ← Monthly GR with periods=3 and lag=1
          x_LV           ← LV function (no frequency prefix)
        """
        # 1) Handle functools.partial with a 'periods' keyword
        if isinstance(self.transform_fn, functools.partial):
            func = self.transform_fn.func
            base = func.__name__
            period = self.transform_fn.keywords.get('periods', None)
            if isinstance(period, int) and period > 1:
                fn_name = f"{base}{period}"
            else:
                fn_name = base
        else:
            # 2) Direct callables or alias functions
            fn_name = getattr(self.transform_fn, "__name__", "transform")

        # 3) Add frequency prefix for non-LV functions
        if fn_name != "LV" and self.freq is not None:
            if self.freq == "M":
                fn_name = f"MM{fn_name}"  # Month-over-month
            elif self.freq == "Q":
                fn_name = f"QQ{fn_name}"  # Quarter-over-quarter

        parts = [fn_name]
        if self.lag > 0:
            parts.append(f"L{self.lag}")

        return "_".join([self.var] + parts)
    def lookup_map(self) -> Dict[str, Any]:
        """
        Map the attribute 'var_series' to the variable name for lookup().
        """
        return {"var_series": self.var}

    def apply(self, *dfs: pd.DataFrame) -> pd.Series:
        """
        Resolve input series, apply lag, and transform.

        Parameters
        ----------
        *dfs : pandas.DataFrame
            DataFrame sources for variable lookup.

        Returns
        -------
        pandas.Series
            Transformed series named by self.name.

        Raises
        ------
        KeyError
            If the variable is not found in provided DataFrames.
        """
        # Resolve the input series via lookup()
        self.lookup(*dfs)
        series = self.var_series

        # Detect and cache frequency from series index only if not already specified
        if self.freq is None:
            if hasattr(series.index, 'freq') and series.index.freq is not None:
                freq_str = str(series.index.freq)
            else:
                freq_str = pd.infer_freq(series.index)
            
            if freq_str and freq_str.startswith('M'):
                self.freq = "M"
            elif freq_str and freq_str.startswith('Q'):
                self.freq = "Q"
            else:
                self.freq = None

        # Apply lag if requested
        series = series.shift(self.lag)

        # Apply the transformation function
        result = self.transform_fn(series)

        # Set the result name and return
        result.name = self.name
        # record column name
        self.output_names = [self.name]
        return result

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

def ROLLAVG4(series: pd.Series, periods: int = 4) -> pd.Series:
    """Rolling average over specified periods."""
    return series.rolling(periods).mean()


def DIV_ROLLAVG4(series: pd.Series, periods: int = 4) -> pd.Series:
    """Difference from rolling average: series - rolling average."""
    return series - ROLLAVG(series, periods)

# Alias functions for common lags (no need to add to type_tsfm.yaml)

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