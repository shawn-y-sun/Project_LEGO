# TECHNIC/test.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd
from statsmodels.stats.stattools import jarque_bera, adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import shapiro

from .model import ModelBase


class ModelTestBase(ABC):
    """
    Abstract base class for model testing frameworks.

    Attributes
    ----------
    model : Optional[ModelBase]
        A fitted model instance.
    X : Optional[pd.DataFrame]
        Feature matrix used for testing (when applicable).
    y : Optional[pd.Series]
        Target series used for testing (when applicable).
    test_dict : Dict[str, Any]
        Configuration dict mapping test names to functions and pass rules.
    """
    def __init__(
        self,
        model: Optional[ModelBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        test_dict: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.X = X
        self.y = y
        self.test_dict = test_dict or {}

    @property
    @abstractmethod
    def test_result(self) -> pd.DataFrame:
        """Compute and return a DataFrame of test outcomes."""
        ...

    @property
    @abstractmethod
    def test_filter(self) -> Dict[str, Any]:
        """Return overall pass/fail status and detailed test results."""
        ...


# Separate dictionaries for normality and stationarity tests
normality_tests: Dict[str, Dict[str, Any]] = {
    'Jarque_Bera': {'func': jarque_bera, 'pass_if_greater': True},
    'Shapiro':      {'func': shapiro,      'pass_if_greater': True}
}

stationarity_tests: Dict[str, Dict[str, Any]] = {
    'ADF':  {'func': adfuller, 'pass_if_greater': False},
    'KPSS': {'func': kpss,      'pass_if_greater': True}
}


class NormalityTest(ModelTestBase):
    """
    Performs normality diagnostics on residuals using multiple tests.
    """
    def __init__(
        self,
        model: Optional[ModelBase] = None,
        resid: Optional[pd.Series] = None,
        threshold: Union[float, Dict[str, float]] = 0.05
    ):
        # Initialize base with no X/y, just the configuration dict
        super().__init__(model=model, X=None, y=None, test_dict=normality_tests)
        # Extract residuals
        self.resid = resid if resid is not None else getattr(model, 'resid', None)
        self.threshold = threshold

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Runs configured normality tests and returns a DataFrame:
        ['test_name', 'test_pass', 'test_stat', 'p_value']
        """
        if self.resid is None:
            return pd.DataFrame(columns=["test_name", "test_pass", "test_stat", "p_value"])

        records = []
        for name, config in self.test_dict.items():
            stat, pvalue = config['func'](self.resid)[:2]
            thresh = self.threshold[name] if isinstance(self.threshold, dict) else self.threshold
            passed = pvalue > thresh if config['pass_if_greater'] else pvalue < thresh
            records.append({
                "test_name": name,
                "test_pass": passed,
                "test_stat": float(stat),
                "p_value": float(pvalue)
            })
        return pd.DataFrame(records)

    @property
    def test_filter(self) -> Dict[str, Any]:
        """
        Determines overall pass/fail by majority vote.
        """
        df = self.test_result
        overall = df['test_pass'].sum() >= (len(df) / 2) if not df.empty else False
        return {"overall_pass": overall, "detail": df}


class StationarityTest(ModelTestBase):
    """
    Performs stationarity diagnostics on a time series using multiple tests.
    """
    def __init__(
        self,
        model: Optional[ModelBase] = None,
        series: Optional[pd.Series] = None,
        threshold: Union[float, Dict[str, float]] = 0.05
    ):
        super().__init__(model=model, X=None, y=None, test_dict=stationarity_tests)
        self.series = series if series is not None else getattr(model, 'fittedvalues', None)
        self.threshold = threshold

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Runs configured stationarity tests and returns a DataFrame:
        ['test_name', 'test_pass', 'test_stat', 'p_value']
        """
        if self.series is None:
            return pd.DataFrame(columns=["test_name", "test_pass", "test_stat", "p_value"])

        records = []
        for name, config in self.test_dict.items():
            stat, pvalue = config['func'](self.series)[:2]
            thresh = self.threshold[name] if isinstance(self.threshold, dict) else self.threshold
            passed = pvalue > thresh if config['pass_if_greater'] else pvalue < thresh
            records.append({
                "test_name": name,
                "test_pass": passed,
                "test_stat": float(stat),
                "p_value": float(pvalue)
            })
        return pd.DataFrame(records)

    @property
    def test_filter(self) -> Dict[str, Any]:
        """
        Determines overall pass/fail by majority vote.
        """
        df = self.test_result
        overall = df['test_pass'].sum() >= (len(df) / 2) if not df.empty else False
        return {"overall_pass": overall, "detail": df}
