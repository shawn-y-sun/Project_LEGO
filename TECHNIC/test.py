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

    Parameters
    ----------
    model : Optional[ModelBase]
        Fitted model instance to test.
    X : Optional[pd.DataFrame]
        Feature matrix for testing.
    y : Optional[pd.Series]
        Target vector for testing.
    test_dict : Optional[Dict[str, Any]]
        Dictionary of test configuration parameters.
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
    def test_result(self) -> Any:
        """Returns the computed test result (outcome) of the testing."""
        ...

    @property
    @abstractmethod
    def test_filter(self) -> Any:
        """Returns the overall test decision plus detailed test results."""
        ...


# Test Dictionary
normality_tests Dict[str, Dict[str, Any]] = {
    'Jarque_Bera': {'func': jarque_bera, 'pass_if_greater': True},
    'Shapiro': {'func': shapiro, 'pass_if_greater': True}
}

stationarity_tests: Dict[str, Dict[str, Any]] = {
    'ADF': {'func': adfuller, 'pass_if_greater': False},
    'KPSS': {'func': kpss, 'pass_if_greater': True}
}
class NormalityTest(ModelTestBase):
    """
    Concrete implementation for normality diagnostics on residuals.
    """

    def __init__(
        self,
        model: Optional[ModelBase] = None,
        resid: Optional[pd.Series] = None,
        threshold: Union[float, Dict[str, float]] = 0.05
    ):
        self.model = model
        self.resid = resid if resid is not None else getattr(model, 'resid', None)
        self.threshold = threshold
        self.test_dict = normality_tests

    @property
    def test_result(self) -> pd.DataFrame:
        if self.resid is None:
            return pd.DataFrame(columns=["test_name", "test_pass_flag", "test_stat", "p_value"])

        records = []
        for name, config in self.test_dict.items():
            stat, pvalue = config['func'](self.resid)[:2]
            threshold = self.threshold[name] if isinstance(self.threshold, dict) else self.threshold
            passed = pvalue > threshold if config['pass_if_greater'] else pvalue < threshold
            records.append({
                "test_name": name,
                "test_pass_flag": passed,
                "test_stat": float(stat),
                "p_value": float(pvalue)
            })
        return pd.DataFrame(records)

    @property
    def test_filter(self) -> Dict[str, Any]:
        df = self.test_result
        majority_pass = df['test_pass_flag'].sum() >= (len(df) / 2) if not df.empty else False
        return {"overall_pass": majority_pass, "detail": df}

class StationarityTest(ModelTestBase):
    """
    Concrete implementation for stationarity diagnostics on time series.
    """

    def __init__(
        self,
        model: Optional[ModelBase] = None,
        series: Optional[pd.Series] = None,
        threshold: Union[float, Dict[str, float]] = 0.05
    ):
        self.model = model
        self.series = series if series is not None else getattr(model, 'fittedvalues', None)
        self.threshold = threshold
        self.test_dict = stationarity_tests

    @property
    def test_result(self) -> pd.DataFrame:
        if self.series is None:
            return pd.DataFrame(columns=["test_name", "test_pass_flag", "test_stat", "p_value"])

        records = []
        for name, config in self.test_dict.items():
            stat, pvalue = config['func'](self.series)[:2]
            threshold = self.threshold[name] if isinstance(self.threshold, dict) else self.threshold
            passed = pvalue > threshold if config['pass_if_greater'] else pvalue < threshold
            records.append({
                "test_name": name,
                "test_pass_flag": passed,
                "test_stat": float(stat),
                "p_value": float(pvalue)
            })
        return pd.DataFrame(records)

    @property
    def test_filter(self) -> Dict[str, Any]:
        df = self.test_result
        majority_pass = df['test_pass_flag'].sum() >= (len(df) / 2) if not df.empty else False
        return {"overall_pass": majority_pass, "detail": df}
