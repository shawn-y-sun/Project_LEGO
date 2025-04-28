# TECHNIC/test.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd
from statsmodels.stats.stattools import jarque_bera
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


# Dictionary of normality diagnostic tests
normality_test_dict: Dict[str, callable] = {
    'Jarque_Bera': jarque_bera,
    'Shapiro': shapiro
}

# New! Criteria for interpreting p-value:
normality_pass_criteria: Dict[str, bool] = {
    'Jarque_Bera': True,  # pass if p-value > threshold
    'Shapiro': True       # pass if p-value > threshold
}

class NormalityTest(ModelTestBase):
    """
    Concrete ModelTestBase implementation for normality diagnostics on residuals.
    """

    def __init__(
        self,
        model: Optional[ModelBase] = None,
        test_dict: Dict[str, Any] = normality_test_dict,
        pass_criteria: Dict[str, bool] = normality_pass_criteria,
        resid: Optional[pd.Series] = None,
        threshold: Union[float, Dict[str, float]] = 0.05
    ):
        super().__init__(model=model, X=None, y=None, test_dict=test_dict)
        self.pass_criteria = pass_criteria
        self.threshold = threshold
        
        # Set residuals
        if resid is not None:
            self.resid = resid
        elif self.model is not None and hasattr(self.model, 'resid'):
            self.resid = getattr(self.model, 'resid')
        else:
            self.resid = None

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Runs all normality tests and returns a DataFrame:
        Columns: ['test_name', 'test_pass_flag', 'test_stat', 'p_value']
        """
        records = []
        if self.resid is None:
            return pd.DataFrame(columns=["test_name", "test_pass_flag", "test_stat", "p_value"])
        
        for name, test_func in self.test_dict.items():
            stat, pvalue = test_func(self.resid)[:2]
            threshold = self.threshold[name] if isinstance(self.threshold, dict) else self.threshold
            criteria = self.pass_criteria.get(name, True)  # default assume p > threshold means pass
            if criteria:
                passed = pvalue > threshold
            else:
                passed = pvalue < threshold
            records.append({
                "test_name": name,
                "test_pass_flag": passed,
                "test_stat": float(stat),
                "p_value": float(pvalue)
            })
        return pd.DataFrame(records)

    @property
    def test_filter(self) -> Dict[str, Any]:
        """
        Majority rule: returns a dictionary with:
          - 'overall_pass' : bool, majority of tests passed
          - 'detail' : DataFrame from test_result
        """
        df = self.test_result
        if df.empty:
            return {"overall_pass": False, "detail": df}
        
        majority_pass = df['test_pass_flag'].sum() >= (len(df) / 2)
        return {
            "overall_pass": majority_pass,
            "detail": df
        }
