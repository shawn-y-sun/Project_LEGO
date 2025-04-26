# TECHNIC/test.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd

from .model import ModelBase
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro


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
        """
        Returns the computed test result (outcome) of the testing.
        """
        ...

    @property
    @abstractmethod
    def test_filter(self) -> bool:
        """
        Returns True if test conditions are met, False otherwise.
        """
        ...


# Dictionary of normality diagnostic tests
normality_test_dict: Dict[str, callable] = {
    'Jarque_Bera': jarque_bera,
    'Shapiro': shapiro
}

class NormalityTest(ModelTestBase):
    """
    Concrete ModelTestBase implementation for normality diagnostics on residuals.

    Provides access to model residuals and performs specified normality tests.
    """
    def __init__(
        self,
        model: Optional[ModelBase] = None,
        test_dict: Dict[str, Any] = normality_test_dict,
        resid: Optional[pd.Series] = None,
        threshold: Union[float, Dict[str, float]] = 0.05
    ):
        super().__init__(model=model, X=None, y=None, test_dict=test_dict)
        self.threshold = threshold
        # Use provided residuals if given, otherwise extract from model
        if resid is not None:
            self.resid = resid
        elif self.model is not None and hasattr(self.model, 'resid'):
            self.resid = getattr(self.model, 'resid')
        else:
            self.resid = None

    @property
    def test_filter(self) -> bool:
        """
        Returns True if all tests in test_result are passed, False otherwise.
        """
        results = self.test_result
        if not results:
            return False
        return all(
            test_info.get('passed', False) for test_info in results.values()
        )