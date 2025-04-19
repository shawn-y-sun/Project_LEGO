# TECHNIC/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable, Dict, Any

class MoedlBase(ABC):
    """
    Abstract base class for statistical models.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.coefs_ = None
        self.fitted = False

    @abstractmethod
    def fit(self):
        """Fit the model to X and y."""
        pass

    @abstractmethod
    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data."""
        pass


class MeasureBase(ABC):
    """
    Abstract base for model measures.

    Parameters:
      model: fitted model object
      X: DataFrame of predictors
      y: Series of target values
      filtering_funcs: dict[name → function(model, X, y)]
      performance_funcs: dict[name → function(model, X, y)]
      testing_funcs: dict[name → function(model, X, y)]
    """

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        filtering_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]],
        performance_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]],
        testing_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]]
    ):
        self.model = model
        self.X = X
        self.y = y
        self.filtering_funcs = filtering_funcs
        self.performance_funcs = performance_funcs
        self.testing_funcs = testing_funcs

    @property
    def filtering_measures(self) -> Dict[str, Any]:
        """Computed values used to screen candidate models."""
        return {
            name: fn(self.model, self.X, self.y)
            for name, fn in self.filtering_funcs.items()
        }

    @property
    def performance_measures(self) -> Dict[str, Any]:
        """Key performance metrics for champion selection."""
        return {
            name: fn(self.model, self.X, self.y)
            for name, fn in self.performance_funcs.items()
        }

    @property
    def testing_measures(self) -> Dict[str, Any]:
        """Residual, assumption, and scenario testing results."""
        return {
            name: fn(self.model, self.X, self.y)
            for name, fn in self.testing_funcs.items()
        }
