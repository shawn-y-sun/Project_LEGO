# TECHNIC/measure.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor


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



def compute_vif(model, X, y):
    Xc = sm.add_constant(X)
    return {col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)}

class OLS_Measures(MeasureBase):
    def __init__(self, model, X: pd.DataFrame, y: pd.Series):
        filtering = {
            "max_pvalue": lambda m, X, y: float(m.pvalues.drop("const", errors="ignore").max())
        }
        performance = {
            "r2":      lambda m, X, y: float(m.rsquared),
            "adj_r2":  lambda m, X, y: float(m.rsquared_adj),
            "rmse":    lambda m, X, y: float(np.sqrt(((y - m.fittedvalues) ** 2).mean()))
        }
        testing = {
            "jb_stat":   lambda m, X, y: float(jarque_bera(m.resid)[0]),
            "jb_pvalue": lambda m, X, y: float(jarque_bera(m.resid)[1])
        }
        super().__init__(model, X, y, filtering, performance, testing)