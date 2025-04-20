# TECHNIC/model.py
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


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class OLS(ModelBase):
    """
    Ordinary Least Squares regression wrapper that stores
    coefficients, p-values, and VIFs for each predictor.
    """

    def fit(self):
        Xc = sm.add_constant(self.X)
        model = sm.OLS(self.y, Xc).fit()
        self.coefs_   = model.params
        self.pvalues_ = model.pvalues

        # compute VIFs (including constant if present)
        vif_dict = {
            col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)
        }
        self.vifs_ = pd.Series(vif_dict)

        self.fitted = True
        return model

    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
        Xc_new = sm.add_constant(X_new)
        return Xc_new.dot(self.coefs_)