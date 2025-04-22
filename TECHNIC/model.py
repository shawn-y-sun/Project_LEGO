# TECHNIC/model.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable, Dict, Any
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class ModelBase(ABC):
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




class OLS(ModelBase):
    """
    Ordinary Least Squares regression wrapper that stores
    model summary statistics for each predictor.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        super().__init__(X, y)
        # placeholders for results
        self.results = None
        self.params = None
        self.pvalues = None
        self.rsquared = None
        self.rsquared_adj = None
        self.fittedvalues = None
        self.resid = None
        self.bse = None
        self.vif = None

    def fit(self):
        """
        Fit OLS using statsmodels, extract statistics, and return self.
        """
        # add constant
        Xc = sm.add_constant(self.X)
        # fit model
        result = sm.OLS(self.y, Xc).fit()
        # store results
        self.results = result
        self.params = result.params
        self.pvalues = result.pvalues
        self.rsquared = result.rsquared
        self.rsquared_adj = result.rsquared_adj
        self.fittedvalues = result.fittedvalues
        self.resid = result.resid
        self.bse = result.bse
        # compute VIFs including constant
        vif_dict = {
            col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)
        }
        self.vif = pd.Series(vif_dict)
        self.fitted = True
        return self

    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """
        Predict using the fitted statsmodels results.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
        # ensure constant is added
        Xc_new = sm.add_constant(X_new, has_constant='add')
        return self.results.predict(Xc_new)