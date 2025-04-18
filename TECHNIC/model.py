# TECHNIC/model.py
from .base import ModelBase
import pandas as pd
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