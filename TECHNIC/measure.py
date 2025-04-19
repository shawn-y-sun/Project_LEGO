# TECHNIC/measure.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .base import MeasureBase

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
            "jb_pvalue": lambda m, X, y: float(jarque_bera(m.resid)[1]),
            "vif":       compute_vif
        }
        super().__init__(model, X, y, filtering, performance, testing)