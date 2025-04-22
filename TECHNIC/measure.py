# TECHNIC/measure.py

from abc import ABC
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

class MeasureBase(ABC):
    """
    Abstract base for model measures, supporting separate in-sample
    and out-of-sample performance evaluations.

    Parameters:
      model: fitted model object
      X: DataFrame of in-sample predictors
      y: Series of in-sample target values
      X_out: optional DataFrame of out-of-sample predictors
      y_out: optional Series of out-of-sample target values
      y_pred_out: optional Series of predicted out-of-sample target values
      filter_funcs: dict[name → function(model, X, y)]
      perf_in_funcs: dict[name → function(model, X, y)]
      perf_out_funcs: dict[name → function(model, X_out, y_out)]
      test_funcs: dict[name → function(model, X, y)]
    """
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_out: Optional[pd.DataFrame] = None,
        y_out: Optional[pd.Series] = None,
        y_pred_out: Optional[pd.Series] = None,
        filter_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]] = None,
        perf_in_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]] = None,
        perf_out_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]] = None,
        test_funcs: Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]] = None
    ):
        self.model = model
        self.X = X
        self.y = y
        self.X_out = X_out if X_out is not None else pd.DataFrame()
        self.y_out = y_out if y_out is not None else pd.Series(dtype=float)
        self.y_pred_out = y_pred_out
        self.filter_funcs = filter_funcs or {}
        self.perf_in_funcs = perf_in_funcs or {}
        self.perf_out_funcs = perf_out_funcs or {}
        self.test_funcs = test_funcs or {}

    @property
    def filter_measures(self) -> Dict[str, Any]:
        """Values used to screen candidate models via filtering."""
        return {name: fn(self.model, self.X, self.y)
                for name, fn in self.filter_funcs.items()}

    @property
    def in_perf_measures(self) -> Dict[str, Any]:
        """Performance metrics on the in-sample (training) data."""
        return {name: fn(self.model, self.X, self.y)
                for name, fn in self.perf_in_funcs.items()}

    @property
    def out_perf_measures(self) -> Dict[str, Any]:
        """Performance metrics on the out-of-sample (holdout) data."""
        if not self.X_out.empty and not self.y_out.empty:
            return {name: fn(self.model, self.X_out, self.y_out)
                    for name, fn in self.perf_out_funcs.items()}
        return {}

    @property
    def test_measures(self) -> Dict[str, Any]:
        """Test results (residuals, assumptions, scenario) on in-sample data."""
        return {name: fn(self.model, self.X, self.y)
                for name, fn in self.test_funcs.items()}

    @property
    def param_measures(self) -> Dict[str, Dict[str, Any]]:
        """Parameter details: dict of variable → metrics dict (coef, pvalue, vif, std)."""
        # Default empty; override in subclasses if unsupported
        return {}


def compute_vif(model, X, y):
    """Compute variance inflation factors including intercept."""
    Xc = sm.add_constant(X)
    return {col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)}

class OLS_Measures(MeasureBase):
    """
    Measure class for OLS models: filtering, in-sample performance,
    out-of-sample performance, and testing.
    Allows optional out-of-sample predictions via y_pred_out.
    """
    def __init__(self,
                 model,
                 X: pd.DataFrame,
                 y: pd.Series,
                 X_out: Optional[pd.DataFrame] = None,
                 y_out: Optional[pd.Series] = None,
                 y_pred_out: Optional[pd.Series] = None):
        # Store optional predictions
        self.y_pred_out = y_pred_out
        # filtering: max p-value in-sample
        filter_funcs = {
            "max_pvalue": lambda m, X, y: float(
                m.pvalues.drop("const", errors="ignore").max()
            )
        }
        # in-sample performance functions
        perf_in_funcs = {
            "r2": lambda m, X, y: float(m.rsquared),
            "adj_r2": lambda m, X, y: float(m.rsquared_adj),
            "rmse": lambda m, X, y: float(
                np.sqrt(((y - m.fittedvalues) ** 2).mean())
            )
        }
        # out-of-sample performance functions
        perf_out_funcs = {
            "me": lambda m, Xo, yo: float(
                np.max(np.abs(yo - (self.y_pred_out if self.y_pred_out is not None else m.predict(Xo))))
            ),
            "mae": lambda m, Xo, yo: float(
                np.mean(np.abs(yo - (self.y_pred_out if self.y_pred_out is not None else m.predict(Xo))))
            ),
            "rmse": lambda m, Xo, yo: float(
                np.sqrt(((yo - (self.y_pred_out if self.y_pred_out is not None else m.predict(Xo))) ** 2).mean())
            )
        }
        # testing: residual and assumption tests
        test_funcs = {
            "jb_stat": lambda m, X, y: float(jarque_bera(m.resid)[0]),
            "jb_pvalue": lambda m, X, y: float(jarque_bera(m.resid)[1]),
            "vif": compute_vif
        }
        super().__init__(
            model=model,
            X=X,
            y=y,
            X_out=X_out,
            y_out=y_out,
            filter_funcs=filter_funcs,
            perf_in_funcs=perf_in_funcs,
            perf_out_funcs=perf_out_funcs,
            test_funcs=test_funcs
        )

    @property
    def param_measures(self) -> Dict[str, Dict[str, Any]]:
        """Return dict of parameter statistics: coef, pvalue, vif, std for each variable."""
        # collect from statsmodels
        params = self.model.params
        pvals = self.model.pvalues
        ses = getattr(self.model, 'bse', pd.Series(np.nan, index=params.index))
        # compute VIF including intercept
        vif_dict = compute_vif(self.model, self.X, self.y)
        result = {}
        for var in params.index:
            result[var] = {
                'coef': float(params.get(var, np.nan)),
                'pvalue': float(pvals.get(var, np.nan)),
                'vif': float(vif_dict.get(var, np.nan)),
                'std': float(ses.get(var, np.nan))
            }
        return result
