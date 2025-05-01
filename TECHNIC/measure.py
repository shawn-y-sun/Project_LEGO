# TECHNIC/measure.py

from abc import ABC
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional, Type
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .testset import TestSetBase, PPNR_OLS_TestSet
from .report import ModelReportBase, OLS_ModelReport

class MeasureBase(ABC):
    """
    Abstract base for model measures, supporting performance evaluations and testing.
    """
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_out: Optional[pd.DataFrame] = None,
        y_out: Optional[pd.Series] = None,
        y_pred_out: Optional[pd.Series] = None,
        testset_cls: Optional[TestSetBase] = None,
        report_cls: Optional[Type[ModelReportBase]] = None,
        perf_in_funcs: Optional[Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]]] = None,
        perf_out_funcs: Optional[Dict[str, Callable[[Any, pd.DataFrame, pd.Series], Any]]] = None,
    ):
        self.model = model
        self.X = X
        self.y = y
        self.X_out = X_out or pd.DataFrame()
        self.y_out = y_out or pd.Series(dtype=float)
        self.y_pred_out = y_pred_out
        self.perf_in_funcs = perf_in_funcs or {}
        self.perf_out_funcs = perf_out_funcs or {}
        self.testset_cls = testset_cls
        self.report_cls = report_cls

    @property
    def in_perf_measures(self) -> Dict[str, Any]:
        return {name: fn(self.model, self.X, self.y)
                for name, fn in self.perf_in_funcs.items()}

    @property
    def out_perf_measures(self) -> Dict[str, Any]:
        if not self.X_out.empty and not self.y_out.empty:
            return {name: fn(self.model, self.X_out, self.y_out)
                    for name, fn in self.perf_out_funcs.items()}
        return {}

    @property
    def test_measures(self) -> Dict[str, Any]:
        return self.testset_cls.test_results if self.testset_cls else {}

    def create_report(self, **kwargs) -> ModelReportBase:
        if not self.report_cls:
            raise ValueError("No report_cls provided for measure report generation.")
        return self.report_cls(self, **kwargs)


def compute_vif(model, X, y):
    """Compute variance inflation factors including intercept."""
    Xc = sm.add_constant(X)
    return {col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)}

class OLS_Measures(MeasureBase):
    """
    Measure class for OLS models: performance and diagnostics.
    """
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_out: Optional[pd.DataFrame] = None,
        y_out: Optional[pd.Series] = None,
        y_pred_out: Optional[pd.Series] = None,
        testset_cls: Type[TestSetBase] = PPNR_OLS_TestSet,
        report_cls: Type[ModelReportBase] = OLS_ModelReport,
    ):
        # Performance functions
        perf_in = {
            "r2": lambda m, X, y: float(m.rsquared),
            "adj_r2": lambda m, X, y: float(m.rsquared_adj),
            "me": lambda m, X, y: float(np.max(np.abs(y - m.fittedvalues))),
            "mae": lambda m, X, y: float(np.mean(np.abs(y - m.fittedvalues))),
            "rmse": lambda m, X, y: float(np.sqrt(((y - m.fittedvalues) ** 2).mean())),
        }
        perf_out = {
            "me": lambda m, Xo, yo: float(np.max(np.abs(yo - (y_pred_out or m.predict(Xo))))),
            "mae": lambda m, Xo, yo: float(np.mean(np.abs(yo - (y_pred_out or m.predict(Xo))))),
            "rmse": lambda m, Xo, yo: float(np.sqrt(((yo - (y_pred_out or m.predict(Xo))) ** 2).mean())),
        }
        # Instantiate default testset_cls and report
        testset_cls = testset_cls()
        super().__init__(
            model=model,
            X=X,
            y=y,
            X_out=X_out,
            y_out=y_out,
            y_pred_out=y_pred_out,
            testset_cls=testset_cls,
            report_cls=report_cls,
            perf_in_funcs=perf_in,
            perf_out_funcs=perf_out,
        )

    @property
    def param_measures(self) -> Dict[str, Dict[str, Any]]:
        params = self.model.params
        pvals = self.model.pvalues
        ses = getattr(self.model, 'bse', pd.Series(np.nan, index=params.index))
        vif_dict = compute_vif(self.model, self.X, self.y)
        result: Dict[str, Dict[str, Any]] = {}
        for var in params.index:
            result[var] = {
                'coef': float(params.get(var, np.nan)),
                'pvalue': float(pvals.get(var, np.nan)),
                'vif': float(vif_dict.get(var, np.nan)),
                'std': float(ses.get(var, np.nan)),
            }
        return result