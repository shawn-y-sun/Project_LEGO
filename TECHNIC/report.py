import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Dict
from .plot import *

class ModelReportBase(ABC):
    """
    Abstract base for model-specific reports.
    """
    def __init__(
        self,
        X: DataFrame,
        y: Series,
        y_fitted_in: Series,
        X_out: Optional[DataFrame] = None,
        y_out: Optional[Series] = None,
        y_pred_out: Optional[Series] = None,
        in_perf_measures: Dict[str, Any] = None,
        out_perf_measures: Dict[str, Any] = None,
        test_measures: Dict[str, Any] = None,
        param_measures: Dict[str, Dict[str, Any]] = None,
        perf_set_plot_fn: Optional[Callable[[Dict[str, 'ModelReportBase']], Any]] = None
    ):
        # Store model data and measures
        self.X = X
        self.y = y
        self.y_fitted_in = y_fitted_in
        self.X_out = X_out
        self.y_out = y_out
        self.y_pred_out = y_pred_out
        self.in_perf_measures = in_perf_measures or {}
        self.out_perf_measures = out_perf_measures or {}
        self.test_measures = test_measures or {}
        self.param_measures = param_measures or {}
        self._perf_set_plot_fn = perf_set_plot_fn

    @abstractmethod
    def show_in_perf_tbl(self) -> DataFrame:
        """Return in-sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_out_perf_tbl(self) -> DataFrame:
        """Return out-of-sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_test_tbl(self) -> DataFrame:
        """Return testing measures as a DataFrame."""
        ...

    @abstractmethod
    def show_params_tbl(self) -> DataFrame:
        """Return parameter measures as a DataFrame."""
        ...

    @abstractmethod
    def plot_perf(self, **kwargs) -> Any:
        """Plot performance metrics."""
        ...

class ReportSet:
    """
    Aggregates multiple ModelReportBase instances into consolidated performance tables and plots.

    :param reports: dict mapping model_id to ModelReportBase.
    """
    def __init__(
        self,
        reports: Dict[str, ModelReportBase]
    ):
        if not isinstance(reports, dict) or not all(isinstance(r, ModelReportBase) for r in reports.values()):
            raise TypeError("`reports` must be a dict mapping model_id to ModelReportBase instances.")
        self._reports = reports

    def show_in_perf_set(self, **kwargs) -> pd.DataFrame:
        """
        Return a DataFrame with each model's in-sample performance as separate rows,
        keeping the same columns from individual reports and using model_id as index.
        """
        dfs = []
        for mid, rpt in self._reports.items():
            df = rpt.show_in_perf_tbl(**kwargs).copy()
            df.index = [mid] * len(df)
            dfs.append(df)
        if dfs:
            result = pd.concat(dfs, axis=0)
            result.index.name = 'Model'
            return result
        return pd.DataFrame()

    def show_out_perf_set(self, **kwargs) -> pd.DataFrame:
        """
        Return a DataFrame with each model's out-of-sample performance as separate rows,
        keeping the same columns from individual reports and using model_id as index.
        """
        dfs = []
        for mid, rpt in self._reports.items():
            df = rpt.show_out_perf_tbl(**kwargs).copy()
            df.index = [mid] * len(df)
            dfs.append(df)
        if dfs:
            result = pd.concat(dfs, axis=0)
            result.index.name = 'Model'
            return result
        return pd.DataFrame()

    def plot_perf_set(
        self,
        plot_fn: Callable[[Dict[str, ModelReportBase]], Any] = None,
        **kwargs
    ) -> Any:
        """
        Delegate to a plot function (defaulting to each report's own perf_set_plot_fn)
        that accepts the full mapping of model_id -> ModelReportBase and any additional kwargs.
        """
        # Use provided plot_fn, or default to the first report's perf_set_plot_fn
        fn = plot_fn
        if fn is None:
            # assume all reports share the same default, take from first
            fn = next(iter(self._reports.values())).perf_set_plot_fn
        return fn(self._reports, **kwargs)



class OLS_ModelReport(ModelReportBase):
    """
    Report for OLS models: displays performance, tests, and parameters.
    """
    def __init__(
        self,
        X: DataFrame,
        y: Series,
        y_fitted_in: Series,
        X_out: Optional[DataFrame] = None,
        y_out: Optional[Series] = None,
        y_pred_out: Optional[Series] = None,
        in_perf_measures: Dict[str, Any] = None,
        out_perf_measures: Dict[str, Any] = None,
        test_measures: Dict[str, Any] = None,
        param_measures: Dict[str, Dict[str, Any]] = None,
        perf_plot_fn: Callable[..., Any] = ols_model_perf_plot,
        test_plot_fn: Callable[..., Any] = ols_model_test_plot,
        perf_set_plot_fn: Callable[[Dict[str, 'ModelReportBase']], Any] = ols_plot_perf_set
    ):
        super().__init__(
            X=X,
            y=y,
            y_fitted_in=y_fitted_in,
            X_out=X_out,
            y_out=y_out,
            y_pred_out=y_pred_out,
            in_perf_measures=in_perf_measures,
            out_perf_measures=out_perf_measures,
            test_measures=test_measures,
            param_measures=param_measures,
            perf_set_plot_fn=perf_set_plot_fn
        )
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn

    def show_in_perf_tbl(self) -> pd.DataFrame:
        """In-sample performance metrics as a single-row DataFrame."""
        return pd.DataFrame([self.in_perf_measures])

    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Out-of-sample performance metrics as a single-row DataFrame (empty if none)."""
        out = self.out_perf_measures
        return pd.DataFrame([out]) if out else pd.DataFrame()

    def show_test_tbl(self) -> pd.DataFrame:
        """
        Flatten self.test_measures into a DataFrame, with a MultiIndex
        (TestCategory, Test) and one column per metric found in the innermost dicts.
        Works for arbitrary metrics.
        """
        records = []
        for test_name, subtests in self.test_measures.items():
            for subtest_name, metrics in subtests.items():
                # metrics is any dict: {"statistic":…, "pvalue":…, "foo":…, ...}
                row = {"TestCategory": test_name, "Test": subtest_name}
                row.update(metrics)
                records.append(row)
        if not records:
            return pd.DataFrame()  # no tests to show
        df = pd.DataFrame.from_records(records)
        # Set a clean MultiIndex
        return df.set_index(["TestCategory", "Test"])

    def show_params_tbl(self) -> pd.DataFrame:
        """Parameter table with columns: Variable, Coef, Pvalue, Sig, VIF, Std."""
        pm = self.param_measures
        df = pd.DataFrame.from_dict(pm, orient='index')
        df.index.name = 'Variable'
        df = df.reset_index()
        df = df.rename(columns={
            'coef': 'Coef',
            'pvalue': 'Pvalue',
            'sig': 'Sig',
            'vif': 'VIF',
            'std': 'Std'
        })
        cols = ['Variable', 'Coef', 'Pvalue', 'Sig', 'VIF', 'Std']
        cols_existing = [c for c in cols if c in df.columns]
        return df[cols_existing]

    def plot_perf(self, **kwargs) -> Any:
        """Plot actual vs fitted/in-sample and predicted/out-of-sample values."""
        return self.perf_plot_fn(
            self.X,
            self.y,
            X_out=self.X_out,
            y_out=self.y_out,
            y_fitted_in=self.y_fitted_in,
            y_pred_out=self.y_pred_out,
            **kwargs
        )

    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        Display report sections sequentially:
          1) In-sample performance
          2) Optional out-of-sample performance
          3) Parameter table
          4) In-sample performance plot
          5) Optional testing metrics & plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        if not getattr(self, 'out_perf_measures', None):
            show_out = False

        print('=== In-Sample Performance ===')
        print(self.show_in_perf_tbl().to_string(index=False))

        if show_out:
            print('\n=== Out-of-Sample Performance ===')
            print(self.show_out_perf_tbl().to_string(index=False))

        # Parameters
        def fmt_coef(x):
            try:
                val = float(x)
            except:
                return str(x)
            if abs(val) >= 1e5 or (abs(val) > 0 and abs(val) < 1e-3):
                return f"{val:.4e}"
            return f"{val:.4f}"

        def fmt_std(x):
            try:
                val = float(x)
            except:
                return str(x)
            if abs(val) >= 1e5 or (abs(val) > 0 and abs(val) < 1e-3):
                return f"{val:.4e}"
            return f"{val:.4f}"

        print("\n=== Model Parameters ===")
        params_df = self.show_params_tbl()
        print(
            params_df.to_string(
                index=False,
                formatters={
                    'Coef':   fmt_coef,
                    'Pvalue': '{:.3f}'.format,
                    'VIF':    '{:.2f}'.format,
                    'Std':    fmt_std
                }
            )
        )

        # Performance plot
        fig1 = self.plot_perf(**(perf_kwargs or {}))
        plt.show()

        if show_tests:
            print('\n=== Model Testing ===')
            print(self.show_test_tbl().to_string(index=True))