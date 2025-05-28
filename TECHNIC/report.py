# TECHNIC/report.py
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
    Abstract base for model-specific reports, now initialized with a ModelBase.

    Parameters
    ----------
    model : ModelBase
        Fitted ModelBase instance containing data, predictions, and measures.
    perf_set_plot_fn : callable, optional
        Function to plot performance across multiple reports.
    """
    def __init__(
        self,
        model: Any,
        perf_set_plot_fn: Optional[Callable[[Dict[str, 'ModelReportBase']], Any]] = None
    ):
        # Store model for attribute access
        self.model = model
        # Optional performance-set plotting function
        self.perf_set_plot_fn = perf_set_plot_fn

    @abstractmethod
    def show_in_perf_tbl(self) -> DataFrame:
        """Return in-sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_out_perf_tbl(self) -> DataFrame:
        """Return out-of-sample performance measures as a DataFrame."""
        ...

    def show_test_tbl(self) -> None:
        """
        Print all test results from the model's TestSet in a reader-friendly format.
        """
        # Gather all test results (both active and inactive)
        results = self.model.testset.all_test_results
        for test_name, result in results.items():
            print(f"--- {test_name} ---")
            # Print DataFrame or Series, fallback to default print
            if hasattr(result, 'to_string'):
                print(result.to_string())
            else:
                print(result)
            print()

    @abstractmethod
    def show_params_tbl(self) -> DataFrame:
        """Return parameter measures as a DataFrame."""
        ...

    @abstractmethod
    def plot_perf(self, **kwargs) -> Any:
        """Plot performance metrics for this model."""
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
    
    def show_report(
        self,
        show_out: bool = True,
        show_params: bool = False,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        params_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        Sequentially display:
          1) In-sample performance (always)
          2) Out-of-sample performance (if show_out)
          3) Performance plot
          4) Parameter tables per model (if show_params)
          5) Testing tables per model (if show_tests)
        """
        perf_kwargs   = perf_kwargs or {}
        params_kwargs = params_kwargs or {}
        test_kwargs   = test_kwargs or {}

        # 1) In-sample performance
        print("=== In-Sample Performance ===")
        df_in = self.show_in_perf_set(**perf_kwargs)
        print(df_in.to_string())

        # 2) Optional out-of-sample performance
        if show_out:
            print("\n=== Out-of-Sample Performance ===")
            df_out = self.show_out_perf_set(**perf_kwargs)
            print(df_out.to_string())

        # 3) Performance plot
        print("\n=== Performance Plot ===")
        fig = self.plot_perf_set(**perf_kwargs)
        plt.show()

        # 4) Optional parameter tables per model
        if show_params:
            for model_id, report in self._reports.items():
                print(f"\n=== Model: {model_id} — Parameters ===")
                df_params = report.show_params_tbl(**params_kwargs)
                print(df_params.to_string())

        # 5) Optional per-model testing metrics
        if show_tests:
            for model_id, report in self._reports.items():
                print(f"\n=== Model: {model_id} — Testing Metrics ===\n")
                report.show_test_tbl(**test_kwargs)


class OLS_ModelReport(ModelReportBase):
    """
    Report for OLS models: displays performance, tests, and parameter tables.
    """
    def __init__(
        self,
        model: Any,
        perf_plot_fn: Callable[['ModelReportBase'], Any] = ols_model_perf_plot,
        test_plot_fn: Callable[['ModelReportBase'], Any] = ols_model_test_plot,
        perf_set_plot_fn: Callable[[Dict[str, 'ModelReportBase']], Any] = ols_plot_perf_set
    ):
        super().__init__(model=model, perf_set_plot_fn=perf_set_plot_fn)
        # Store specific plot routines
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn

    def show_in_perf_tbl(self) -> pd.DataFrame:
        """Single-row DataFrame of in-sample metrics."""
        return pd.DataFrame([self.model.in_perf_measures])

    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Single-row DataFrame of out-of-sample metrics (empty if none)."""
        out = self.model.out_perf_measures
        return pd.DataFrame([out]) if out else pd.DataFrame()

    def show_params_tbl(self) -> pd.DataFrame:
        """Parameter table with columns: Variable, Coef, Pvalue, Sig, VIF, Std."""
        pm = self.model.param_measures
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
            self.model.X,
            self.model.y,
            X_out=self.model.X_out,
            y_out=self.model.y_out,
            y_fitted_in=self.model.y_fitted_in,
            y_pred_out=self.model.y_pred_out,
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
            self.show_test_tbl()