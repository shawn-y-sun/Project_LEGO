import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, Any
from .measure import *
from .plot import ols_perf_plot, ols_test_plot

class ModelReportBase(ABC):
    """
    Abstract base for model‐specific reports. Subclasses must implement
    methods to display in‑sample performance, out‑of‑sample performance,
    testing measures, parameter tables, and plotting.
    """
    def __init__(
        self,
        measure,
        perf_plot_fn: Callable[..., Any],
        test_plot_fn: Callable[..., Any]
    ):
        self.measure = measure
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn

    @abstractmethod
    def show_perf_tbl(self) -> pd.DataFrame:
        """Return in‑sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Return out‑of‑sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_test_tbl(self) -> pd.DataFrame:
        """Return in‑sample testing measures as a DataFrame."""
        ...

    @abstractmethod
    def show_params_tbl(self) -> pd.DataFrame:
        """Return parameter measures (coef, pvalue, sig, VIF, Std) as a DataFrame."""
        ...

    @abstractmethod
    def plot_perf(self, **kwargs) -> Any:
        """Plot performance metrics; implementation must be provided by subclass."""
        ...

    def plot_tests(self, **kwargs) -> Any:
        """Render diagnostic plot via injected test_plot_fn."""
        return self.test_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )
    
class OLSReport(ModelReportBase):
    """
    OLS-specific report: implements display methods for in-sample performance,
    out-of-sample performance, testing measures, parameter tables, and plotting.
    """
    def __init__(self, measures: OLS_Measures):
        super().__init__(
            measure=measures,
            perf_plot_fn=ols_perf_plot,
            test_plot_fn=ols_test_plot
        )

    def show_perf_tbl(self) -> pd.DataFrame:
        """In-sample performance metrics as a single-row DataFrame."""
        return pd.DataFrame([self.measure.in_perf_measures])

    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Out-of-sample performance metrics as a single-row DataFrame (empty if none)."""
        out = self.measure.out_perf_measures
        return pd.DataFrame([out]) if out else pd.DataFrame()

    def show_test_tbl(self) -> pd.DataFrame:
        """In-sample testing measures as a single-row DataFrame."""
        return pd.json_normalize(self.measure.test_measures)

    def show_params_tbl(self) -> pd.DataFrame:
        """Parameter table with columns: Variable, Coef, Pvalue, Sig, VIF, Std."""
        pm = self.measure.param_measures
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
        """
        Plot actual vs. fitted (in-sample) and predicted (out-of-sample) values,
        with absolute errors as a bar chart (alpha=0.7).
        """
        # Build full target series
        y_list = [self.measure.y]
        if getattr(self.measure, 'y_out', None) is not None:
            y_list.append(self.measure.y_out)
        y_full = pd.concat(y_list).sort_index()

        return self.perf_plot_fn(
            self.measure.model,
            self.measure.X,
            y_full,
            X_out=getattr(self.measure, 'X_out', None),
            y_pred_out=getattr(self.measure, 'y_pred_out', None),
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

        if not getattr(self.measure, 'out_perf_measures', None):
            show_out = False

        print('=== In-Sample Performance ===')
        print(self.show_perf_tbl().to_string(index=False))

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
            print('\n=== Test Metrics ===')
            print(self.show_test_tbl().to_string(index=False))
            fig2 = self.plot_tests(**(test_kwargs or {}))
            plt.show()
