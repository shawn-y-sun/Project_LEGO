# TECHNIC/report.py
from abc import ABC
from typing import Callable, Any
import pandas as pd
import numpy as np
from .measure import *
from .plot import *

class ModelReportBase(ABC):
    """
    Abstract base for model‐specific reports. Subclasses must implement
    methods to display in‑sample performance, out‑of‑sample performance,
    testing measures, and parameter tables.
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

    def plot_perf(self, **kwargs) -> Any:
        """Render the in‑sample performance plot via injected function."""
        return self.perf_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )

    def plot_tests(self, **kwargs) -> Any:
        """Render the test diagnostics plot via injected function."""
        return self.test_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )
    

class OLSReport(ModelReportBase):
    """
    OLS-specific report: implements display methods for in-sample performance,
    out-of-sample performance, testing measures, and parameter tables.
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
        # Rename to proper column headers
        df = df.rename(columns={
            'coef': 'Coef',
            'pvalue': 'Pvalue',
            'sig': 'Sig',
            'vif': 'VIF',
            'std': 'Std'
        })
        # Standardize column order
        cols = ['Variable', 'Coef', 'Pvalue', 'Sig', 'VIF', 'Std']
        cols_existing = [c for c in cols if c in df.columns]
        return df[cols_existing]

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
          2) Optional out-of-sample performance (default on if data exists)
          3) Parameter table
          4) In-sample performance plot
          5) Optional testing metrics & plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # Always check if out-of-sample data exists; disable if not
        if not self.measure.out_perf_measures:
            show_out = False
        # else keep the user's preference (default True)

        # 1) In-sample performance
        print('=== In-Sample Performance ===')
        print(self.show_perf_tbl().to_string(index=False))

        # 2) Out-of-sample performance
        if show_out:
            print('\n=== Out-of-Sample Performance ===')
            print(self.show_out_perf_tbl().to_string(index=False))

        # 3) Parameters
        # Custom formatters: scientific notation for large/small values
        def fmt_coef(x):
            try:
                val = float(x)
            except Exception:
                return str(x)
            if abs(val) >= 1e5 or (abs(val) > 0 and abs(val) < 1e-3):
                return f"{val:.4e}"
            return f"{val:.4f}"

        def fmt_std(x):
            try:
                val = float(x)
            except Exception:
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

        # 4) In-sample performance plot
        fig1 = self.plot_perf(**perf_kwargs)
        plt.show()

        # 5) Optional testing metrics & plot
        if show_tests:
            print('\n=== Test Metrics ===')
            print(self.show_test_tbl().to_string(index=False))
            fig2 = self.plot_tests(**test_kwargs)
            plt.show()

