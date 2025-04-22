# TECHNIC/report.py
from abc import ABC
from typing import Callable, Any
import pandas as pd
import numpy as np
from .measure import *
from .plot import *

class ModelReportBase(ABC):
    """
    Abstract template for model‐specific report classes.

    Parameters:
      measure: MeasureBase instance (stores model, X, y and provides perf & testing measures)
      perf_plot_fn: Callable[[Any, pd.DataFrame, pd.Series], Any]
      test_plot_fn: Callable[[Any, pd.DataFrame, pd.Series], Any]
    """

    def __init__(
        self,
        measure: MeasureBase,
        perf_plot_fn: Callable[..., Any],
        test_plot_fn: Callable[..., Any]
    ):
        self.measure = measure
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn

    @abstractmethod
    def show_perf_tbl(self):
        """Return a DataFrame of performance measures."""
        ...

    @abstractmethod
    def show_test_tbl(self):
        """Return a DataFrame of diagnostic/testing measures."""
        ...

    def plot_perf(self, **kwargs) -> Any:
        """Render performance plot via injected function."""
        return self.perf_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )

    def plot_tests(self, **kwargs) -> Any:
        """Render diagnostics plot via injected function."""
        return self.test_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )
    
    @abstractmethod
    def show_report(self):
        """
        Print tables and render plots in one call.
        """
        ...
    

class OLSReport(ModelReportBase):
    """
    OLS‐specific report: implements table methods and its own show_report().
    """

    def __init__(self, measures: OLS_Measures):
        super().__init__(
            measure=measures,
            perf_plot_fn=ols_perf_plot,
            test_plot_fn=ols_test_plot
        )

    def show_perf_tbl(self) -> pd.DataFrame:
        return pd.DataFrame([self.measure.performance_measures])

    def show_test_tbl(self) -> pd.DataFrame:
        return pd.json_normalize(self.measure.testing_measures)

    def show_params(self) -> pd.DataFrame:
        """
        Return exact parameter estimates, p‑values, VIFs, and standard errors.
        No rounding is applied here.
        """
        model = self.measure.model
        X = self.measure.X

        # pull statsmodels objects directly
        params = model.params           # pandas Series
        pvals  = model.pvalues         # pandas Series
        ses    = getattr(model, "bse", pd.Series(np.nan, index=params.index))

        # assemble base DataFrame
        df = pd.DataFrame({
            "Variable": params.index,
            "Coef":      params.values,
            "Pvalue":    pvals.values,
            "Std":       ses.values
        })

        # compute VIF for features (const/intercept excluded if index name is "const")
        features = [v for v in params.index if v in X.columns]
        vif_vals = [
            variance_inflation_factor(X.values, i)
            for i, v in enumerate(X.columns) if v in features
        ]
        vif = pd.Series(vif_vals, index=features)

        # map VIF back onto full df (others as NaN)
        df["VIF"] = df["Variable"].map(lambda v: vif.get(v, np.nan))

        # add significance flag
        df["Sig"] = df["Pvalue"] <= 0.05

        # ensure column order
        return df[["Variable", "Coef", "Pvalue", "Sig", "VIF", "Std"]]

    def show_report(
        self,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        1) Print performance metrics
        2) Print formatted parameter table
        3) Render performance plot
        4) Optionally: print test metrics & render test plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

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

        # 1) Performance metrics
        print("=== Performance Metrics ===")
        print(self.show_perf_tbl().to_string(index=False))

        # 2) Model parameters formatted for display
        print("\n=== Model Parameters ===")
        params_df = self.show_params()
        print(
            params_df.to_string(
                index=False,
                formatters={
                    "Coef":   fmt_coef,
                    "Pvalue": "{:.3f}".format,
                    "VIF":    "{:.2f}".format,
                    "Std":    fmt_std
                }
            )
        )

        # 3) Performance plot
        fig1 = self.plot_perf(**perf_kwargs)
        plt.show()

        # 4) Optional tests
        if show_tests:
            print("\n=== Test Metrics ===")
            print(self.show_test_tbl().to_string(index=False))
            fig2 = self.plot_tests(**test_kwargs)
            plt.show()
