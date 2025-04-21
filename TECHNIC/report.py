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
    OLS‐specific report class that uses an OLSMeasures object.
    """

    def __init__(self, measures: OLS_Measures):
        super().__init__(
            measure=measures,
            perf_plot_fn=ols_perf_plot,
            test_plot_fn=ols_test_plot,
        )

    def show_perf_tbl(self) -> pd.DataFrame:
        # Wrap the performance_measures dict into a single‐row DF
        df = pd.DataFrame([self.measure.performance_measures])
        return df

    def show_test_tbl(self) -> pd.DataFrame:
        # Normalize the testing_measures dict into a single‐row DF
        df = pd.DataFrame([self.measure.testing_measures])
        return df

    def show_params(self) -> pd.DataFrame:
        model = self.measure.model
        X = self.measure.X

        # Collect names and coef values
        if hasattr(model, "coef_"):
            names = list(X.columns)
            coefs = np.asarray(model.coef_)
        elif hasattr(model, "params"):
            names = list(model.params.index)
            coefs = np.asarray(model.params.values)
        else:
            raise AttributeError("Model has neither .coef_ nor .params")

        # Collect p-values
        if hasattr(model, "pvalues_"):
            pvals = np.asarray(model.pvalues_)
        elif hasattr(model, "pvalues"):
            pvals = np.asarray(model.pvalues.values)
        else:
            pvals = np.full_like(coefs, np.nan, dtype=float)

        # Build parameter rows
        rows = []
        if hasattr(model, "intercept_"):
            rows.append({
                "driver": "intercept",
                "coef": float(model.intercept_),
                "pvalue": np.nan,
                "significant": False
            })

        for name, coef, p in zip(names, coefs, pvals):
            sig = bool((p <= 0.05) if not np.isnan(p) else False)
            rows.append({
                "driver": name,
                "coef": float(coef),
                "pvalue": float(p) if not np.isnan(p) else np.nan,
                "significant": sig
            })

        return pd.DataFrame(rows)

    def show_report(
        self,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ):
        """
        Print tables, parameters, and render plots.
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # 1) Print performance & test tables
        print("=== Performance Metrics ===")
        print(self.show_perf_tbl().to_string(index=False))
        print("\n=== Test Metrics ===")
        print(self.show_test_tbl().to_string(index=False))

        # 2) Print parameter table
        print("\n=== Model Parameters ===")
        print(self.show_params().to_string(index=False))

        # 3) Render plots
        fig1 = self.plot_perf(**perf_kwargs)
        plt.show()
        fig2 = self.plot_tests(**test_kwargs)
        plt.show()
    
