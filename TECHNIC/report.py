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
        Return a DataFrame of model parameters including VIF.
        Columns: Variable, Coef (4 dp), Pvalue (3 dp), Sig, VIF (2 dp)
        """
        model = self.measure.model
        X = self.measure.X

        # 1) Names & coefficients
        if hasattr(model, "coef_"):
            names = list(X.columns)
            coefs = np.array(model.coef_)
        elif hasattr(model, "params"):
            names = list(model.params.index)
            coefs = np.array(model.params.values)
        else:
            raise AttributeError("Model has neither .coef_ nor .params")

        # 2) P‑values
        if hasattr(model, "pvalues_"):
            pvals = np.array(model.pvalues_)
        elif hasattr(model, "pvalues"):
            pvals = np.array(model.pvalues.values)
        else:
            pvals = np.full_like(coefs, np.nan, dtype=float)

        # 3) VIFs
        vifs = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]

        # 4) Build rows (no intercept row here; assume intercept not in X.columns)
        rows = []
        if hasattr(model, "intercept_"):
            rows.append({
                "Variable": "Intercept",
                "Coef": round(float(model.intercept_), 4),
                "Pvalue": np.nan,
                "Sig": False,
                "VIF": np.nan
            })

        for name, coef, p, vif in zip(names, coefs, pvals, vifs):
            rows.append({
                "Variable": name,
                "Coef": round(float(coef), 4),
                "Pvalue": round(float(p), 3) if not np.isnan(p) else np.nan,
                "Sig": bool(p <= 0.05) if not np.isnan(p) else False,
                "VIF": round(float(vif), 2)
            })

        return pd.DataFrame(rows, columns=["Variable", "Coef", "Pvalue", "Sig", "VIF"])

    def show_report(
        self,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        1) Print performance metrics
        2) Print parameter table
        3) Render performance plot
        4) Optionally: print test metrics & render test plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # 1) Performance metrics
        print("=== Performance Metrics ===")
        print(self.show_perf_tbl().to_string(index=False))

        # 2) Parameter table
        print("\n=== Model Parameters ===")
        print(self.show_params().to_string(index=False))

        # 3) Performance plot
        fig1 = self.plot_perf(**perf_kwargs)
        plt.show()

        # 4) Optional testing
        if show_tests:
            print("\n=== Test Metrics ===")
            print(self.show_test_tbl().to_string(index=False))
            fig2 = self.plot_tests(**test_kwargs)
            plt.show()
