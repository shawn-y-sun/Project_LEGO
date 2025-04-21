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
        Build a DataFrame of all parameters:
           Variable | Coef  | Pvalue | Sig   | VIF   | Std
        with rounding: Coef 3dp, Pvalue 3dp, VIF 2dp, Std 2dp.
        """
        model = self.measure.model
        X = self.measure.X

        # Must be a statsmodels RegressionResults
        params = model.params          # pandas Series
        pvals  = model.pvalues        # pandas Series
        ses    = getattr(model, "bse", pd.Series(np.nan, index=params.index))

        # Assemble into DataFrame
        df = pd.DataFrame({
            "Coef":   params,
            "Pvalue": pvals,
            "Std":    ses
        })

        # Compute VIF only for the features (exclude intercept if named "const")
        features = list(X.columns)
        vif_vals = [
            variance_inflation_factor(X.values, i)
            for i in range(len(features))
        ]
        vif = pd.Series(vif_vals, index=features)

        # Map VIF into the params DataFrame
        df["VIF"] = df.index.map(lambda name: vif.get(name, np.nan))

        # Significance flag
        df["Sig"] = df["Pvalue"] <= 0.05

        # Reset index into a column called "Variable"
        df = df.reset_index().rename(columns={"index": "Variable"})

        # Reorder & format columns
        df = df[["Variable", "Coef", "Pvalue", "Sig", "VIF", "Std"]]
        df["Coef"]   = df["Coef"].round(3)
        df["Pvalue"] = df["Pvalue"].round(3)
        df["VIF"]    = df["VIF"].round(2)
        df["Std"]    = df["Std"].round(2)

        return df

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
