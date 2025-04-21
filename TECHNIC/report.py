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

    def show_report(
        self,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ):
        """
        Print tables and render plots in one call.
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # 1) Print tables
        perf_df = self.show_perf_tbl()
        test_df = self.show_test_tbl()

        print("=== Performance Metrics ===")
        print(perf_df.to_string(index=False))
        print("\n=== Test Metrics ===")
        print(test_df.to_string(index=False))

        # 2) Render plots
        fig1 = self.plot_perf(**perf_kwargs)
        plt.show()
        fig2 = self.plot_tests(**test_kwargs)
        plt.show()

        # Return objects in case the user wants to inspect or save them
        return perf_df, test_df, fig1, fig2
    
