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
        test_plot_fn: Callable[..., Any],
    ):
        self.measure = measure
        self.perf_plot_fn  = perf_plot_fn
        self.test_plot_fn  = test_plot_fn

    def show_perf_tbl(self) -> pd.DataFrame:
        """Return a single‐row DataFrame of performance measures from the Measure object."""
        return pd.DataFrame([self.measure.performance_measures])

    def show_test_tbl(self) -> pd.DataFrame:
        """Return a single‐row DataFrame of testing measures from the Measure object."""
        return pd.json_normalize(self.measure.testing_measures)

    def plot_perf(self, **kwargs) -> Any:
        """Return the performance plot via the supplied function."""
        return self.perf_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )

    def plot_tests(self, **kwargs) -> Any:
        """Return the testing plot via the supplied function."""
        return self.test_plot_fn(
            self.measure.model,
            self.measure.X,
            self.measure.y,
            **kwargs
        )
    

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