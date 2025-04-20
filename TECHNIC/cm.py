# TECHNIC/cm.py

import pandas as pd
from typing import (
    Type, List, Dict, Any, Union
)
from .internal import *
from .mev import *
from .model import *
from .measure import *
from .transform import TSFM
from .report import *


class CM:
    """
    Candidate Model wrapper.

    Parameters:
      model_id: unique identifier for this candidate model (e.g. "CM1")
      data_manager: DataManager instance with loaded data
      model_cls: subclass of ModelBase (e.g., OLS)
      measure_cls: subclass of MeasureBase (e.g., OLSMeasures)
      report_cls: optional subclass of ReportBase (e.g., OLSReport)
      target: name of the dependent variable (must be in internal_data)
    """
    def __init__(
        self,
        model_id: str,
        data_manager: Any,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
        target: str,
        report_cls: Optional[Type[ModelReportBase]] = None
    ):
        self.model_id = model_id
        self.dm = data_manager
        self.model_cls = model_cls
        self.measure_cls = measure_cls
        self.report_cls = report_cls
        self.target = target

        # placeholders
        self.X = self.y = None
        self.X_in = self.y_in = None
        self.X_full = self.y_full = None

        self.model_in = self.model_full = None
        self.measure_in = self.measure_full = None
        self.report_in = self.report_full = None

    def build(self, specs: List[Union[str, Dict[str, TSFM]]]) -> None:
        """
        Define drivers, split data, fit models, and build measures/reports.
        """
        # build predictors and target
        self.X = self.dm.build_indep_vars(specs)
        self.y = self.dm.internal_data[self.target].copy()

        # split in-sample / full
        cutoff = self.dm.in_sample_end
        if cutoff is not None:
            self.X_in = self.X.loc[:cutoff]
            self.y_in = self.y.loc[:cutoff]
        else:
            self.X_in, self.y_in = self.X, self.y
        self.X_full, self.y_full = self.X, self.y

        # fit models
        self.model_in = self.model_cls(self.X_in, self.y_in)
        fitted_in = self.model_in.fit()
        self.model_full = self.model_cls(self.X_full, self.y_full)
        fitted_full = self.model_full.fit()

        # build measures
        self.measure_in = self.measure_cls(fitted_in, self.X_in, self.y_in)
        self.measure_full = self.measure_cls(fitted_full, self.X_full, self.y_full)

        # build reports if requested
        if self.report_cls:
            self.report_in = self.report_cls(self.measure_in)
            self.report_full = self.report_cls(self.measure_full)

    def summary_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Returns performance & test tables for in-sample and full-sample.
        """
        return {
            f"{self.model_id}_in_perf_tbl":  pd.DataFrame([self.measure_in.performance_measures]),
            f"{self.model_id}_in_test_tbl":  pd.json_normalize(self.measure_in.testing_measures),
            f"{self.model_id}_full_perf_tbl":pd.DataFrame([self.measure_full.performance_measures]),
            f"{self.model_id}_full_test_tbl":pd.json_normalize(self.measure_full.testing_measures),
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        If report_cls was provided, returns all tables and plots:
          {
            'in': {'perf_tbl': ..., 'test_tbl': ..., 'perf_plot': ..., 'test_plot': ...},
            'full': {...}
          }
        """
        if not self.report_cls:
            raise ValueError("No report_cls provided at init.")
        return {
            'in': {
                'perf_tbl':  self.report_in.show_perf_tbl(),
                'test_tbl':  self.report_in.show_test_tbl(),
                'perf_plot': self.report_in.plot_perf(),
                'test_plot': self.report_in.plot_tests(),
            },
            'full': {
                'perf_tbl':  self.report_full.show_perf_tbl(),
                'test_tbl':  self.report_full.show_test_tbl(),
                'perf_plot': self.report_full.plot_perf(),
                'test_plot': self.report_full.plot_tests(),
            }
        }