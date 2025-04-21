# TECHNIC/cm.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from pandas.api.types import is_numeric_dtype
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
      target: name of the dependent variable (must be in internal_data)
      data_manager: DataManager instance with loaded data
      model_cls: subclass of ModelBase (e.g., OLS)
      measure_cls: subclass of MeasureBase (e.g., OLSMeasures)
      report_cls: optional subclass of ReportBase (e.g., OLSReport)
    """
    def __init__(
        self,
        model_id: str,
        target: str,
        data_manager: Any,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
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
        Define drivers, clip to internal index range, validate data,
        split data, fit models, and build measures/reports.
        """
        # Build predictors and target
        X = self.dm.build_indep_vars(specs)
        y = self.dm.internal_data[self.target].copy()

        # Restrict to the internal data index range
        idx = self.dm.internal_data.index
        X = X.reindex(idx).astype(float)
        y = y.reindex(idx).astype(float)

        # Assign
        self.X, self.y = X, y

        # --- Validate X for NaNs and Infs ---
        nan_cols = []
        inf_cols = []

        for col in self.X.columns:
            ser = self.X[col]
            if ser.isna().any():
                nan_cols.append(col)
            elif is_numeric_dtype(ser):
                if not np.isfinite(ser.dropna()).all():
                    inf_cols.append(col)

        # --- Validate y for NaNs and Infs ---
        nan_y = False
        inf_y = False

        if self.y.isna().any():
            nan_y = True
        elif is_numeric_dtype(self.y):
            if not np.isfinite(self.y.dropna()).all():
                inf_y = True

        # --- Raise combined error if any issues found ---
        errors = []
        if nan_cols:
            errors.append(f"X contains NaNs in columns: {nan_cols}")
        if inf_cols:
            errors.append(f"X contains infinite values in columns: {inf_cols}")
        if nan_y:
            errors.append("y (target) contains NaN values")
        if inf_y:
            errors.append("y (target) contains infinite values")

        if errors:
            raise ValueError("Data validation error – " + "; ".join(errors))

        # Split in‑sample / full
        cutoff = self.dm.in_sample_end
        if cutoff is not None:
            self.X_in = self.X.loc[:cutoff]
            self.y_in = self.y.loc[:cutoff]
        else:
            self.X_in, self.y_in = self.X, self.y
        self.X_full, self.y_full = self.X, self.y

        # Fit in‑sample and full models
        self.model_in   = self.model_cls(self.X_in,  self.y_in)
        fitted_in       = self.model_in.fit()
        self.model_full = self.model_cls(self.X_full, self.y_full)
        fitted_full     = self.model_full.fit()

        # Compute measures
        self.measure_in   = self.measure_cls(fitted_in,  self.X_in,  self.y_in)
        self.measure_full = self.measure_cls(fitted_full,self.X_full,self.y_full)

        # Build reports if requested
        if self.report_cls:
            self.report_in   = self.report_cls(self.measure_in)
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

    def show_report(
        self,
        show_tests: bool = False,
        show_full: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        Delegate to in‑sample (always) and full‑sample (optional) reports’ show_report().

        Parameters
        ----------
        show_tests : bool
            If True, include test metrics & test plots in each report.
        show_full : bool
            If True, also show the full‑sample report. Default is False.
        perf_kwargs : dict, optional
            Passed to each report’s performance plot.
        test_kwargs : dict, optional
            Passed to each report’s test plot.
        """
        if not self.report_cls:
            raise ValueError("No report_cls provided at init.")

        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # In‑sample report (always shown)
        print(f"--- {self.model_id} — In‑Sample Report ---")
        self.report_in.show_report(
            show_tests=show_tests,
            perf_kwargs=perf_kwargs,
            test_kwargs=test_kwargs
        )

        # Full‑sample report (only if requested)
        if show_full:
            print(f"\n--- {self.model_id} — Full‑Sample Report ---")
            self.report_full.show_report(
                show_tests=show_tests,
                perf_kwargs=perf_kwargs,
                test_kwargs=test_kwargs
            )