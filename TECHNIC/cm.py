# TECHNIC/cm.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from typing import Type, List, Dict, Any, Optional, Union

from .internal import InternalDataLoader
from .mev import MEVLoader
from .model import ModelBase
from .measure import MeasureBase
from .transform import TSFM
from .report import scenario_rank_test  # assuming this exists

# Advanced statistical tests
from scipy.stats import shapiro, kstest, cramervonmises
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
# Unit-root and variance-ratio tests
from arch.unitroot import PhillipsPerron, VarianceRatio, DFGLS

class CM:
    """
    Candidate Model wrapper.

    Manages in-sample, out-of-sample, full-sample fitting,
    scenario analysis, pseudo-out-of-sample checks,
    parameter/input sensitivity, filtering and reporting.
    """
    def __init__(
        self,
        model_id: str,
        target: str,
        data_manager: Any,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
        report_cls: Optional[Type] = None
    ):
        self.model_id = model_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.measure_cls = measure_cls
        self.report_cls = report_cls
        # placeholders
        self.X = self.y = None
        self.X_in = self.y_in = None
        self.X_out = self.y_out = None
        self.X_full = self.y_full = None
        self.model_in = self.model_full = None
        self.measure_in = self.measure_full = None
        self.report_in = self.report_full = None

    def build(self, specs: List[Union[str, Dict[str, TSFM]]]) -> None:
        """
        Build features/target, validate, split, fit models,
        compute measures and instantiate reports.
        """
        X = self.dm.build_indep_vars(specs)
        y = self.dm.internal_data[self.target].copy()
        idx = self.dm.internal_data.index
        X = X.reindex(idx).astype(float)
        y = y.reindex(idx).astype(float)
        self.X, self.y = X, y

        # data validation
        nan_cols, inf_cols = [], []
        for col in X.columns:
            s = X[col]
            if s.isna().any(): nan_cols.append(col)
            elif is_numeric_dtype(s) and not np.isfinite(s.dropna()).all(): inf_cols.append(col)
        if nan_cols or inf_cols or y.isna().any() or (
            is_numeric_dtype(y) and not np.isfinite(y.dropna()).all()
        ):
            msgs = []
            if nan_cols: msgs.append(f"X contains NaNs: {nan_cols}")
            if inf_cols: msgs.append(f"X contains infinite: {inf_cols}")
            if y.isna().any(): msgs.append("y contains NaNs")
            if is_numeric_dtype(y) and not np.isfinite(y.dropna()).all(): msgs.append("y contains infinite values")
            raise ValueError("Data validation error: " + "; ".join(msgs))

        # split
        cutoff = self.dm.in_sample_end
        if cutoff is not None:
            self.X_in, self.y_in = X.loc[:cutoff], y.loc[:cutoff]
            self.X_out, self.y_out = (
                X.loc[cutoff + pd.Timedelta(days=1):],
                y.loc[cutoff + pd.Timedelta(days=1):]
            )
        else:
            self.X_in, self.y_in = X, y
            self.X_out, self.y_out = pd.DataFrame(), pd.Series(dtype=float)
        self.X_full, self.y_full = X, y

        # fit models
        self.model_in = self.model_cls(self.X_in, self.y_in).fit()
        self.model_full = self.model_cls(self.X_full, self.y_full).fit()

        # compute measures
        self.measure_in = self.measure_cls(
            self.model_in, self.X_in, self.y_in,
            X_out=self.X_out, y_out=self.y_out,
            y_pred_out=(self.model_in.predict(self.X_out) if not self.X_out.empty else None)
        )
        self.measure_full = self.measure_cls(self.model_full, self.X_full, self.y_full)

        # instantiate reports
        if self.report_cls:
            self.report_in = self.report_cls(self.measure_in)
            self.report_full = self.report_cls(self.measure_full)


    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: Dict[str, Any] = None,
        test_kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Delegate to the in-sample report to show performance and optional tests,
        and optionally show out-of-sample report.
        """
        if not self.report_in:
            raise RuntimeError("report_in is not defined. Call build() with report_cls before show_report().")

        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # In-sample and out-of-sample via report_in
        self.report_in.show_report(
            show_out=show_out,
            show_tests=show_tests,
            perf_kwargs=perf_kwargs,
            test_kwargs=test_kwargs
        )
