# TECHNIC/cm.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from typing import Type, List, Dict, Any, Union, Optional

from .internal import InternalDataLoader
from .mev import MEVLoader
from .model import ModelBase
from .measure import MeasureBase
from .transform import TSFM

class CM:
    """
    Candidate Model wrapper.

    Manages in-sample, out-of-sample, and full-sample model fitting,
    measures, and reporting.

    Parameters:
      model_id: unique identifier (e.g. "CM1")
      target: name of dependent variable
      data_manager: DataManager instance
      model_cls: subclass of ModelBase
      measure_cls: subclass of MeasureBase
      report_cls: optional subclass of report for display
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
        compute measures using MeasureBase (handles both in- and out-of-sample),
        and instantiate reports.
        """
        # 1) Prepare X and y
        X = self.dm.build_indep_vars(specs)
        y = self.dm.internal_data[self.target].copy()
        idx = self.dm.internal_data.index
        X = X.reindex(idx).astype(float)
        y = y.reindex(idx).astype(float)
        self.X, self.y = X, y

        # 2) Data validation
        nan_cols, inf_cols = [], []
        for col in X.columns:
            ser = X[col]
            if ser.isna().any(): nan_cols.append(col)
            elif is_numeric_dtype(ser) and not np.isfinite(ser.dropna()).all(): inf_cols.append(col)
        nan_y = y.isna().any()
        inf_y = is_numeric_dtype(y) and not np.isfinite(y.dropna()).all()
        errors = []
        if nan_cols: errors.append(f"X contains NaNs: {nan_cols}")
        if inf_cols: errors.append(f"X contains infinite: {inf_cols}")
        if nan_y: errors.append("y contains NaNs")
        if inf_y: errors.append("y contains infinite values")
        if errors: raise ValueError("Data validation error – " + "; ".join(errors))

        # 3) Split data
        cutoff = self.dm.in_sample_end
        if cutoff is not None:
            self.X_in = X.loc[:cutoff];   self.y_in = y.loc[:cutoff]
            self.X_out = X.loc[cutoff + pd.Timedelta(days=1):]; self.y_out = y.loc[cutoff + pd.Timedelta(days=1):]
        else:
            self.X_in, self.y_in = X, y
            self.X_out, self.y_out = pd.DataFrame(), pd.Series(dtype=float)
        self.X_full, self.y_full = X, y

                # 4) Fit models
        # fit in-sample model and store directly
        self.model_in = self.model_cls(self.X_in, self.y_in).fit()
        # fit full-sample model
        self.model_full = self.model_cls(self.X_full, self.y_full).fit()

                # 5) Compute measures
        # use fitted model objects directly
        self.measure_in = self.measure_cls(
            self.model_in, self.X_in, self.y_in,
            X_out=self.X_out,
            y_out=self.y_out,
            y_pred_out=(self.model_in.predict(self.X_out) if not self.X_out.empty else None)
        )
        self.measure_full = self.measure_cls(
            self.model_full, self.X_full, self.y_full
        )

        # 6) Instantiate reports
        if self.report_cls:
            self.report_in = self.report_cls(self.measure_in)
            self.report_full = self.report_cls(self.measure_full)

    def summary_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Return performance and test tables for in-sample, out-of-sample, and full-sample.
        """
        tables = {
            f"{self.model_id}_in_perf":  pd.DataFrame([self.measure_in.in_perf_measures]),
            f"{self.model_id}_in_test":  pd.json_normalize(self.measure_in.test_measures),
            f"{self.model_id}_out_perf": pd.DataFrame([self.measure_in.out_perf_measures]) if self.measure_in.out_perf_measures else pd.DataFrame(),
            f"{self.model_id}_full_perf": pd.DataFrame([self.measure_full.in_perf_measures]),
            f"{self.model_id}_full_test": pd.json_normalize(self.measure_full.test_measures)
        }
        return tables

    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        Display sequentially:
          1) In-sample performance
          2) Optional out-of-sample performance
          3) Model parameters
          4) In-sample performance plot
          5) Optional testing metrics & plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # disable out-of-sample if none
        if not self.measure_in.out_perf_measures:
            show_out = False

        # 1) In-sample performance
        print(f"--- {self.model_id} — In-Sample Performance ---")
        print(self.report_in.show_perf_tbl().to_string(index=False))

        # 2) Out-of-sample performance
        if show_out:
            print(f"\n--- {self.model_id} — Out-of-Sample Performance ---")
            print(self.report_in.show_out_perf_tbl().to_string(index=False))

        # 3) Parameters
        def fmt_coef(x):
            try: val = float(x)
            except: return str(x)
            if abs(val)>=1e5 or (0<abs(val)<1e-3): return f"{val:.4e}"
            return f"{val:.4f}"
        def fmt_std(x):
            try: val = float(x)
            except: return str(x)
            if abs(val)>=1e5 or (0<abs(val)<1e-3): return f"{val:.4e}"
            return f"{val:.4f}"

        print(f"\n--- {self.model_id} — Model Parameters ---")
        params_df = self.report_in.show_params_tbl()
        print(params_df.to_string(index=False, formatters={
            'Coef': fmt_coef, 'Pvalue': '{:.3f}'.format,
            'VIF': '{:.2f}'.format, 'Std': fmt_std
        }))

        # 4) In-sample performance plot
        fig1 = self.report_in.plot_perf(**perf_kwargs)
        plt.show()

        # 5) Optional tests
        if show_tests:
            print(f"\n--- {self.model_id} — Test Metrics ---")
            print(self.report_in.show_test_tbl().to_string(index=False))
            fig2 = self.report_in.plot_tests(**test_kwargs)
            plt.show()