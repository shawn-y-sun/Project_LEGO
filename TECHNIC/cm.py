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
from .transform import TSFM

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
        testset_cls: Optional[Type[Any]] = None,
        report_cls: Optional[Type[Any]] = None
    ):
        self.model_id = model_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.testset_cls = testset_cls
        self.report_cls = report_cls
        # placeholders for data and models
        self.X = self.y = None
        self.X_in = self.y_in = None
        self.X_out = self.y_out = None
        self.X_full = self.y_full = None
        self.model_in: Optional[ModelBase] = None
        self.model_full: Optional[ModelBase] = None

    def __repr__(self) -> str:
        '''
        Return a formula representation of the model without spaces: 'target~C+var1+var2+...'.
        '''
        # Determine which feature set to use
        if self.model_in is not None and hasattr(self, 'X_in'):
            cols = list(self.X_in.columns)
        elif self.X is not None:
            cols = list(self.X.columns)
        else:
            return f"<CM{self.model_id}:no_model_data>"
        # Build formula string without spaces
        formula = f"{self.target}~C"
        if cols:
            formula += "+" + "+".join(cols)
        return formula

    @property
    def formula(self) -> str:
        '''
        Expose the formula string (same as __repr__).
        '''
        return self.__repr__()

    def build(
        self,
        specs: List[Union[str, Dict[str, Any]]],
        sample: str = 'both'
    ) -> None:
        '''
        Build features/target, validate, split, and fit models.
        :param specs: feature specifications for DataManager.
        :param sample: which sample(s) to build: 'in', 'full', or 'both'.
        '''
        # Determine which samples to build
        if sample not in {'in', 'full', 'both'}:
            raise ValueError("sample must be one of 'in', 'full', or 'both'.")
        build_in = sample in {'in', 'both'}
        build_full = sample in {'full', 'both'}

        # Prepare data
        X = self.dm.build_indep_vars(specs)
        y = self.dm.internal_data[self.target].copy()
        idx = self.dm.internal_data.index
        X = X.reindex(idx).astype(float)
        y = y.reindex(idx).astype(float)
        self.X, self.y = X, y

        # Data validation
        nan_cols, inf_cols = [], []
        for col in X.columns:
            s = X[col]
            if s.isna().any():
                nan_cols.append(col)
            elif is_numeric_dtype(s) and not np.isfinite(s.dropna()).all():
                inf_cols.append(col)
        if nan_cols or inf_cols or y.isna().any() or (
            is_numeric_dtype(y) and not np.isfinite(y.dropna()).all()
        ):
            msgs = []
            if nan_cols:
                msgs.append(f"X contains NaNs: {nan_cols}")
            if inf_cols:
                msgs.append(f"X contains infinite: {inf_cols}")
            if y.isna().any():
                msgs.append("y contains NaNs")
            if is_numeric_dtype(y) and not np.isfinite(y.dropna()).all():
                msgs.append("y contains infinite values")
            raise ValueError("Data validation error: " + "; ".join(msgs))

        # Split in- and out-of-sample
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

        # Fit models
        if build_in:
            self.model_in = self.model_cls(
                self.X_in,
                self.y_in,
                X_out=self.X_out,
                y_out=self.y_out,
                testset_cls=self.testset_cls,
                report_cls=self.report_cls
            ).fit()
        if build_full:
            self.model_full = self.model_cls(
                self.X_full,
                self.y_full,
                X_out=self.X_out,
                y_out=self.y_out,
                testset_cls=self.testset_cls,
                report_cls=self.report_cls
            ).fit()

    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: Dict[str, Any] = None,
        test_kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Delegate to the in-sample model's report to show performance and optional tests.
        """
        if not self.report_cls:
            raise RuntimeError(
                "report_cls not provided. Instantiate CM with report_cls before calling show_report()."
            )

        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        report = self.model_in.report
        report.show_report(
            show_out=show_out,
            show_tests=show_tests,
            perf_kwargs=perf_kwargs,
            test_kwargs=test_kwargs
        )

    @property
    def report_in(self) -> Any:
        '''
        Expose the in-sample report property at the CM level.
        '''
        return self.model_in.report

    @property
    def tests_in(self) -> Any:
        '''
        Expose the in-sample tests property at the CM level.
        '''
        return self.model_in.tests

    @property
    def report_full(self) -> Any:
        '''
        Expose the full-sample report property at the CM level.
        '''
        return self.model_full.report

    @property
    def tests_full(self) -> Any:
        '''
        Expose the full-sample tests property at the CM level.
        '''
        return self.model_full.tests