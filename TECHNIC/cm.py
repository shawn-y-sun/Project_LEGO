# TECHNIC/cm.py

import pandas as pd
from typing import (
    Type, List, Dict, Any, Union
)
from .internal import InternalDataLoader
from .mev import MEVLoader
from .base import ModelBase, MeasureBase
from .transform import TSFM


class CM:
    """
    Candidate Model wrapper.

    Parameters:
      data_manager: DataManager instance with loaded data.
      model_cls: subclass of ModelBase (e.g., OLS).
      measure_cls: subclass of MeasureBase (e.g., OLSMeasures).
      target: name of the dependent variable (must be in internal_data).
    """
    def __init__(
        self,
        data_manager: Any,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
        target: str
    ):
        self.dm = data_manager
        self.model_cls = model_cls
        self.measure_cls = measure_cls
        self.target = target

        # Placeholders for data splits and fitted models/measures
        self.X = None
        self.y = None
        self.X_in = None
        self.y_in = None
        self.X_full = None
        self.y_full = None

        self.model_in = None
        self.model_full = None

        self.measure_in = None
        self.measure_full = None

        self.perf_in: Dict[str, Any] = {}
        self.test_in: Dict[str, Any] = {}
        self.perf_full: Dict[str, Any] = {}
        self.test_full: Dict[str, Any] = {}

    def specify(self, specs: List[Union[str, Dict[str, TSFM]]]) -> None:
        """
        Define model drivers, split data, fit in‑sample and full models,
        and compute measures.

        specs: list of variable names or {var_name: TSFM} dicts (nested lists allowed).
        """
        # Build predictor matrix
        self.X = self.dm.build_indep_vars(specs)
        # Dependent series
        self.y = self.dm.internal_data[self.target].copy().normalize()

        # Split by dm.in_sample_end
        cutoff = self.dm.in_sample_end
        if cutoff is None:
            # Use all data as in‑sample
            self.X_in, self.y_in = self.X, self.y
            self.X_full, self.y_full = self.X, self.y
        else:
            self.X_in = self.X.loc[:cutoff]
            self.y_in = self.y.loc[:cutoff]
            # Full includes both in‑ and out‑sample
            self.X_full = self.X
            self.y_full = self.y

        # Fit in‑sample model
        self.model_in = self.model_cls(self.X_in, self.y_in)
        fitted_in = self.model_in.fit()

        # Fit full sample model
        self.model_full = self.model_cls(self.X_full, self.y_full)
        fitted_full = self.model_full.fit()

        # Compute measures
        self.measure_in = self.measure_cls(fitted_in, self.X_in, self.y_in)
        self.perf_in = self.measure_in.performance_measures
        self.test_in = self.measure_in.testing_measures

        self.measure_full = self.measure_cls(fitted_full, self.X_full, self.y_full)
        self.perf_full = self.measure_full.performance_measures
        self.test_full = self.measure_full.testing_measures

    def summary_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Return performance and test results as DataFrames:
          - 'in_sample_perf', 'in_sample_test'
          - 'full_sample_perf', 'full_sample_test'
        """
        return {
            'in_sample_perf': pd.DataFrame([self.perf_in]),
            'in_sample_test': pd.json_normalize(self.test_in),
            'full_sample_perf': pd.DataFrame([self.perf_full]),
            'full_sample_test': pd.json_normalize(self.test_full),
        }
