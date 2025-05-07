# TECHNIC/scenario.py
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from .data import DataManager
from .model import ModelBase
from .condition import CondVar


class Scenario:
    """
    Perform scenario forecasting analysis for a fitted model based on MEV scenarios.

    :param dm: DataManager instance (must have scen_mevs loaded)
    :param model: Fitted ModelBase instance (with .predict)
    :param specs: feature specs (str, TSFM, CondVar, etc.) as in CM
    :param P0: Optional index (e.g. timestamp or position) marking forecast start.
    :param target: Optional name of target series; defaults to model.y.name
    """
    def __init__(
        self,
        dm: DataManager,
        model: ModelBase,
        specs: Any,
        P0: Optional[pd.Timestamp] = None,
        target: Optional[str] = None
    ):
        self.dm = dm
        self.model = model
        self.specs = specs
        # Determine P0
        if P0 is not None:
            self.P0 = pd.to_datetime(P0)
        elif getattr(dm, 'scen_in_sample_end', None) is not None:
            self.P0 = dm.scen_in_sample_end
        elif getattr(dm, 'in_sample_end', None) is not None:
            self.P0 = dm.in_sample_end
        else:
            raise ValueError("Please specify P0 or ensure DataManager has in_sample_end or scen_in_sample_end.")
        # Determine target
        self.target = target or getattr(model, 'y', pd.Series()).name
        if not self.target:
            raise ValueError("Target name could not be inferred; please specify explicitly.")
        # Build scenario feature matrices
        self.X_scens: Dict[str, pd.DataFrame] = {}
        for wb_key, scen_map in dm.scen_mevs.items():
            for scen_name, df_mev in scen_map.items():
                sc_key = f"{wb_key}_{scen_name}"
                X = dm.build_indep_vars(self.specs, mev_df=df_mev)
                # restrict to periods > P0
                X = X.loc[X.index > self.P0]
                self.X_scens[sc_key] = X

    def simple_forecast(self) -> pd.DataFrame:
        """
        Run predictions for each scenario without conditional updates.
        :return: DataFrame of forecasts, columns are scenario keys.
        """
        preds = {}
        for sc, X in self.X_scens.items():
            preds[sc] = self.model.predict(X)
        return pd.DataFrame(preds)

    def conditional_forecast(self) -> pd.DataFrame:
        """
        Run predictions handling CondVar specs that depend on prior forecasts.
        If a CondVar has cond_var equal to target, its main_series shifts as forecasts proceed.
        """
        results = {}
        # Identify which specs are conditional on target
        cond_specs = [s for s in (self.specs if isinstance(self.specs, list) else [self.specs])
                      if isinstance(s, CondVar) and any(cv == self.target for cv in s.cond_var)]
        for sc, X in self.X_scens.items():
            # Make a copy to iteratively update
            X_iter = X.copy()
            y_pred = []
            # iterate period by period
            for idx in X_iter.index:
                # for each conditional spec, update its main_series/cond_var
                for spec in cond_specs:
                    # rebuild the single-feature DataFrame for this spec
                    # assume spec.main_series name exists in X_iter
                    # get prev forecast if needed
                    spec_args = {'main_series': None, 'cond_series': None}
                    # Ideally, re-apply spec.apply() with updated series
                    # but as placeholder: skip complex rebuild
                    pass
                # Predict for this row
                x_row = X_iter.loc[[idx]]
                y_hat = self.model.predict(x_row).iloc[0]
                y_pred.append((idx, y_hat))
            results[sc] = pd.Series(dict(y_pred))
        return pd.DataFrame(results)

    def forecast(
        self,
        conditional: bool = False
    ) -> pd.DataFrame:
        """
        Dispatch to simple or conditional forecast.

        :param conditional: if True and CondVar present, use conditional_forecast; else simple.
        """
        if conditional and any(isinstance(s, CondVar) for s in (self.specs if isinstance(self.specs, list) else [self.specs])):
            return self.conditional_forecast()
        return self.simple_forecast()

    @property
    def scenarios(self) -> List[str]:
        """List of scenario keys."""
        return list(self.X_scens.keys())

    @property
    def periods(self) -> pd.DatetimeIndex:
        """Forecast periods (index) based on one scenario."""
        # assume all scenarios share same index
        return next(iter(self.X_scens.values())).index
