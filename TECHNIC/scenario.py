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
        # Resolve P0
        if P0 is not None:
            self.P0 = pd.to_datetime(P0)
        elif getattr(dm, 'scen_in_sample_end', None) is not None:
            self.P0 = dm.scen_in_sample_end
        elif getattr(dm, 'in_sample_end', None) is not None:
            self.P0 = dm.in_sample_end
        else:
            raise ValueError("Please specify P0 or ensure DataManager has in_sample_end or scen_in_sample_end.")
        # Resolve target
        if target:
            self.target = target
        elif hasattr(model, 'y') and model.y is not None:
            self.target = model.y.name
        else:
            raise ValueError("Target name could not be inferred; please specify explicitly.")
        # Detect conditional specs globally
        specs_list = self.specs if isinstance(self.specs, list) else [self.specs]
        self.cond_specs: List[CondVar] = [
            spec for spec in specs_list
            if isinstance(spec, CondVar)
            and any(
                (cv == self.target) if isinstance(cv, str)
                else (cv.name == self.target)
                for cv in spec.cond_var
            )
        ]
        self._has_cond = bool(self.cond_specs)
        # Build nested scenario feature matrices: {workbook_key: {scenario_name: X_df}}
        self.X_scens: Dict[str, Dict[str, pd.DataFrame]] = {}
        for wb_key, scen_map in dm.scen_mevs.items():
            self.X_scens[wb_key] = {}
            for scen_name, df_mev in scen_map.items():
                X_full = dm.build_indep_vars(self.specs, mev_df=df_mev)
                X_trunc = X_full.loc[X_full.index >= self.P0].copy()
                self.X_scens[wb_key][scen_name] = X_trunc

    def simple_forecast(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict target for a single scenario feature table X.
        :param X: feature DataFrame (e.g. from X_scens)
        :return: Series of predicted values indexed like X
        """
        return self.model.predict(X)

    def conditional_forecast(
        self,
        X: pd.DataFrame,
        y0: pd.Series
    ) -> pd.Series:
        """
        Iteratively forecast a single scenario: starting from P0 and initial y0 series,
        rebuild any CondVar specs depending on the target at each step.

        :param X: feature DataFrame starting from P0
        :param y0: target Series up to and including P0 (index must include P0)
        :return: Series of forecasts indexed by X.index (periods > P0)
        """
        if not isinstance(y0, pd.Series) or self.P0 not in y0.index:
            raise ValueError("y0 must be a pandas Series with its index including P0")
        P0_inferred = y0.index.max()
        X_inter = X.loc[X.index >= P0_inferred].copy()
        y_series = y0.copy()
        preds: List[tuple] = []
        for idx in X_inter.index:
            for spec in self.cond_specs:
                spec.main_series = X_inter[spec.main_name]
                updated_cond: List[pd.Series] = []
                for cv in spec.cond_var:
                    if isinstance(cv, str) and cv == self.target:
                        updated_cond.append(y_series)
                    elif isinstance(cv, pd.Series):
                        updated_cond.append(cv)
                    else:
                        updated_cond.append(self.dm.internal_data[cv])
                spec.cond_var = updated_cond
                new_series = spec.apply()
                X_inter[new_series.name] = new_series
            y_hat = self.model.predict(X_inter.loc[[idx]]).iloc[0]
            preds.append((idx, y_hat))
            y_series.loc[idx] = y_hat
        return pd.Series(dict(preds), name=self.target)

    def forecast(
        self,
        X: pd.DataFrame,
        y0: Optional[pd.Series] = None,
        conditional: bool = False
    ) -> pd.Series:
        """
        Forecast a single scenario: simple or conditional based on flag.

        :param X: feature DataFrame starting from P0
        :param y0: initial target Series up to P0 (needed if conditional=True)
        :param conditional: If True and conditional specs exist, uses conditional_forecast
        :return: Series of predictions indexed like X
        """
        if conditional and self._has_cond:
            if y0 is None:
                y0 = self.dm.internal_data[self.target].loc[:self.P0]
            return self.conditional_forecast(X, y0)
        return self.simple_forecast(X)

    @property
    def y_scens(self) -> Dict[str, Dict[str, pd.Series]]:
        """
        Nested forecast results for all scenarios using `forecast`.
        """
        y0_series = self.dm.internal_data[self.target].loc[:self.P0]
        results: Dict[str, Dict[str, pd.Series]] = {}
        for wb_key, scen_dict in self.X_scens.items():
            results[wb_key] = {}
            for scen_name, X in scen_dict.items():
                results[wb_key][scen_name] = self.forecast(X, y0_series, conditional=self._has_cond)
        return results

    @property
    def scenarios(self) -> List[str]:
        """List of scenario set keys."""
        return list(self.X_scens.keys())