# =============================================================================
# module: backtesting.py
# Purpose: Rolling in-sample backtesting for base variable conversion
# Key Types/Classes: BacktestingTest
# Key Functions: results_df
# Dependencies: pandas, numpy
# =============================================================================

from typing import Optional, Any, List
import pandas as pd
import numpy as np


class BacktestingTest:
    """
    Rolling in-sample backtesting for converting fitted target values into base values.

    The backtest uses each historical in-sample date as a jump-off (P0), then
    converts the next ``horizon`` fitted target values into base values using
    the model's base predictor. Each route is stored in a long-format DataFrame
    with a shared date index and NaNs outside of the route window.

    Parameters
    ----------
    model : ModelBase
        Fitted model instance with in-sample predictions.
    horizon : int, default 9
        Number of periods to forecast for each rolling route.
    """

    def __init__(self, model: Any, horizon: int = 9) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.model = model
        self.dm = model.dm
        self.horizon = horizon

    @staticmethod
    def _format_month(dt: pd.Timestamp) -> str:
        return dt.strftime('%b%Y')

    @staticmethod
    def _format_quarter(dt: pd.Timestamp) -> str:
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{quarter}"

    def _format_route(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        freq = getattr(self.dm, 'freq', 'M')
        if freq == 'Q':
            return f"{self._format_quarter(start)}-{self._format_quarter(end)}"
        return f"{self._format_month(start)}-{self._format_month(end)}"

    def _get_model_id(self) -> str:
        for attr in ("model_id", "_model_id"):
            model_id = getattr(self.model, attr, None)
            if model_id:
                return str(model_id)
        return self.model.__class__.__name__

    def _get_actual_series(self, idx: pd.Index) -> pd.Series:
        base_series = getattr(self.model, 'y_base_full', pd.Series(dtype=float))
        if isinstance(base_series, pd.Series) and not base_series.empty:
            return base_series.reindex(idx)
        return self.model.y_full.reindex(idx)

    @property
    def results_df(self) -> Optional[pd.DataFrame]:
        """
        Return backtesting results in long format.

        Columns: ['date', 'model', 'route', 'value']
        - route 'Actual' contains the in-sample actual series.
        - Each backtesting route is labeled by its jump-off start and end.
        """
        if self.dm is None:
            return None

        in_sample_idx = pd.Index(getattr(self.dm, 'in_sample_idx', [])).sort_values()
        if in_sample_idx.empty:
            return None

        model_id = self._get_model_id()
        actual_series = self._get_actual_series(in_sample_idx)

        blocks: List[pd.DataFrame] = [
            pd.DataFrame({
                'date': in_sample_idx,
                'model': model_id,
                'route': 'Actual',
                'value': actual_series.values
            })
        ]

        y_fitted = getattr(self.model, 'y_fitted_in', None)
        if y_fitted is None or y_fitted.empty:
            return pd.concat(blocks, ignore_index=True)

        max_start = len(in_sample_idx) - self.horizon
        if max_start <= 0:
            return pd.concat(blocks, ignore_index=True)

        for i in range(max_start):
            p0 = in_sample_idx[i]
            horizon_idx = in_sample_idx[i + 1:i + 1 + self.horizon]
            if len(horizon_idx) < self.horizon:
                break

            y_pred = y_fitted.reindex(horizon_idx)
            base_pred = pd.Series(index=horizon_idx, dtype=float)

            if y_pred.notna().any():
                try:
                    if getattr(self.model, 'base_predictor', None) is not None:
                        base_pred = self.model.base_predictor.predict_base(y_pred, p0)
                    else:
                        base_pred = y_pred.copy()
                except Exception:
                    base_pred = pd.Series(index=horizon_idx, dtype=float)

            route_series = pd.Series(index=in_sample_idx, dtype=float)
            route_series.loc[horizon_idx] = base_pred.reindex(horizon_idx).values

            route_label = self._format_route(p0, horizon_idx[-1])
            blocks.append(pd.DataFrame({
                'date': in_sample_idx,
                'model': model_id,
                'route': route_label,
                'value': route_series.values
            }))

        return pd.concat(blocks, ignore_index=True)
