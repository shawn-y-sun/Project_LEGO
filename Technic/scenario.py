import pandas as pd
from typing import Any, Dict, List, Optional, Union
import warnings
import matplotlib.pyplot as plt
import os

from .internal import TimeSeriesLoader, PanelLoader
from .data import DataManager
from .model import ModelBase
from .condition import CondVar
from .feature import DumVar

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
from matplotlib.ticker import FuncFormatter


class ScenManager:
    """
    Manages scenario forecasting analysis for fitted models based on MEV scenarios.
    This class is designed to be used as a component of ModelBase instances, providing
    scenario forecasting capabilities and test data.

    Parameters
    ----------
    dm : DataManager
        DataManager instance (must have scen_mevs loaded and internal loader with scen_p0 set)
    model : ModelBase
        Fitted ModelBase instance (with .predict)
    specs : Any
        Feature specs (str, TSFM, CondVar, etc.) as in CM
    horizon : int, default=12
        Number of quarters to forecast after P0 (e.g., 9 or 12 quarters)
    target : str, optional
        Name of target series; defaults to model.y.name
    qtr_method : str, default='mean'
        Method to aggregate monthly results to quarterly frequency.
        Options: 'mean', 'sum', 'last', 'first'

    Attributes
    ----------
    y_scens : Dict[str, Dict[str, pd.Series]]
        Nested forecast results for all scenarios
    scenarios : List[str]
        List of scenario set keys
    scenario_colors : List[str]
        Color scheme for plotting scenarios in order

    Example
    -------
    # Create ScenManager for a fitted model with 9-quarter forecast horizon
    scen_mgr = ScenManager(dm, model, specs, horizon=9)
    
    # Create ScenManager with sum aggregation for quarterly reporting
    scen_mgr = ScenManager(dm, model, specs, horizon=9, qtr_method='sum')
    
    # Access scenario forecasts
    forecasts = scen_mgr.y_scens
    """
    
    # Class attribute for consistent scenario colors across all plotting methods
    scenario_colors = ['orange', 'grey', 'dodgerblue', 'purple', 'brown', 'pink', 'olive', 'cyan']
    def __init__(
        self,
        dm: DataManager,
        model: ModelBase,
        specs: Any,
        horizon: int = 12,
        target: Optional[str] = None,
        qtr_method: str = 'mean'
    ):
        self.dm = dm
        self.model = model
        self.specs = specs
        
        # Validate forecast horizon
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        
        # Validate quarterly aggregation method
        valid_qtr_methods = ['mean', 'sum', 'last', 'first']
        if qtr_method not in valid_qtr_methods:
            raise ValueError(f"qtr_method must be one of {valid_qtr_methods}")
        self.qtr_method = qtr_method
        
        # Get P0 from DataManager's internal loader
        self.P0 = self.dm.scen_p0
        if self.P0 is None:
            raise ValueError("Internal data loader must have scen_p0 set for scenario analysis.")
            
        # Calculate horizon end date (P0 + horizon quarters)
        self.horizon_end = self.P0 + pd.offsets.QuarterEnd(self.horizon)
            
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

    @property
    def X_scens(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Build and return scenario feature matrices on demand.
        
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Nested scenario feature matrices: {scenario_set: {scenario_name: X_df}}
            
        Raises
        ------
        ValueError
            If scenario data is inconsistent between MEVs and internal data
        """
        X_scens: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Get scenario sets from MEVs
        for scen_set, scen_map in self.dm.scen_mevs.items():
            X_scens[scen_set] = {}
            
            # Check if we have internal data for this scenario set
            has_internal = bool(self.dm.scen_internal_data)
            if has_internal and scen_set not in self.dm.scen_internal_data:
                raise ValueError(
                    f"Scenario set '{scen_set}' found in MEVs but not in internal data. "
                    "Please provide corresponding internal data or clean scen_internal_data."
                )
            
            for scen_name, df_mev in scen_map.items():
                # If we have internal data, validate scenario exists
                if has_internal:
                    if scen_name not in self.dm.scen_internal_data[scen_set]:
                        raise ValueError(
                            f"Scenario '{scen_name}' found in MEVs but not in internal data "
                            f"for set '{scen_set}'. Please provide corresponding internal data "
                            "or clean scen_internal_data."
                        )
                    internal_df = self.dm.scen_internal_data[scen_set][scen_name]
                else:
                    internal_df = pd.DataFrame()
                
                # Build features using both MEVs and internal data if available
                try:
                    X_full = self.dm.build_features(
                        self.specs,
                        internal_df=internal_df,
                        mev_df=df_mev
                    )
                except KeyError as e:
                    # Extract the missing variable name from the error message
                    missing_var = str(e).strip("'")
                    if internal_df is not None and internal_df.empty:
                        raise ValueError(
                            f"Scenario '{scen_name}' in set '{scen_set}' has empty internal data. "
                            "Please ensure all scenario internal data is populated with the necessary variables."
                        ) from e
                    else:
                        raise ValueError(
                            f"Variable '{missing_var}' not found in scenario '{scen_name}' internal data (set: '{scen_set}'). "
                            "Please ensure all required internal variables are available in the scenario data."
                        ) from e
                
                # Filter to forecast period (from P0 onwards and within horizon)
                if isinstance(self.dm._internal_loader, PanelLoader):
                    # For panel data, filter based on date column
                    date_col = self.dm._internal_loader.date_col
                    if date_col in X_full.columns:
                        X_filtered = X_full[
                            (X_full[date_col] >= self.P0) & 
                            (X_full[date_col] <= self.horizon_end)
                        ].copy()
                    else:
                        X_filtered = X_full.copy()
                else:  # TimeSeriesLoader
                    # For time series, filter based on DatetimeIndex
                    if isinstance(X_full.index, pd.DatetimeIndex):
                        X_filtered = X_full[
                            (X_full.index >= self.P0) & 
                            (X_full.index <= self.horizon_end)
                        ].copy()
                    else:
                        X_filtered = X_full.copy()
                    
                X_scens[scen_set][scen_name] = X_filtered.astype(float)
                
        return X_scens

    def simple_forecast(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict target for a single scenario feature table X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame (e.g. from X_scens)

        Returns
        -------
        pd.Series
            Series of predicted values indexed like X, truncated to horizon and after P0
        """
        # Get predictions
        preds = self.model.predict(X)
        
        # Filter to forecast period (after P0 and within horizon) based on data structure
        if isinstance(self.dm._internal_loader, PanelLoader):
            # For panel data, filter based on date column
            date_col = self.dm._internal_loader.date_col
            if date_col in X.columns:
                # Get the indices where dates are after P0 and within horizon
                valid_idx = X[(X[date_col] > self.P0) & (X[date_col] <= self.horizon_end)].index
                preds = preds[valid_idx]
        else:  # TimeSeriesLoader
            # For time series, filter based on DatetimeIndex
            if isinstance(preds.index, pd.DatetimeIndex):
                preds = preds[(preds.index > self.P0) & (preds.index <= self.horizon_end)]
            
        return preds

    def conditional_forecast(
        self,
        X: pd.DataFrame,
        y0: pd.Series
    ) -> pd.Series:
        """
        Iteratively forecast a single scenario: starting from P0 and initial y0 series,
        rebuild any CondVar specs depending on the target at each step.

        Currently only supports time series data (TimeSeriesLoader). Panel data (PanelLoader)
        is not yet supported.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame starting from P0
        y0 : pd.Series
            Target Series up to and including P0 (index must include P0)

        Returns
        -------
        pd.Series
            Series of forecasts indexed by X.index (periods > P0), truncated to horizon

        Raises
        ------
        ValueError
            - If y0 is not a pandas Series or doesn't include P0
            - If using PanelLoader (not yet supported)
        """
        if isinstance(self.dm._internal_loader, PanelLoader):
            raise ValueError(
                "Conditional forecasting is not yet supported for panel data. "
                "Please use simple_forecast() instead."
            )

        if not isinstance(y0, pd.Series) or self.P0 not in y0.index:
            raise ValueError("y0 must be a pandas Series with its index including P0")
            
        # Only process data up to horizon
        X_horizon = X[X.index <= self.horizon_end].copy()
        
        P0_inferred = y0.index.max()
        X_iter = X_horizon.loc[X_horizon.index >= P0_inferred].copy()
        y_series = y0.copy()
        preds: List[tuple] = []
        for idx in X_iter.index:
            for spec in self.cond_specs:
                spec.main_series = X_iter[spec.name]
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
                X_iter[new_series.name] = new_series
            y_hat = self.model.predict(X_iter.loc[[idx]]).iloc[0]
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

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame starting from P0
        y0 : pd.Series, optional
            Initial target Series up to P0 (needed if conditional=True)
        conditional : bool, default=False
            If True and conditional specs exist, uses conditional_forecast

        Returns
        -------
        pd.Series
            Series of predictions indexed like X, truncated to horizon

        Raises
        ------
        ValueError
            If conditional=True is used with panel data (not supported)
        """
        if conditional and self._has_cond:
            if y0 is None:
                # Get y0 based on data structure
                if isinstance(self.dm._internal_loader, PanelLoader):
                    raise ValueError(
                        "Conditional forecasting is not yet supported for panel data. "
                        "Please use simple_forecast() instead."
                    )
                else:  # TimeSeriesLoader
                    y0 = self.dm.internal_data[self.target].loc[:self.P0]
            return self.conditional_forecast(X, y0)
        return self.simple_forecast(X)

    @property
    def y_base_scens(self) -> Dict[str, Dict[str, pd.Series]]:
        """
        Nested base forecast results for all scenarios using base predictor.

        Returns
        -------
        Dict[str, Dict[str, pd.Series]]
            Nested scenario base forecast results: {scenario_set: {scenario_name: y_base_series}}
            Each series contains base forecasts up to horizon quarters after P0 using base_predictor.

        Notes
        -----
        For time series data:
            - Uses the scenario jumpoff date (scen_p0) for base prediction conversion
        For panel data:
            - Only supports simple forecasting (no conditional forecasting)
            - Base forecasts are made for each entity-date combination
        """
        if not hasattr(self.model, 'base_predictor') or self.model.base_predictor is None:
            return {}
            
        # Get scenario target forecasts
        target_forecasts = self.y_scens
        
        results: Dict[str, Dict[str, pd.Series]] = {}
        for scen_set, scen_dict in target_forecasts.items():
            results[scen_set] = {}
            for scen_name, y_pred in scen_dict.items():
                if y_pred.empty:
                    results[scen_set][scen_name] = pd.Series([], name=self.model.target_base or self.target)
                    continue
                    
                try:
                    # Use scen_p0 as the jumpoff date for base prediction
                    base_pred = self.model.base_predictor.predict_base(y_pred, self.P0)
                    results[scen_set][scen_name] = base_pred
                except Exception:
                    # Return empty series if conversion fails
                    results[scen_set][scen_name] = pd.Series([], name=self.model.target_base or self.target)
        
        return results

    @property
    def forecast_y_base_df(self) -> Dict[str, pd.DataFrame]:
        """
        Organize scenario base forecasting results into DataFrames with period indicators.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping scenario set names to DataFrames.
            Each DataFrame contains:
            - Period: Indicator column ('Pre-P0', 'P0', 'P1', etc.)
              Note: Period is always in quarterly frequency. For monthly data,
              all months within the same quarter share the same period indicator.
            - Fitted: In-sample fitted base values from model (if available)
            - One column per scenario containing base forecast values
            
        Notes
        -----
        - For time series data, index is DatetimeIndex
        - For panel data, index includes both entity and date information
        - Periods before P0 are marked as 'Pre-P0'
        - P0 and subsequent periods are numbered sequentially (P0, P1, P2, etc.)
        - Period indicators are always quarterly, even for monthly data
        - All time indices are normalized to dates only (no time component)
        """
        # Get base scenario forecasts
        base_scen_results = self.y_base_scens
        
        if not base_scen_results:
            return {}
        
        # Get fitted base values from model
        fitted_base_values = None
        if hasattr(self.model, 'y_base_fitted_in') and self.model.y_base_fitted_in is not None:
            fitted_base_values = self.model.y_base_fitted_in
            fitted_base_values.index = pd.to_datetime(fitted_base_values.index).normalize()
        
        # Function to get quarter-end date
        def get_quarter_end(date):
            return pd.Period(date, freq='Q').end_time.normalize()
        
        # Function to assign period indicator
        def assign_period(date, p0_quarter_end):
            date_quarter_end = get_quarter_end(date)
            if date_quarter_end < p0_quarter_end:
                return 'Pre-P0'
            elif date_quarter_end == p0_quarter_end:
                return 'P0'
            else:
                # Calculate quarters difference
                quarters_diff = (date_quarter_end.to_period('Q') - p0_quarter_end.to_period('Q')).n
                return f'P{quarters_diff}'
        
        results: Dict[str, pd.DataFrame] = {}
        for scen_set, scen_dict in base_scen_results.items():
            # Start with an empty DataFrame
            if isinstance(self.dm._internal_loader, TimeSeriesLoader):
                # For time series, get all unique dates
                all_dates = pd.Index([])
                if fitted_base_values is not None:
                    all_dates = all_dates.union(fitted_base_values.index)
                for scen_series in scen_dict.values():
                    # Normalize scenario series index
                    scen_series.index = pd.to_datetime(scen_series.index).normalize()
                    all_dates = all_dates.union(scen_series.index)
                all_dates = all_dates.sort_values()
                
                # Create DataFrame with DatetimeIndex
                df = pd.DataFrame(index=all_dates)
                
                # Get P0 quarter-end date
                p0_quarter_end = get_quarter_end(self.P0)
                
                # Add Period indicator
                df['Period'] = df.index.map(lambda x: assign_period(x, p0_quarter_end))
                
                # Add fitted base values if available
                if fitted_base_values is not None:
                    df['Fitted'] = fitted_base_values
                
                # Add scenario base forecasts
                for scen_name, scen_series in scen_dict.items():
                    df[scen_name] = scen_series
                    
            else:  # PanelLoader
                # For panel data, get all unique entity-date combinations
                entity_col = self.dm._internal_loader.entity_col
                date_col = self.dm._internal_loader.date_col
                
                # Collect all entity-date pairs
                all_pairs = set()
                if fitted_base_values is not None:
                    all_pairs.update(zip(fitted_base_values.index.get_level_values(entity_col),
                                      fitted_base_values.index.get_level_values(date_col).normalize()))
                for scen_series in scen_dict.values():
                    # Normalize dates in the index
                    dates = scen_series.index.get_level_values(date_col).normalize()
                    entities = scen_series.index.get_level_values(entity_col)
                    all_pairs.update(zip(entities, dates))
                
                # Create MultiIndex DataFrame
                entities, dates = zip(*sorted(all_pairs))
                idx = pd.MultiIndex.from_tuples(list(zip(entities, dates)),
                                              names=[entity_col, date_col])
                df = pd.DataFrame(index=idx)
                
                # Get P0 quarter-end date
                p0_quarter_end = get_quarter_end(self.P0)
                
                # Add Period indicator
                date_series = pd.Series(dates, index=idx)
                df['Period'] = date_series.map(lambda x: assign_period(x, p0_quarter_end))
                
                # Add fitted base values if available
                if fitted_base_values is not None:
                    df['Fitted'] = fitted_base_values
                
                # Add scenario base forecasts
                for scen_name, scen_series in scen_dict.items():
                    # Normalize dates in the scenario series index
                    dates = scen_series.index.get_level_values(date_col).normalize()
                    entities = scen_series.index.get_level_values(entity_col)
                    scen_series.index = pd.MultiIndex.from_tuples(
                        list(zip(entities, dates)),
                        names=[entity_col, date_col]
                    )
                    df[scen_name] = scen_series
            
            # Store the result
            results[scen_set] = df
            
        return results

    @property
    def forecast_y_base_qtr_df(self) -> Dict[str, pd.DataFrame]:
        """
        Organize scenario base forecasting results into quarterly DataFrames with period indicators.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping scenario set names to DataFrames.
            Each DataFrame contains quarterly base data with:
            - Index: Period indicators ('P0', 'P1', 'P2', etc. for P0+ periods; 
                    'YY-MM' format for Pre-P0 quarters like '21-03', '21-06')
            - Fitted: In-sample fitted base values in quarterly frequency
            - One column per scenario containing quarterly base forecast values
            
        Notes
        -----
        - All results are converted to quarterly frequency using self.qtr_method
        - For monthly data, conversion is applied according to qtr_method parameter
        - For quarterly data, data is used as-is
        - Period indicators: P0, P1, P2, etc. for forecast periods; YY-MM for historical
        - YY-MM format uses quarter-end month (03, 06, 09, 12)
        """
        # Get base scenario forecasts
        base_scen_results = self.y_base_scens
        
        if not base_scen_results:
            return {}
        
        # Get fitted base values from model
        fitted_base_values = None
        if hasattr(self.model, 'y_base_fitted_in') and self.model.y_base_fitted_in is not None:
            fitted_base_values = self.model.y_base_fitted_in
            fitted_base_values.index = pd.to_datetime(fitted_base_values.index).normalize()
        
        # Function to get quarter-end date
        def get_quarter_end(date):
            return pd.Period(date, freq='Q').end_time.normalize()
        
        # Function to assign period indicator with new format
        def assign_period_label(date, p0_quarter_end):
            date_quarter_end = get_quarter_end(date)
            if date_quarter_end < p0_quarter_end:
                # Pre-P0: use YY-MM format (quarter end month)
                year_2digit = str(date_quarter_end.year)[-2:]
                month = f"{date_quarter_end.month:02d}"
                return f'{year_2digit}-{month}'
            elif date_quarter_end == p0_quarter_end:
                return 'P0'
            else:
                # Calculate quarters difference
                quarters_diff = (date_quarter_end.to_period('Q') - p0_quarter_end.to_period('Q')).n
                return f'P{quarters_diff}'
        
        # Function to aggregate data to quarterly
        def aggregate_to_quarterly(series, qtr_method):
            """Aggregate series to quarterly frequency."""
            if self.dm.freq == 'Q':
                # Already quarterly, return as-is
                return series
            
            # Convert to quarterly
            series_copy = series.copy()
            series_copy.index = pd.to_datetime(series_copy.index)
            
            # Group by quarter and aggregate
            quarterly_grouped = series_copy.groupby(pd.Grouper(freq='Q'))
            
            if qtr_method == 'mean':
                result = quarterly_grouped.mean()
            elif qtr_method == 'sum':
                result = quarterly_grouped.sum()
            elif qtr_method == 'last':
                result = quarterly_grouped.last()
            elif qtr_method == 'first':
                result = quarterly_grouped.first()
            else:
                raise ValueError(f"Unsupported aggregation method: {qtr_method}")
            
            # Convert index to quarter-end dates
            result.index = result.index.to_period('Q').to_timestamp(how='end').normalize()
            return result
        
        results: Dict[str, pd.DataFrame] = {}
        for scen_set, scen_dict in base_scen_results.items():
            # Collect all data first
            all_data = {}
            
            # Process fitted base values
            if fitted_base_values is not None:
                fitted_quarterly = aggregate_to_quarterly(fitted_base_values, self.qtr_method)
                all_data['Fitted'] = fitted_quarterly
            
            # Process scenario base forecasts
            for scen_name, scen_series in scen_dict.items():
                # Normalize scenario series index
                scen_series.index = pd.to_datetime(scen_series.index).normalize()
                scen_quarterly = aggregate_to_quarterly(scen_series, self.qtr_method)
                all_data[scen_name] = scen_quarterly
            
            # Get all unique quarter-end dates
            all_quarters = pd.Index([])
            for series in all_data.values():
                all_quarters = all_quarters.union(series.index)
            all_quarters = all_quarters.sort_values()
            
            # Create quarterly DataFrame
            df = pd.DataFrame(index=all_quarters)
            
            # Add all data series
            for col_name, series in all_data.items():
                df[col_name] = series
            
            # Get P0 quarter-end date
            p0_quarter_end = get_quarter_end(self.P0)
            
            # Create period labels for index
            period_labels = [assign_period_label(date, p0_quarter_end) for date in df.index]
            
            # Set period labels as index
            df.index = period_labels
            df.index.name = 'Period'
            
            results[scen_set] = df
            
        return results

    @property
    def y_scens(self) -> Dict[str, Dict[str, pd.Series]]:
        """
        Nested forecast results for all scenarios using `forecast`.

        Returns
        -------
        Dict[str, Dict[str, pd.Series]]
            Nested scenario forecast results: {scenario_set: {scenario_name: y_series}}
            Each series contains forecasts up to horizon quarters after P0.

        Notes
        -----
        For time series data:
            - Uses the target series up to P0 for conditional forecasting
        For panel data:
            - Only supports simple forecasting (no conditional forecasting)
            - Forecasts are made for each entity-date combination
        """
        # Get initial conditions based on data structure
        if isinstance(self.dm._internal_loader, TimeSeriesLoader):
            y0_series = self.dm.internal_data[self.target].loc[:self.P0]
        else:  # PanelLoader
            y0_series = None  # Panel data doesn't support conditional forecasting
            if self._has_cond:
                warnings.warn(
                    "Conditional forecasting is not supported for panel data. "
                    "Using simple_forecast() instead.",
                    UserWarning
                )

        results: Dict[str, Dict[str, pd.Series]] = {}
        for scen_set, scen_dict in self.X_scens.items():
            results[scen_set] = {}
            for scen_name, X in scen_dict.items():
                # For panel data or when no conditional specs, use simple forecast
                if isinstance(self.dm._internal_loader, PanelLoader) or not self._has_cond:
                    results[scen_set][scen_name] = self.forecast(X, conditional=False)
                else:
                    # For time series with conditional specs, use conditional forecast
                    results[scen_set][scen_name] = self.forecast(X, y0_series, conditional=True)
        return results

    @property
    def forecast_y_qtr_df(self) -> Dict[str, pd.DataFrame]:
        """
        Organize scenario forecasting results into quarterly DataFrames with period indicators.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping scenario set names to DataFrames.
            Each DataFrame contains quarterly data with:
            - Index: Period indicators ('P0', 'P1', 'P2', etc. for P0+ periods; 
                    'YY-MM' format for Pre-P0 quarters like '21-03', '21-06')
            - Fitted: In-sample fitted values in quarterly frequency
            - One column per scenario containing quarterly forecast values
            
        Notes
        -----
        - All results are converted to quarterly frequency using self.qtr_method
        - For monthly data, conversion is applied according to qtr_method parameter
        - For quarterly data, data is used as-is
        - Period indicators: P0, P1, P2, etc. for forecast periods; YY-MM for historical
        - YY-MM format uses quarter-end month (03, 06, 09, 12)
        """
        # Get scenario forecasts
        scen_results = self.y_scens
        
        # Get fitted values from model
        fitted_values = self.model.y_fitted_in if hasattr(self.model, 'y_fitted_in') else None
        if fitted_values is not None:
            fitted_values.index = pd.to_datetime(fitted_values.index).normalize()
        
        # Function to get quarter-end date
        def get_quarter_end(date):
            return pd.Period(date, freq='Q').end_time.normalize()
        
        # Function to assign period indicator with new format
        def assign_period_label(date, p0_quarter_end):
            date_quarter_end = get_quarter_end(date)
            if date_quarter_end < p0_quarter_end:
                # Pre-P0: use YY-MM format (quarter end month)
                year_2digit = str(date_quarter_end.year)[-2:]
                month = f"{date_quarter_end.month:02d}"
                return f'{year_2digit}-{month}'
            elif date_quarter_end == p0_quarter_end:
                return 'P0'
            else:
                # Calculate quarters difference
                quarters_diff = (date_quarter_end.to_period('Q') - p0_quarter_end.to_period('Q')).n
                return f'P{quarters_diff}'
        
        # Function to aggregate data to quarterly
        def aggregate_to_quarterly(series, qtr_method):
            """Aggregate series to quarterly frequency."""
            if self.dm.freq == 'Q':
                # Already quarterly, return as-is
                return series
            
            # Convert to quarterly
            series_copy = series.copy()
            series_copy.index = pd.to_datetime(series_copy.index)
            
            # Group by quarter and aggregate
            quarterly_grouped = series_copy.groupby(pd.Grouper(freq='Q'))
            
            if qtr_method == 'mean':
                result = quarterly_grouped.mean()
            elif qtr_method == 'sum':
                result = quarterly_grouped.sum()
            elif qtr_method == 'last':
                result = quarterly_grouped.last()
            elif qtr_method == 'first':
                result = quarterly_grouped.first()
            else:
                raise ValueError(f"Unsupported aggregation method: {qtr_method}")
            
            # Convert index to quarter-end dates
            result.index = result.index.to_period('Q').to_timestamp(how='end').normalize()
            return result
        
        results: Dict[str, pd.DataFrame] = {}
        for scen_set, scen_dict in scen_results.items():
            # Collect all data first
            all_data = {}
            
            # Process fitted values
            if fitted_values is not None:
                fitted_quarterly = aggregate_to_quarterly(fitted_values, self.qtr_method)
                all_data['Fitted'] = fitted_quarterly
            
            # Process scenario forecasts
            for scen_name, scen_series in scen_dict.items():
                # Normalize scenario series index
                scen_series.index = pd.to_datetime(scen_series.index).normalize()
                scen_quarterly = aggregate_to_quarterly(scen_series, self.qtr_method)
                all_data[scen_name] = scen_quarterly
            
            # Get all unique quarter-end dates
            all_quarters = pd.Index([])
            for series in all_data.values():
                all_quarters = all_quarters.union(series.index)
            all_quarters = all_quarters.sort_values()
            
            # Create quarterly DataFrame
            df = pd.DataFrame(index=all_quarters)
            
            # Add all data series
            for col_name, series in all_data.items():
                df[col_name] = series
            
            # Get P0 quarter-end date
            p0_quarter_end = get_quarter_end(self.P0)
            
            # Create period labels for index
            period_labels = [assign_period_label(date, p0_quarter_end) for date in df.index]
            
            # Set period labels as index
            df.index = period_labels
            df.index.name = 'Period'
            
            results[scen_set] = df
            
        return results

    @property
    def forecast_y_df(self) -> Dict[str, pd.DataFrame]:
        """
        Organize scenario forecasting results into DataFrames with period indicators.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping scenario set names to DataFrames.
            Each DataFrame contains:
            - Period: Indicator column ('Pre-P0', 'P0', 'P1', etc.)
              Note: Period is always in quarterly frequency. For monthly data,
              all months within the same quarter share the same period indicator.
            - Fitted: In-sample fitted values from model
            - One column per scenario containing forecast values
            
        Notes
        -----
        - For time series data, index is DatetimeIndex
        - For panel data, index includes both entity and date information
        - Periods before P0 are marked as 'Pre-P0'
        - P0 and subsequent periods are numbered sequentially (P0, P1, P2, etc.)
        - Period indicators are always quarterly, even for monthly data
        - All time indices are normalized to dates only (no time component)
        """
        # Get scenario forecasts
        scen_results = self.y_scens
        
        # Get fitted values from model
        fitted_values = self.model.y_fitted_in if hasattr(self.model, 'y_fitted_in') else None
        if fitted_values is not None:
            fitted_values.index = pd.to_datetime(fitted_values.index).normalize()
        
        # Function to get quarter-end date
        def get_quarter_end(date):
            return pd.Period(date, freq='Q').end_time.normalize()
        
        # Function to assign period indicator
        def assign_period(date, p0_quarter_end):
            date_quarter_end = get_quarter_end(date)
            if date_quarter_end < p0_quarter_end:
                return 'Pre-P0'
            elif date_quarter_end == p0_quarter_end:
                return 'P0'
            else:
                # Calculate quarters difference
                quarters_diff = (date_quarter_end.to_period('Q') - p0_quarter_end.to_period('Q')).n
                return f'P{quarters_diff}'
        
        results: Dict[str, pd.DataFrame] = {}
        for scen_set, scen_dict in scen_results.items():
            # Start with an empty DataFrame
            if isinstance(self.dm._internal_loader, TimeSeriesLoader):
                # For time series, get all unique dates
                all_dates = pd.Index([])
                if fitted_values is not None:
                    all_dates = all_dates.union(fitted_values.index)
                for scen_series in scen_dict.values():
                    # Normalize scenario series index
                    scen_series.index = pd.to_datetime(scen_series.index).normalize()
                    all_dates = all_dates.union(scen_series.index)
                all_dates = all_dates.sort_values()
                
                # Create DataFrame with DatetimeIndex
                df = pd.DataFrame(index=all_dates)
                
                # Get P0 quarter-end date
                p0_quarter_end = get_quarter_end(self.P0)
                
                # Add Period indicator
                df['Period'] = df.index.map(lambda x: assign_period(x, p0_quarter_end))
                
                # Add fitted values if available
                if fitted_values is not None:
                    df['Fitted'] = fitted_values
                
                # Add scenario forecasts
                for scen_name, scen_series in scen_dict.items():
                    df[scen_name] = scen_series
                    
            else:  # PanelLoader
                # For panel data, get all unique entity-date combinations
                entity_col = self.dm._internal_loader.entity_col
                date_col = self.dm._internal_loader.date_col
                
                # Collect all entity-date pairs
                all_pairs = set()
                if fitted_values is not None:
                    all_pairs.update(zip(fitted_values.index.get_level_values(entity_col),
                                      fitted_values.index.get_level_values(date_col).normalize()))
                for scen_series in scen_dict.values():
                    # Normalize dates in the index
                    dates = scen_series.index.get_level_values(date_col).normalize()
                    entities = scen_series.index.get_level_values(entity_col)
                    all_pairs.update(zip(entities, dates))
                
                # Create MultiIndex DataFrame
                entities, dates = zip(*sorted(all_pairs))
                idx = pd.MultiIndex.from_tuples(list(zip(entities, dates)),
                                              names=[entity_col, date_col])
                df = pd.DataFrame(index=idx)
                
                # Get P0 quarter-end date
                p0_quarter_end = get_quarter_end(self.P0)
                
                # Add Period indicator
                date_series = pd.Series(dates, index=idx)
                df['Period'] = date_series.map(lambda x: assign_period(x, p0_quarter_end))
                
                # Add fitted values if available
                if fitted_values is not None:
                    df['Fitted'] = fitted_values
                
                # Add scenario forecasts
                for scen_name, scen_series in scen_dict.items():
                    # Normalize dates in the scenario series index
                    dates = scen_series.index.get_level_values(date_col).normalize()
                    entities = scen_series.index.get_level_values(entity_col)
                    scen_series.index = pd.MultiIndex.from_tuples(
                        list(zip(entities, dates)),
                        names=[entity_col, date_col]
                    )
                    df[scen_name] = scen_series
            
            # Store the result
            results[scen_set] = df
            
        return results

    def plot_forecasts(
        self,
        scen_set: str,
        forecast_data: Optional[pd.DataFrame] = None,
        figsize: tuple = (8, 4),
        style: Optional[Dict[str, Dict[str, Any]]] = None,
        title_prefix: str = "",
        save_path: Optional[str] = None,
        show_qtr: bool = True
    ) -> plt.Figure:
        """
        Plot forecasting results for a single scenario set.
        
        Creates a plot showing:
        - Fitted values for Pre-P0 and P0 periods
        - Forecast values for each scenario after P0
        
        Parameters
        ----------
        scen_set : str
            Name of the scenario set being plotted
        forecast_data : pd.DataFrame, optional
            DataFrame containing forecast data. If None, will use either
            forecast_y_qtr_df or forecast_y_df based on show_qtr parameter
        figsize : tuple, default=(8, 4)
            Figure size as (width, height)
        style : Dict[str, Dict[str, Any]], optional
            Styling options for each scenario. Format:
            {scenario_name: {style_param: value}}
            Example: {'Base': {'color': 'blue', 'linestyle': '-'},
                     'Adverse': {'color': 'red', 'linestyle': '--'}}
            If not provided, uses default styles
        title_prefix : str, default=""
            Prefix to add to plot titles
        save_path : str, optional
            If provided, saves plots to this directory path
        show_qtr : bool, default=True
            If True, use quarterly frequency data (forecast_y_qtr_df).
            If False, use original frequency data (forecast_y_df).
            Only used when forecast_data is None.
            
        Returns
        -------
        plt.Figure
            The plot figure for the scenario set
            
        Notes
        -----
        - Creates new figures without affecting existing plots
        - Works with both quarterly and original frequency data
        - Handles both YY-MM format and 'Pre-P0' period indicators
        """
        # Get forecast data if not provided
        if forecast_data is None:
            if show_qtr:
                forecast_data = self.forecast_y_qtr_df[scen_set]
            else:
                forecast_data = self.forecast_y_df[scen_set]
        
        df = forecast_data.copy()
        
        # Create new figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Handle different data structures
        if 'Period' not in df.columns:
            # Aggregated data with period labels as index
            period_labels = df.index.tolist()
            x_positions = list(range(len(period_labels)))
            
            # Get scenario names and sort for consistent ordering
            scenarios = sorted([col for col in df.columns if col != 'Fitted'])
            
            # Identify historical periods and forecast periods
            p0_position = None
            if 'P0' in period_labels:
                p0_position = period_labels.index('P0')
            
            # Plot fitted values for historical periods and P0
            if 'Fitted' in df.columns:
                fitted_data = df['Fitted'].dropna()
                if not fitted_data.empty:
                    fitted_x = []
                    fitted_y = []
                    for i, (period, value) in enumerate(fitted_data.items()):
                        if not pd.isna(value) and (not period.startswith('P') or period == 'P0'):
                            fitted_x.append(x_positions[period_labels.index(period)])
                            fitted_y.append(value)
                    
                    if fitted_x:
                        ax.plot(fitted_x, fitted_y, color='black', linestyle='-', 
                               label='Fitted', linewidth=2)
            
            # Plot each scenario (forecast periods only)
            for i, scenario in enumerate(scenarios):
                # Get style for this scenario
                if style and scenario in style:
                    scen_style = style[scenario]
                    if 'linewidth' not in scen_style:
                        scen_style['linewidth'] = 2
                else:
                    color = self.scenario_colors[i % len(self.scenario_colors)]
                    scen_style = {
                        'color': color,
                        'linestyle': '-',
                        'label': scenario,
                        'linewidth': 2
                    }
                
                # Plot scenario data for forecast periods
                scenario_data = df[scenario].dropna()
                if not scenario_data.empty:
                    forecast_x = []
                    forecast_y = []
                    for period, value in scenario_data.items():
                        if not pd.isna(value) and period.startswith('P'):
                            forecast_x.append(x_positions[period_labels.index(period)])
                            forecast_y.append(value)
                    
                    if forecast_x:
                        ax.plot(forecast_x, forecast_y, **scen_style)
            
            # Add vertical line at P0
            if p0_position is not None:
                ax.axvline(x=p0_position, color='gray', linestyle='--', alpha=0.7)
                ax.text(p0_position, ax.get_ylim()[1], 'P0',
                       rotation=0, ha='center', va='bottom', fontsize=8)
            
            # Set x-axis ticks and labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(period_labels, rotation=45, ha='right', fontsize=8)
            
        else:
            # Original frequency data with DatetimeIndex and Period column
            scenarios = sorted([col for col in df.columns if col not in ['Period', 'Fitted']])
            
            # Plot fitted values for Pre-P0 and P0 periods
            if 'Fitted' in df.columns:
                fitted_mask = df['Period'].isin(['Pre-P0', 'P0'])
                fitted_data = df[fitted_mask]['Fitted'].dropna()
                if not fitted_data.empty:
                    ax.plot(fitted_data.index, fitted_data.values, color='black', 
                           linestyle='-', label='Fitted', linewidth=2)
            
            # Plot each scenario (P0 and forecast periods)
            for i, scenario in enumerate(scenarios):
                if style and scenario in style:
                    scen_style = style[scenario]
                    if 'linewidth' not in scen_style:
                        scen_style['linewidth'] = 2
                else:
                    color = self.scenario_colors[i % len(self.scenario_colors)]
                    scen_style = {
                        'color': color,
                        'linestyle': '-',
                        'label': scenario,
                        'linewidth': 2
                    }
                
                # Plot scenario data for forecast periods
                forecast_mask = df['Period'].str.startswith('P')
                scenario_data = df[forecast_mask][scenario].dropna()
                if not scenario_data.empty:
                    ax.plot(scenario_data.index, scenario_data.values, **scen_style)
            
            # Add vertical line at P0
            p0_date = df[df['Period'] == 'P0'].index
            if len(p0_date) > 0:
                ax.axvline(x=p0_date[0], color='gray', linestyle='--', alpha=0.7)
                ax.text(p0_date[0], ax.get_ylim()[1], 'P0',
                       rotation=0, ha='center', va='bottom', fontsize=8)
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Customize plot
        title = f"{title_prefix}Scenario Forecast - {scen_set}" if title_prefix else f"Scenario Forecast - {scen_set}"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{scen_set.replace(' ', '_')}_forecast.png"
            fig.savefig(os.path.join(save_path, filename))
        
        return fig

    def plot_scenario_variables(
        self,
        scen_set: str,
        scen_dict: Dict[str, pd.DataFrame],
        title_prefix: str = "",
        save_path: Optional[str] = None,
        subplot_width: float = 5.0,
        subplot_height: float = 3.5
    ) -> plt.Figure:
        """
        Plot individual variables across scenarios for a single scenario set.
        
        Creates a grid plot showing each non-dummy variable across all scenarios.
        Each subplot shows one variable with different scenario lines starting 
        from P0 onwards (no Pre-P0 periods included).
        
        Parameters
        ----------
        scen_set : str
            Name of the scenario set being plotted
        scen_dict : Dict[str, pd.DataFrame]
            Dictionary mapping scenario names to their feature DataFrames
        title_prefix : str, default=""
            Prefix to add to plot titles
        save_path : str, optional
            If provided, saves plots to this directory path
        subplot_width : float, default=5.0
            Standard width for each subplot in inches
        subplot_height : float, default=3.5
            Standard height for each subplot in inches
            
        Returns
        -------
        plt.Figure
            The plot figure for the scenario set
            
        Notes
        -----
        - Only plots non-dummy variables (excludes DumVar features)
        - Each variable is plotted from P0 onwards only
        - Uses same color scheme as plot_forecasts method
        - Grid layout with maximum 3 plots per row
        - Each subplot has standardized size regardless of grid layout
        """
        # Get scenario feature matrices
        X_scens = self.X_scens
        
        # Store figures for return
        figures = {}
        
        # Get all variable names from the first scenario
        first_scenario = list(scen_dict.keys())[0]
        all_variables = list(scen_dict[first_scenario].columns)
        
        # Filter out dummy variables by checking if they match DumVar patterns
        non_dummy_vars = []
        specs_list = self.specs if isinstance(self.specs, list) else [self.specs]
        
        # Get dummy variable patterns from specs
        dummy_patterns = set()
        for spec in specs_list:
            if isinstance(spec, DumVar):
                # Get the column names that would be generated by this DumVar
                dummy_cols = spec.get_feature_names() if hasattr(spec, 'get_feature_names') else []
                dummy_patterns.update(dummy_cols)
            elif isinstance(spec, list):
                for sub_spec in spec:
                    if isinstance(sub_spec, DumVar):
                        dummy_cols = sub_spec.get_feature_names() if hasattr(sub_spec, 'get_feature_names') else []
                        dummy_patterns.update(dummy_cols)
        
        # Filter variables - exclude dummy variables and common non-feature columns
        for var in all_variables:
            # Skip if it's a dummy variable pattern (starts with common dummy prefixes)
            if any(var.startswith(prefix) for prefix in ['M:', 'Q:', 'D:', 'W:']):
                continue
            # Skip if it matches known dummy patterns
            if var in dummy_patterns:
                continue
            # Skip common non-feature columns
            if var in ['Period', 'Fitted']:
                continue
            non_dummy_vars.append(var)
        
        if not non_dummy_vars:
            print(f"No non-dummy variables found for scenario set '{scen_set}'")
            return plt.figure()  # Return empty figure
        
        # Calculate grid dimensions (max 3 columns)
        n_vars = len(non_dummy_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
        
        # Calculate figure size based on standardized subplot size
        fig_width = n_cols * subplot_width
        fig_height = n_rows * subplot_height
        
        # Create figure with subplots using calculated size
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Get sorted scenario names for consistent coloring
        scenarios = sorted(scen_dict.keys())
        
        # Plot each variable
        for i, var in enumerate(non_dummy_vars):
            ax = axes[i]
            
            # Plot each scenario for this variable
            for j, scenario in enumerate(scenarios):
                X_scenario = scen_dict[scenario]
                
                # Filter data to P0 onwards only
                if isinstance(self.dm._internal_loader, PanelLoader):
                    # For panel data, filter based on date column
                    date_col = self.dm._internal_loader.date_col
                    if date_col in X_scenario.columns:
                        mask = X_scenario[date_col] >= self.P0
                        X_filtered = X_scenario[mask]
                        if var in X_filtered.columns:
                            dates = pd.to_datetime(X_filtered[date_col]).normalize()
                            values = X_filtered[var]
                            # Group by date and take mean for panel data
                            data_series = pd.Series(values.values, index=dates).groupby(level=0).mean()
                        else:
                            continue
                    else:
                        continue
                else:  # TimeSeriesLoader
                    # For time series, filter based on index
                    mask = X_scenario.index >= self.P0
                    X_filtered = X_scenario[mask]
                    if var in X_filtered.columns:
                        data_series = X_filtered[var]
                        data_series.index = pd.to_datetime(data_series.index).normalize()
                    else:
                        continue
                
                # Plot the series
                if not data_series.empty:
                    color = self.scenario_colors[j % len(self.scenario_colors)]
                    ax.plot(data_series.index, data_series.values,
                           color=color, linestyle='-', label=scenario, linewidth=2)
            
            # Add vertical line at P0
            ax.axvline(x=self.P0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize subplot
            ax.set_title(f'{var}', fontsize=10, pad=15)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Format X-axis (simplified for subplots)
            from matplotlib.dates import DateFormatter, YearLocator
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        # Overall title and layout
        title = f"{title_prefix}Scenario Variables - {scen_set}" if title_prefix else f"Scenario Variables - {scen_set}"
        fig.suptitle(title, fontsize=12, y=0.98)
        
        # Adjust spacing between subplots for better visual appeal
        plt.subplots_adjust(
            left=0.06,      # Left margin (reduced since no y-axis labels)
            bottom=0.08,    # Bottom margin (reduced since no x-axis labels)
            right=0.96,     # Right margin (reduced since no y-axis labels)
            top=0.80,       # Top margin (leave more room for main title above subplots)
            wspace=0.20,    # Width spacing between subplots (reduced)
            hspace=0.40     # Height spacing between subplots (reduced since main title is now properly above)
        )
        
        # Save figure if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{scen_set.replace(' ', '_')}_variables.png"
            fig.savefig(os.path.join(save_path, filename))
        
        return fig

    def plot_all(
        self,
        figsize: tuple = (8, 4),
        style: Optional[Dict[str, Dict[str, Any]]] = None,
        title_prefix: str = "",
        save_path: Optional[str] = None,
        subplot_width: float = 5.0,
        subplot_height: float = 3.5,
        show_qtr: bool = True
    ) -> Dict[str, Dict[str, plt.Figure]]:
        """
        Plot both forecasts and scenario variables for all scenario sets.
        
        This method runs plot_forecasts() and plot_scenario_variables() for each
        scenario set, providing a comprehensive view of all scenario analysis.
        
        Parameters
        ----------
        figsize : tuple, default=(8, 4)
            Figure size for forecast plots as (width, height)
        style : Dict[str, Dict[str, Any]], optional
            Styling options for each scenario in forecast plots
        title_prefix : str, default=""
            Prefix to add to plot titles
        save_path : str, optional
            If provided, saves plots to this directory path
        subplot_width : float, default=5.0
            Standard width for each subplot in variable plots
        subplot_height : float, default=3.5
            Standard height for each subplot in variable plots
        show_qtr : bool, default=True
            If True, use quarterly frequency data (forecast_y_qtr_df).
            If False, use original frequency data (forecast_y_df).
            
        Returns
        -------
        Dict[str, Dict[str, plt.Figure]]
            Nested dictionary with structure:
            {scenario_set: {'forecasts': forecast_fig, 'variables': variables_fig}}
            
        Notes
        -----
        - Creates both forecast and variable plots for each scenario set
        - Handles both time series and panel data appropriately
        - Uses consistent color schemes across all plots
        """
        # Get forecast DataFrames and scenario feature matrices
        forecast_dfs = self.forecast_y_qtr_df if show_qtr else self.forecast_y_df
        X_scens = self.X_scens
        
        # Store all figures for return
        all_figures = {}
        
        for scen_set in forecast_dfs.keys():
            all_figures[scen_set] = {}
            
            # Create forecast plot
            forecast_fig = self.plot_forecasts(
                scen_set=scen_set,
                forecast_data=forecast_dfs[scen_set],
                figsize=figsize,
                style=style,
                title_prefix=title_prefix,
                save_path=save_path,
                show_qtr=show_qtr
            )
            all_figures[scen_set]['forecasts'] = forecast_fig
            
            # Create variable plots if scenario data exists
            if scen_set in X_scens:
                variables_fig = self.plot_scenario_variables(
                    scen_set=scen_set,
                    scen_dict=X_scens[scen_set],
                    title_prefix=title_prefix,
                    save_path=save_path,
                    subplot_width=subplot_width,
                    subplot_height=subplot_height
                )
                all_figures[scen_set]['variables'] = variables_fig
        
        return all_figures