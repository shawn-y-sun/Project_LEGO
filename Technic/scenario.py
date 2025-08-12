import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import matplotlib.pyplot as plt
import os

from .internal import TimeSeriesLoader, PanelLoader
from .data import DataManager
from .model import ModelBase
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
    model : ModelBase
        Fitted ModelBase instance (with .predict, dm, specs, and target attributes)
    horizon : int, default=12
        Number of quarters to forecast after P0 (e.g., 9 or 12 quarters)
    qtr_method : str, default='mean'
        Method to aggregate monthly results to quarterly frequency.
        Options: 'mean', 'sum', 'end' (quarter-end value). If 'avg' is
        provided, it's treated as 'mean'.

    Attributes
    ----------
    horizon_frame : Tuple[pd.Timestamp, pd.Timestamp]
        Prediction horizon time frame as (start_date, end_date) tuple
    y_scens : Dict[str, Dict[str, pd.Series]]
        Nested forecast results for all scenarios
    scenarios : List[str]
        List of scenario set keys
    scenario_colors : List[str]
        Color scheme for plotting scenarios in order

    Example
    -------
    # Create ScenManager for a fitted model with 9-quarter forecast horizon
    scen_mgr = ScenManager(model, horizon=9)
    
    # Create ScenManager with sum aggregation for quarterly reporting
    scen_mgr = ScenManager(model, horizon=9, qtr_method='sum')
    
    # Access scenario forecasts
    forecasts = scen_mgr.y_scens
    """
    
    # Class attribute for consistent scenario colors across all plotting methods
    scenario_colors = ['orange', 'grey', 'dodgerblue', 'purple', 'brown', 'pink', 'olive', 'cyan']
    def __init__(
        self,
        model: ModelBase,
        horizon: int = 12,
        qtr_method: str = 'mean'
    ):
        # Reference parameters from ModelBase
        self.model = model
        self.dm = self.model.dm
        self.specs = self.model.specs
        self.target = self.model.target
        
        # Validate forecast horizon
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        
        # Validate quarterly aggregation method
        # Normalize and validate quarterly aggregation method
        if qtr_method == 'avg':
            qtr_method = 'mean'
        valid_qtr_methods = ['mean', 'sum', 'end']
        if qtr_method not in valid_qtr_methods:
            raise ValueError(f"qtr_method must be one of {valid_qtr_methods}")
        self.qtr_method = qtr_method
        
        # Get P0 from DataManager's internal loader
        self.P0 = self.dm.scen_p0
        if self.P0 is None:
            raise ValueError("Internal data loader must have scen_p0 set for scenario analysis.")
            
        # Calculate horizon end date (P0 + horizon quarters)
        self.horizon_end = self.P0 + pd.offsets.QuarterEnd(self.horizon)

    @property
    def horizon_frame(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the prediction horizon time frame as a tuple of start and end dates.
        
        The start date is the first available date after P0 from scenario MEV data,
        and the end date is the cutoff date after enough quarters are included 
        as specified by self.horizon.
        
        Returns
        -------
        Tuple[pd.Timestamp, pd.Timestamp]
            (start_date, end_date) where:
            - start_date: First available date after P0 from scenario data
            - end_date: Cutoff date after horizon quarters from P0
        """
        # Get the first scenario MEV dataframe to determine available date range
        if not self.dm.scen_mevs:
            raise ValueError("No scenario MEV data available to determine horizon frame.")
        
        # Get the first scenario set and first scenario
        first_scen_set = list(self.dm.scen_mevs.keys())[0]
        first_scen_name = list(self.dm.scen_mevs[first_scen_set].keys())[0]
        first_mev_df = self.dm.scen_mevs[first_scen_set][first_scen_name]
        
        # Determine start date based on data structure
        if isinstance(self.dm._internal_loader, PanelLoader):
            # For panel data, get dates from date column
            date_col = self.dm._internal_loader.date_col
            if date_col in first_mev_df.columns:
                available_dates = pd.to_datetime(first_mev_df[date_col])
            else:
                # Fallback to theoretical calculation
                start_date = self.P0 + pd.Timedelta(days=1)
                end_date = self.horizon_end
                return (start_date, end_date)
        else:  # TimeSeriesLoader
            # For time series, get dates from index
            if isinstance(first_mev_df.index, pd.DatetimeIndex):
                available_dates = first_mev_df.index
            else:
                # Fallback to theoretical calculation
                start_date = self.P0 + pd.Timedelta(days=1)
                end_date = self.horizon_end
                return (start_date, end_date)
        
        # Filter to dates after P0
        dates_after_p0 = available_dates[available_dates > self.P0]
        
        if dates_after_p0.empty:
            raise ValueError(f"No dates after P0 ({self.P0}) found in scenario data.")
        
        # Start date is the first available date after P0
        start_date = dates_after_p0.min()
        
        # End date is the horizon end date (theoretical calculation)
        end_date = self.horizon_end
        
        # Ensure end date doesn't exceed available data
        if dates_after_p0.max() < end_date:
            end_date = dates_after_p0.max()
        
        return (start_date, end_date)

    @property
    def sens_test(self):
        """
        Get a SensitivityTest instance initialized with this ScenManager.
        
        This property provides easy access to sensitivity testing capabilities
        for the current scenario manager and its associated model.
        
        Returns
        -------
        SensitivityTest
            SensitivityTest instance initialized with this ScenManager
            
        Example
        -------
        >>> scen_mgr = ScenManager(model, horizon=9)
        >>> sens_test = scen_mgr.sens_test
        >>> 
        >>> # Run parameter sensitivity testing
        >>> param_results = sens_test.y_param_shock
        >>> 
        >>> # Plot sensitivity results
        >>> sens_test.plot_all_param_shock()
        """
        from .sensitivity import SensitivityTest
        return SensitivityTest(self)

    @property
    def X_scens(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Build and return scenario feature matrices on demand.
        
        For models with lookback variables, uses rolling_predict() to generate
        features that account for conditional variable effects. For models without
        lookback variables, builds features using MEVs and internal data.

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
        
        # If model has lookback variables, use rolling_predict for all scenarios
        if self.model.has_lookback_var:
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
                    
                    # Use rolling_predict to get features for models with lookback variables
                    _, X_features = self.model.rolling_predict(
                        df_internal=internal_df,
                        df_mev=df_mev,
                        y=self.model.y_full,
                        time_frame=self.horizon_frame
                    )
                    
                    X_scens[scen_set][scen_name] = X_features.astype(float)
        else:
            # Original logic for models without lookback variables
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
                    
                    # Filter to forecast period using horizon_frame
                    start_date, end_date = self.horizon_frame
                    
                    if isinstance(self.dm._internal_loader, PanelLoader):
                        # For panel data, filter based on date column
                        date_col = self.dm._internal_loader.date_col
                        if date_col in X_full.columns:
                            X_filtered = X_full[
                                (X_full[date_col] >= start_date) & 
                                (X_full[date_col] <= end_date)
                            ].copy()
                        else:
                            X_filtered = X_full.copy()
                    else:  # TimeSeriesLoader
                        # For time series, filter based on DatetimeIndex
                        if isinstance(X_full.index, pd.DatetimeIndex):
                            X_filtered = X_full[
                                (X_full.index >= start_date) & 
                                (X_full.index <= end_date)
                            ].copy()
                        else:
                            X_filtered = X_full.copy()
                        
                    X_scens[scen_set][scen_name] = X_filtered.astype(float)
                
        return X_scens

    def forecast(
        self,
        X: pd.DataFrame
    ) -> pd.Series:
        """
        Forecast a single scenario using the model's prediction methods.
        
        This method leverages the model's .predict() and .rolling_predict() methods
        based on whether the model has lookback variables.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame for the scenario

        Returns
        -------
        pd.Series
            Series of predictions indexed like X, filtered to horizon_frame
        """
        # Check if model has lookback variables
        if hasattr(self.model, 'has_lookback_var') and self.model.has_lookback_var:
            # Use rolling_predict for models with lookback variables
            # Get scenario internal data and MEV data
            scen_internal = None
            scen_mev = None
            
            # Find the scenario data that corresponds to this X
            for scen_set, scen_dict in self.dm.scen_mevs.items():
                for scen_name, mev_df in scen_dict.items():
                    # Check if this MEV data corresponds to the X we're forecasting
                    # This is a simplified check - in practice, you might need more sophisticated matching
                    if hasattr(self.dm, 'scen_internal_data') and self.dm.scen_internal_data:
                        if scen_set in self.dm.scen_internal_data and scen_name in self.dm.scen_internal_data[scen_set]:
                            scen_internal = self.dm.scen_internal_data[scen_set][scen_name]
                            scen_mev = mev_df
                            break
                if scen_internal is not None:
                    break
            
            if scen_internal is None:
                # Fallback: use internal data and first available MEV
                scen_internal = self.dm.internal_data
                scen_mev = list(self.dm.scen_mevs.values())[0][list(self.dm.scen_mevs.values())[0].keys()][0]
            
            # Get time frame as string tuple for rolling_predict
            start_date, end_date = self.horizon_frame
            time_frame = (str(start_date), str(end_date))
            
            # Use rolling_predict
            predictions, _ = self.model.rolling_predict(
                df_internal=scen_internal,
                df_mev=scen_mev,
                y=self.dm.internal_data[self.target],
                time_frame=time_frame
            )
            
            return predictions
            
        else:
            # Use standard predict for models without lookback variables
            predictions = self.model.predict(X)
            
            # Filter to forecast period using horizon_frame
            start_date, end_date = self.horizon_frame
            
            if isinstance(self.dm._internal_loader, PanelLoader):
                # For panel data, filter based on date column
                date_col = self.dm._internal_loader.date_col
                if date_col in X.columns:
                    # Get the indices where dates are within horizon_frame
                    valid_idx = X[(X[date_col] >= start_date) & (X[date_col] <= end_date)].index
                    predictions = predictions[valid_idx]
            else:  # TimeSeriesLoader
                # For time series, filter based on DatetimeIndex
                if isinstance(predictions.index, pd.DatetimeIndex):
                    predictions = predictions[(predictions.index >= start_date) & (predictions.index <= end_date)]
                
            return predictions

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
        - Uses the model's .predict() method for standard forecasting
        - Uses the model's .rolling_predict() method for models with lookback variables
        - Automatically handles both time series and panel data
        """
        results: Dict[str, Dict[str, pd.Series]] = {}
        for scen_set, scen_dict in self.X_scens.items():
            results[scen_set] = {}
            for scen_name, X in scen_dict.items():
                # Use the simplified forecast method
                results[scen_set][scen_name] = self.forecast(X)
        return results

    @property
    def forecast_y_qtr_df(self) -> Dict[str, pd.DataFrame]:
        """Deprecated: quarterly aggregation for target forecasts is no longer exposed."""
        raise AttributeError("forecast_y_qtr_df has been removed. Use forecast_y_df for target (monthly) and forecast_y_base_qtr_df for base quarterly aggregation.")

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
            - Fitted_IS: In-sample fitted values from model
            - Pred_OOS: Out-of-sample predictions
            - Actual: Actual target values
            - One column per scenario containing forecast values
            
        Notes
        -----
        - For time series data, index is DatetimeIndex
        - For panel data, index includes both entity and date information
        - Periods before P0 are marked as 'Pre-P0'
        - P0 and subsequent periods are numbered sequentially (P0, P1, P2, etc.)
        - Period indicators are always quarterly, even for monthly data
        - All time indices are normalized to dates only (no time component)

        Example output structure
        ------------------------
        For a single scenario set (monthly frequency):
        ┌──────────────┬─────────┬───────────┬──────────┬────────┬────────┐
        │ Index (Date) │ Period  │ Fitted_IS │ Pred_OOS │ Actual │ Base   │
        ├──────────────┼─────────┼───────────┼──────────┼────────┼────────┤
        │ 2024-12-31   │ P0      │ 1.02      │ NaN      │ 1.01   │ 1.01   │
        │ 2025-01-31   │ P1      │ NaN       │ 1.05     │ NaN    │ 1.05   │
        │ 2025-02-28   │ P1      │ NaN       │ 1.06     │ NaN    │ 1.06   │
        └──────────────┴─────────┴───────────┴──────────┴────────┴────────┘
        """
        # Get scenario forecasts
        scen_results = self.y_scens
        
        # Collect actual, fitted (IS), and OOS predictions
        actual_values = getattr(self.model, 'y_full', None)
        fitted_values = getattr(self.model, 'y_fitted_in', None)
        pred_oos = getattr(self.model, 'y_pred_out', None)
        if actual_values is not None:
            actual_values = actual_values.copy()
            actual_values.index = pd.to_datetime(actual_values.index).normalize()
        if fitted_values is not None:
            fitted_values = fitted_values.copy()
            fitted_values.index = pd.to_datetime(fitted_values.index).normalize()
        if pred_oos is not None and not pred_oos.empty:
            pred_oos = pred_oos.copy()
            pred_oos.index = pd.to_datetime(pred_oos.index).normalize()
        
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
                
                # Add Actual / Fitted_IS / Pred_OOS if available
                if actual_values is not None:
                    df['Actual'] = actual_values
                if fitted_values is not None:
                    df['Fitted_IS'] = fitted_values
                if pred_oos is not None and not pred_oos.empty:
                    df['Pred_OOS'] = pred_oos
                
                # Add scenario forecasts
                for scen_name, scen_series in scen_dict.items():
                    df[scen_name] = scen_series
                    
                # If monthly frequency, ensure we keep monthly detail (target stays monthly)
                # No additional quarterly aggregation for target here

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
                
                # Add Fitted_IS aligned by index if available
                if fitted_values is not None:
                    df['Fitted_IS'] = fitted_values
                # Add Actual aligned to date level
                if actual_values is not None:
                    actual_aligned = actual_values.reindex(pd.to_datetime(dates).normalize())
                    df['Actual'] = actual_aligned.values
                
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
                
                # Add Pred_OOS aligned to date level
                if pred_oos is not None and not pred_oos.empty:
                    pred_aligned = pred_oos.reindex(pd.to_datetime(df.index.get_level_values(date_col)).normalize())
                    df['Pred_OOS'] = pred_aligned.values
            
            # Store the result
            results[scen_set] = df
            
        return results

    @property
    def forecast_y_base_df(self) -> Dict[str, pd.DataFrame]:
        """
        Organize scenario base forecasting results into DataFrames with period indicators.
        Always aligned to original data frequency; when monthly, shows monthly base forecasts.

        Example output structure
        ------------------------
        For a single scenario set (monthly frequency):
        ┌──────────────┬─────────┬───────────┬──────────┬────────┬────────┐
        │ Index (Date) │ Period  │ Fitted_IS │ Pred_OOS │ Actual │ Base   │
        ├──────────────┼─────────┼───────────┼──────────┼────────┼────────┤
        │ 2024-12-31   │ P0      │ 100.0     │ NaN      │ 100.0  │ 100.0  │
        │ 2025-01-31   │ P1      │ NaN       │ 101.2    │ NaN    │ 101.2  │
        │ 2025-02-28   │ P1      │ NaN       │ 101.6    │ NaN    │ 101.6  │
        └──────────────┴─────────┴───────────┴──────────┴────────┴────────┘
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
                
                # Add fitted/actual/pred_oos if available
                if fitted_base_values is not None:
                    df['Fitted_IS'] = fitted_base_values
                # Actual base if available from model (y_base_full), else omit
                y_base_full = getattr(self.model, 'y_base_full', None)
                if y_base_full is not None and not y_base_full.empty:
                    df['Actual'] = y_base_full.reindex(df.index)
                y_base_pred_out = getattr(self.model, 'y_base_pred_out', None)
                if y_base_pred_out is not None and not y_base_pred_out.empty:
                    df['Pred_OOS'] = y_base_pred_out.reindex(df.index)
                
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
                
                # Add fitted/actual/pred_oos if available
                if fitted_base_values is not None:
                    df['Fitted_IS'] = fitted_base_values
                y_base_full = getattr(self.model, 'y_base_full', None)
                if y_base_full is not None and not y_base_full.empty:
                    df['Actual'] = y_base_full
                y_base_pred_out = getattr(self.model, 'y_base_pred_out', None)
                if y_base_pred_out is not None and not y_base_pred_out.empty:
                    df['Pred_OOS'] = y_base_pred_out
                
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
        When internal frequency is monthly, this converts monthly base forecasts to quarterly
        using the configured qtr_method ('mean', 'sum', 'end').

        Example output structure
        ------------------------
        For a single scenario set (quarterly aggregation):
        ┌──────────────┬─────────┬───────────┬──────────┬────────┬────────┐
        │ Index (Date) │ Period  │ Fitted_IS │ Pred_OOS │ Actual │ Base   │
        ├──────────────┼─────────┼───────────┼──────────┼────────┼────────┤
        │ 2024-12-31   │ 24-12   │ 100.0     │ NaN      │ 100.0  │ 100.0  │
        │ 2025-03-31   │ P0      │ 101.0     │ NaN      │ NaN    │ 101.0  │
        │ 2025-06-30   │ P1      │ NaN       │ 102.3    │ NaN    │ 102.3  │
        └──────────────┴─────────┴───────────┴──────────┴────────┴────────┘
        """
        # Get base scenario forecasts
        base_scen_results = self.y_base_scens
        
        if not base_scen_results:
            return {}
        
        # Get fitted base values and actual/pred_oos from model
        fitted_base_values = getattr(self.model, 'y_base_fitted_in', None)
        if fitted_base_values is not None:
            fitted_base_values = fitted_base_values.copy()
            fitted_base_values.index = pd.to_datetime(fitted_base_values.index).normalize()
        y_base_full = getattr(self.model, 'y_base_full', None)
        if y_base_full is not None and not y_base_full.empty:
            y_base_full = y_base_full.copy()
            y_base_full.index = pd.to_datetime(y_base_full.index).normalize()
        y_base_pred_out = getattr(self.model, 'y_base_pred_out', None)
        if y_base_pred_out is not None and not y_base_pred_out.empty:
            y_base_pred_out = y_base_pred_out.copy()
            y_base_pred_out.index = pd.to_datetime(y_base_pred_out.index).normalize()
        
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
                # Avoid rendering artificial zeros when a quarter has only NaNs
                try:
                    # pandas >= 1.1 supports min_count
                    result = quarterly_grouped.sum(min_count=1)
                except TypeError:
                    # Fallback: explicit apply to ensure NaN if all values are NaN
                    result = quarterly_grouped.apply(lambda s: s.sum() if s.notna().any() else np.nan)
            elif qtr_method == 'end':
                # Use last value in quarter (quarter-end)
                result = quarterly_grouped.last()
            else:
                raise ValueError(f"Unsupported aggregation method: {qtr_method}")
            
            # Convert index to quarter-end dates
            result.index = result.index.to_period('Q').to_timestamp(how='end').normalize()
            return result
        
        results: Dict[str, pd.DataFrame] = {}
        for scen_set, scen_dict in base_scen_results.items():
            # Collect all data first
            all_data = {}
            
            # Build a combined series for Fitted_IS (IS window) and Pred_OOS (OOS window)
            combined_series = None
            if fitted_base_values is not None:
                combined_series = fitted_base_values.copy()
            if y_base_pred_out is not None and not y_base_pred_out.empty:
                if combined_series is None:
                    combined_series = y_base_pred_out.copy()
                else:
                    # Prefer OOS where available
                    combined_series = combined_series.combine_first(y_base_pred_out)
            # Aggregate combined series to quarterly for partitioning
            if combined_series is not None:
                combined_quarterly = aggregate_to_quarterly(combined_series, self.qtr_method)
                # Separately aggregate IS and OOS for mask
                fitted_quarterly = aggregate_to_quarterly(fitted_base_values, self.qtr_method) if fitted_base_values is not None else pd.Series(dtype=float)
                pred_oos_quarterly = aggregate_to_quarterly(y_base_pred_out, self.qtr_method) if y_base_pred_out is not None and not y_base_pred_out.empty else pd.Series(dtype=float)
            else:
                combined_quarterly = pd.Series(dtype=float)
                fitted_quarterly = pd.Series(dtype=float)
                pred_oos_quarterly = pd.Series(dtype=float)
            
            # Process scenario base forecasts
            for scen_name, scen_series in scen_dict.items():
                # Normalize scenario series index
                scen_series.index = pd.to_datetime(scen_series.index).normalize()
                scen_quarterly = aggregate_to_quarterly(scen_series, self.qtr_method)
                all_data[scen_name] = scen_quarterly
            
            # Get all unique quarter-end dates (include IS/OOS so fitted appears on the plot)
            all_quarters = pd.Index([])
            for series in all_data.values():
                all_quarters = all_quarters.union(series.index)
            if not combined_quarterly.empty:
                all_quarters = all_quarters.union(combined_quarterly.index)
            if not y_base_full is None and not y_base_full.empty:
                # Include actuals' quarterly index to keep alignment if needed
                all_quarters = all_quarters.union(
                    aggregate_to_quarterly(y_base_full, self.qtr_method).index
                )
            all_quarters = all_quarters.sort_values()
            
            # Create quarterly DataFrame
            df = pd.DataFrame(index=all_quarters)
            
            # Add core series
            df['Fitted_IS'] = np.nan
            df['Pred_OOS'] = np.nan
            if not combined_quarterly.empty:
                # Initialize Fitted_IS with fitted_quarterly where available
                if not fitted_quarterly.empty:
                    df.loc[fitted_quarterly.index, 'Fitted_IS'] = pd.to_numeric(fitted_quarterly, errors='coerce')
                # Fill Pred_OOS with pred_oos_quarterly
                if not pred_oos_quarterly.empty:
                    df.loc[pred_oos_quarterly.index, 'Pred_OOS'] = pd.to_numeric(pred_oos_quarterly, errors='coerce')
                # Where Pred_OOS is present, clear Fitted_IS to avoid overlap
                df.loc[df['Pred_OOS'].notna(), 'Fitted_IS'] = np.nan
            # Add scenario series
            for col_name, series in all_data.items():
                df[col_name] = pd.to_numeric(series, errors='coerce')
            
            # Get P0 quarter-end date
            p0_quarter_end = get_quarter_end(self.P0)
            
            # Create period labels as a column; keep index as actual dates
            df['Period'] = [assign_period_label(date, p0_quarter_end) for date in df.index]
            # Add Actual quarterly if available
            if y_base_full is not None and not y_base_full.empty:
                actual_q = aggregate_to_quarterly(y_base_full, self.qtr_method).reindex(all_quarters)
                df['Actual'] = pd.to_numeric(actual_q, errors='coerce')
            
            results[scen_set] = df
            
        return results

    def plot_forecasts(
        self,
        scen_set: str,
        forecast_data: Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]] = None,
        figsize: tuple = (8, 4),
        style: Optional[Dict[str, Dict[str, Any]]] = None,
        title_prefix: str = "",
        save_path: Optional[str] = None,
        show_qtr: bool = False,
        show_actual: bool = False
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
        forecast_data : Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]], optional
            DataFrame containing forecast data, or tuple of (target_df, base_df) for side-by-side plots.
            If None, will use either forecast_y_qtr_df or forecast_y_df based on show_qtr parameter.
            If tuple is provided, creates side-by-side plots: left for target variable, right for base variable.
        figsize : tuple, default=(8, 4)
            Figure size as (width, height). For side-by-side plots, width is automatically doubled.
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
        show_actual : bool, default=False
            If True, draw the Actual line on plots; otherwise do not include Actual
            
        Returns
        -------
        plt.Figure
            The plot figure for the scenario set
            
        Notes
        -----
        - Creates new figures without affecting existing plots
        - Works with both quarterly and original frequency data
        - Handles both YY-MM format and 'Pre-P0' period indicators
        - Supports side-by-side plotting when forecast_data is a tuple
        """
        # Get required DataFrames: always original frequency on first row
        if forecast_data is None:
            target_df = self.forecast_y_df[scen_set].copy()
            base_df = self.forecast_y_base_df.get(scen_set)
            base_qtr_df = None
            # Show quarterly base row whenever requested and available
            if show_qtr:
                base_qtr_df = self.forecast_y_base_qtr_df.get(scen_set)
        else:
            # Backward-compat: if provided, infer left/right
            if isinstance(forecast_data, tuple):
                target_df, base_df = forecast_data
            else:
                target_df, base_df = forecast_data, None
            base_qtr_df = None

        # Build subplot grid
        add_quarterly_row = base_qtr_df is not None and not base_qtr_df.empty
        nrows = 2 if add_quarterly_row else 1
        ncols = 2
        fig_width = figsize[0] * ncols
        fig_height = figsize[1] * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        if nrows == 1:
            axes_flat = np.ravel(axes)
            ax_target, ax_base = axes_flat[0], axes_flat[1]
        else:
            ax_target = axes[0, 0]
            ax_base = axes[0, 1]
            ax_empty = axes[1, 0]
            ax_qtr = axes[1, 1]
            ax_empty.set_visible(False)

        def plot_df(ax, df: pd.DataFrame, title: str):
            if df is None or df.empty:
                ax.set_visible(False)
                return
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df = df.copy()
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass
            # Plot Actual
            if show_actual and 'Actual' in df.columns:
                actual_series = pd.to_numeric(df['Actual'], errors='coerce').dropna()
                if not actual_series.empty:
                    ax.plot(actual_series.index, actual_series.values, color='red', linestyle='-', linewidth=2, alpha=0.3, label='Actual')
            # Plot Fitted_IS
            fitted_color = 'black'
            if 'Fitted_IS' in df.columns:
                fitted_series = pd.to_numeric(df['Fitted_IS'], errors='coerce').dropna()
                if not fitted_series.empty:
                    ax.plot(fitted_series.index, fitted_series.values, color=fitted_color, linestyle='-', linewidth=2, label='Fitted_IS')
            # Plot Pred_OOS (same color as Fitted_IS, dashed)
            if 'Pred_OOS' in df.columns:
                pred_series = pd.to_numeric(df['Pred_OOS'], errors='coerce').dropna()
                if not pred_series.empty:
                    ax.plot(pred_series.index, pred_series.values, color=fitted_color, linestyle='--', linewidth=2, label='Pred_OOS')
            # Plot scenarios
            scenario_cols = [c for c in df.columns if c not in {'Period', 'Fitted_IS', 'Pred_OOS', 'Actual'}]
            for i, scenario in enumerate(sorted(scenario_cols)):
                series = pd.to_numeric(df[scenario], errors='coerce').dropna()
                if series.empty:
                    continue
                if style and scenario in style:
                    scen_style = style[scenario]
                    scen_style.setdefault('linewidth', 2)
                else:
                    color = self.scenario_colors[i % len(self.scenario_colors)]
                    scen_style = {'color': color, 'linestyle': '-', 'label': scenario, 'linewidth': 2}
                ax.plot(series.index, series.values, **scen_style)
            # Axes formatting: YY-MM, auto monthly/quarterly/yearly density
            if isinstance(df.index, pd.DatetimeIndex):
                n = len(df.index)
                # Slightly denser ticks for left plot to improve readability
                if n > 90:
                    ax.xaxis.set_major_locator(YearLocator())
                elif n > 24:
                    ax.xaxis.set_major_locator(MonthLocator(bymonth=[3, 6, 9, 12]))
                else:
                    ax.xaxis.set_major_locator(MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(DateFormatter('%y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot first row
        plot_df(ax_target, target_df, f"{title_prefix}Target Variable" if title_prefix else "Target Variable")
        plot_df(ax_base, base_df, f"{title_prefix}Base Variable" if title_prefix else "Base Variable")
        # Plot second row quarterly base if requested
        if nrows == 2:
            plot_df(ax_qtr, base_qtr_df, f"{title_prefix}Base Variable (Quarterly)" if title_prefix else "Base Variable (Quarterly)")
        
        # Set overall title for dual plots with variable names
        target_name = getattr(self, 'target', getattr(self.model, 'target', 'Target'))
        base_name = getattr(self.model, 'target_base', None)
        if not base_name:
            base_name = 'Base'
        base_fragment = f" | Base Variable: {base_name}" if base_name else ""
        title_core = f"Scenario Forecast - {scen_set} | Target Variable: {target_name}{base_fragment}"
        overall_title = f"{title_prefix}{title_core}" if title_prefix else title_core
        fig.suptitle(overall_title, fontsize=12)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
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
            
            # Format X-axis: always show quarter-end ticks for scenario variables
            from matplotlib.dates import DateFormatter, MonthLocator
            ax.xaxis.set_major_locator(MonthLocator(bymonth=[3, 6, 9, 12]))
            ax.xaxis.set_major_formatter(DateFormatter('%y-%m'))
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
        show_qtr: bool = True,
        show_actual: bool = False
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
        show_actual : bool, default=False
            If True, draw the Actual line on plots; otherwise do not include Actual
            
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
        # Target forecasts are always shown in original frequency (monthly if M, quarterly if Q)
        forecast_dfs = self.forecast_y_df
        # Base forecasts for side-by-side when not showing quarterly row
        base_forecast_dfs = self.forecast_y_base_df
        X_scens = self.X_scens
        
        # Store all figures for return
        all_figures = {}
        
        for scen_set in forecast_dfs.keys():
            all_figures[scen_set] = {}
            
            # Prepare forecast data
            if show_qtr:
                # Let plot_forecasts compute both monthly and quarterly views
                forecast_data = None
            else:
                target_df = forecast_dfs[scen_set]
                base_df = base_forecast_dfs.get(scen_set) if base_forecast_dfs else None
                if base_df is not None and not base_df.empty:
                    forecast_data = (target_df, base_df)
                else:
                    forecast_data = target_df
            
            # Create forecast plot
            forecast_fig = self.plot_forecasts(
                scen_set=scen_set,
                forecast_data=forecast_data,
                figsize=figsize,
                style=style,
                title_prefix=title_prefix,
                save_path=save_path,
                show_qtr=show_qtr,
                show_actual=show_actual
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