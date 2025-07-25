# =============================================================================
# module: data.py
# Purpose: Manage and combine internal and MEV data for modeling
# Dependencies: pandas, typing, DataLoader, MEVLoader, TSFM, CondVar, DumVar
# =============================================================================
import os
from pathlib import Path
import pandas as pd
import warnings
import yaml
import math
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import numpy as np
from scipy.interpolate import CubicSpline

from .internal import *
from .mev import MEVLoader
from .transform import TSFM
from .feature import Feature
from . import transform as transform_module
from .condition import CondVar
import inspect
import functools

warnings.simplefilter(action="ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# DataManager class
# ----------------------------------------------------------------------------

class DataManager:
    """
    Manage and combine internal and MEV data for modeling.

    The DataManager class serves as a central hub for managing and combining data from
    different sources (internal data and MEV data). It provides functionality for:
    - Accessing the latest data from loaders
    - Interpolating quarterly MEV data to monthly frequency
    - Building features from specifications
    - Applying transforms to data
    - Managing in-sample/out-of-sample splits
    - Refreshing or replacing data loaders

    Parameters
    ----------
    internal_loader : DataLoader
        Pre-loaded DataLoader instance with internal data. This loader should already
        have data loaded and sample splits defined.
    mev_loader : MEVLoader
        Pre-loaded MEVLoader instance with MEV data. This loader should already have
        both model and scenario MEV data loaded.

    Examples
    --------
    Basic Usage:
    >>> # Initialize loaders
    >>> internal_loader = TimeSeriesLoader(freq='M')
    >>> internal_loader.load(source='internal_data.csv', date_col='date')
    >>> mev_loader = MEVLoader()
    >>> mev_loader.load()
    >>> 
    >>> # Create DataManager
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> 
    >>> # Access data
    >>> internal_data = dm.internal_data
    >>> model_mev = dm.model_mev
    >>> scenarios = dm.scen_mevs
    >>> 
    >>> # Refresh data after loader updates
    >>> dm.refresh()
    >>> 
    >>> # Or replace loaders entirely
    >>> new_internal = TimeSeriesLoader(freq='M')
    >>> new_internal.load(source='updated_data.csv', date_col='date')
    >>> dm.refresh(internal_loader=new_internal)

    Building Features:
    >>> # Simple feature from raw variables
    >>> features = dm.build_features(['GDP', 'UNRATE'])
    >>> 
    >>> # Using transforms
    >>> from .transform import TSFM, diff, pct_change
    >>> specs = [
    ...     TSFM('GDP', diff),           # First difference of GDP
    ...     TSFM('UNRATE', pct_change),  # Percent change in unemployment
    ...     'CPI'                        # Raw CPI values
    ... ]
    >>> features = dm.build_features(specs)

    Applying Functions to Data:
    >>> # Add a new column to internal data
    >>> def add_gdp_growth(df):
    ...     df['GDP_growth'] = df['GDP'].pct_change()
    ...     return None  # In-place modification
    >>> dm.apply_to_internal(add_gdp_growth)
    >>> 
    >>> # Add features to MEV data
    >>> def add_mev_features(mev_df, internal_df):
    ...     mev_df['GDP_to_UNRATE'] = mev_df['GDP'] / mev_df['UNRATE']
    ...     return mev_df
    >>> dm.apply_to_mevs(add_mev_features)

    Working with Transforms:
    >>> # Generate transform specifications for variables
    >>> specs = dm.build_tsfm_specs(
    ...     specs=['GDP', 'UNRATE'],
    ...     max_lag=2,        # Include up to 2 lags
    ...     max_periods=3     # For transforms that take periods parameter
    ... )
    >>> # Results in transforms like:
    >>> # GDP: [GDP, diff(GDP), diff(GDP,2), lag(GDP,1), lag(GDP,2)]
    >>> # UNRATE: [UNRATE, pct_change(UNRATE), lag(UNRATE,1), lag(UNRATE,2)]

    Notes
    -----
    - The DataManager maintains caches for all data to improve performance and isolation
    - All data modifications through apply_* methods are made to cached data, not loaders
    - The class provides dynamic access to cached data, ensuring consistency
    - Sample splits are managed by the internal_loader and accessed through properties
    - Use refresh() to update cached data after loader modifications or to replace loaders

    See Also
    --------
    DataLoader : Base class for loading internal data
    MEVLoader : Class for loading and managing MEV data
    TSFM : Transform wrapper for feature engineering
    """
    def __init__(
        self,
        internal_loader: DataLoader,
        mev_loader: MEVLoader,
    ):
        # Store loaders
        self._internal_loader = internal_loader
        self._mev_loader = mev_loader

        # Cache for interpolated MEV data
        self._mev_cache: Dict[str, pd.DataFrame] = {}
        self._scen_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Cache for data copies that can be modified
        self._internal_data_cache: Optional[pd.DataFrame] = None
        self._scen_internal_data_cache: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        self._model_mev_cache: Optional[pd.DataFrame] = None
        self._scen_mevs_cache: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        
        # Frequency cache
        self._freq_cache: Optional[str] = None
        
        # Check if both monthly and quarterly MEVs exist
        if not (self._mev_loader.model_mev_mth.empty or self._mev_loader.model_mev_qtr.empty):
            overlap_mevs = set(self._mev_loader.model_mev_mth.columns) & set(self._mev_loader.model_mev_qtr.columns)
            if overlap_mevs:
                warnings.warn(
                    "Both monthly and quarterly MEVs detected with overlapping codes: "
                    f"{sorted(overlap_mevs)}. For monthly frequency data, interpolated "
                    "quarterly values will be suffixed with '_Q'.",
                    UserWarning
                )

    def refresh(
        self,
        internal_loader: Optional[DataLoader] = None,
        mev_loader: Optional[MEVLoader] = None
    ) -> None:
        """
        Refresh cached data from loaders or replace loaders entirely.
        
        This method serves two purposes:
        1. Clear cached data to force reloading from existing loaders
        2. Replace one or both loaders with new instances
        
        Use this method when:
        - Loaders have been updated with new data
        - You want to switch to different loaders
        - You need to ensure cached data is up-to-date
        
        Parameters
        ----------
        internal_loader : DataLoader, optional
            New internal data loader to replace existing one.
            If None, keeps existing loader but clears caches.
        mev_loader : MEVLoader, optional
            New MEV loader to replace existing one.
            If None, keeps existing loader but clears caches.
            
        Examples
        --------
        >>> # Refresh data from existing loaders
        >>> dm.refresh()
        >>> 
        >>> # Replace internal loader only
        >>> new_internal = TimeSeriesLoader(freq='M')
        >>> new_internal.load(source='new_data.csv', date_col='date')
        >>> dm.refresh(internal_loader=new_internal)
        >>> 
        >>> # Replace both loaders
        >>> new_mev = MEVLoader()
        >>> new_mev.load(source='new_mevs.xlsx')
        >>> dm.refresh(
        ...     internal_loader=new_internal,
        ...     mev_loader=new_mev
        ... )
        """
        # Update loaders if new ones provided
        if internal_loader is not None:
            self._internal_loader = internal_loader
        if mev_loader is not None:
            self._mev_loader = mev_loader

        # Clear all caches to force reloading
        self._mev_cache.clear()
        self._scen_cache.clear()
        self._internal_data_cache = None
        self._scen_internal_data_cache = None
        self._model_mev_cache = None
        self._scen_mevs_cache = None
        self._freq_cache = None

    @property
    def internal_data(self) -> pd.DataFrame:
        """
        Get the cached internal data, creating a copy from the loader if needed.

        Returns
        -------
        pd.DataFrame
            Cached internal data. Any modifications made through
            apply_to_internal() will be reflected in this data.

        Example
        -------
        >>> internal = dm.internal_data
        >>> print(f"Available variables: {internal.columns.tolist()}")
        >>> print(f"Date range: {internal.index.min()} to {internal.index.max()}")
        """
        if self._internal_data_cache is None:
            self._internal_data_cache = self._internal_loader.internal_data.copy()
        return self._internal_data_cache

    @property
    def scen_internal_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get the cached scenario internal data, creating copies from the loader if needed.

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Dictionary mapping scenario set names to scenario dictionaries.
            Each scenario DataFrame contains the scenario-specific internal data.

        Example
        -------
        >>> scenarios = dm.scen_internal_data
        >>> # Access base scenario data
        >>> if 'EWST2024' in scenarios and 'Base' in scenarios['EWST2024']:
        ...     base_data = scenarios['EWST2024']['Base']
        ...     print(f"Base scenario variables: {base_data.columns.tolist()}")
        """
        if self._scen_internal_data_cache is None:
            # Create deep copies of scenario internal data
            self._scen_internal_data_cache = {}
            for set_name, scen_dict in self._internal_loader.scen_internal_data.items():
                self._scen_internal_data_cache[set_name] = {}
                for scen_name, df in scen_dict.items():
                    self._scen_internal_data_cache[set_name][scen_name] = df.copy()
        return self._scen_internal_data_cache

    @property
    def freq(self) -> str:
        """
        Get the frequency of the internal data.
        
        Returns
        -------
        str
            Data frequency: 'M' for monthly, 'Q' for quarterly.
            The frequency is inferred from the internal data index once and cached.
        
        Example
        -------
        >>> freq = dm.freq
        >>> print(f"Data frequency: {freq}")  # 'M' or 'Q'
        """
        if self._freq_cache is None:
            # Infer frequency from internal data
            freq_str = pd.infer_freq(self.internal_data.index)
            if freq_str and freq_str.startswith('M'):
                self._freq_cache = "M"
            elif freq_str and freq_str.startswith('Q'):
                self._freq_cache = "Q"
            else:
                # Default to monthly if unable to determine
                self._freq_cache = "M"
        return self._freq_cache

    def _combine_mevs(self, qtr_data: pd.DataFrame, mth_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine quarterly and monthly MEV data based on internal data frequency.
        
        Parameters
        ----------
        qtr_data : pd.DataFrame
            Quarterly MEV data
        mth_data : pd.DataFrame
            Monthly MEV data
            
        Returns
        -------
        pd.DataFrame
            Combined MEV data in the appropriate frequency
            
        Notes
        -----
        If internal data is monthly:
            - Interpolates quarterly data to monthly
            - Appends to monthly data (if exists)
            - For overlapping MEVs only, uses monthly data and adds '_Q' suffix to interpolated quarterly
            
        If internal data is quarterly:
            - Computes quarterly averages of monthly data (if exists)
            - Appends to quarterly data
            - For overlapping MEVs only, adds '_M' suffix to monthly-derived columns
        """
        # Get internal data frequency
        is_monthly = self.freq == 'M'
        
        if is_monthly:
            # Monthly frequency case
            # First interpolate quarterly data
            mev_qtr_monthly = self._interpolate_df(qtr_data) if not qtr_data.empty else pd.DataFrame()
            
            if mth_data.empty:
                # If no monthly data, use interpolated quarterly data as-is (no suffix)
                return mev_qtr_monthly
            
            if mev_qtr_monthly.empty:
                return mth_data
            
            # Find overlapping columns
            overlap_cols = set(mev_qtr_monthly.columns) & set(mth_data.columns)
            non_overlap_cols = set(mev_qtr_monthly.columns) - set(mth_data.columns)
            
            # Start with monthly data
            result = mth_data.copy()
            
            # For overlapping columns, keep monthly data and add interpolated quarterly with '_Q' suffix
            for col in overlap_cols:
                result[f"{col}_Q"] = mev_qtr_monthly[col]
            
            # Add non-overlapping columns from quarterly data without suffix
            for col in non_overlap_cols:
                result[col] = mev_qtr_monthly[col]
            
            # Update MEV map only for overlapping columns that got a suffix
            if overlap_cols:
                derived_cols = [f"{col}_Q" for col in overlap_cols]
                self._update_mev_map_with_derived(derived_cols, '_Q')
            
            return result
            
        else:
            # Quarterly frequency case
            if mth_data.empty:
                return qtr_data
                
            # Convert monthly data to quarterly averages
            # First convert index to PeriodIndex for proper quarterly grouping
            mth_data = mth_data.copy()
            mth_data.index = pd.PeriodIndex(mth_data.index, freq='M')
            
            # Group by quarter and compute averages
            mth_quarterly = mth_data.groupby(mth_data.index.asfreq('Q')).mean()
            
            # Convert index to quarter-end timestamps and normalize to midnight
            mth_quarterly.index = mth_quarterly.index.to_timestamp(how='end').normalize()
            
            if qtr_data.empty:
                return mth_quarterly
            
            # Find overlapping columns
            overlap_cols = set(mth_quarterly.columns) & set(qtr_data.columns)
            non_overlap_cols = set(mth_quarterly.columns) - set(qtr_data.columns)
            
            # Start with quarterly data
            result = qtr_data.copy()
            
            # For overlapping columns, add monthly-derived with '_M' suffix
            for col in overlap_cols:
                result[f"{col}_M"] = mth_quarterly[col]
            
            # Add non-overlapping columns from monthly data without suffix
            for col in non_overlap_cols:
                result[col] = mth_quarterly[col]
            
            # Update MEV map only for overlapping columns that got a suffix
            if overlap_cols:
                derived_cols = [f"{col}_M" for col in overlap_cols]
                self._update_mev_map_with_derived(derived_cols, '_M')
            
            return result

    def _update_mev_map_with_derived(self, derived_cols: List[str], suffix: str) -> None:
        """
        Update the MEV map with derived MEV codes (those with _Q or _M suffix).
        
        Parameters
        ----------
        derived_cols : List[str]
            List of derived column names (with suffix)
        suffix : str
            The suffix used ('_Q' or '_M')
            
        Notes
        -----
        For each derived MEV:
        - Uses the same type as the original MEV
        - Adds a note to the description about the derivation method
        """
        mev_map = self._mev_loader._mev_map  # Access the underlying map directly
        
        for col in derived_cols:
            # Get original MEV code by removing suffix
            orig_code = col[:-len(suffix)]
            
            # Skip if original MEV not in map
            if orig_code not in mev_map:
                continue
                
            # Copy original MEV info
            orig_info = mev_map[orig_code].copy()
            
            # Add derivation note to description
            if suffix == '_Q':
                note = " (Interpolated from quarterly)"
            else:  # '_M'
                note = " (Averaged from monthly)"
            orig_info['description'] = orig_info['description'] + note
            
            # Add to MEV map
            mev_map[col] = orig_info

    @property
    def model_mev(self) -> pd.DataFrame:
        """
        Get the cached model MEV data, combining quarterly and monthly data appropriately.
        Creates a copy from the loader if needed and caches it for future modifications.

        The process depends on internal data frequency:
        
        For monthly internal data:
        1. Interpolates quarterly data to monthly frequency
        2. Combines with monthly data if available
        3. For overlapping MEVs, uses union of monthly data and interpolated quarterly
        
        For quarterly internal data:
        1. Computes quarterly averages of monthly data (complete quarters only)
        2. Combines with quarterly data
        3. For overlapping MEVs, adds '_M' suffix to monthly-derived columns
        
        Returns
        -------
        pd.DataFrame
            Cached combined MEV data matching internal data frequency.
            For monthly data: Includes both interpolated quarterly and raw monthly data.
            For quarterly data: Includes both raw quarterly and averaged monthly data.

        Example
        -------
        >>> mev = dm.model_mev
        >>> print("MEV variables:", mev.columns.tolist())
        >>> # Check if we have both quarterly and monthly versions
        >>> monthly_vars = [col for col in mev.columns if col.endswith('_M')]
        >>> print("Monthly-derived variables:", monthly_vars)
        """
        if self._model_mev_cache is None:
            # Get current data from loader
            current_qtr = self._mev_loader.model_mev_qtr
            current_mth = self._mev_loader.model_mev_mth
            
            # Combine the data based on frequency
            df = self._combine_mevs(current_qtr, current_mth)
            
            # Add month and quarter indicators
            df['M'] = df.index.month
            df['Q'] = df.index.quarter
            
            self._model_mev_cache = df
        
        return self._model_mev_cache

    @property
    def scen_mevs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get the cached scenario MEV data, combining quarterly and monthly data appropriately.
        Creates copies from the loader if needed and caches them for future modifications.

        The method maintains a three-level structure:
        {scenario_set: {scenario_name: DataFrame}}

        For example:
        - 'EWST2024': {'Base': df1, 'Adverse': df2}
        - 'GRST2024': {'Base': df3, 'Severe': df4}

        For each scenario DataFrame:
        - If internal data is monthly:
            * Interpolates quarterly data to monthly frequency
            * Combines with monthly data if available
            * For overlapping MEVs, uses monthly data and adds '_Q' suffix to interpolated quarterly
        - If internal data is quarterly:
            * Computes quarterly averages of monthly data (complete quarters only)
            * Combines with quarterly data
            * For overlapping MEVs, adds '_M' suffix to monthly-derived columns

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Nested dictionary of cached combined scenario data.
            Outer key: scenario set name (e.g., 'EWST2024')
            Inner key: scenario name (e.g., 'Base', 'Adverse')
            Value: DataFrame with combined MEV data matching internal data frequency

        Example
        -------
        >>> scenarios = dm.scen_mevs
        >>> # Access base scenario from EWST2024
        >>> if 'EWST2024' in scenarios and 'Base' in scenarios['EWST2024']:
        ...     base_ewst = scenarios['EWST2024']['Base']
        ...     print(f"EWST Base scenario range: {base_ewst.index.min()} to {base_ewst.index.max()}")
        >>> 
        >>> # Compare GDP across scenarios
        >>> for scen_name, scen_df in scenarios.get('EWST2024', {}).items():
        ...     print(f"{scen_name} GDP mean: {scen_df['GDP'].mean():.2f}")
        """
        if self._scen_mevs_cache is None:
            # Get current data from loader
            current_qtr = self._mev_loader.scen_mev_qtr
            current_mth = self._mev_loader.scen_mev_mth
            
            # Process each scenario set and scenario
            processed = {}
            
            # Get all unique scenario sets
            all_sets = set(current_qtr.keys()) | set(current_mth.keys() if current_mth else {})
            
            for scen_set in all_sets:
                processed[scen_set] = {}
                
                # Get quarterly and monthly data for this set
                qtr_dict = current_qtr.get(scen_set, {})
                mth_dict = current_mth.get(scen_set, {}) if current_mth else {}
                
                # Get all unique scenarios in this set
                all_scens = set(qtr_dict.keys()) | set(mth_dict.keys())
                
                for scen_name in all_scens:
                    # Get quarterly and monthly data for this scenario
                    qtr_df = qtr_dict.get(scen_name, pd.DataFrame())
                    mth_df = mth_dict.get(scen_name, pd.DataFrame())
                    
                    # Combine the data using the same method as model_mev
                    combined_df = self._combine_mevs(qtr_df, mth_df)
                    
                    # Add month and quarter indicators
                    combined_df['M'] = combined_df.index.month
                    combined_df['Q'] = combined_df.index.quarter
                    
                    processed[scen_set][scen_name] = combined_df
            
            self._scen_mevs_cache = processed
            
        return self._scen_mevs_cache

     # Modeling in‑sample/out‑of‑sample splits
    @property
    def internal_in(self) -> pd.DataFrame:
        """
        Get in-sample internal data using DataLoader's in_sample_idx.

        This property provides access to the training data subset based on
        the sample split defined in the internal_loader.

        Returns
        -------
        pd.DataFrame
            In-sample portion of internal data.

        Example
        -------
        >>> in_sample = dm.internal_in
        >>> print(f"Training data shape: {in_sample.shape}")
        >>> print(f"Training period: {in_sample.index.min()} to {in_sample.index.max()}")
        """
        return self.internal_data.loc[self._internal_loader.in_sample_idx]

    @property
    def internal_out(self) -> pd.DataFrame:
        """
        Get out-of-sample internal data using DataLoader's out_sample_idx.

        This property provides access to the testing/validation data subset
        based on the sample split defined in the internal_loader.

        Returns
        -------
        pd.DataFrame
            Out-of-sample portion of internal data.

        Example
        -------
        >>> out_sample = dm.internal_out
        >>> print(f"Testing data shape: {out_sample.shape}")
        >>> print(f"Testing period: {out_sample.index.min()} to {out_sample.index.max()}")
        """
        return self.internal_data.loc[self._internal_loader.out_sample_idx]
    
    def build_features(
        self,
        specs: List[Union[str, Feature]],
        internal_df: Optional[pd.DataFrame] = None,
        mev_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build feature DataFrame from specifications, which may include raw variable names
        (str) or Feature instances (TSFM, CondVar, DummyVar, etc.).

        This method combines features from both internal and MEV data sources,
        applying any specified transformations in the process. It handles both time series
        and panel data formats intelligently.

        Parameters
        ----------
        specs : list
            Each element can be:
            - str: A column name from either internal_data or model_mev
            - Feature: A feature object (TSFM, CondVar, etc.) that defines a transformation
            - list/tuple: Nested specs will be flattened
        internal_df : DataFrame, optional
            Override for internal data; defaults to self.internal_data
        mev_df : DataFrame, optional
            Override for model MEV; defaults to self.model_mev

        Returns
        -------
        DataFrame
            Combined features without entity and date columns, ready for model fitting.
            For time series data: Features indexed by date.
            For panel data: Features in the same order as the original data.

        Examples
        --------
        >>> # Time Series Data Example
        >>> features = dm.build_features(['GDP', 'UNRATE', 'CPI'])
        >>> 
        >>> # Panel Data Example
        >>> specs = [
        ...     'GDP',                     # MEV feature
        ...     'balance',                 # Internal feature
        ...     TSFM('GDP', diff),        # Transformed MEV
        ...     ('CPI', 'HOUSING')        # Group of MEV features
        ... ]
        >>> features = dm.build_features(specs)  # Returns only the specified features

        Notes
        -----
        - Features are built in the order specified
        - For panel data, MEV features are joined based on date alignment
        - All dates are normalized to midnight UTC
        - Missing values in raw variables are preserved
        - Transform features may introduce additional NaN values
        - The method flattens nested lists/tuples in specs
        - Entity and date columns are used internally for alignment but removed from final output
        """
        data_int = internal_df if internal_df is not None else self.internal_data
        data_mev = mev_df if mev_df is not None else self.model_mev

        # Determine if we're working with panel data
        is_panel = isinstance(self._internal_loader, PanelLoader)
        date_col = self._internal_loader.date_col if is_panel else None
        entity_col = self._internal_loader.entity_col if is_panel else None

        # Flatten nested spec lists and tuples
        def _flatten(items):
            for it in items:
                # treat tuples just like lists so group-tuples 
                # get unpacked into their member specs
                if isinstance(it, (list, tuple)):
                    yield from _flatten(it)
                else:
                    yield it
        flat_specs = list(_flatten(specs))
        
        # Initialize lists to collect features
        internal_pieces = []
        mev_pieces = []
        feature_pieces = []
        
        for spec in flat_specs:
            if isinstance(spec, Feature):
                # For TSFM instances, ensure frequency consistency
                if isinstance(spec, TSFM):
                    if spec.freq is None:
                        spec.freq = self.freq
                    elif spec.freq != self.freq:
                        warnings.warn(
                            f"TSFM instance for '{spec.var}' has frequency '{spec.freq}' "
                            f"but DataManager has frequency '{self.freq}'. "
                            f"Updating TSFM frequency to match DataManager.",
                            UserWarning
                        )
                        spec.freq = self.freq
                
                # For Features, we need to handle the result differently based on data type
                feature_result = spec.apply(data_int, data_mev)
                
                if is_panel:
                    # For panel data, we need to ensure we have the entity and date columns
                    if isinstance(feature_result, pd.Series):
                        # Convert Series to DataFrame
                        feature_result = feature_result.to_frame()
                    
                    if isinstance(feature_result, pd.DataFrame):
                        if date_col not in feature_result.columns:
                            # For panel data, we need to preserve the original entity-date structure
                            # Create a mapping DataFrame with entity and date columns
                            date_mapping = data_int[[entity_col, date_col]].copy()
                            # Add the feature result columns using the original index alignment
                            for col in feature_result.columns:
                                date_mapping[col] = feature_result[col].values
                            feature_result = date_mapping
                    feature_pieces.append(feature_result)
                else:
                    # For time series, just collect the result
                    feature_pieces.append(feature_result)
                    
            elif isinstance(spec, str):
                # Raw variable - collect in appropriate list
                if spec in data_int.columns:
                    if is_panel:
                        # For panel data, we need the entity/date cols temporarily for alignment
                        temp_df = data_int[[entity_col, date_col, spec]].copy()
                        internal_pieces.append(temp_df)
                    else:
                        internal_pieces.append(data_int[spec])
                elif spec in data_mev.columns:
                    if is_panel:
                        # For panel data, we'll need to merge MEV features later
                        mev_pieces.append(spec)
                    else:
                        mev_pieces.append(data_mev[spec])
                else:
                    raise KeyError(f"Feature '{spec}' not found in data sources.")
            else:
                raise TypeError(f"Invalid spec type after flatten(): {type(spec)}")

        # Combine features based on data type
        if is_panel:
            # For panel data, first combine internal features
            if internal_pieces:
                # Merge all internal pieces on entity and date columns
                result = pd.concat(internal_pieces, axis=1).drop_duplicates([entity_col, date_col])
            else:
                # Create empty DataFrame with entity and date columns
                result = data_int[[entity_col, date_col]].copy()

            # Add MEV features if any exist
            if mev_pieces:
                # Prepare MEV data - normalize index to midnight
                mev_subset = data_mev[mev_pieces].copy()
                mev_subset.index = mev_subset.index.normalize()
                
                # Convert date column to datetime and normalize
                result[date_col] = pd.to_datetime(result[date_col]).dt.normalize()
                
                # Merge MEV features based on date alignment
                result = result.merge(
                    mev_subset,
                    left_on=date_col,
                    right_index=True,
                    how='left'
                )
            
            # Add feature pieces if any exist
            if feature_pieces:
                # Merge each feature piece with the result
                for piece in feature_pieces:
                    # Drop any duplicate entity/date columns that might have been added
                    cols_to_use = [col for col in piece.columns 
                                 if col not in [entity_col, date_col]]
                    if cols_to_use:  # Only merge if we have features to add
                        result = result.merge(
                            piece[[entity_col, date_col] + cols_to_use],
                            on=[entity_col, date_col],
                            how='left'
                        )
            
            # Remove entity and date columns from final result
            result = result.drop(columns=[entity_col, date_col])
        else:
            # For time series data, combine all pieces
            pieces = []
            if internal_pieces:
                pieces.extend(internal_pieces)
            if mev_pieces:
                pieces.extend(mev_pieces)
            if feature_pieces:
                pieces.extend(feature_pieces)
            
            # Concatenate all features
            result = pd.concat(pieces, axis=1)
            result.index = result.index.normalize()

        return result

    def build_tsfm_specs(
        self,
        specs: List[Union[str, TSFM]],
        max_lag: int = 0,
        max_periods: Union[int, List[int]] = 1,
        exp_sign_map: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Union[str, TSFM]]]:
        """
        Generate TSFM specification lists for each variable based on their type.
        Returns a mapping of variable names to lists of transform specifications.

        This method uses the MEV type mapping and transform mapping from the MEVLoader
        to automatically generate appropriate transforms for each variable.

        Parameters
        ----------
        specs : list
            List of variable names or TSFM instances to generate specs for.
            - str: Variable names will be mapped to transforms based on their type
            - TSFM: Transform instances will be used as-is
        max_lag : int, default=0
            Generate transform entries for lags 0 through max_lag.
            Must be non-negative.
        max_periods : Union[int, List[int]], default=1
            For transforms that accept a 'periods' parameter:
            - If int: generate entries for periods 1 through max_periods
            - If List[int]: generate entries for the specific periods provided
            Must be positive (if int) or contain only positive values (if list).
            
            Special handling for monthly data: When internal data has monthly frequency
            and max_periods > 3, automatically creates periods [1, 2, 3, 6, 9, 12, ...]
            (multiples of 3 beyond period 3) instead of [1, 2, 3, 4, 5, 6, ...].
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            - Keys: MEV variable names (str)
            - Values: Expected signs (int): 1 for positive, -1 for negative, 0 for no expectation
            If provided, TSFM instances created from matching variable names will use 
            the specified exp_sign value. Variables not in the map default to exp_sign=0.

        Returns
        -------
        Dict[str, List[Union[str, TSFM]]]
            Mapping of variable names to lists of specifications.
            - Keys: Variable names from input specs
            - Values: Lists containing either:
                - str: For unmapped variables
                - TSFM: Transform instances for mapped variables

        Examples
        --------
        >>> # Basic usage with default parameters
        >>> specs = dm.build_tsfm_specs(['GDP', 'UNRATE'])
        >>> # Result example:
        >>> # {
        >>> #     'GDP': [TSFM(GDP, log), TSFM(GDP, diff)],
        >>> #     'UNRATE': [TSFM(UNRATE, diff)]
        >>> # }
        >>> 
        >>> # With lags and multiple periods (int)
        >>> specs = dm.build_tsfm_specs(
        ...     specs=['GDP', 'UNRATE'],
        ...     max_lag=2,
        ...     max_periods=2
        ... )
        >>> # Result includes variations like:
        >>> # GDP: [
        >>> #     TSFM(GDP, log),
        >>> #     TSFM(GDP, diff, periods=1), TSFM(GDP, diff, periods=2),
        >>> #     TSFM(GDP, log, lag=1), TSFM(GDP, log, lag=2)
        >>> # ]
        >>> 
        >>> # With specific periods (list) - useful for monthly data
        >>> specs = dm.build_tsfm_specs(
        ...     specs=['GDP', 'UNRATE'],
        ...     max_lag=1,
        ...     max_periods=[1, 2, 3, 6, 9, 12]
        ... )
        >>> # Result includes transforms with specific periods like:
        >>> # GDP: [
        >>> #     TSFM(GDP, log),
        >>> #     TSFM(GDP, diff, periods=1), TSFM(GDP, diff, periods=2),
        >>> #     TSFM(GDP, diff, periods=3), TSFM(GDP, diff, periods=6),
        >>> #     TSFM(GDP, diff, periods=9), TSFM(GDP, diff, periods=12),
        >>> #     (plus lagged versions)
        >>> # ]
        >>> 
        >>> # Monthly data automatic behavior (max_periods > 3)
        >>> # For monthly internal data with max_periods=13:
        >>> specs = dm.build_tsfm_specs(['GDP'], max_periods=13)
        >>> # Automatically creates periods [1, 2, 3, 6, 9, 12] instead of [1...13]

        Notes
        -----
        - Variables not found in MEV type mapping will only use raw values
        - Transform functions must exist in transform_module
        - The method warns about unmapped variables but continues processing
        - Transform order is preserved within each variable's list
        """
        if max_lag < 0:
            raise ValueError("max_lag must be >= 0")
        
        # Validate max_periods
        if isinstance(max_periods, int):
            if max_periods < 1:
                raise ValueError("max_periods must be >= 1")
        elif isinstance(max_periods, list):
            if not max_periods:
                raise ValueError("max_periods list cannot be empty")
            if any(p < 1 for p in max_periods):
                raise ValueError("all values in max_periods list must be >= 1")
        else:
            raise TypeError("max_periods must be int or List[int]")

        # Apply special period logic for monthly data
        # When internal_data is monthly, periods > 3 should only include multiples of 3
        effective_max_periods = max_periods
        if self.freq == 'M' and isinstance(max_periods, int) and max_periods > 3:
            # Create periods list: (1, 2, 3, 6, 9, 12, ...) up to max_periods
            periods_list = [1, 2, 3]
            for p in range(6, max_periods + 1, 3):  # multiples of 3 starting from 6
                periods_list.append(p)
            effective_max_periods = periods_list

        vt_map = self._mev_loader.mev_map
        tf_map = self._mev_loader.tsfm_map
        specs_map: Dict[str, List[Union[str, TSFM]]] = {}
        missing: List[str] = []
    
        for spec in specs:
            if isinstance(spec, TSFM):
                specs_map[spec.var] = [spec]

            elif isinstance(spec, str):
                var_name = spec
                var_info = vt_map.get(spec)
                if var_info is None:
                    missing.append(spec)
                    specs_map[var_name] = [spec]
                else:
                    # Get the type from the var_info dictionary
                    var_type = var_info['type']
                    fnames = tf_map.get(var_type, [])
                    tsfms: List[Union[str, TSFM]] = []
                    for name in fnames:
                        fn = getattr(transform_module, name, None)
                        if not callable(fn):
                            continue
                        sig = inspect.signature(fn)
                        if 'periods' in sig.parameters:
                            if isinstance(effective_max_periods, int):
                                pvals = list(range(1, effective_max_periods + 1))
                            else:  # List[int]
                                pvals = effective_max_periods
                        else:
                            pvals = [None]
                        for p in pvals:
                            base_fn = functools.partial(fn, periods=p) if p else fn
                            for lag in range(max_lag+1):
                                # Get expected sign from map if provided
                                exp_sign = 0
                                if exp_sign_map and spec in exp_sign_map:
                                    exp_sign = exp_sign_map[spec]
                                tsfms.append(TSFM(spec, base_fn, lag, exp_sign=exp_sign, freq=self.freq))
                    specs_map[var_name] = tsfms
            else:
                raise ValueError(f"Invalid spec: {spec!r}")

        if missing:
            warnings.warn(
                f"No type mapping for variables: {missing!r}, using raw-only", UserWarning
            )
        return specs_map

    def build_search_vars(
        self,
        specs: List[Union[str, TSFM]],
        max_lag: int = 0,
        max_periods: Union[int, List[int]] = 1,
        exp_sign_map: Optional[Dict[str, int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Build a DataFrame for each variable by generating transform specifications
        and applying them to the data.

        This is a convenience method that combines build_tsfm_specs() and build_features()
        to create a dictionary of transformed DataFrames, one for each input variable.

        Parameters
        ----------
        specs : list
            List of variable names or TSFM instances to process.
            See build_tsfm_specs() for details.
        max_lag : int, default=0
            Maximum lag to include in transforms. Must be non-negative.
        max_periods : Union[int, List[int]], default=1
            Maximum periods for transforms that accept this parameter.
            Must be positive (if int) or contain only positive values (if list).
            For monthly data with max_periods > 3, automatically uses 
            [1, 2, 3, 6, 9, 12, ...] instead of [1, 2, 3, 4, 5, 6, ...].
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            See build_tsfm_specs() for details.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of variable names to DataFrames containing all transforms.
            Each DataFrame contains the raw variable and its transforms.

        Examples
        --------
        >>> # Basic usage
        >>> var_dfs = dm.build_search_vars(['GDP', 'UNRATE'])
        >>> gdp_df = var_dfs['GDP']
        >>> print("GDP transforms:", gdp_df.columns.tolist())
        >>> 
        >>> # With lags and multiple periods
        >>> var_dfs = dm.build_search_vars(
        ...     specs=['GDP', 'UNRATE'],
        ...     max_lag=2,
        ...     max_periods=2
        ... )
        >>> # Access specific transforms
        >>> gdp_changes = var_dfs['GDP']['GDP_diff']
        >>> gdp_2period = var_dfs['GDP']['GDP_diff_2']

        See Also
        --------
        build_tsfm_specs : Generate transform specifications
        build_features : Build features from specifications
        """
        tsfm_specs = self.build_tsfm_specs(
            specs,
            max_lag=max_lag,
            max_periods=max_periods,
            exp_sign_map=exp_sign_map
        )
        var_df_map: Dict[str, pd.DataFrame] = {}
        for var, tsfms in tsfm_specs.items():
            var_df_map[var] = self.build_features(tsfms)
        return var_df_map
    
    @staticmethod
    def _interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate quarterly MEV DataFrames to match monthly frequency using cubic spline.
        The interpolation ensures that quarterly averages of the monthly values match
        the original quarterly values.

        The process:
        1. Forward fills any NA values in the quarterly series
        2. Extends the quarterly series by 4 quarters by holding the last value flat
        3. Performs cubic spline interpolation on the extended series
        4. Scales interpolated values to preserve quarterly averages
        5. Returns only the original date range (removes the extended quarters)

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with datetime index. If not quarterly, returns original data.

        Returns
        -------
        pd.DataFrame
            If input is quarterly: Interpolated monthly DataFrame with scaled values
            If input is not quarterly: Original DataFrame

        Notes
        -----
        - Uses scipy's CubicSpline with natural boundary conditions
        - Scales interpolated values to preserve quarterly averages
        - Handles missing values by preserving them in output
        - Normalizes all dates to midnight UTC
        - For quarterly data, creates a monthly date range from first quarter start to last quarter end
        """
        if df.empty:
            return df

        # Normalize index to midnight UTC
        df2 = df.copy()
        df2.index = pd.DatetimeIndex(pd.to_datetime(df2.index)).normalize()
        freq_mev = pd.infer_freq(df2.index)
        
        # Only interpolate Q -> M
        if freq_mev and freq_mev.startswith('Q'):
            # Get the first and last quarters for complete coverage
            first_qtr = pd.Period(df2.index[0], freq='Q')
            last_qtr = pd.Period(df2.index[-1], freq='Q')
            
            # Create monthly index from first month of first quarter to last month of last quarter
            start_month = first_qtr.start_time
            end_month = last_qtr.end_time
            monthly_index = pd.date_range(start=start_month, end=end_month, freq='M')
            
            # Initialize result DataFrame with NaN
            monthly_df = pd.DataFrame(index=monthly_index)
            
            # Process each column separately
            for col in df2.columns:
                q_series = df2[col]
                
                # Skip if all values are NA
                if q_series.isnull().all():
                    monthly_df[col] = np.nan
                    continue
                
                # Find continuous non-NA segments
                non_na_mask = ~q_series.isnull()
                valid_indices = q_series.index[non_na_mask]
                
                if len(valid_indices) == 0:
                    monthly_df[col] = np.nan
                    continue
                
                # Get the first and last non-NA indices
                first_valid_idx = valid_indices[0]
                last_valid_idx = valid_indices[-1]
                
                # Get the non-NA segment
                valid_series = q_series.loc[first_valid_idx:last_valid_idx].dropna()
                
                if len(valid_series) < 4:  # Need at least 4 points for cubic spline
                    # Use linear interpolation for short segments
                    valid_series = valid_series.reindex(monthly_index, method='linear')
                    monthly_df[col] = valid_series
                    continue
                
                # Extend the valid series by 4 quarters
                last_value = valid_series.iloc[-1]
                extended_qtrs = pd.date_range(
                    start=pd.Period(valid_series.index[-1], freq='Q').end_time + pd.Timedelta(days=1),
                    periods=4,
                    freq='Q'
                )
                extended_data = pd.Series([last_value] * 4, index=extended_qtrs)
                extended_series = pd.concat([valid_series, extended_data])
                
                # Convert dates to numeric for spline interpolation
                x = extended_series.index.map(pd.Timestamp.toordinal)
                y = extended_series.values
                
                # Fit cubic spline
                spline = CubicSpline(x, y, bc_type='natural')
                
                # Get monthly points within the valid range
                valid_start = pd.Period(valid_series.index[0], freq='Q').start_time
                valid_end = pd.Period(valid_series.index[-1], freq='Q').end_time
                valid_months = pd.date_range(start=valid_start, end=valid_end, freq='M')
                
                # Evaluate spline at monthly points
                monthly_x = valid_months.map(pd.Timestamp.toordinal)
                monthly_y = spline(monthly_x)
                
                # Create initial monthly series
                m_series = pd.Series(monthly_y, index=valid_months)
                
                # Scale values to preserve quarterly averages
                scaled_series = m_series.copy()
                
                # Map months to corresponding quarter ends for scaling
                month_to_qtr = pd.PeriodIndex(valid_months, freq='Q').end_time
                
                # Apply scaling for each quarter
                for qtr_end in valid_series.index:
                    mask = month_to_qtr == qtr_end
                    if not mask.any():
                        continue
                    
                    interpolated_avg = m_series[mask].mean()
                    observed_value = valid_series.loc[qtr_end]
                    
                    # Handle zero or near-zero averages
                    if np.isclose(interpolated_avg, 0):
                        scale_factor = 1.0
                    else:
                        scale_factor = observed_value / interpolated_avg
                    
                    scaled_series.loc[mask] = m_series.loc[mask] * scale_factor
                
                # Assign the scaled values to the result DataFrame
                monthly_df[col] = scaled_series
                
                # Preserve original NA values
                na_qtrs = q_series[q_series.isnull()].index
                for qtr in na_qtrs:
                    qtr_period = pd.Period(qtr, freq='Q')
                    qtr_start = qtr_period.start_time
                    qtr_end = qtr_period.end_time
                    na_months = monthly_df.loc[qtr_start:qtr_end].index
                    monthly_df.loc[na_months, col] = np.nan
            
            # Add month and quarter indicators
            monthly_df['M'] = pd.DatetimeIndex(monthly_df.index).month
            monthly_df['Q'] = pd.DatetimeIndex(monthly_df.index).quarter
            
            return monthly_df

        return df
    
    def apply_to_mevs(
        self,
        fn: Callable[
            [pd.DataFrame, pd.DataFrame],
            pd.DataFrame
        ]
    ) -> None:
        """
        Apply a feature engineering function to all cached MEV tables (model and scenarios).

        This method allows you to add derived features to all MEV tables at once.
        The function is applied to both the model MEVs and all scenario MEVs.
        Changes are made to the cached data in the DataManager, not the original loaders.

        Parameters
        ----------
        fn : callable
            Function that takes two arguments:
            1. df_mev: DataFrame of MEV data to modify
            2. internal_df: DataFrame of internal data for reference
            Must return a DataFrame with new/modified columns.

        Examples
        --------
        >>> # Add GDP/UNRATE ratio to all MEV tables
        >>> def add_gdp_ratio(mev_df, internal_df):
        ...     mev_df['GDP_UNRATE_RATIO'] = mev_df['GDP'] / mev_df['UNRATE']
        ...     return mev_df
        >>> dm.apply_to_mevs(add_gdp_ratio)
        >>> 
        >>> # Add multiple features using internal data
        >>> def add_features(mev_df, internal_df):
        ...     # Relative to historical average
        ...     hist_mean = internal_df['GDP'].mean()
        ...     mev_df['GDP_REL'] = mev_df['GDP'] / hist_mean
        ...     # Composite indicator
        ...     mev_df['COMPOSITE'] = mev_df['GDP'] * mev_df['UNRATE']
        ...     return mev_df
        >>> dm.apply_to_mevs(add_features)

        Notes
        -----
        - The function must return a DataFrame (modified or new)
        - Changes are applied to cached MEV data in DataManager
        - Original loaders remain unchanged
        - The function has access to internal data for reference calculations
        """
        internal_df = self.internal_data

        def _apply_mev(mev_df: pd.DataFrame):
            result = fn(mev_df.copy(), internal_df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"apply_to_mevs: fn must return a DataFrame, got {type(result)}")
            for col in result.columns:
                mev_df[col] = result[col].reindex(mev_df.index)

        # Apply to cached model MEV data
        model_mev_df = self.model_mev  # This ensures cache is created
        _apply_mev(model_mev_df)

        # Apply to cached scenario MEV data
        scen_mevs_dict = self.scen_mevs  # This ensures cache is created
        for scen_set, scen_dict in scen_mevs_dict.items():
            for scen_name, scen_df in scen_dict.items():
                _apply_mev(scen_df)
    
    def apply_to_internal(
        self,
        fn: Callable[
            [pd.DataFrame],
            Optional[Union[pd.Series, pd.DataFrame]]
        ]
    ) -> None:
        """
        Apply a feature engineering function to all cached internal data (main and scenarios).

        This method allows you to add derived features to both the main internal data
        and all scenario internal data. Changes can be made either in-place or by 
        returning new data to merge. Changes are made to the cached data in the 
        DataManager, not the original loader.

        Parameters
        ----------
        fn : callable
            Function that takes one argument:
            - internal_df: DataFrame of internal data to modify
            Can return:
            - None: If modifications were made in-place
            - Series: Single new column to add
            - DataFrame: Multiple new columns to add

        Examples
        --------
        >>> # In-place modification applied to all internal data
        >>> def add_growth_rate(df):
        ...     df['GROWTH'] = df['VALUE'].pct_change()
        ...     # No return needed for in-place changes
        >>> dm.apply_to_internal(add_growth_rate)
        >>> 
        >>> # Return new features for all internal data
        >>> def create_indicators(df):
        ...     return pd.DataFrame({
        ...         'HIGH_VALUE': df['VALUE'] > df['VALUE'].mean(),
        ...         'LOW_VALUE': df['VALUE'] < df['VALUE'].mean()
        ...     })
        >>> dm.apply_to_internal(create_indicators)
        >>> 
        >>> # Return single feature as Series for all internal data
        >>> def moving_average(df):
        ...     return df['VALUE'].rolling(window=3).mean().rename('MA3')
        >>> dm.apply_to_internal(moving_average)

        Notes
        -----
        - The function is applied to both main internal data and all scenario internal data
        - The function can modify data in-place and/or return new features
        - Returned Series must have a name
        - All new columns are aligned to the respective DataFrame indices
        - Changes are made to cached data in DataManager, not the original loader
        - If no scenario data exists, a warning is issued but main data is still processed
        """
        def _apply_internal(internal_df: pd.DataFrame):
            """Helper function to apply the user function to a single DataFrame."""
            ret = fn(internal_df)
            if ret is None:
                return
            if isinstance(ret, pd.Series):
                internal_df[ret.name] = ret.reindex(internal_df.index)
            elif isinstance(ret, pd.DataFrame):
                for col in ret.columns:
                    internal_df[col] = ret[col].reindex(internal_df.index)
            else:
                raise TypeError(
                    f"apply_to_internal(): fn must return None, Series or DataFrame, got {type(ret)}"
                )

        # Apply to cached main internal data
        main_internal_df = self.internal_data  # This ensures cache is created
        _apply_internal(main_internal_df)

        # Apply to cached scenario internal data
        scen_internal_dict = self.scen_internal_data  # This ensures cache is created
        
        if not scen_internal_dict:
            warnings.warn(
                "No scenario internal data found. Function only applied to main internal data. "
                "Load scenario data using load_scens() if you want to apply the function to scenarios as well.",
                UserWarning
            )
        else:
            for scen_set, scen_dict in scen_internal_dict.items():
                for scen_name, scen_df in scen_dict.items():
                    _apply_internal(scen_df)

    @property
    def var_map(self) -> Dict[str, Dict[str, str]]:
        """
        Get the variable type mapping for codes that exist in either model_mev or internal_data.
        This includes both original MEVs and any derived MEVs (e.g., with '_Q' suffix),
        as well as internal data variables that have been added to the variable map.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary mapping variable codes to their type and description information.
            Includes codes that exist in either model_mev columns or internal_data columns.

        Example
        -------
        >>> var_info = dm.var_map
        >>> # Shows info for MEVs that exist in model_mev
        >>> print(var_info['GDP'])  # {'type': 'level', 'description': 'Gross Domestic Product'}
        >>> # Also shows internal variables added to variable map
        >>> print(var_info['balance'])  # {'type': 'level', 'description': 'Account Balance'}
        >>> # If GDP_Q exists in model_mev, it will be included
        >>> if 'GDP_Q' in dm.model_mev.columns:
        ...     print(var_info['GDP_Q'])  # {'type': 'level', 'description': 'GDP (Interpolated from quarterly)'}
        """
        # Get all variable codes from the loader's map
        full_var_map = self._mev_loader.mev_map
        
        # Get available codes from both model_mev and internal_data
        available_mev_codes = set(self.model_mev.columns)
        available_internal_codes = set(self.internal_data.columns)
        all_available_codes = available_mev_codes | available_internal_codes
        
        # Filter the map to only include codes that exist in either data source
        filtered_map = {
            code: info for code, info in full_var_map.items()
            if code in all_available_codes
        }
        
        return filtered_map

    @property
    def in_sample_end(self) -> Optional[pd.Timestamp]:
        """
        Get the in-sample end date from the internal loader.

        Returns
        -------
        Optional[pd.Timestamp]
            The end date of the in-sample period, or None if not set.
        """
        if isinstance(self._internal_loader, TimeSeriesLoader):
            return self._internal_loader.in_sample_end
        return None

    @property
    def full_sample_end(self) -> Optional[pd.Timestamp]:
        """
        Get the full sample end date from the internal loader.

        Returns
        -------
        Optional[pd.Timestamp]
            The end date of the full sample period, or None if not set.
        """
        return self._internal_loader.full_sample_end

    @property
    def scen_p0(self) -> Optional[pd.Timestamp]:
        """
        Get the scenario jumpoff date from the internal loader.

        Returns
        -------
        Optional[pd.Timestamp]
            The scenario jumpoff date, or None if not set.
        """
        return self._internal_loader.scen_p0

    @property
    def in_sample_idx(self) -> pd.Index:
        """
        Get the in-sample index from the internal loader.

        Returns
        -------
        pd.Index
            Index of in-sample observations.
        """
        return self._internal_loader.in_sample_idx

    @property
    def out_sample_idx(self) -> pd.Index:
        """
        Get the out-of-sample index from the internal loader.

        Returns
        -------
        pd.Index
            Index of out-of-sample observations.
        """
        return self._internal_loader.out_sample_idx

    @property
    def scen_in_sample_idx(self) -> Optional[pd.Index]:
        """
        Get the scenario in-sample index from the internal loader.

        Returns
        -------
        Optional[pd.Index]
            Index of scenario in-sample observations, or None if not available.
        """
        return self._internal_loader.scen_in_sample_idx

    @property
    def scen_out_sample_idx(self) -> Optional[pd.Index]:
        """
        Get the scenario out-of-sample index from the internal loader.

        Returns
        -------
        Optional[pd.Index]
            Index of scenario out-of-sample observations, or None if not available.
        """
        return self._internal_loader.scen_out_sample_idx

    def update_var_map(self, updates: Dict[str, Dict[str, Optional[str]]]) -> None:
        """
        Update the variable mapping with new or modified variable codes.
        
        This method provides a convenient way to update variable mappings directly through
        the DataManager interface. It delegates to the underlying MEVLoader's
        update_mev_map method while preserving any cached MEV data that was modified
        through apply_to_mevs(). This can be used for both MEV and internal variables.
        
        For new variable codes, it's highly recommended to specify both 'type' and 'category'.
        Description is optional if you can remember what the variable code means.
        
        Parameters
        ----------
        updates : dict
            Dictionary where keys are variable codes and values are dictionaries
            containing the attributes to update. Supported attributes are:
            - 'type': Variable type (e.g., 'level', 'rate')
            - 'description': Human-readable description
            - 'category': Variable category (e.g., 'GDP', 'Job Market', 'Inflation')
            
            For existing variable codes, only the specified attributes will be updated;
            unspecified attributes will remain unchanged.
            
            For new variable codes, unspecified attributes will be set to None.
            
        Examples
        --------
        >>> # Typical workflow: add new MEV columns then update mapping
        >>> def add_custom_mev(mev_df, internal_df):
        ...     mev_df['CUSTOM_GDP'] = mev_df['GDP'] * 1.1  # Custom GDP calculation
        ...     return mev_df
        >>> dm.apply_to_mevs(add_custom_mev)
        >>> 
        >>> # Now update the mapping for the new variable
        >>> dm.update_var_map({
        ...     'CUSTOM_GDP': {
        ...         'type': 'level',
        ...         'description': 'Custom GDP Measure',
        ...         'category': 'GDP'
        ...     }
        ... })
        >>> 
        >>> # Both the new column and mapping are preserved
        >>> print('CUSTOM_GDP' in dm.model_mev.columns)  # True
        >>> print(dm.var_map['CUSTOM_GDP'])  # Shows the mapping info
        >>> 
        >>> # Update existing variable code (only specified attributes)
        >>> dm.update_var_map({
        ...     'GDP': {
        ...         'category': 'Economic Growth'  # Only update category
        ...         # type and description remain unchanged
        ...     }
        ... })
        >>> 
        >>> # Add internal variable to mapping
        >>> dm.update_var_map({
        ...     'balance': {
        ...         'type': 'level',
        ...         'description': 'Account Balance',
        ...         'category': 'Internal'
        ...     }
        ... })
        
        Notes
        -----
        - Changes are made to the MEVLoader's in-memory variable map
        - Cached MEV data is preserved, including any columns added via apply_to_mevs()
        - To persist mapping changes, you would need to update the Excel file manually
        - For new variable codes, 'type' and 'category' are highly recommended
        - Valid attributes: 'type', 'description', 'category'
        - Can be used for both MEV and internal variables
        
        See Also
        --------
        MEVLoader.update_mev_map : The underlying method that performs the update
        apply_to_mevs : Method for adding new MEV columns
        """
        # Delegate to the MEVLoader's update_mev_map method
        # This updates the mapping without affecting cached data
        self._mev_loader.update_mev_map(updates)
