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
from typing import Any, Dict, List, Optional, Callable, Union

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
    - The DataManager maintains caches for interpolated MEV data to improve performance
    - All data modifications through apply_* methods are made to the original data in loaders
    - The class provides dynamic access to loader data, ensuring you always have the latest data
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

    @property
    def internal_data(self) -> pd.DataFrame:
        """
        Get the latest internal data from the internal loader.

        Returns
        -------
        pd.DataFrame
            The most recent internal data. Any modifications made through
            apply_to_internal() will be reflected in this data.

        Example
        -------
        >>> internal = dm.internal_data
        >>> print(f"Available variables: {internal.columns.tolist()}")
        >>> print(f"Date range: {internal.index.min()} to {internal.index.max()}")
        """
        return self._internal_loader.internal_data

    @property
    def scen_internal_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get the latest scenario internal data from the internal loader.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping scenario names to their corresponding internal data.
            Each DataFrame contains the scenario-specific internal data.

        Example
        -------
        >>> scenarios = dm.scen_internal_data
        >>> # Access base scenario data
        >>> base_data = scenarios['Base']
        >>> print(f"Base scenario variables: {base_data.columns.tolist()}")
        >>> print(f"Base scenario range: {base_data.index.min()} to {base_data.index.max()}")
        """
        return self._internal_loader.scen_internal_data

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
        internal_freq = pd.infer_freq(self.internal_data.index)
        is_monthly = internal_freq and internal_freq.startswith('M')
        
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
        Get the latest model MEV data, combining quarterly and monthly data appropriately.
        Uses caching to avoid recomputing unless data has changed.

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
            Combined MEV data matching internal data frequency.
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
        # Get current data from loader
        current_qtr = self._mev_loader.model_mev_qtr
        current_mth = self._mev_loader.model_mev_mth
        
        # Create cache key from both quarterly and monthly data
        cache_key = hash(str(current_qtr) + str(current_mth))
        
        if cache_key not in self._mev_cache:
            # Combine the data based on frequency
            df = self._combine_mevs(current_qtr, current_mth)
            
            # Add month and quarter indicators
            df['M'] = df.index.month
            df['Q'] = df.index.quarter
            
            self._mev_cache[cache_key] = df
        
        return self._mev_cache[cache_key]

    @property
    def scen_mevs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get the latest scenario MEV data, combining quarterly and monthly data appropriately.
        Uses caching to avoid recomputing unless data has changed.

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
            Nested dictionary of combined scenario data.
            Outer key: scenario set name (e.g., 'EWST2024')
            Inner key: scenario name (e.g., 'Base', 'Adverse')
            Value: DataFrame with combined MEV data matching internal data frequency

        Example
        -------
        >>> scenarios = dm.scen_mevs
        >>> # Access base scenario from EWST2024
        >>> base_ewst = scenarios['EWST2024']['Base']
        >>> print(f"EWST Base scenario range: {base_ewst.index.min()} to {base_ewst.index.max()}")
        >>> 
        >>> # Compare GDP across scenarios
        >>> for scen_name, scen_df in scenarios['EWST2024'].items():
        ...     print(f"{scen_name} GDP mean: {scen_df['GDP'].mean():.2f}")
        >>> 
        >>> # Check if we have both quarterly and monthly versions
        >>> monthly_vars = [col for col in base_ewst.columns if col.endswith('_M')]
        >>> print("Monthly-derived variables:", monthly_vars)
        """
        # Get current data from loader
        current_qtr = self._mev_loader.scen_mev_qtr
        current_mth = self._mev_loader.scen_mev_mth
        
        # Create cache key from both quarterly and monthly data
        cache_key = hash(str(current_qtr) + str(current_mth))
        
        if cache_key not in self._scen_cache:
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
            
            self._scen_cache[cache_key] = processed
            
        return self._scen_cache[cache_key]

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
        applying any specified transformations in the process.

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
            Combined features indexed by date. The columns will be:
            - Raw variables: Same name as input
            - Transformed features: Names defined by Feature objects

        Examples
        --------
        >>> # Simple raw variables
        >>> features = dm.build_features(['GDP', 'UNRATE', 'CPI'])
        >>> 
        >>> # Mix of raw and transformed features
        >>> from .transform import TSFM, diff, pct_change
        >>> specs = [
        ...     'GDP',                     # Raw GDP
        ...     TSFM('GDP', diff),        # GDP change
        ...     TSFM('UNRATE', pct_change, lag=1),  # Lagged unemployment change
        ...     ('CPI', 'HOUSING')        # Group of raw features
        ... ]
        >>> features = dm.build_features(specs)
        >>> 
        >>> # Using conditional features
        >>> from .condition import CondVar
        >>> specs = [
        ...     CondVar('GDP', lambda x: x > 0),  # Positive GDP indicator
        ...     'UNRATE'
        ... ]
        >>> features = dm.build_features(specs)

        Notes
        -----
        - Features are built in the order specified
        - All dates are normalized to midnight UTC
        - Missing values in raw variables are preserved
        - Transform features may introduce additional NaN values
        - The method flattens nested lists/tuples in specs
        """
        data_int = internal_df if internal_df is not None else self.internal_data
        data_mev = mev_df if mev_df is not None else self.model_mev

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
        pieces = []
        
        for spec in flat_specs:
            if isinstance(spec, Feature):
                # Apply feature transform
                pieces.append(spec.apply(data_int, data_mev))
            elif isinstance(spec, str):
                # Raw variable
                if spec in data_int.columns:
                    pieces.append(data_int[spec])
                elif spec in data_mev.columns:
                    pieces.append(data_mev[spec])
                else:
                    raise KeyError(f"Feature '{spec}' not found in data sources.")
            else:
                raise TypeError(f"Invalid spec type after flatten(): {type(spec)}")

        X = pd.concat(pieces, axis=1)
        X.index = X.index.normalize()
        return X

    def build_tsfm_specs(
        self,
        specs: List[Union[str, TSFM]],
        max_lag: int = 0,
        max_periods: int = 1
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
        max_periods : int, default=1
            For transforms that accept a 'periods' parameter, generate entries
            for periods 1 through max_periods. Must be positive.

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
        >>> # With lags and multiple periods
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

        Notes
        -----
        - Variables not found in MEV type mapping will only use raw values
        - Transform functions must exist in transform_module
        - The method warns about unmapped variables but continues processing
        - Transform order is preserved within each variable's list
        """
        if max_lag < 0:
            raise ValueError("max_lag must be >= 0")
        if max_periods < 1:
            raise ValueError("max_periods must be >= 1")

        vt_map = self._mev_loader.mev_map
        tf_map = self._mev_loader.tsfm_map
        specs_map: Dict[str, List[Union[str, TSFM]]] = {}
        missing: List[str] = []
    
        for spec in specs:
            if isinstance(spec, TSFM):
                specs_map[spec.feature_name] = [spec]

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
                        pvals = list(range(1, max_periods+1)) if 'periods' in sig.parameters else [None]
                        for p in pvals:
                            base_fn = functools.partial(fn, periods=p) if p else fn
                            for lag in range(max_lag+1):
                                tsfms.append(TSFM(spec, base_fn, lag))
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
        max_periods: int = 1
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
        max_periods : int, default=1
            Maximum periods for transforms that accept this parameter.
            Must be positive.

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
            max_periods=max_periods
        )
        var_df_map: Dict[str, pd.DataFrame] = {}
        for var, tsfms in tsfm_specs.items():
            var_df_map[var] = self.build_features(tsfms)
        return var_df_map
    
    @staticmethod
    def _interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate quarterly MEV DataFrames to match monthly frequency.

        This method performs cubic interpolation on quarterly data to estimate
        monthly values. If the input is not quarterly, it returns the original data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with datetime index

        Returns
        -------
        pd.DataFrame
            If input is quarterly: Interpolated monthly DataFrame
            If input is not quarterly: Original DataFrame

        Notes
        -----
        - Uses pandas' cubic interpolation
        - Preserves the original column names and types
        - Normalizes all dates to midnight UTC
        - For quarterly data, creates a monthly date range from min to max date
        """
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        freq_mev = pd.infer_freq(df2.index)
        # Only interpolate Q -> M
        if freq_mev and freq_mev.startswith('Q'):
            start = df2.index.min()
            end = df2.index.max()
            target_idx = pd.date_range(start=start, end=end, freq='M')
            df2 = df2.reindex(target_idx).astype(float)
            df2 = df2.interpolate(method='cubic')
            df2.index = df2.index.normalize()
            return df2
        return df
    
    def apply_to_mevs(
        self,
        fn: Callable[
            [pd.DataFrame, pd.DataFrame],
            pd.DataFrame
        ]
    ) -> None:
        """
        Apply a feature engineering function to all MEV tables (model and scenarios).

        This method allows you to add derived features to all MEV tables at once.
        The function is applied to both the model MEVs and all scenario MEVs.
        Changes are made in-place to the original data in the MEVLoader.

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
        - Changes are applied to both model_mev_qtr and scen_mev_qtr
        - The MEV cache is cleared after applying changes
        - The function has access to internal data for reference calculations
        """
        internal_df = self.internal_data

        def _apply_mev(mev_df: pd.DataFrame):
            result = fn(mev_df.copy(), internal_df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"apply_to_mevs: fn must return a DataFrame, got {type(result)}")
            for col in result.columns:
                mev_df[col] = result[col].reindex(mev_df.index)

        # Apply to main MEV in loader
        _apply_mev(self._mev_loader.model_mev_qtr)

        # Apply to each scenario in loader
        for wb_key, scen_map in self._mev_loader.scen_mev_qtr.items():
            for scen_name, df in scen_map.items():
                _apply_mev(df)
                
        # Clear caches since data has changed
        self._mev_cache.clear()
        self._scen_cache.clear()
    
    def apply_to_internal(
        self,
        fn: Callable[
            [pd.DataFrame],
            Optional[Union[pd.Series, pd.DataFrame]]
        ]
    ) -> None:
        """
        Apply a feature engineering function to internal data.

        This method allows you to add derived features to the internal data.
        Changes can be made either in-place or by returning new data to merge.

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
        >>> # In-place modification
        >>> def add_growth_rate(df):
        ...     df['GROWTH'] = df['VALUE'].pct_change()
        ...     # No return needed for in-place changes
        >>> dm.apply_to_internal(add_growth_rate)
        >>> 
        >>> # Return new features
        >>> def create_indicators(df):
        ...     return pd.DataFrame({
        ...         'HIGH_VALUE': df['VALUE'] > df['VALUE'].mean(),
        ...         'LOW_VALUE': df['VALUE'] < df['VALUE'].mean()
        ...     })
        >>> dm.apply_to_internal(create_indicators)
        >>> 
        >>> # Return single feature as Series
        >>> def moving_average(df):
        ...     return df['VALUE'].rolling(window=3).mean().rename('MA3')
        >>> dm.apply_to_internal(moving_average)

        Notes
        -----
        - The function can modify data in-place and/or return new features
        - Returned Series must have a name
        - All new columns are aligned to the internal data index
        - Changes are made directly to the loader's internal_data
        """
        internal_df = self._internal_loader.internal_data

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

    @property
    def mev_map(self) -> Dict[str, Dict[str, str]]:
        """
        Get the latest MEV type mapping from the MEV loader.
        This includes both original MEVs and any derived MEVs (e.g., with '_Q' suffix).

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary mapping MEV codes to their type and description information.
            Includes both original and derived MEVs.

        Example
        -------
        >>> mev_info = dm.mev_map
        >>> # Original MEV info
        >>> print(mev_info['GDP'])  # {'type': 'level', 'description': 'Gross Domestic Product'}
        >>> # Derived quarterly MEV info
        >>> print(mev_info['GDP_Q'])  # {'type': 'level', 'description': 'GDP (Interpolated from quarterly)'}
        """
        return self._mev_loader.mev_map