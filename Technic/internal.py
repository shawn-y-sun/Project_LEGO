# =============================================================================
# module: internal.py
# Purpose: Internal data loaders for different modeling projects (PPNR, SMR)
# Dependencies: pandas, numpy
# =============================================================================

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict
from pathlib import Path
from enum import Enum
import warnings


class SplitMethod(Enum):
    """
    Enumeration of available data splitting methods.
    
    Methods
    -------
    RANDOM : str
        Split data randomly across all observations
    STRATIFIED : str
        Split data while maintaining time period distribution
    TIME_CUTOFF : str
        Split data based on a specific cutoff date
    """
    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_CUTOFF = "time_cutoff"


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the common interface and shared functionality
    for loading and processing structured data across different 
    modeling projects.
    
    Parameters
    ----------
    freq : str, default='M'
        Frequency code ('M' for monthly, 'Q' for quarterly)
    full_sample_start : Optional[str], optional
        Start date for full sample period (YYYY-MM-DD)
    full_sample_end : Optional[str], optional
        End date for full sample period (YYYY-MM-DD)
    scen_p0 : Optional[str], optional
        The month-end date that serves as the jumpoff date for scenario forecasting.
        All data after this date are considered in the forecasting horizon.
        Must be a month-end date if provided.
        
    Example
    -------
    >>> loader = CustomLoader(freq="M", scen_p0="2023-12-31")
    >>> loader.load(source="data.csv", date_col="date")
    >>> in_sample = df.loc[loader.in_sample_idx]
    >>> forecast_horizon = df.loc[loader.forecast_horizon_idx]
    """
    
    def __init__(
        self,
        freq: Optional[str] = None,
        full_sample_start: Optional[str] = None,
        full_sample_end: Optional[str] = None,
        scen_p0: Optional[str] = None
    ):
        # Allow freq to be None initially - will be inferred from data if not specified
        if freq is not None and freq not in ['M', 'Q']:
            raise ValueError("freq must be either 'M' (monthly) or 'Q' (quarterly)")
        self.freq = freq
        self.full_sample_start = pd.to_datetime(full_sample_start).normalize() if full_sample_start else None
        self.full_sample_end = pd.to_datetime(full_sample_end).normalize() if full_sample_end else None
        
        # Handle scen_p0 (scenario jumpoff date)
        if scen_p0 is not None:
            scen_p0_date = pd.to_datetime(scen_p0).normalize()
            # Convert to month-end if not already
            self.scen_p0 = pd.Timestamp(scen_p0_date.year, scen_p0_date.month, 1) + pd.offsets.MonthEnd(0)
        else:
            self.scen_p0 = None
            
        self._internal_data : Optional[pd.DataFrame] = None
        self._in_sample_idx: Optional[pd.Index] = None
        self._out_sample_idx: Optional[pd.Index] = None
        # Initialize scenario data container
        self._scen_internal_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Initialize cached scenario indices
        self._cached_scen_out_sample_idx: Optional[pd.Index] = None
        self._cached_scen_in_sample_idx: Optional[pd.Index] = None

    def _load_from_file(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Read a CSV or Excel file into a DataFrame.

        Parameters
        ----------
        path : str
            File path ending with .csv, .xls, or .xlsx
        **kwargs : dict
            Additional arguments passed to pd.read_csv or pd.read_excel
            For Excel files:
            - sheet_name: Name or index of sheet to read. Defaults to 0 (first sheet)

        Returns
        -------
        pd.DataFrame
            Raw DataFrame from file

        Raises
        ------
        ValueError
            If file extension is not supported
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(path, **kwargs)
        elif ext in ['.xls', '.xlsx']:
            # Handle sheet_name parameter
            sheet_name = kwargs.pop('sheet_name', 0)  # Default to first sheet
            
            # Read the Excel file with explicit sheet_name
            df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
            
            # If we got a dict (multiple sheets), take the specified sheet or first one
            if isinstance(df, dict):
                if isinstance(sheet_name, (int, str)):
                    df = df[sheet_name]
                else:
                    # If sheet_name was None or a list, take the first sheet
                    sheet_name = list(df.keys())[0]
                    df = df[sheet_name]
            
            return df
        else:
            raise ValueError(f"Unsupported file type: {path}")

    @abstractmethod
    def load(
        self,
        source: Union[str, pd.DataFrame],
        date_col: Optional[str] = None,
        sheet: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Load time series data from various sources.
        
        Parameters
        ----------
        source : str or DataFrame
            Can be:
            - Path to Excel/CSV file
            - Pre-loaded DataFrame with datetime index or date column
        date_col : str, optional
            Name of date column if source is DataFrame or file without datetime index.
            Not required if DataFrame already has datetime index.
        sheet : str, optional
            Sheet name if source is Excel file. If None, uses first sheet.
        **kwargs : dict
            Additional arguments passed to file reading functions
        """
        pass

    @property
    def internal_data(self) -> pd.DataFrame:
        """
        Accessor for the processed internal data.
        
        Returns
        -------
        pd.DataFrame
            The processed internal data
            
        Raises
        ------
        ValueError
            If `load()` has not been called yet
        """
        if self._internal_data is None:
            raise ValueError("Internal data not loaded. Call `load()` first.")
        return self._internal_data

    @property
    def in_sample_idx(self) -> pd.Index:
        """
        Get the index for in-sample data.
        
        Returns
        -------
        pd.Index
            Index of in-sample observations
            
        Raises
        ------
        ValueError
            If sample indices have not been set
        """
        if self._in_sample_idx is None:
            raise ValueError("Sample indices not set. Ensure data is loaded and split.")
        return self._in_sample_idx

    @property
    def out_sample_idx(self) -> pd.Index:
        """
        Get the index for out-of-sample data.
        
        Returns
        -------
        pd.Index
            Index of out-of-sample observations
            
        Raises
        ------
        ValueError
            If sample indices have not been set
        """
        if self._out_sample_idx is None:
            raise ValueError("Sample indices not set. Ensure data is loaded and split.")
        return self._out_sample_idx

    def load_scens(
        self,
        source: Union[str, Dict[str, Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]],
        scens: Optional[Dict[str, str]] = None,
        set_name: Optional[str] = None,
        date_col: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Load scenario data from various sources.
        
        Parameters
        ----------
        source : str or Dict
            Can be:
            - Path to Excel file with scenarios in different sheets
            - Dictionary of scenario DataFrames for a single set
            - Dictionary of dictionaries for multiple scenario sets
        scens : Dict[str, str], optional
            Mapping of sheet names to scenario names when loading from Excel
        set_name : str, optional
            Name for the scenario set when loading from Excel or single dictionary
        date_col : str, optional
            Name of date column if loading from Excel
        **kwargs : dict
            Additional arguments passed to file reading functions
            
        Example
        -------
        >>> # Load scenarios from Excel file
        >>> loader = TimeSeriesLoader(scen_p0="2023-12-31")
        >>> loader.load("historical.csv", date_col="date")
        >>> loader.load_scens(
        ...     source="scenarios.xlsx",
        ...     scens={"Base": "Base_Scenario", "Adverse": "Adverse_Scenario"},
        ...     set_name="CCAR2024",
        ...     date_col="forecast_date"
        ... )
        
        >>> # Load from pre-processed DataFrames
        >>> base_scen = pd.DataFrame({
        ...     "date": pd.date_range("2024-01-01", "2025-12-31", freq="M"),
        ...     "gdp_growth": np.random.normal(2, 0.5, 24),
        ...     "unemployment": np.random.normal(4, 0.3, 24)
        ... })
        >>> loader.load_scens(
        ...     source={"Base": base_scen},
        ...     set_name="Custom_Set",
        ...     date_col="date"
        ... )
        
        >>> # Load multiple scenario sets
        >>> scenario_dict = {
        ...     "EWST2024": {"Base": base_scen.copy()},
        ...     "Internal": {"Expected": base_scen.copy()}
        ... }
        >>> loader.load_scens(source=scenario_dict)
        """
        # Clear cached indices since we're loading new scenario data
        self._clear_cached_indices()
        
        if self._internal_data is None:
            raise ValueError("Internal data must be loaded before loading scenarios")

        # Determine if input is three-layer or two-layer dictionary
        if isinstance(source, dict):
            is_three_layer = any(isinstance(v, dict) for v in source.values())
            
            if is_three_layer:
                update_dict = source
            else:
                # Convert two-layer to three-layer
                if set_name is None:
                    raise ValueError("set_name required when using two-layer dictionary")
                update_dict = {set_name: source}
        else:  # Excel file
            if not scens:
                raise ValueError("scens mapping required when loading from Excel")
            if set_name is None:
                set_name = Path(source).stem
            
            # Load scenarios from Excel sheets
            update_dict = {set_name: {}}
            for scen_name, sheet in scens.items():
                df = self._load_from_file(source, sheet_name=sheet, **kwargs)
                if date_col:
                    df = self._standardize_index(df, date_col)
                update_dict[set_name][scen_name] = df

        # Process each scenario set
        for curr_set_name, scen_dict in update_dict.items():
            # Validate columns are consistent within set
            all_cols = [set(df.columns) for df in scen_dict.values()]
            if not all(cols == all_cols[0] for cols in all_cols):
                raise ValueError(
                    f"All scenarios in set '{curr_set_name}' must have the same columns"
                )

            # Validate at least one common column with internal data
            common_cols = all_cols[0] & set(self._internal_data.columns)
            if not common_cols:
                raise ValueError(
                    f"No common columns between scenario set '{curr_set_name}' "
                    "and internal data"
                )

            # Store scenarios
            self._scen_internal_data[curr_set_name] = {}
            for scen_name, df in scen_dict.items():
                # Ensure proper datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if date_col is None:
                        raise ValueError(
                            f"date_col required for scenario '{scen_name}' without datetime index"
                        )
                    df = self._standardize_index(df, date_col)
                
                self._scen_internal_data[curr_set_name][scen_name] = df

    @property
    def scen_internal_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get scenario internal data.

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Nested dictionary of scenario data.
            Outer key: scenario set name (e.g., 'EWST2024')
            Inner key: scenario name (e.g., 'Base', 'Adverse')
            Value: DataFrame with scenario data

        Example
        -------
        >>> scenarios = loader.scen_internal_data
        >>> # Access base scenario from EWST2024
        >>> base_ewst = scenarios['EWST2024']['Base']
        >>> print(f"Common columns with internal data: "
        ...       f"{set(base_ewst.columns) & set(loader.internal_data.columns)}")
        """
        return self._scen_internal_data

    def clean_scens(self, set_name: Optional[str] = None) -> None:
        """
        Remove scenario data.
        
        Parameters
        ----------
        set_name : str, optional
            Name of specific scenario set to remove.
            If None, removes all scenario data.
            
        Example
        -------
        >>> # Load multiple scenario sets
        >>> loader = TimeSeriesLoader(scen_p0="2023-12-31")
        >>> loader.load("historical.csv", date_col="date")
        >>> # Create sample scenarios
        >>> scen_data = {
        ...     "CCAR2024": {
        ...         "Base": pd.DataFrame({"value": [1, 2, 3]}),
        ...         "Adverse": pd.DataFrame({"value": [0, -1, -2]})
        ...     },
        ...     "Internal": {
        ...         "Expected": pd.DataFrame({"value": [1.5, 2.5, 3.5]})
        ...     }
        ... }
        >>> loader.load_scens(source=scen_data)
        >>> print(f"Initial scenario sets: {list(loader.scen_internal_data.keys())}")
        
        >>> # Remove specific set
        >>> loader.clean_scens(set_name="CCAR2024")
        >>> print(f"After removing CCAR2024: {list(loader.scen_internal_data.keys())}")
        
        >>> # Remove all scenarios
        >>> loader.clean_scens()
        >>> print(f"After removing all: {list(loader.scen_internal_data.keys())}")
        """
        # Clear cached indices since we're modifying scenario data
        self._clear_cached_indices()
        
        if set_name is None:
            self._scen_internal_data.clear()
        elif set_name in self._scen_internal_data:
            del self._scen_internal_data[set_name]

    def _clear_cached_indices(self) -> None:
        """Clear cached scenario indices."""
        self._cached_scen_out_sample_idx = None
        self._cached_scen_in_sample_idx = None

    def _calculate_scenario_indices(self) -> Tuple[pd.Index, pd.Index]:
        """
        Calculate both in-sample and out-sample scenario indices.
        
        Returns
        -------
        Tuple[pd.Index, pd.Index]
            Tuple of (in_sample_idx, out_sample_idx)
        
        Raises
        ------
        ValueError
            If no scenario data is loaded or required columns are missing
            
        Example
        -------
        >>> # Setup loader with scenario data
        >>> loader = TimeSeriesLoader(scen_p0="2023-12-31")
        >>> # Create sample historical and scenario data
        >>> historical = pd.DataFrame({
        ...     "date": pd.date_range("2023-01-01", "2023-12-31", freq="M"),
        ...     "value": range(12)
        ... }).set_index("date")
        >>> forecast = pd.DataFrame({
        ...     "date": pd.date_range("2024-01-01", "2024-12-31", freq="M"),
        ...     "value": range(12, 24)
        ... }).set_index("date")
        >>> # Load data
        >>> loader.load(historical)
        >>> loader.load_scens(
        ...     source={"Base": forecast},
        ...     set_name="Test"
        ... )
        >>> # Calculate indices
        >>> in_idx, out_idx = loader._calculate_scenario_indices()
        >>> print("\nHistorical (in-sample) dates:")
        >>> print(f"From {in_idx.min():%Y-%m-%d} to {in_idx.max():%Y-%m-%d}")
        >>> print("\nForecast (out-of-sample) dates:")
        >>> print(f"From {out_idx.min():%Y-%m-%d} to {out_idx.max():%Y-%m-%d}")
        """
        if not self._scen_internal_data:
            raise ValueError("No scenario data loaded. Call `load_scens()` first.")
        if self.scen_p0 is None:
            return pd.Index([]), pd.Index([])
            
        in_sample_points = set()
        out_sample_points = set()
        
        for scen_set in self._scen_internal_data.values():
            for scenario in scen_set.values():
                # For panel data, use stored date_col
                if isinstance(self, PanelLoader):
                    if self.date_col not in scenario.columns:
                        raise ValueError(f"date_col '{self.date_col}' not found in scenario data")
                    time_points = pd.to_datetime(scenario[self.date_col])
                    in_sample_points.update(time_points[time_points <= self.scen_p0])
                    out_sample_points.update(time_points[time_points > self.scen_p0])
                # For time series data, use index
                else:
                    in_sample_points.update(scenario.index[scenario.index <= self.scen_p0])
                    out_sample_points.update(scenario.index[scenario.index > self.scen_p0])
        
        return (
            pd.Index(sorted(in_sample_points)),
            pd.Index(sorted(out_sample_points))
        )
        
    @property
    def scen_out_sample_idx(self) -> Optional[pd.Index]:
        """
        Get the index for data in the scenario forecasting horizon (after scen_p0_date).
        This index is derived from scenario data tables, not internal_data.
        Uses cached values for better performance.
        
        For TimeSeriesLoader, uses the DataFrame index directly.
        For PanelLoader, uses the stored date_col to determine time points.
        
        Returns
        -------
        Optional[pd.Index]
            Index of observations after the scenario jumpoff date if set and scenario data exists,
            None otherwise
            
        Example
        -------
        >>> # Setup loader with scenario data
        >>> loader = TimeSeriesLoader(scen_p0="2023-12-31")
        >>> # Create sample data
        >>> historical = pd.DataFrame({
        ...     "date": pd.date_range("2023-01-01", "2023-12-31", freq="M"),
        ...     "value": range(12)
        ... }).set_index("date")
        >>> base_scen = pd.DataFrame({
        ...     "date": pd.date_range("2024-01-01", "2024-12-31", freq="M"),
        ...     "value": range(12, 24)
        ... }).set_index("date")
        >>> adverse_scen = pd.DataFrame({
        ...     "date": pd.date_range("2024-01-01", "2024-12-31", freq="M"),
        ...     "value": range(-12, 0)
        ... }).set_index("date")
        >>> # Load data
        >>> loader.load(historical)
        >>> loader.load_scens(
        ...     source={
        ...         "Base": base_scen,
        ...         "Adverse": adverse_scen
        ...     },
        ...     set_name="Test"
        ... )
        >>> # Get forecast horizon dates
        >>> forecast_dates = loader.scen_out_sample_idx
        >>> print(f"Forecast starts: {forecast_dates.min():%Y-%m-%d}")
        >>> print(f"Forecast ends: {forecast_dates.max():%Y-%m-%d}")
        >>> print(f"Number of forecast periods: {len(forecast_dates)}")
        """
        if self._cached_scen_out_sample_idx is None:
            self._cached_scen_in_sample_idx, self._cached_scen_out_sample_idx = self._calculate_scenario_indices()
        return self._cached_scen_out_sample_idx
        
    @property
    def scen_in_sample_idx(self) -> Optional[pd.Index]:
        """
        Get the index for historical scenario data (on or before scen_p0_date).
        This index is derived from scenario data tables, not internal_data.
        Uses cached values for better performance.
        
        For TimeSeriesLoader, uses the DataFrame index directly.
        For PanelLoader, uses the stored date_col to determine time points.
        
        Returns
        -------
        Optional[pd.Index]
            Index of observations on or before the scenario jumpoff date if set and scenario data exists,
            None otherwise
            
        Example
        -------
        >>> # Setup panel loader with scenario data
        >>> loader = PanelLoader(
        ...     entity_col="firm_id",
        ...     date_col="report_date",
        ...     scen_p0="2023-12-31"
        ... )
        >>> # Create sample panel data
        >>> historical = pd.DataFrame({
        ...     "firm_id": [1, 1, 2, 2] * 6,
        ...     "report_date": pd.date_range("2023-01-01", periods=12, freq="M").repeat(2),
        ...     "value": range(24)
        ... })
        >>> base_scen = pd.DataFrame({
        ...     "firm_id": [1, 1, 2, 2] * 6,
        ...     "report_date": pd.date_range("2024-01-01", periods=12, freq="M").repeat(2),
        ...     "value": range(24, 48)
        ... })
        >>> # Load data
        >>> loader.load(historical)
        >>> loader.load_scens(
        ...     source={"Base": base_scen},
        ...     set_name="Test"
        ... )
        >>> # Get historical dates
        >>> hist_dates = loader.scen_in_sample_idx
        >>> print(f"Historical period: {hist_dates.min():%Y-%m-%d} to {hist_dates.max():%Y-%m-%d}")
        >>> print(f"Number of unique dates: {len(hist_dates)}")
        >>> print(f"Number of firms per period: "
        ...       f"{len(historical.groupby('report_date')['firm_id'].nunique().unique())}")
        """
        if self._cached_scen_in_sample_idx is None:
            self._cached_scen_in_sample_idx, self._cached_scen_out_sample_idx = self._calculate_scenario_indices()
        return self._cached_scen_in_sample_idx


class TimeSeriesLoader(DataLoader):
    """
    Base class for time series data loading.
    
    Handles single time series data with standardized period-end dates.
    Supports both monthly and quarterly frequencies.
    
    Parameters
    ----------
    in_sample_start : str, optional
        Start date for in-sample period (YYYY-MM-DD)
    in_sample_end : str, optional
        End date for in-sample period (YYYY-MM-DD)
    full_sample_start : str, optional
        Start date for full sample period (YYYY-MM-DD)
    full_sample_end : str, optional
        End date for full sample period (YYYY-MM-DD)
    freq : str, default='M'
        Frequency code ('M' for monthly, 'Q' for quarterly)
    scen_p0 : str, optional
        The month-end date that serves as the jumpoff date for scenario forecasting
        
    Example
    -------
    >>> # Load monthly time series with sample splitting
    >>> loader = TimeSeriesLoader(
    ...     in_sample_start="2020-01-01",
    ...     in_sample_end="2022-12-31",
    ...     full_sample_end="2023-12-31",
    ...     freq="M"
    ... )
    >>> loader.load("monthly_data.csv", date_col="date")
    >>> in_sample = loader.internal_data.loc[loader.in_sample_idx]
    >>> out_sample = loader.internal_data.loc[loader.out_sample_idx]
    
    >>> # Load quarterly data from DataFrame
    >>> dates = pd.date_range("2020-01-01", "2023-12-31", freq="Q")
    >>> df = pd.DataFrame({"value": range(len(dates))}, index=dates)
    >>> loader = TimeSeriesLoader(freq="Q")
    >>> loader.load(df)  # DataFrame already has datetime index
    
    >>> # Load from Excel with scenarios
    >>> loader = TimeSeriesLoader(
    ...     in_sample_start="2020-01-01",
    ...     in_sample_end="2022-12-31",
    ...     scen_p0="2023-12-31"
    ... )
    >>> loader.load("data.xlsx", date_col="Date", sheet="Historical")
    >>> loader.load_scens(
    ...     source="scenarios.xlsx",
    ...     scens={"Base": "Base_Scenario", "Adverse": "Adverse_Scenario"},
    ...     set_name="EWST2024"
    ... )
    >>> historical = loader.internal_data.loc[loader.scen_in_sample_idx]
    >>> forecast = loader.scen_internal_data["EWST2024"]["Base"].loc[loader.scen_out_sample_idx]
    """
    
    def __init__(
        self,
        in_sample_start: Optional[str] = None,
        in_sample_end: Optional[str] = None,
        full_sample_start: Optional[str] = None,
        full_sample_end: Optional[str] = None,
        freq: str = 'M',
        scen_p0: Optional[str] = None
    ):
        """
        Initialize TimeSeriesLoader.
        
        Parameters
        ----------
        in_sample_start : str, optional
            Start date for in-sample period (YYYY-MM-DD)
        in_sample_end : str, optional
            End date for in-sample period (YYYY-MM-DD)
        full_sample_start : str, optional
            Start date for full sample period (YYYY-MM-DD)
        full_sample_end : str, optional
            End date for full sample period (YYYY-MM-DD)
        freq : str, default='M'
            Frequency code ('M' for monthly, 'Q' for quarterly)
        scen_p0 : str, optional
            The month-end date that serves as the jumpoff date for scenario forecasting
        """
        super().__init__(
            freq=freq,
            full_sample_start=full_sample_start,
            full_sample_end=full_sample_end,
            scen_p0=scen_p0
        )
        self.in_sample_start = pd.to_datetime(in_sample_start).normalize() if in_sample_start else None
        self.in_sample_end = pd.to_datetime(in_sample_end).normalize() if in_sample_end else None
        
        if self.in_sample_start and self.in_sample_end and self.in_sample_start > self.in_sample_end:
            raise ValueError("in_sample_start must be before in_sample_end")

    def load(
        self,
        source: Union[str, pd.DataFrame],
        date_col: Optional[str] = None,
        sheet: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Load time series data from various sources.
        
        Parameters
        ----------
        source : str or DataFrame
            Can be:
            - Path to Excel/CSV file
            - Pre-loaded DataFrame with datetime index or date column
        date_col : str, optional
            Name of date column if source is DataFrame or file without datetime index.
            Not required if DataFrame already has datetime index.
        sheet : str, optional
            Sheet name if source is Excel file. If None, uses first sheet.
        **kwargs : dict
            Additional arguments passed to file reading functions
            
        Example
        -------
        >>> # Load from CSV file
        >>> loader = TimeSeriesLoader(freq="M")
        >>> loader.load("monthly_data.csv", date_col="transaction_date")
        
        >>> # Load from Excel with specific sheet
        >>> loader.load(
        ...     source="quarterly_data.xlsx",
        ...     date_col="Date",
        ...     sheet="Raw_Data",
        ...     skiprows=2  # Skip header rows
        ... )
        
        >>> # Load from DataFrame
        >>> df = pd.DataFrame({
        ...     "date": pd.date_range("2020-01-01", "2023-12-31", freq="M"),
        ...     "value": np.random.randn(48)
        ... })
        >>> loader.load(df, date_col="date")  # With date column
        >>> loader.load(df.set_index("date"))  # With datetime index
        """
        # Handle different source types
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:  # str path to file
            # Only pass sheet_name if sheet is explicitly provided
            if sheet is not None:
                kwargs['sheet_name'] = sheet
            df = self._load_from_file(source, **kwargs)

        # Process the DataFrame
        df = self._standardize_index(df, date_col)
        
        # Set sample indices based on time cutoff
        if self.in_sample_start and self.in_sample_end:
            self._in_sample_idx = df[
                (df.index >= self.in_sample_start) & 
                (df.index <= self.in_sample_end)
            ].index
            
            # Out-of-sample is data after in_sample_end up to full_sample_end
            out_mask = df.index > self.in_sample_end
            if self.full_sample_end:
                out_mask &= df.index <= self.full_sample_end
            self._out_sample_idx = df[out_mask].index
        else:
            self._in_sample_idx = df.index
            self._out_sample_idx = pd.Index([])

        self._internal_data = df

    def _standardize_index(self, df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """
        Convert a date column into a period-end DatetimeIndex and sort.
        Also validates and potentially updates self.freq based on data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data with either datetime index or date column
        date_col : str, optional
            Name of date column. Not required if df already has datetime index.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized DatetimeIndex
            
        Raises
        ------
        ValueError
            If data frequency doesn't match self.freq or can't be determined
        """
        df = df.copy()
        
        # Handle datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.normalize()
            dates = df.index
        else:
            if date_col is None:
                raise ValueError("date_col required when DataFrame does not have datetime index")
            if date_col not in df.columns:
                raise ValueError(f"date_col '{date_col}' not found in DataFrame")
                
            df[date_col] = pd.to_datetime(df[date_col])
            dates = df[date_col]
        
        # Validate/infer frequency
        inferred_freq = pd.infer_freq(dates.sort_values())
        if inferred_freq is None:
            raise ValueError("Could not infer data frequency. Please ensure data has regular time intervals.")
            
        # Convert inferred frequency to M or Q
        if inferred_freq.startswith('M'):
            data_freq = 'M'
        elif inferred_freq.startswith('Q'):
            data_freq = 'Q'
        else:
            raise ValueError(f"Data frequency {inferred_freq} not supported. Must be monthly (M) or quarterly (Q).")
            
        # Update or validate self.freq
        if self.freq is None:
            self.freq = data_freq
        elif self.freq != data_freq:
            raise ValueError(f"Data frequency ({data_freq}) does not match specified frequency ({self.freq})")
        
        # Standardize to period end dates
        if isinstance(df.index, pd.DatetimeIndex):
            periods = pd.PeriodIndex(df.index, freq=self.freq)
            df.index = periods.to_timestamp(how='end').normalize()
        else:
            periods = pd.PeriodIndex(df[date_col], freq=self.freq)
            idx = periods.to_timestamp(how='end').normalize()
            df.index = idx
            df.drop(columns=[date_col], inplace=True)

        return df.sort_index()

    @property
    def p0(self) -> Optional[pd.Timestamp]:
        """
        Get the date index just ahead of in_sample_start.
        
        Returns
        -------
        pd.Timestamp or None
            Date index just ahead of in_sample_start, or None if 
            in_sample_start is the earliest record or not specified.
            
        Example
        -------
        >>> loader = TimeSeriesLoader(in_sample_start="2020-02-01")
        >>> loader.load("data.csv", date_col="date")
        >>> # If data starts from 2020-01-31, p0 would be 2020-01-31
        >>> # If data starts from 2020-02-01 or later, p0 would be None
        >>> p0_date = loader.p0
        """
        if self.in_sample_start is None or self._internal_data is None:
            return None
            
        # Get dates before in_sample_start
        available_dates = self._internal_data.index
        dates_before = available_dates[available_dates < self.in_sample_start]
        
        if dates_before.empty:
            # in_sample_start is the earliest record
            return None
        else:
            # Return the latest date before in_sample_start
            return dates_before.max()

    @property
    def out_p0(self) -> Optional[pd.Timestamp]:
        """
        Get the date index of in_sample_end.
        
        Returns
        -------
        pd.Timestamp or None
            Date index of in_sample_end if specified, None otherwise.
            
        Example
        -------
        >>> loader = TimeSeriesLoader(
        ...     in_sample_start="2020-01-01",
        ...     in_sample_end="2022-12-31"
        ... )
        >>> loader.load("data.csv", date_col="date")
        >>> out_p0_date = loader.out_p0  # Returns 2022-12-31
        """
        return self.in_sample_end


class PanelLoader(DataLoader):
    """
    Base class for panel/longitudinal data loading.
    
    Handles entity-level time series data with multiple entities.
    Maintains standardized period-end dates and ensures unique entity-date pairs.
    
    Parameters
    ----------
    entity_col : str, default='entity_id'
        Name of the column containing entity identifiers
    date_col : str, default='date'
        Name of the column containing dates
    split_method : Union[str, SplitMethod], default=SplitMethod.RANDOM
        Method to split data into in-sample and out-of-sample
    test_size : float, default=0.2
        Proportion of data for out-of-sample (for random/stratified sampling)
    random_seed : int, default=42
        Random seed for reproducible sampling
    in_sample_start : str, optional
        Start date for in-sample period (if using time_cutoff)
    in_sample_end : str, optional
        End date for in-sample period (if using time_cutoff)
    full_sample_start : str, optional
        Start date for full sample period
    full_sample_end : str, optional
        End date for full sample period
    freq : str, default='M'
        Frequency code ('M' for monthly, 'Q' for quarterly)
    scen_p0 : str, optional
        The month-end date that serves as the jumpoff date for scenario forecasting
        
    Example
    -------
    >>> # Random splitting of panel data
    >>> loader = PanelLoader(
    ...     entity_col="customer_id",
    ...     date_col="transaction_date",
    ...     split_method="random",
    ...     test_size=0.2
    ... )
    >>> loader.load("customer_data.csv")
    >>> in_sample = loader.internal_data.loc[loader.in_sample_idx]
    >>> out_sample = loader.internal_data.loc[loader.out_sample_idx]
    
    >>> # Stratified splitting by time period
    >>> df = pd.DataFrame({
    ...     "account_id": [1, 1, 2, 2] * 3,
    ...     "date": pd.date_range("2023-01-01", periods=12, freq="M").repeat(2),
    ...     "value": np.random.randn(12)
    ... })
    >>> loader = PanelLoader(
    ...     entity_col="account_id",
    ...     split_method="stratified",
    ...     test_size=0.25
    ... )
    >>> loader.load(df, date_col="date")  # Each month will have ~25% of accounts in test set
    
    >>> # Time-based splitting with scenarios
    >>> loader = PanelLoader(
    ...     entity_col="firm_id",
    ...     date_col="report_date",
    ...     split_method="time_cutoff",
    ...     in_sample_start="2020-01-01",
    ...     in_sample_end="2022-12-31",
    ...     scen_p0="2023-12-31"
    ... )
    >>> loader.load("firm_data.xlsx", sheet="Historical")
    >>> loader.load_scens(
    ...     source="scenarios.xlsx",
    ...     scens={"Base": "Base_Scenario", "Stress": "Stress_Scenario"},
    ...     set_name="CCAR2024"
    ... )
    >>> historical = loader.internal_data.loc[loader.scen_in_sample_idx]
    >>> forecast = loader.scen_internal_data["CCAR2024"]["Base"].loc[loader.scen_out_sample_idx]
    """
    
    def __init__(
        self,
        entity_col: str = 'entity_id',
        date_col: str = 'date',
        split_method: Union[str, SplitMethod] = SplitMethod.RANDOM,
        test_size: float = 0.2,
        random_seed: int = 42,
        in_sample_start: Optional[str] = None,
        in_sample_end: Optional[str] = None,
        full_sample_start: Optional[str] = None,
        full_sample_end: Optional[str] = None,
        freq: str = 'M',
        scen_p0: Optional[str] = None
    ):
        """
        Initialize PanelLoader.
        
        Parameters
        ----------
        entity_col : str, default='entity_id'
            Name of the column containing entity identifiers
        date_col : str, default='date'
            Name of the column containing dates
        split_method : str or SplitMethod, default=SplitMethod.RANDOM
            Method to use for splitting data into in/out samples
        test_size : float, default=0.2
            Fraction of data to use for out-of-sample testing
        random_seed : int, default=42
            Random seed for reproducible splits
        in_sample_start : str, optional
            Start date for in-sample period (YYYY-MM-DD)
        in_sample_end : str, optional
            End date for in-sample period (YYYY-MM-DD)
        full_sample_start : str, optional
            Start date for full sample period (YYYY-MM-DD)
        full_sample_end : str, optional
            End date for full sample period (YYYY-MM-DD)
        freq : str, default='M'
            Frequency code ('M' for monthly, 'Q' for quarterly)
        scen_p0 : str, optional
            The month-end date that serves as the jumpoff date for scenario forecasting
        """
        super().__init__(
            freq=freq,
            full_sample_start=full_sample_start,
            full_sample_end=full_sample_end,
            scen_p0=scen_p0
        )
        self.entity_col = entity_col
        self.date_col = date_col
        
        # Convert split_method to enum if string
        if isinstance(split_method, str):
            try:
                split_method = SplitMethod(split_method.lower())
            except ValueError:
                raise ValueError(f"Invalid split_method: {split_method}")
        self.split_method = split_method
        
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.test_size = test_size
        self.random_seed = random_seed
        
        self.in_sample_start = pd.to_datetime(in_sample_start).normalize() if in_sample_start else None
        self.in_sample_end = pd.to_datetime(in_sample_end).normalize() if in_sample_end else None
        
        if self.in_sample_start and self.in_sample_end and self.in_sample_start > self.in_sample_end:
            raise ValueError("in_sample_start must be before in_sample_end")

    def load(
        self,
        source: Union[str, pd.DataFrame],
        date_col: Optional[str] = None,
        sheet: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Load panel data from various sources.
        
        Parameters
        ----------
        source : str or DataFrame
            Can be:
            - Path to Excel/CSV file
            - Pre-loaded DataFrame
        date_col : str, optional
            Name of date column. If provided, overrides the date_col specified in __init__
        sheet : str, optional
            Sheet name if source is Excel file. If None, uses first sheet.
        **kwargs : dict
            Additional arguments passed to file reading functions
            
        Example
        -------
        >>> # Load from CSV with custom date column
        >>> loader = PanelLoader(entity_col="customer_id")
        >>> loader.load("transactions.csv", date_col="transaction_date")
        
        >>> # Load from Excel with specific sheet
        >>> loader = PanelLoader(entity_col="account_id", date_col="report_date")
        >>> loader.load("account_data.xlsx", sheet="Monthly_Data")
        
        >>> # Load from DataFrame with date standardization
        >>> df = pd.DataFrame({
        ...     "customer_id": [1, 1, 2, 2],
        ...     "date": ["2023-01-15", "2023-02-15", "2023-01-15", "2023-02-15"],
        ...     "balance": [100, 150, 200, 250]
        ... })
        >>> loader = PanelLoader(entity_col="customer_id")
        >>> loader.load(df, date_col="date")  # Dates standardized to month-end
        """
        # Update date_col if provided
        if date_col is not None:
            self.date_col = date_col

        # Handle different source types
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:  # str path to file
            # Only pass sheet_name if sheet is explicitly provided
            if sheet is not None:
                kwargs['sheet_name'] = sheet
            df = self._load_from_file(source, **kwargs)

        # Validate and process the data
        self._validate_structure(df, self.date_col)
        df = self._process_data(df, self.date_col)
        
        # Split the data according to the specified method
        self._split_samples(df)
        
        self._internal_data = df

    def _validate_structure(self, df: pd.DataFrame, date_col: str):
        """
        Validate the data structure meets panel data requirements.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data to validate
        date_col : str
            Name of the date column

        Raises
        ------
        ValueError
            If required columns are missing or data contains duplicate entity-date pairs
            
        Example
        -------
        >>> # Valid panel data structure
        >>> df_valid = pd.DataFrame({
        ...     "entity_id": [1, 1, 2, 2],
        ...     "date": ["2023-01-01", "2023-02-01", "2023-01-01", "2023-02-01"],
        ...     "value": [100, 110, 200, 220]
        ... })
        >>> loader = PanelLoader()
        >>> loader._validate_structure(df_valid, "date")  # No error
        
        >>> # Invalid: duplicate entity-date pair
        >>> df_invalid = pd.DataFrame({
        ...     "entity_id": [1, 1, 1],
        ...     "date": ["2023-01-01", "2023-01-01", "2023-02-01"],
        ...     "value": [100, 101, 110]
        ... })
        >>> loader._validate_structure(df_invalid, "date")  # Raises ValueError
        """
        # Check required columns exist
        missing_cols = []
        for col in [date_col, self.entity_col]:
            if col not in df.columns:
                missing_cols.append(col)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Check date column can be parsed
        try:
            pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Invalid date format in column '{date_col}': {str(e)}")
            
        # Check for duplicate date-entity pairs
        duplicates = df.duplicated([date_col, self.entity_col], keep=False)
        if duplicates.any():
            dup_records = df[duplicates].sort_values([self.entity_col, date_col])
            raise ValueError(
                f"Found duplicate date-entity pairs. First few duplicates:\n"
                f"{dup_records[[self.entity_col, date_col]].head()}"
            )

    def _process_data(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Process panel data: standardize dates to period-end and ensure proper structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Validated raw data
        date_col : str
            Name of the date column
            
        Returns
        -------
        pd.DataFrame
            Processed data with standardized dates and entity structure
            
        Example
        -------
        >>> # Monthly data standardization
        >>> df = pd.DataFrame({
        ...     "entity_id": [1, 1, 2, 2],
        ...     "date": ["2023-01-15", "2023-02-15", "2023-01-20", "2023-02-20"],
        ...     "value": [100, 110, 200, 220]
        ... })
        >>> loader = PanelLoader(freq="M")
        >>> processed = loader._process_data(df, "date")  # Dates -> month-end
        
        >>> # Quarterly data standardization
        >>> df_q = pd.DataFrame({
        ...     "entity_id": [1, 1, 2, 2],
        ...     "date": ["2023-02-15", "2023-05-15", "2023-02-28", "2023-05-31"],
        ...     "value": [100, 110, 200, 220]
        ... })
        >>> loader_q = PanelLoader(freq="Q")
        >>> processed_q = loader_q._process_data(df_q, "date")  # Dates -> quarter-end
        """
        df = df.copy()
        
        # Standardize dates to period-end
        df[date_col] = pd.to_datetime(df[date_col])
        df['period'] = df[date_col].dt.to_period(self.freq)
        df[date_col] = df['period'].dt.to_timestamp(how='end').dt.normalize()
        
        # Sort by entity and date, reset index
        df = df.sort_values([self.entity_col, date_col]).reset_index(drop=True)
        
        return df

    def _split_samples(self, df: pd.DataFrame):
        """
        Split the data into in-sample and out-of-sample based on specified method.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed data to split
            
        Example
        -------
        >>> # Random splitting
        >>> df = pd.DataFrame({
        ...     "entity_id": [1, 1, 2, 2] * 3,
        ...     "date": pd.date_range("2023-01-01", periods=12, freq="M").repeat(2),
        ...     "value": range(24)
        ... })
        >>> loader = PanelLoader(split_method="random", test_size=0.25)
        >>> loader._split_samples(df)
        >>> in_sample = df.loc[loader.in_sample_idx]
        >>> out_sample = df.loc[loader.out_sample_idx]
        
        >>> # Stratified splitting (by time period)
        >>> loader = PanelLoader(split_method="stratified", test_size=0.5)
        >>> loader._split_samples(df)  # Each period has ~50% entities in test set
        
        >>> # Time cutoff splitting
        >>> loader = PanelLoader(
        ...     split_method="time_cutoff",
        ...     in_sample_start="2023-01-01",
        ...     in_sample_end="2023-06-30"
        ... )
        >>> loader._split_samples(df)  # Split based on dates
        """
        np.random.seed(self.random_seed)
        
        # First, get the indices within full sample period if specified
        sample_mask = pd.Series(True, index=df.index)
        if self.full_sample_start:
            sample_mask &= df[date_col] >= self.full_sample_start
        if self.full_sample_end:
            sample_mask &= df[date_col] <= self.full_sample_end
        
        eligible_idx = df[sample_mask].index
        
        if self.split_method == SplitMethod.RANDOM:
            # Simple random sampling from eligible indices
            n_samples = len(eligible_idx)
            n_test = int(n_samples * self.test_size)
            test_idx = np.random.choice(eligible_idx, n_test, replace=False)
            self._out_sample_idx = pd.Index(test_idx)
            self._in_sample_idx = eligible_idx[~eligible_idx.isin(self._out_sample_idx)]
            
        elif self.split_method == SplitMethod.STRATIFIED:
            # Stratified sampling by period from eligible data
            eligible_df = df.loc[eligible_idx]
            test_indices = []
            for period in eligible_df['period'].unique():
                period_idx = eligible_df[eligible_df['period'] == period].index
                n_period_test = int(len(period_idx) * self.test_size)
                if n_period_test > 0:  # Only sample if we have data to sample
                    period_test_idx = np.random.choice(period_idx, n_period_test, replace=False)
                    test_indices.extend(period_test_idx)
            self._out_sample_idx = pd.Index(test_indices)
            self._in_sample_idx = eligible_idx[~eligible_idx.isin(self._out_sample_idx)]
            
        else:  # TIME_CUTOFF
            self._in_sample_idx = df[
                (df[date_col] >= self.in_sample_start) & 
                (df[date_col] <= self.in_sample_end)
            ].index
            
            # Out-of-sample is data after in_sample_end up to full_sample_end
            out_mask = df[date_col] > self.in_sample_end
            if self.full_sample_end:
                out_mask &= df[date_col] <= self.full_sample_end
            self._out_sample_idx = df[out_mask].index


class PPNRInternalLoader(TimeSeriesLoader):
    """
    Data loader for PPNR (Pre-Provision Net Revenue) internal data.
    
    This loader handles time series data specific to PPNR modeling,
    including proper date handling and sample splitting.
    
    Parameters
    ----------
    in_sample_start : str, optional
        Start date for in-sample period (YYYY-MM-DD)
    in_sample_end : str, optional
        End date for in-sample period (YYYY-MM-DD)
    full_sample_start : str, optional
        Start date for full sample period (YYYY-MM-DD)
    full_sample_end : str, optional
        End date for full sample period (YYYY-MM-DD)
    freq : str, default='M'
        Frequency code ('M' for monthly, 'Q' for quarterly)
    scen_p0 : str, optional
        The month-end date that serves as the jumpoff date for scenario forecasting
        
    Example
    -------
    >>> # Load PPNR data with scenarios
    >>> loader = PPNRInternalLoader(
    ...     in_sample_start="2020-01-01",
    ...     in_sample_end="2022-12-31",
    ...     scen_p0="2023-12-31"
    ... )
    >>> # Create sample data
    >>> historical = pd.DataFrame({
    ...     "date": pd.date_range("2020-01-01", "2023-12-31", freq="M"),
    ...     "net_interest_income": np.random.normal(1000, 50, 48),
    ...     "non_interest_income": np.random.normal(500, 30, 48),
    ...     "operating_expense": np.random.normal(800, 40, 48)
    ... })
    >>> base_scen = pd.DataFrame({
    ...     "date": pd.date_range("2024-01-01", "2025-12-31", freq="M"),
    ...     "net_interest_income": np.random.normal(1100, 60, 24),
    ...     "non_interest_income": np.random.normal(550, 35, 24),
    ...     "operating_expense": np.random.normal(850, 45, 24)
    ... })
    >>> # Load and process data
    >>> loader.load(historical, date_col="date")
    >>> loader.load_scens(
    ...     source={"Base": base_scen},
    ...     set_name="CCAR2024",
    ...     date_col="date"
    ... )
    >>> # Calculate PPNR components
    >>> historical_ppnr = (
    ...     loader.internal_data["net_interest_income"] +
    ...     loader.internal_data["non_interest_income"] -
    ...     loader.internal_data["operating_expense"]
    ... )
    >>> base_ppnr = (
    ...     loader.scen_internal_data["CCAR2024"]["Base"]["net_interest_income"] +
    ...     loader.scen_internal_data["CCAR2024"]["Base"]["non_interest_income"] -
    ...     loader.scen_internal_data["CCAR2024"]["Base"]["operating_expense"]
    ... )
    """
    
    def __init__(
        self,
        in_sample_start: Optional[str] = None,
        in_sample_end: Optional[str] = None,
        full_sample_start: Optional[str] = None,
        full_sample_end: Optional[str] = None,
        freq: str = 'M',
        scen_p0: Optional[str] = None
    ):
        # Call parent's __init__ with all parameters
        super().__init__(
            in_sample_start=in_sample_start,
            in_sample_end=in_sample_end,
            full_sample_start=full_sample_start,
            full_sample_end=full_sample_end,
            freq=freq,
            scen_p0=scen_p0
        )


class SMRInternalLoader(PanelLoader):
    """
    Legacy class name for PanelLoader.
    Maintained for backward compatibility.
    
    This class implements the same functionality as PanelLoader
    but keeps the original name for existing code compatibility.
    Defaults to using 'account_id' as the entity column name.
    
    Example
    -------
    >>> # Load account-level data with scenarios
    >>> loader = SMRInternalLoader(
    ...     split_method="random",
    ...     test_size=0.2
    ... )
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     "account_id": [1001, 1001, 1002, 1002] * 3,
    ...     "date": pd.date_range("2023-01-01", periods=6, freq="M").repeat(2),
    ...     "balance": np.random.lognormal(10, 0.5, 12),
    ...     "interest_rate": np.random.normal(0.05, 0.01, 12)
    ... })
    >>> base_scen = pd.DataFrame({
    ...     "account_id": [1001, 1001, 1002, 1002] * 2,
    ...     "date": pd.date_range("2024-01-01", periods=4, freq="M").repeat(2),
    ...     "balance": np.random.lognormal(10, 0.6, 8),
    ...     "interest_rate": np.random.normal(0.06, 0.015, 8)
    ... })
    >>> # Load and process data
    >>> loader.load(df, date_col="date")
    >>> loader.load_scens(
    ...     source={"Base": base_scen},
    ...     set_name="CCAR2024"
    ... )
    >>> # Analyze data
    >>> historical = loader.internal_data.loc[loader.scen_in_sample_idx]
    >>> forecast = loader.scen_internal_data["CCAR2024"]["Base"]
    """
    
    def __init__(self, *args, **kwargs):
        # Override entity_col default to maintain backward compatibility
        if 'entity_col' not in kwargs and 'account_col' in kwargs:
            kwargs['entity_col'] = kwargs.pop('account_col')
        elif 'entity_col' not in kwargs:
            kwargs['entity_col'] = 'account_id'
        super().__init__(*args, **kwargs)
