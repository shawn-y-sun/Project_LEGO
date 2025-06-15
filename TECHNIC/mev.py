# =============================================================================
# module: mev.py
# Purpose: Load and manage Macro Economic Variables (MEVs) from various sources
# Dependencies: pandas, yaml, pathlib, typing
# =============================================================================

import os
from pathlib import Path
import pandas as pd
import yaml
import warnings
from typing import Dict, Any, Tuple, Optional, Union, List, Set

# Determine support directory relative to this module file using pathlib
_BASE_DIR = Path(__file__).resolve().parent
_SUPPORT_DIR = _BASE_DIR / 'support'
_DEFAULT_MEV_TYPE_PATH = _SUPPORT_DIR / 'mev_type.xlsx'
_DEFAULT_TSFM_PATH = _SUPPORT_DIR / 'type_tsfm.yaml'

def _process_quarterly_excel(workbook: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Process quarterly MEV data from Excel with standard format.
    
    Parameters
    ----------
    workbook : str
        Path to Excel workbook
    sheet : str, optional
        Sheet name to process. If None, uses first sheet.
        
    Returns
    -------
    DataFrame
        Processed DataFrame with datetime index
    """
    raw = pd.read_excel(workbook, sheet_name=sheet)
    df = raw.copy()
    df_mev = df.iloc[1:]
    df_mev.columns = df_mev.iloc[0]

    df_mev.index = df_mev.iloc[:, 0]
    df_mev = df_mev.iloc[1:, 1:]
    df_mev = df_mev.loc[:, df_mev.columns.notna()].iloc[2:]

    df_mev.columns = df_mev.iloc[0]
    df_mev = df_mev.iloc[1:, :]

    # Reformat index to timestamps at period end
    df_mev.index = [i.replace(':', 'Q') for i in df_mev.index]
    df_mev.index = pd.PeriodIndex(df_mev.index, freq='Q').to_timestamp(how='end')
    df_mev.index = pd.to_datetime(df_mev.index).normalize()
    
    return df_mev

def _process_monthly_excel(workbook: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Process monthly MEV data from Excel with standard format.
    Assumes data starts from row 1 with dates in first column.
    
    Parameters
    ----------
    workbook : str
        Path to Excel workbook
    sheet : str, optional
        Sheet name to process. If None, uses first sheet.
        
    Returns
    -------
    DataFrame
        Processed DataFrame with datetime index
    """
    df = pd.read_excel(workbook, sheet_name=sheet)
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df

class MEVLoader:
    """
    Loader for Macro Economic Variables (MEVs) from various sources.
    Supports loading from Excel files or pre-loaded DataFrames.
    Can handle both monthly and quarterly frequencies for base MEVs.
    Handles scenario MEVs separately with internal processing.
    
    The loader maintains two types of MEVs:
    1. Model MEVs: Base MEVs used for model training/testing
    2. Scenario MEVs: Multiple sets of scenario MEVs for forecasting
    
    Parameters
    ----------
    mev_map : dict, optional
        Direct mapping of MEV codes to type/description dicts.
        If provided, overrides mev_type_path.
        Example: {
            'GDP': {'type': 'level', 'description': 'Gross Domestic Product'},
            'UNRATE': {'type': 'rate', 'description': 'Unemployment Rate'}
        }
    tsfm_map : dict, optional
        Direct mapping of types to transform lists.
        If provided, overrides tsfm_path.
        Example: {
            'level': ['log', 'diff'],
            'rate': ['diff']
        }
    mev_type_path : str or Path, optional
        Path to Excel file containing MEV type mappings.
        Must have columns: mev_code, type, description.
        If None, uses default from support/mev_type.xlsx.
    tsfm_path : str or Path, optional
        Path to YAML file containing transform mappings.
        Must have a 'transforms' key mapping types to transform lists.
        If None, uses default from support/type_tsfm.yaml.
        
    Examples
    --------
    Basic Usage:
    >>> # Initialize loader
    >>> loader = MEVLoader()
    >>> 
    >>> # Load model MEVs from Excel
    >>> loader.load("model_mevs.xlsx")
    >>> 
    >>> # Load scenario MEVs from Excel with multiple scenarios
    >>> loader.load_scens(
    ...     "EWST2024.xlsx",
    ...     scens={"Base": "Base", "Adv": "Adverse", "Sev": "Severe"}
    ... )
    
    Working with Model MEVs:
    >>> # Load from DataFrame
    >>> loader.load(df_mevs, date_col="Date")
    >>> 
    >>> # Access model MEVs
    >>> qtr_mevs = loader.model_mev_qtr  # Quarterly MEVs
    >>> mth_mevs = loader.model_mev_mth  # Monthly MEVs
    >>> 
    >>> # Clean model MEVs
    >>> loader.clean_model_mevs()
    
    Working with Scenario MEVs:
    >>> # Load multiple scenario sets
    >>> loader.load_scens(
    ...     "EWST2024.xlsx",
    ...     scens={"Base": "Base", "Adv": "Adverse"}
    ... )
    >>> loader.load_scens(
    ...     "GRST2024.xlsx",
    ...     scens={"Base": "Base", "Sev": "Severe"}
    ... )
    >>> 
    >>> # Access scenario MEVs
    >>> ewst_scens = loader.scen_mev_qtr["EWST2024"]  # All EWST scenarios
    >>> base_scen = ewst_scens["Base"]  # Base scenario data
    >>> 
    >>> # Update specific scenarios
    >>> loader.update_scen_mevs(
    ...     {"Base": df_update, "Adv": df_update},
    ...     set_name="EWST2024"
    ... )
    >>> 
    >>> # Clean specific scenario set
    >>> loader.clean_scen_mevs(set_name="EWST2024")
    >>> 
    >>> # Clean all scenarios
    >>> loader.clean_scen_mevs()
    
    Advanced Usage:
    >>> # Load external data with custom preprocessing
    >>> loader.load(
    ...     "external_mevs.csv",
    ...     external=True,
    ...     date_col="DATE"
    ... )
    >>> 
    >>> # Update multiple scenario sets at once
    >>> loader.update_scen_mevs({
    ...     "EWST2024": {
    ...         "Base": df_ewst_base,
    ...         "Adv": df_ewst_adv
    ...     },
    ...     "GRST2024": {
    ...         "Base": df_grst_base,
    ...         "Sev": df_grst_sev
    ...     }
    ... })
    >>> 
    >>> # Get MEV metadata
    >>> mev_info = loader.get_mev_info("GDP")
    >>> mev_type = mev_info["type"]
    >>> mev_desc = mev_info["description"]
    
    Notes
    -----
    - All dates are normalized to midnight UTC
    - Supports both monthly and quarterly frequencies
    - Automatically validates MEV codes against mev_map
    - Handles index alignment when updating data
    - Maintains separate containers for model and scenario MEVs
    - Scenario MEVs are organized in a three-layer structure:
      {set_name: {scenario_name: DataFrame}}
    
    See Also
    --------
    DataManager : Higher-level class that combines MEVs with internal data
    """
    def __init__(
        self,
        mev_map: Optional[Dict[str, Dict[str, str]]] = None,
        tsfm_map: Optional[Dict[str, List[str]]] = None,
        mev_type_path: Optional[Union[str, Path]] = None,
        tsfm_path: Optional[Union[str, Path]] = None
    ):
        # Load or set mapping tables first
        self._mev_map = self._load_mev_map(mev_type_path) if mev_map is None else mev_map
        self._tsfm_map = self._load_tsfm_map(tsfm_path) if tsfm_map is None else tsfm_map
        
        # Initialize empty data containers for model MEVs
        self._model_mev_qtr: Optional[pd.DataFrame] = None
        self._model_mev_mth: Optional[pd.DataFrame] = None
        
        # Initialize empty containers for scenario MEVs
        self._scen_mev_qtr: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._scen_mev_mth: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Track all MEV codes
        self._mev_codes: Set[str] = set()
        
    def _load_mev_map(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, str]]:
        """Load MEV type mapping from Excel file."""
        file_path = Path(path) if path else _DEFAULT_MEV_TYPE_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"MEV type mapping file not found: {file_path}")
            
        df = pd.read_excel(file_path)
        required_cols = {'mev_code', 'type', 'description'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Expected columns {required_cols} in {file_path}")
            
        return {
            code: {'type': type_, 'description': desc}
            for code, type_, desc in zip(df['mev_code'], df['type'], df['description'])
        }
        
    def _load_tsfm_map(self, path: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
        """Load transform mapping from YAML file."""
        file_path = Path(path) if path else _DEFAULT_TSFM_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"Transform mapping file not found: {file_path}")
            
        with open(file_path) as f:
            spec = yaml.safe_load(f)
            
        if 'transforms' not in spec or not isinstance(spec['transforms'], dict):
            raise KeyError(f"Key 'transforms' missing or invalid format in {file_path}")
            
        return spec['transforms']
        
    def load(
        self,
        source: Union[str, pd.DataFrame],
        sheet: Optional[str] = None,
        freq: Optional[str] = None,
        external: bool = False,
        date_col: Optional[str] = None
    ) -> None:
        """
        Load base MEV data from various sources. If MEVs with same codes already exist,
        they will be updated with new values. New MEV codes will be appended.
        
        Parameters
        ----------
        source : str or DataFrame
            Can be:
            - Path to Excel/CSV file
            - Pre-loaded DataFrame with datetime index or date column
        sheet : str, optional
            Sheet name if source is Excel file. If None, uses first sheet.
        freq : str, optional
            Expected frequency ('M' for monthly, 'Q' for quarterly).
            If not provided, will be inferred from data.
        external : bool, default False
            If True, treats file source as external data without preprocessing.
            If False and source is a file, uses preprocessing functions.
        date_col : str, optional
            Name of date column if source is DataFrame or external file without datetime index.
            Not required if DataFrame already has datetime index.
        """
        # Handle different source types
        if isinstance(source, pd.DataFrame):
            data = self._load_from_df(source, date_col, freq)
        else:  # str path to file
            if external:
                # Load external file directly
                if source.lower().endswith('.csv'):
                    data = pd.read_csv(source)
                    data = self._load_from_df(data, date_col, freq)
                else:  # Excel
                    data = pd.read_excel(source, sheet_name=sheet)
                    data = self._load_from_df(data, date_col, freq)
            else:
                # Use preprocessing functions
                data = self._load_from_excel(source, sheet, freq)
        
        # Store data in appropriate container based on frequency
        if data.index.inferred_freq.startswith('Q'):
            if self._model_mev_qtr is None:
                self._model_mev_qtr = data
            else:
                # Update existing MEVs and append new ones
                existing_cols = self._model_mev_qtr.columns
                new_cols = data.columns
                
                # Align indices for proper update
                combined_idx = self._model_mev_qtr.index.union(data.index)
                self._model_mev_qtr = self._model_mev_qtr.reindex(combined_idx)
                data = data.reindex(combined_idx)
                
                # Update existing columns and append new ones
                self._model_mev_qtr[new_cols] = data[new_cols]
                
        elif data.index.inferred_freq.startswith('M'):
            if self._model_mev_mth is None:
                self._model_mev_mth = data
            else:
                # Update existing MEVs and append new ones
                existing_cols = self._model_mev_mth.columns
                new_cols = data.columns
                
                # Align indices for proper update
                combined_idx = self._model_mev_mth.index.union(data.index)
                self._model_mev_mth = self._model_mev_mth.reindex(combined_idx)
                data = data.reindex(combined_idx)
                
                # Update existing columns and append new ones
                self._model_mev_mth[new_cols] = data[new_cols]
                
        else:
            raise ValueError(
                f"Unsupported data frequency: {data.index.inferred_freq}. "
                "Only monthly (M) and quarterly (Q) frequencies are supported."
            )
            
        # Update MEV codes and validate
        self._mev_codes.update(data.columns)
        self._validate_mev_codes()

    def load_scens(
        self,
        source: Union[str, Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]]],
        scens: Optional[Dict[str, str]] = None,
        set_name: Optional[str] = None,
        freq: Optional[str] = None
    ) -> None:
        """
        Load scenario MEVs from either an Excel source or a dictionary structure.
        Creates or updates a scenario set containing multiple sub-scenarios.
        
        The method supports two loading modes:
        1. From Excel file:
           - source: path to Excel workbook
           - scens: mapping of scenario names to sheet names
           - set_name: optional name for the scenario set
           
        2. From dictionary (3-layer structure):
           - source: dictionary with structure {set_name: {scen_name: DataFrame}}
           - scens and set_name parameters are ignored
        
        Parameters
        ----------
        source : str or dict
            Either:
            - Path to Excel workbook containing scenario data, or
            - Dictionary with structure {set_name: {scen_name: DataFrame}}
        scens : dict, optional
            Required only when loading from Excel.
            Mapping of scenario names to sheet names
            Example: {"Base": "BaseSheet", "Adv": "AdverseSheet"}
        set_name : str, optional
            Used only when loading from Excel.
            Name of the scenario set. If None, uses the source filename without extension.
            Example: If source is 'EWST2024.xlsx', set_name defaults to 'EWST2024'
        freq : str, optional
            Expected frequency ('M' for monthly, 'Q' for quarterly).
            If not provided, will be inferred from data.
            
        Examples
        --------
        >>> loader = MEVLoader()
        >>> # 1. Load from Excel file
        >>> loader.load_scens(
        ...     "EWST2024.xlsx",
        ...     scens={"Base": "Base", "Adv": "Adverse"}
        ... )
        >>> 
        >>> # 2. Load from dictionary
        >>> loader.load_scens({
        ...     "EWST2024": {
        ...         "Base": df_base,
        ...         "Adv": df_adverse
        ...     }
        ... })
        """
        if isinstance(source, str):
            # Loading from Excel file
            if scens is None:
                raise ValueError("'scens' parameter required when loading from Excel file")
                
            # Determine scenario set name if not provided
            if set_name is None:
                set_name = Path(source).stem
                
            # Initialize containers for this scenario set if not exist
            if set_name not in self._scen_mev_qtr:
                self._scen_mev_qtr[set_name] = {}
            if set_name not in self._scen_mev_mth:
                self._scen_mev_mth[set_name] = {}
                
            # Load each sub-scenario from Excel sheets
            for scen_name, sheet in scens.items():
                data = self._load_from_excel(source, sheet, freq)
                
                # Store in appropriate scenario container based on frequency
                if data.index.inferred_freq.startswith('Q'):
                    self._scen_mev_qtr[set_name][scen_name] = data
                elif data.index.inferred_freq.startswith('M'):
                    self._scen_mev_mth[set_name][scen_name] = data
                else:
                    raise ValueError(
                        f"Unsupported data frequency: {data.index.inferred_freq}. "
                        "Only monthly (M) and quarterly (Q) frequencies are supported."
                    )
                    
                # Update MEV codes and validate
                self._mev_codes.update(data.columns)
                
        else:
            # Loading from dictionary structure
            for curr_set_name, scenarios in source.items():
                # Initialize containers for this scenario set if not exist
                if curr_set_name not in self._scen_mev_qtr:
                    self._scen_mev_qtr[curr_set_name] = {}
                if curr_set_name not in self._scen_mev_mth:
                    self._scen_mev_mth[curr_set_name] = {}
                    
                # Process each scenario
                for scen_name, data in scenarios.items():
                    # Convert Series to DataFrame if necessary
                    if isinstance(data, pd.Series):
                        data = data.to_frame()
                        
                    # Validate and store based on frequency
                    if not isinstance(data.index, pd.DatetimeIndex):
                        raise ValueError(
                            f"Data for scenario '{scen_name}' must have datetime index"
                        )
                        
                    inferred_freq = pd.infer_freq(data.index)
                    if inferred_freq is None:
                        raise ValueError(
                            f"Could not infer frequency from data for scenario '{scen_name}'"
                        )
                        
                    # Store in appropriate container based on frequency
                    if inferred_freq.startswith('Q'):
                        self._scen_mev_qtr[curr_set_name][scen_name] = data
                    elif inferred_freq.startswith('M'):
                        self._scen_mev_mth[curr_set_name][scen_name] = data
                    else:
                        raise ValueError(
                            f"Unsupported data frequency: {inferred_freq}. "
                            "Only monthly (M) and quarterly (Q) frequencies are supported."
                        )
                        
                    # Update MEV codes and validate
                    self._mev_codes.update(data.columns)
                    
        self._validate_mev_codes()

    def _load_from_excel(
        self,
        source: str,
        sheet: Optional[str],
        freq: Optional[str]
    ) -> pd.DataFrame:
        """Load and process data from Excel file using preprocessing functions."""
        if freq and freq.upper() not in ['M', 'Q']:
            raise ValueError("freq must be 'M' or 'Q' if specified")
            
        # Try processing as quarterly first if freq not specified
        if not freq or freq.upper() == 'Q':
            try:
                df = _process_quarterly_excel(source, sheet)
                if pd.infer_freq(df.index).startswith('Q'):
                    return df
            except Exception as e:
                if freq and freq.upper() == 'Q':
                    raise ValueError(f"Failed to process quarterly Excel file: {e}")
                
        # Try processing as monthly
        try:
            df = _process_monthly_excel(source, sheet)
            if pd.infer_freq(df.index).startswith('M'):
                return df
            raise ValueError("Data frequency not monthly")
        except Exception as e:
            raise ValueError(f"Failed to process Excel file as either quarterly or monthly: {e}")
            
    def _load_from_df(
        self,
        df: pd.DataFrame,
        date_col: Optional[str],
        freq: Optional[str]
    ) -> pd.DataFrame:
        """
        Process DataFrame data.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame with either datetime index or date column
        date_col : str, optional
            Name of date column. Not required if df already has datetime index.
        freq : str, optional
            Expected frequency ('M' for monthly, 'Q' for quarterly)
            
        Returns
        -------
        DataFrame
            Processed DataFrame with datetime index
        """
        if freq and freq.upper() not in ['M', 'Q']:
            raise ValueError("freq must be 'M' or 'Q' if specified")
            
        # Create copy of input
        data = df.copy()
        
        # Handle datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            # Already has datetime index, just normalize
            data.index = data.index.normalize()
        else:
            # Need to set index from date column
            if date_col is None:
                raise ValueError(
                    "date_col required when DataFrame does not have datetime index"
                )
            if date_col not in data.columns:
                raise ValueError(f"date_col '{date_col}' not found in DataFrame")
                
            data.set_index(date_col, inplace=True)
            data.index = pd.to_datetime(data.index).normalize()
        
        # Validate frequency
        inferred_freq = pd.infer_freq(data.index)
        if not inferred_freq:
            raise ValueError("Could not infer frequency from data")
            
        if freq:
            if not inferred_freq.startswith(freq.upper()):
                raise ValueError(
                    f"Data frequency ({inferred_freq}) doesn't match specified ({freq})"
                )
        elif not inferred_freq.startswith(('M', 'Q')):
            raise ValueError(
                f"Unsupported data frequency: {inferred_freq}. "
                "Only monthly (M) and quarterly (Q) frequencies are supported."
            )
            
        return data
        
    def _validate_mev_codes(self) -> None:
        """Validate that all MEV codes are in the MEV_MAP."""
        missing_codes = [code for code in self._mev_codes if code not in self._mev_map]
        if missing_codes:
            warnings.warn(
                f"The following MEV codes are not in MEV_MAP: {missing_codes}\n"
                "Please add them to mev_type.xlsx with appropriate type and description.",
                UserWarning
            )
            
    @property
    def model_mev_qtr(self) -> pd.DataFrame:
        """Get base quarterly MEV data."""
        if self._model_mev_qtr is None:
            return pd.DataFrame()
        return self._model_mev_qtr
        
    @property
    def model_mev_mth(self) -> pd.DataFrame:
        """Get base monthly MEV data."""
        if self._model_mev_mth is None:
            return pd.DataFrame()
        return self._model_mev_mth
        
    @property
    def scen_mev_qtr(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get quarterly scenario MEVs.
        
        Returns
        -------
        dict
            A nested dictionary with structure:
            {set_name: {scenario_name: DataFrame}}
        """
        return self._scen_mev_qtr
        
    @property
    def scen_mev_mth(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get monthly scenario MEVs.
        
        Returns
        -------
        dict
            A nested dictionary with structure:
            {set_name: {scenario_name: DataFrame}}
        """
        return self._scen_mev_mth
        
    @property
    def mev_codes(self) -> List[str]:
        """Get list of all MEV codes."""
        return sorted(list(self._mev_codes))
        
    @property
    def mev_map(self) -> Dict[str, Dict[str, str]]:
        """Get the MEV type mapping."""
        return self._mev_map
        
    @property
    def tsfm_map(self) -> Dict[str, List[str]]:
        """Get the transform mapping."""
        return self._tsfm_map
        
    def get_mev_info(self, mev_code: str) -> Dict[str, str]:
        """
        Get type and description for a MEV code.
        
        Parameters
        ----------
        mev_code : str
            The MEV code to look up
            
        Returns
        -------
        dict
            Dictionary with 'type' and 'description' keys
            
        Raises
        ------
        KeyError
            If mev_code not found in MEV_MAP
        """
        if mev_code not in self._mev_map:
            raise KeyError(f"MEV code '{mev_code}' not found in MEV_MAP")
        return self._mev_map[mev_code]

    def clean_model_mevs(self) -> None:
        """
        Clear cached model MEVs (both quarterly and monthly).
        """
        self._model_mev_qtr = None
        self._model_mev_mth = None

    def clean_scen_mevs(self, set_name: Optional[str] = None) -> None:
        """
        Clear cached scenario MEVs.
        
        Parameters
        ----------
        set_name : str, optional
            Name of the scenario set to clean. If None, cleans all scenario sets.
            Example: 'EWST2024' will only clean that specific scenario set.
        """
        if set_name is None:
            # Clean all scenario sets
            self._scen_mev_qtr.clear()
            self._scen_mev_mth.clear()
        else:
            # Clean specific scenario set
            if set_name in self._scen_mev_qtr:
                del self._scen_mev_qtr[set_name]
            if set_name in self._scen_mev_mth:
                del self._scen_mev_mth[set_name]

    def clean_all(self) -> None:
        """
        Clear all cached MEVs (both model and scenario MEVs).
        This includes:
        - Model MEVs (quarterly and monthly)
        - All scenario sets
        - MEV codes tracking
        """
        # Clean model MEVs
        self.clean_model_mevs()
        
        # Clean all scenario MEVs
        self.clean_scen_mevs()
        
        # Clean MEV codes tracking
        self._mev_codes.clear()

    def update_scen_mevs(
        self,
        updates: Dict[str, Union[pd.DataFrame, pd.Series, Dict[str, Union[pd.DataFrame, pd.Series]]]],
        set_name: Optional[str] = None
    ) -> None:
        """
        Update scenario MEVs with new data. Supports both two-layer and three-layer dictionary inputs.
        
        Parameters
        ----------
        updates : dict
            Either:
            1. Two-layer dictionary mapping scenario names to update data:
               {"Base": df_update1, "Adv": df_update2, "Sev": df_update3}
               In this case, set_name must be provided if multiple scenario sets exist.
            2. Three-layer dictionary matching the structure of scen_mev_qtr/mth:
               {"EWST2024": {"Base": df_update1, "Adv": df_update2, "Sev": df_update3}}
               In this case, set_name parameter is ignored.
            
            The update values can be either DataFrames or Series. If Series, they will
            be converted to DataFrames.
            
        set_name : str, optional
            Required only when using two-layer dictionary and multiple scenario sets exist.
            Specifies which scenario set to update.
            
        Examples
        --------
        >>> # Update with two-layer dictionary
        >>> loader.update_scen_mevs(
        ...     {"Base": df_base_update, "Adv": df_adv_update},
        ...     set_name="EWST2024"
        ... )
        >>> 
        >>> # Update with three-layer dictionary
        >>> loader.update_scen_mevs({
        ...     "EWST2024": {
        ...         "Base": df_base_update,
        ...         "Adv": df_adv_update
        ...     }
        ... })
        >>> 
        >>> # Update with Series
        >>> loader.update_scen_mevs(
        ...     {"Base": new_gdp_series, "Adv": new_gdp_series},
        ...     set_name="EWST2024"
        ... )
        
        Raises
        ------
        ValueError
            If using two-layer dictionary without set_name when multiple sets exist,
            or if trying to update non-existent scenarios.
        """
        # Determine if input is three-layer or two-layer dictionary
        is_three_layer = any(isinstance(v, dict) for v in updates.values())
        
        if is_three_layer:
            update_dict = updates
        else:
            # Convert two-layer to three-layer
            if set_name is None:
                if len(self._scen_mev_qtr) + len(self._scen_mev_mth) > 1:
                    raise ValueError(
                        "set_name must be provided when using two-layer dictionary "
                        "and multiple scenario sets exist"
                    )
                # If only one set exists, use its name
                qtr_sets = set(self._scen_mev_qtr.keys())
                mth_sets = set(self._scen_mev_mth.keys())
                set_name = list(qtr_sets.union(mth_sets))[0]
            
            update_dict = {set_name: updates}
        
        # Process each scenario set
        for curr_set_name, scen_updates in update_dict.items():
            # Validate scenario set exists
            if (curr_set_name not in self._scen_mev_qtr and 
                curr_set_name not in self._scen_mev_mth):
                raise ValueError(f"Scenario set '{curr_set_name}' does not exist")
            
            # Process each scenario update
            for scen_name, update_data in scen_updates.items():
                # Convert Series to DataFrame if necessary
                if isinstance(update_data, pd.Series):
                    update_data = update_data.to_frame()
                
                # Determine frequency and update appropriate container
                freq = pd.infer_freq(update_data.index)
                if freq is None:
                    raise ValueError(
                        f"Could not infer frequency from update data for scenario '{scen_name}'"
                    )
                
                if freq.startswith('Q'):
                    if curr_set_name not in self._scen_mev_qtr:
                        self._scen_mev_qtr[curr_set_name] = {}
                    if scen_name not in self._scen_mev_qtr[curr_set_name]:
                        self._scen_mev_qtr[curr_set_name][scen_name] = update_data
                    else:
                        # Update existing DataFrame
                        existing_df = self._scen_mev_qtr[curr_set_name][scen_name]
                        
                        # Align indices
                        combined_idx = existing_df.index.union(update_data.index)
                        existing_df = existing_df.reindex(combined_idx)
                        update_data = update_data.reindex(combined_idx)
                        
                        # Update columns
                        for col in update_data.columns:
                            existing_df[col] = update_data[col]
                        
                        self._scen_mev_qtr[curr_set_name][scen_name] = existing_df
                        
                elif freq.startswith('M'):
                    if curr_set_name not in self._scen_mev_mth:
                        self._scen_mev_mth[curr_set_name] = {}
                    if scen_name not in self._scen_mev_mth[curr_set_name]:
                        self._scen_mev_mth[curr_set_name][scen_name] = update_data
                    else:
                        # Update existing DataFrame
                        existing_df = self._scen_mev_mth[curr_set_name][scen_name]
                        
                        # Align indices
                        combined_idx = existing_df.index.union(update_data.index)
                        existing_df = existing_df.reindex(combined_idx)
                        update_data = update_data.reindex(combined_idx)
                        
                        # Update columns
                        for col in update_data.columns:
                            existing_df[col] = update_data[col]
                        
                        self._scen_mev_mth[curr_set_name][scen_name] = existing_df
                else:
                    raise ValueError(
                        f"Unsupported frequency '{freq}' in update data for scenario '{scen_name}'. "
                        "Only monthly (M) and quarterly (Q) frequencies are supported."
                    )
                
                # Update MEV codes tracking
                self._mev_codes.update(update_data.columns)
        
        # Validate all MEV codes
        self._validate_mev_codes()
