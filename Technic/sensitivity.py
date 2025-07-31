import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import matplotlib.pyplot as plt
import os

from .scenario import ScenManager
from .model import ModelBase
from .data import DataManager
from .internal import PanelLoader, TimeSeriesLoader


class SensitivityTest:
    """
    Manages parameter and input sensitivity testing for scenario forecasting models.
    This class provides comprehensive sensitivity analysis capabilities for fitted models
    by systematically varying parameters and inputs to assess their impact on forecasts.

    Parameters
    ----------
    scen_manager : ScenManager
        Fitted ScenManager instance with scenario forecasting capabilities
    param_shock : List[float], default=[1]
        List of parameter shock multipliers to apply during sensitivity testing.
        Values represent multipliers (e.g., [0.5, 1, 1.5] for 50% reduction, baseline, 50% increase)
    input_shock : List[float], default=[1, 2]
        List of input shock multipliers to apply during sensitivity testing.
        Values represent multipliers for input variable variations

    Attributes
    ----------
    scen_manager : ScenManager
        Reference to the provided ScenManager instance
    dm : DataManager
        Data manager from ScenManager
    model : ModelBase
        Fitted model from ScenManager
    target : str
        Target variable name from ScenManager
    horizon_frame : Tuple[pd.Timestamp, pd.Timestamp]
        Forecast horizon time frame from ScenManager
    param_shock : List[float]
        Parameter shock multipliers for sensitivity testing
    input_shock : List[float]
        Input shock multipliers for sensitivity testing
    param_sensitivity_results : Dict[str, Dict[str, pd.DataFrame]]
        Results from parameter sensitivity testing
    input_sensitivity_results : Dict[str, Dict[str, pd.DataFrame]]
        Results from input sensitivity testing
    param_names : List[str]
        Parameter names suitable for sensitivity testing (non-constant, non-dummy variables)
    X_std : pd.Series
        Standard deviations for parameters suitable for input sensitivity testing
    y_param_shock : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Structured results of parameter shock testing across scenarios and parameters
    y_input_shock : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Structured results of input shock testing across scenarios and parameters
    param_shock_df : Dict[str, Dict[str, pd.DataFrame]]
        Summarized DataFrames combining baseline and parameter shock results for each scenario
    input_shock_df : Dict[str, Dict[str, pd.DataFrame]]
        Summarized DataFrames combining baseline and input shock results for each scenario

    Example
    -------
    # Create ScenManager first
    scen_mgr = ScenManager(model, horizon=9)
    
    # Create SensitivityTest with default parameters
    sens_test = SensitivityTest(scen_mgr)
    
    # Create SensitivityTest with custom shock values
    sens_test = SensitivityTest(
        scen_mgr, 
        param_shock=[0.5, 1, 1.5], 
        input_shock=[0.8, 1, 1.2]
    )
    
    # Run parameter sensitivity testing
    param_results = sens_test.run_param_shock(X_new, 'pricing')
    
    # Run input sensitivity testing
    input_results = sens_test.run_input_shock(X_new, 'pricing')
    
    # Access shock results
    param_shock_results = sens_test.y_param_shock
    input_shock_results = sens_test.y_input_shock
    
    # Plot sensitivity results
    sens_test.plot_param_shock('EWST_2024', 'base')
    sens_test.plot_input_shock('EWST_2024', 'base')
    """
    
    def __init__(
        self,
        scen_manager: ScenManager,
        param_shock: List[float] = [1],
        input_shock: List[float] = [1, 2, 3]
    ):
        # Validate ScenManager instance
        if not isinstance(scen_manager, ScenManager):
            raise ValueError("scen_manager must be a ScenManager instance")
        
        # Store ScenManager reference and extract key attributes
        self.scen_manager = scen_manager
        self.dm = scen_manager.dm
        self.model = scen_manager.model
        self.target = scen_manager.target
        self.horizon_frame = scen_manager.horizon_frame
        
        # Validate and store shock parameters
        if not isinstance(param_shock, list) or not param_shock:
            raise ValueError("param_shock must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in param_shock):
            raise ValueError("All param_shock values must be numeric")
        self.param_shock = param_shock
        
        if not isinstance(input_shock, list) or not input_shock:
            raise ValueError("input_shock must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in input_shock):
            raise ValueError("All input_shock values must be numeric")
        self.input_shock = input_shock
        
        # Initialize results storage
        self.param_sensitivity_results = {}
        self.input_sensitivity_results = {}
        
        # Get scenario information for reference
        self.scenarios = list(scen_manager.y_scens.keys()) if scen_manager.y_scens else []
        self.scenario_colors = scen_manager.scenario_colors

    @property
    def param_names(self) -> List[str]:
        """
        Get parameter names suitable for sensitivity testing.
        
        Returns parameter names from model.spec_map["StationarityTest"], 
        excluding constant and dummy variables.
        
        Returns
        -------
        List[str]
            List of parameter names suitable for sensitivity testing
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> params = sens_test.param_names
        >>> print(params)
        ['pricing', 'GDP', 'UNRATE']
        """
        if not hasattr(self.model, 'spec_map'):
            return []
        
        spec_map = self.model.spec_map
        if "StationarityTest" not in spec_map:
            return []
        
        # Get stationarity test variables (excludes dummies and constants)
        stationarity_vars = spec_map["StationarityTest"]
        
        # Filter to only include variables that exist in model parameters
        available_params = []
        if hasattr(self.model, 'params'):
            model_params = self.model.params.keys()
        elif hasattr(self.model, 'coefficients'):
            model_params = self.model.coefficients.keys()
        else:
            return []
        
        for var in stationarity_vars:
            if var in model_params:
                available_params.append(var)
        
        return available_params

    @property
    def X_std(self) -> pd.Series:
        """
        Get standard deviations for all parameters suitable for input sensitivity testing.
        
        Calculates standard deviations from self.model.X_full up to self.dm.scen_p0
        for parameters in self.param_names.
        
        Returns
        -------
        pd.Series
            Standard deviations indexed by parameter names
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> stds = sens_test.X_std
        >>> print(stds)
        pricing    0.15
        GDP        0.25
        UNRATE     0.12
        """
        if not self.param_names:
            return pd.Series(dtype=float)
        
        if not hasattr(self.dm, 'scen_p0') or self.dm.scen_p0 is None:
            # If scen_p0 is not available, use all available data
            X_data = self.model.X_full
        else:
            # Filter data up to scen_p0
            X_data = self.model.X_full[self.model.X_full.index <= self.dm.scen_p0]
        
        # Calculate standard deviations for parameters in param_names
        std_values = {}
        for param in self.param_names:
            if param in X_data.columns:
                std_values[param] = X_data[param].std()
            else:
                std_values[param] = 0.0  # Default to 0 if parameter not found
        
        return pd.Series(std_values)

    def run_param_shock(self, X_new: pd.DataFrame, param: str) -> pd.DataFrame:
        """
        Run parameter shock testing for a specific parameter.
        
        This method applies different shock multipliers to a model parameter and
        returns predictions for both positive and negative shocks.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            Input features for prediction (typically from self.scen_manager.X_scens)
        param : str
            Name of the parameter to shock
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each shock level (e.g., 'pricing+1se', 'pricing-1se')
            Each column contains predictions for that specific parameter shock
            
        Example
        -------
        >>> # Run parameter shock testing for 'pricing' parameter
        >>> X_new = scen_manager.X_scens['EWST_2024']['base']
        >>> results = sens_test.run_param_shock(X_new, 'pricing')
        >>> print(results.columns)
        Index(['pricing+1se', 'pricing+2se', 'pricing-1se', 'pricing-2se'])
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted before running parameter shock testing")
        
        if param not in self.param_names:
            raise ValueError(f"Parameter '{param}' not suitable for sensitivity testing")
        
        results = {}
        
        # Loop through each shock value in self.param_shock
        for shock in self.param_shock:
            # Apply positive shock
            pos_shock_name = f"{param}+{shock}se"
            try:
                pos_predictions = self.model.predict_param_shock(X_new, param, shock)
                results[pos_shock_name] = pos_predictions
            except Exception as e:
                warnings.warn(f"Failed to apply positive shock {shock} to parameter '{param}': {e}")
                results[pos_shock_name] = pd.Series([np.nan] * len(X_new), index=X_new.index)
            
            # Apply negative shock
            neg_shock_name = f"{param}-{shock}se"
            try:
                neg_predictions = self.model.predict_param_shock(X_new, param, -shock)
                results[neg_shock_name] = neg_predictions
            except Exception as e:
                warnings.warn(f"Failed to apply negative shock -{shock} to parameter '{param}': {e}")
                results[neg_shock_name] = pd.Series([np.nan] * len(X_new), index=X_new.index)
        
        return pd.DataFrame(results)

    def run_input_shock(self, X_new: pd.DataFrame, param: str) -> pd.DataFrame:
        """
        Run input shock testing for a specific parameter.
        
        This method applies different shock multipliers to input variable values and
        returns predictions for both positive and negative shocks.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            Input features for prediction (typically from self.scen_manager.X_scens)
        param : str
            Name of the input parameter to shock
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each shock level (e.g., 'pricing+1sd', 'pricing-1sd')
            Each column contains predictions for that specific input shock
            
        Example
        -------
        >>> # Run input shock testing for 'pricing' parameter
        >>> X_new = scen_manager.X_scens['EWST_2024']['base']
        >>> results = sens_test.run_input_shock(X_new, 'pricing')
        >>> print(results.columns)
        Index(['pricing+1sd', 'pricing+2sd', 'pricing-1sd', 'pricing-2sd'])
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted before running input shock testing")
        
        if param not in self.param_names:
            raise ValueError(f"Parameter '{param}' not suitable for sensitivity testing")
        
        if param not in X_new.columns:
            raise ValueError(f"Parameter '{param}' not found in X_new columns")
        
        # Get standard deviation for this parameter
        if param not in self.X_std.index:
            raise ValueError(f"Standard deviation not available for parameter '{param}'")
        
        param_std = self.X_std[param]
        if param_std == 0:
            warnings.warn(f"Standard deviation is zero for parameter '{param}', using 1.0 as default")
            param_std = 1.0
        
        results = {}
        
        # Loop through each shock value in self.input_shock
        for shock in self.input_shock:
            # Apply positive shock
            pos_shock_name = f"{param}+{shock}sd"
            try:
                pos_predictions = self.model.predict_input_shock(X_new, param, shock, param_std)
                results[pos_shock_name] = pos_predictions
            except Exception as e:
                warnings.warn(f"Failed to apply positive input shock {shock} to parameter '{param}': {e}")
                results[pos_shock_name] = pd.Series([np.nan] * len(X_new), index=X_new.index)
            
            # Apply negative shock
            neg_shock_name = f"{param}-{shock}sd"
            try:
                neg_predictions = self.model.predict_input_shock(X_new, param, -shock, param_std)
                results[neg_shock_name] = neg_predictions
            except Exception as e:
                warnings.warn(f"Failed to apply negative input shock -{shock} to parameter '{param}': {e}")
                results[neg_shock_name] = pd.Series([np.nan] * len(X_new), index=X_new.index)
        
        return pd.DataFrame(results)

    @property
    def y_param_shock(self) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Calculate target variable (y) after applying each parameter shock separately.
        
        This property computes shocked predictions for all relevant parameters
        across all scenario sets and scenarios using the run_param_shock() method.
        
        Returns
        -------
        Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
            4-layer nested dictionary structure:
            - 1st layer: scenario set name (e.g., 'EWST_2024')
            - 2nd layer: scenario name (e.g., 'base', 'adv', 'sev')
            - 3rd layer: parameter name (e.g., 'pricing', 'GDP')
            - 4th layer: DataFrame from run_param_shock() for that parameter
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> shock_results = sens_test.y_param_shock
        >>> 
        >>> # Access results for specific scenario set, scenario, and parameter
        >>> base_pricing_results = shock_results['EWST_2024']['base']['pricing']
        >>> print(base_pricing_results.columns)
        Index(['pricing+1se', 'pricing+2se', 'pricing-1se', 'pricing-2se'])
        """
        if not hasattr(self.scen_manager, 'X_scens') or not self.scen_manager.X_scens:
            return {}
        
        if not self.param_names:
            warnings.warn("No parameters available for sensitivity testing")
            return {}
        
        results = {}
        
        # Loop through each scenario set
        for scen_set_name, scen_set_data in self.scen_manager.X_scens.items():
            results[scen_set_name] = {}
            
            # Loop through each scenario in the set
            for scen_name, X_new in scen_set_data.items():
                results[scen_set_name][scen_name] = {}
                
                # Loop through each parameter
                for param in self.param_names:
                    try:
                        # Run parameter shock testing for this parameter
                        param_results = self.run_param_shock(X_new, param)
                        results[scen_set_name][scen_name][param] = param_results
                    except Exception as e:
                        warnings.warn(f"Failed to run parameter shock for '{param}' in scenario '{scen_name}': {e}")
                        # Create empty DataFrame with expected columns
                        expected_cols = []
                        for shock in self.param_shock:
                            expected_cols.extend([f"{param}+{shock}se", f"{param}-{shock}se"])
                        results[scen_set_name][scen_name][param] = pd.DataFrame(
                            {col: [np.nan] * len(X_new) for col in expected_cols},
                            index=X_new.index
                        )
        
        return results

    @property
    def y_input_shock(self) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Calculate target variable (y) after applying each input shock separately.
        
        This property computes shocked predictions for all relevant parameters
        across all scenario sets and scenarios using the run_input_shock() method.
        
        Returns
        -------
        Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
            4-layer nested dictionary structure:
            - 1st layer: scenario set name (e.g., 'EWST_2024')
            - 2nd layer: scenario name (e.g., 'base', 'adv', 'sev')
            - 3rd layer: parameter name (e.g., 'pricing', 'GDP')
            - 4th layer: DataFrame from run_input_shock() for that parameter
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> shock_results = sens_test.y_input_shock
        >>> 
        >>> # Access results for specific scenario set, scenario, and parameter
        >>> base_pricing_results = shock_results['EWST_2024']['base']['pricing']
        >>> print(base_pricing_results.columns)
        Index(['pricing+1sd', 'pricing+2sd', 'pricing-1sd', 'pricing-2sd'])
        """
        if not hasattr(self.scen_manager, 'X_scens') or not self.scen_manager.X_scens:
            return {}
        
        if not self.param_names:
            warnings.warn("No parameters available for input sensitivity testing")
            return {}
        
        results = {}
        
        # Loop through each scenario set
        for scen_set_name, scen_set_data in self.scen_manager.X_scens.items():
            results[scen_set_name] = {}
            
            # Loop through each scenario in the set
            for scen_name, X_new in scen_set_data.items():
                results[scen_set_name][scen_name] = {}
                
                # Loop through each parameter
                for param in self.param_names:
                    try:
                        # Run input shock testing for this parameter
                        param_results = self.run_input_shock(X_new, param)
                        results[scen_set_name][scen_name][param] = param_results
                    except Exception as e:
                        warnings.warn(f"Failed to run input shock for '{param}' in scenario '{scen_name}': {e}")
                        # Create empty DataFrame with expected columns
                        expected_cols = []
                        for shock in self.input_shock:
                            expected_cols.extend([f"{param}+{shock}sd", f"{param}-{shock}sd"])
                        results[scen_set_name][scen_name][param] = pd.DataFrame(
                            {col: [np.nan] * len(X_new) for col in expected_cols},
                            index=X_new.index
                        )
        
        return results

    @property
    def param_shock_df(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create summarized DataFrames for each scenario combining baseline predictions with parameter shock results.
        
        Each DataFrame contains:
        - First column: baseline prediction from self.scen_manager.y_scens with name '{scenario_set}_{scenario}'
        - Remaining columns: all parameter shock results concatenated from self.y_param_shock
        
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            3-layer nested dictionary structure:
            - 1st layer: scenario set name (e.g., 'EWST_2024')
            - 2nd layer: scenario name (e.g., 'base', 'adv', 'sev')
            - 3rd layer: summarized DataFrame with baseline + all parameter shocks
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> summary_dfs = sens_test.param_shock_df
        >>> 
        >>> # Access summarized DataFrame for specific scenario
        >>> base_summary = summary_dfs['EWST_2024']['base']
        >>> print(base_summary.columns)
        Index(['EWST_2024_base', 'pricing+1se', 'pricing+2se', 'pricing-1se', 
               'pricing-2se', 'GDP+1se', 'GDP+2se', 'GDP-1se', 'GDP-2se'])
        """
        if not hasattr(self.scen_manager, 'y_scens') or not self.scen_manager.y_scens:
            return {}
        
        # Get parameter shock results
        param_shock_results = self.y_param_shock
        if not param_shock_results:
            return {}
        
        results = {}
        
        # Loop through each scenario set
        for scen_set_name, scen_set_data in self.scen_manager.y_scens.items():
            results[scen_set_name] = {}
            
            # Loop through each scenario in the set
            for scen_name, baseline_pred in scen_set_data.items():
                # Create the baseline column name
                baseline_col_name = f"{scen_set_name}_{scen_name}"
                
                # Start with baseline predictions as first column
                # Ensure baseline_pred is a pandas Series
                if not isinstance(baseline_pred, pd.Series):
                    if hasattr(baseline_pred, '__len__') and len(baseline_pred) > 0:
                        # Convert numpy array or list to pandas Series
                        baseline_pred = pd.Series(baseline_pred, name=baseline_col_name)
                    else:
                        # Handle empty or scalar values
                        baseline_pred = pd.Series([], name=baseline_col_name)
                
                summary_df = pd.DataFrame({baseline_col_name: baseline_pred})
                
                # Check if we have parameter shock results for this scenario
                if (scen_set_name in param_shock_results and 
                    scen_name in param_shock_results[scen_set_name]):
                    
                    # Get all parameter shock DataFrames for this scenario
                    param_results = param_shock_results[scen_set_name][scen_name]
                    
                    # Concatenate all parameter shock columns
                    for param_name, param_df in param_results.items():
                        # Ensure the parameter DataFrame has the same index as baseline
                        if not param_df.empty and param_df.index.equals(summary_df.index):
                            # Add all columns from this parameter's shock results
                            for col in param_df.columns:
                                summary_df[col] = param_df[col]
                        else:
                            # Handle case where indices don't match or DataFrame is empty
                            warnings.warn(f"Index mismatch or empty DataFrame for parameter '{param_name}' in scenario '{scen_name}'")
                            # Add columns with NaN values
                            for col in param_df.columns:
                                summary_df[col] = np.nan
                
                results[scen_set_name][scen_name] = summary_df
        
        return results

    @property
    def input_shock_df(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create summarized DataFrames for each scenario combining baseline predictions with input shock results.
        
        Each DataFrame contains:
        - First column: baseline prediction from self.scen_manager.y_scens with name '{scenario_set}_{scenario}'
        - Remaining columns: all input shock results concatenated from self.y_input_shock
        
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            3-layer nested dictionary structure:
            - 1st layer: scenario set name (e.g., 'EWST_2024')
            - 2nd layer: scenario name (e.g., 'base', 'adv', 'sev')
            - 3rd layer: summarized DataFrame with baseline + all input shocks
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> summary_dfs = sens_test.input_shock_df
        >>> 
        >>> # Access summarized DataFrame for specific scenario
        >>> base_summary = summary_dfs['EWST_2024']['base']
        >>> print(base_summary.columns)
        Index(['EWST_2024_base', 'pricing+1sd', 'pricing+2sd', 'pricing-1sd', 
               'pricing-2sd', 'GDP+1sd', 'GDP+2sd', 'GDP-1sd', 'GDP-2sd'])
        """
        if not hasattr(self.scen_manager, 'y_scens') or not self.scen_manager.y_scens:
            return {}
        
        # Get input shock results
        input_shock_results = self.y_input_shock
        if not input_shock_results:
            return {}
        
        results = {}
        
        # Loop through each scenario set
        for scen_set_name, scen_set_data in self.scen_manager.y_scens.items():
            results[scen_set_name] = {}
            
            # Loop through each scenario in the set
            for scen_name, baseline_pred in scen_set_data.items():
                # Create the baseline column name
                baseline_col_name = f"{scen_set_name}_{scen_name}"
                
                # Start with baseline predictions as first column
                # Ensure baseline_pred is a pandas Series
                if not isinstance(baseline_pred, pd.Series):
                    if hasattr(baseline_pred, '__len__') and len(baseline_pred) > 0:
                        # Convert numpy array or list to pandas Series
                        baseline_pred = pd.Series(baseline_pred, name=baseline_col_name)
                    else:
                        # Handle empty or scalar values
                        baseline_pred = pd.Series([], name=baseline_col_name)
                
                summary_df = pd.DataFrame({baseline_col_name: baseline_pred})
                
                # Check if we have input shock results for this scenario
                if (scen_set_name in input_shock_results and 
                    scen_name in input_shock_results[scen_set_name]):
                    
                    # Get all input shock DataFrames for this scenario
                    param_results = input_shock_results[scen_set_name][scen_name]
                    
                    # Concatenate all input shock columns
                    for param_name, param_df in param_results.items():
                        # Ensure the parameter DataFrame has the same index as baseline
                        if not param_df.empty and param_df.index.equals(summary_df.index):
                            # Add all columns from this parameter's shock results
                            for col in param_df.columns:
                                summary_df[col] = param_df[col]
                        else:
                            # Handle case where indices don't match or DataFrame is empty
                            warnings.warn(f"Index mismatch or empty DataFrame for input parameter '{param_name}' in scenario '{scen_name}'")
                            # Add columns with NaN values
                            for col in param_df.columns:
                                summary_df[col] = np.nan
                
                results[scen_set_name][scen_name] = summary_df
        
        return results

    def plot_shock(self, scenario_set: str, scenario_name: str, shock_data: Dict[str, Dict[str, pd.DataFrame]], 
                   shock_type: str = "param") -> None:
        """
        Shared plotting method for parameter and input shock results.
        
        Creates a grid of plots (3 columns per row) showing baseline predictions
        and shock results for each parameter. Each plot shows:
        - Baseline prediction as solid blue line
        - Shock results as dashed lines with different colors
        
        Parameters
        ----------
        scenario_set : str
            Name of the scenario set to plot
        scenario_name : str
            Name of the specific scenario to plot
        shock_data : Dict[str, Dict[str, pd.DataFrame]]
            Shock data dictionary (either param_shock_df or input_shock_df)
        shock_type : str, default="param"
            Type of shock for identification ("param" or "input")
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> sens_test.plot_shock('EWST_2024', 'base', sens_test.param_shock_df, 'param')
        """
        # Get the summarized DataFrame for this scenario
        if scenario_set not in shock_data or scenario_name not in shock_data[scenario_set]:
            raise ValueError(f"Scenario '{scenario_name}' not found in scenario set '{scenario_set}'")
        
        df = shock_data[scenario_set][scenario_name]
        if df.empty:
            warnings.warn(f"No data available for scenario '{scenario_name}' in scenario set '{scenario_set}'")
            return
        
        # Get baseline column name
        baseline_col = f"{scenario_set}_{scenario_name}"
        if baseline_col not in df.columns:
            warnings.warn(f"Baseline column '{baseline_col}' not found in data")
            return
        
        # Get parameter names from the data based on shock type
        param_names = []
        suffix = "se" if shock_type == "param" else "sd"
        for col in df.columns:
            if col != baseline_col and f'+' in col and col.endswith(suffix):
                # Extract parameter name from shock column (e.g., 'pricing+1se' -> 'pricing')
                param_name = col.split('+')[0]
                if param_name not in param_names:
                    param_names.append(param_name)
        
        if not param_names:
            warnings.warn(f"No {shock_type} shock columns found in data")
            return
        
        # Calculate grid dimensions
        n_params = len(param_names)
        n_cols = min(3, n_params)  # Don't create more columns than needed
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure and subplots with appropriate width and reduced height
        fig_width = 5 * n_cols  # Adjust width based on actual number of columns
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 3 * n_rows))
        
        # Handle axes array dimensions
        # When we have only 1 subplot, plt.subplots returns a single Axes object, not an array
        if n_params == 1:
            axes = [axes]  # Convert single Axes to list for consistent indexing
        elif n_rows == 1 and n_cols > 1:
            # Single row, multiple columns - axes is already 1D array
            pass  # No reshaping needed
        elif n_cols == 1 and n_rows > 1:
            # Single column, multiple rows - axes is already 1D array  
            pass  # No reshaping needed
        else:
            # Multiple rows and columns - axes is already 2D array
            pass  # No reshaping needed
        
        # Define colors for shock lines
        shock_colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        # Plot each parameter
        for i, param in enumerate(param_names):
            if n_params == 1:
                # Single parameter - axes is now a list with one element
                ax = axes[0]
            elif n_rows == 1:
                # Single row - axes is 1D array, index directly
                ax = axes[i]
            elif n_cols == 1:
                # Single column - axes is 1D array, index directly
                ax = axes[i]
            else:
                # Multiple rows and columns - axes is 2D array
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
            
            # Plot baseline with simplified label (just scenario name)
            baseline_data = df[baseline_col]
            
            # Ensure baseline_data is a pandas Series with proper index
            if not isinstance(baseline_data, pd.Series):
                # Convert to pandas Series if it's a numpy array
                baseline_data = pd.Series(baseline_data, index=df.index, name=baseline_col)
            
            ax.plot(baseline_data.index, baseline_data.values, 'b-', linewidth=2, label=scenario_name)
            
            # Plot shock results with simplified labels (just shock value)
            shock_cols = [col for col in df.columns if col.startswith(param + '+') or col.startswith(param + '-')]
            shock_cols.sort()  # Sort to ensure consistent order
            
            for j, shock_col in enumerate(shock_cols):
                color = shock_colors[j % len(shock_colors)]
                shock_data_vals = df[shock_col]
                
                # Ensure shock_data_vals is a pandas Series with proper index
                if not isinstance(shock_data_vals, pd.Series):
                    # Convert to pandas Series if it's a numpy array
                    shock_data_vals = pd.Series(shock_data_vals, index=df.index, name=shock_col)
                
                # Extract just the shock part (e.g., 'pricing+1se' -> '+1se')
                shock_part = shock_col[len(param):]
                ax.plot(shock_data_vals.index, shock_data_vals.values, '--', color=color, 
                       linewidth=1.5, label=shock_part)
            
            # Set simplified title (just parameter name)
            ax.set_title(param, fontsize=10)
            
            # Remove axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Set smaller font size for tick labels to match x-axis
            ax.tick_params(axis='y', labelsize=8)
            
            # Move legend inside the plot
            ax.legend(loc='upper right', fontsize=8)
            
            # Rotate x-axis labels and format as yy-mm for quarter-end months only
            if not df.empty:
                # Get time index and format it
                time_index = df.index
                if hasattr(time_index, 'strftime'):
                    # If it's a datetime index, format it and filter for quarter-end months
                    quarter_end_labels = []
                    quarter_end_indices = []
                    for k, t in enumerate(time_index):
                        month = t.month
                        if month in [3, 6, 9, 12]:  # Quarter-end months
                            quarter_end_labels.append(t.strftime('%y-%m'))
                            quarter_end_indices.append(k)
                    
                    if quarter_end_indices:
                        ax.set_xticks([time_index[k] for k in quarter_end_indices])
                        ax.set_xticklabels(quarter_end_labels, rotation=45, ha='right', fontsize=8)
                else:
                    # If it's not datetime, try to convert or use as is
                    formatted_labels = [str(t)[:7] for t in time_index]  # Take first 7 chars
                    
                    # Show fewer x-axis labels to prevent crowding
                    n_ticks = min(8, len(time_index))  # Max 8 labels
                    tick_indices = list(range(0, len(time_index), max(1, len(time_index) // n_ticks)))
                    if tick_indices[-1] != len(time_index) - 1:
                        tick_indices.append(len(time_index) - 1)
                    
                    ax.set_xticks([time_index[k] for k in tick_indices])
                    ax.set_xticklabels([formatted_labels[k] for k in tick_indices], rotation=45, ha='right', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        if n_rows * n_cols > n_params:
            for i in range(n_params, n_rows * n_cols):
                if n_params == 1:
                    # Single parameter case - no empty subplots to hide
                    pass
                elif n_rows == 1:
                    # Single row - axes is 1D array
                    axes[i].set_visible(False)
                elif n_cols == 1:
                    # Single column - axes is 1D array
                    axes[i].set_visible(False)
                else:
                    # Multiple rows and columns - axes is 2D array
                    row = i // n_cols
                    col = i % n_cols
                    axes[row, col].set_visible(False)
        
        # Set main title with smaller font
        shock_type_title = "Parameter" if shock_type == "param" else "Input"
        plt.suptitle(f'{shock_type_title} Sensitivity Testing: {scenario_set} - {scenario_name}', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_param_shock(self, scenario_set: str, scenario_name: str) -> None:
        """
        Plot parameter shock results for a specific scenario.
        
        Creates a grid of plots (3 columns per row) showing baseline predictions
        and shock results for each parameter. Each plot shows:
        - Baseline prediction as solid blue line
        - Shock results as dashed lines with different colors
        
        Parameters
        ----------
        scenario_set : str
            Name of the scenario set to plot
        scenario_name : str
            Name of the specific scenario to plot
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> sens_test.plot_param_shock('EWST_2024', 'base')
        """
        self.plot_shock(scenario_set, scenario_name, self.param_shock_df, "param")

    def plot_input_shock(self, scenario_set: str, scenario_name: str) -> None:
        """
        Plot input shock results for a specific scenario.
        
        Creates a grid of plots (3 columns per row) showing baseline predictions
        and shock results for each parameter. Each plot shows:
        - Baseline prediction as solid blue line
        - Shock results as dashed lines with different colors
        
        Parameters
        ----------
        scenario_set : str
            Name of the scenario set to plot
        scenario_name : str
            Name of the specific scenario to plot
            
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> sens_test.plot_input_shock('EWST_2024', 'base')
        """
        self.plot_shock(scenario_set, scenario_name, self.input_shock_df, "input")

    def plot_all_param_shock(self) -> None:
        """
        Plot parameter shock results for all scenarios across all scenario sets.
        
        Runs plot_param_shock() for each scenario in each scenario set,
        with clear separators and labels to distinguish between different scenarios.
        
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> sens_test.plot_all_param_shock()
        """
        if not hasattr(self, 'param_shock_df'):
            warnings.warn("No parameter shock data available. Run y_param_shock first.")
            return
        
        param_shock_data = self.param_shock_df
        if not param_shock_data:
            warnings.warn("No parameter shock data available")
            return
        
        # Print header
        print("Parameter Sensitivity Testing")
        
        # Plot each scenario set and scenario
        for scenario_set in param_shock_data.keys():
            for scenario_name in param_shock_data[scenario_set].keys():
                try:
                    self.plot_param_shock(scenario_set, scenario_name)
                except Exception as e:
                    print(f"Error plotting scenario '{scenario_name}' in scenario set '{scenario_set}': {e}")
                    continue

    def plot_all_input_shock(self) -> None:
        """
        Plot input shock results for all scenarios across all scenario sets.
        
        Runs plot_input_shock() for each scenario in each scenario set,
        with clear separators and labels to distinguish between different scenarios.
        
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> sens_test.plot_all_input_shock()
        """
        if not hasattr(self, 'input_shock_df'):
            warnings.warn("No input shock data available. Run y_input_shock first.")
            return
        
        input_shock_data = self.input_shock_df
        if not input_shock_data:
            warnings.warn("No input shock data available")
            return
        
        # Print header
        print("Input Sensitivity Testing")
        
        # Plot each scenario set and scenario
        for scenario_set in input_shock_data.keys():
            for scenario_name in input_shock_data[scenario_set].keys():
                try:
                    self.plot_input_shock(scenario_set, scenario_name)
                except Exception as e:
                    print(f"Error plotting input shock for scenario '{scenario_name}' in scenario set '{scenario_set}': {e}")
                    continue

    def plot_all(self) -> None:
        """
        Run both parameter and input sensitivity testing plots for all scenarios.
        
        This method executes both plot_all_param_shock() and plot_all_input_shock()
        to provide comprehensive sensitivity analysis visualization.
        
        Example
        -------
        >>> sens_test = SensitivityTest(scen_manager)
        >>> sens_test.plot_all()
        """
        # Run parameter sensitivity testing plots
        self.plot_all_param_shock()
        
        # Run input sensitivity testing plots
        self.plot_all_input_shock()
            

