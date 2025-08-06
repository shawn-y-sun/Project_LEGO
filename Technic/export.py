"""
Export module for LEGO modeling toolkit.

This module provides a flexible and extensible framework for exporting model results
and datasets. It follows a composition-based design pattern with clear separation of
concerns between data extraction, formatting, and output generation.

Key components:
- ExportableModel: Base interface for exportable models
- ModelExportAdapter: Adapter for existing models to implement ExportableModel
- ExportStrategy: Strategy pattern for different export content organization
- ExportFormatHandler: Handler for different output formats
- ExportManager: Orchestrator for the export process
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Type
import pandas as pd
from pathlib import Path
import numpy as np

# Define available export content types as constants
EXPORT_CONTENT_TYPES = {
    'timeseries_data': 'Combined modeling dataset and fit results',
    'staticStats': 'Model statistics and metrics',
    'scenario_testing': 'Scenario testing results',
    'test_results': 'Comprehensive test results from all tests',
    'sensitivity_testing': 'Sensitivity testing results for parameters and inputs'
}

class ExportableModel(ABC):
    """Abstract base class defining the interface for exportable models."""
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return unique identifier for the model."""
        pass
    
    @abstractmethod
    def get_timeseries_data(self) -> pd.DataFrame:
        """Return all time series data (features, actuals, predictions) in long format."""
        pass
    
    @abstractmethod
    def get_model_statistics(self) -> Dict[str, Any]:
        """Return model statistics and metrics."""
        pass
    
    @abstractmethod
    def get_scenario_results(self) -> Optional[pd.DataFrame]:
        """Return scenario testing results if available."""
        pass
    
    @abstractmethod
    def get_test_results(self) -> Optional[pd.DataFrame]:
        """Return comprehensive test results if available."""
        pass
    
    @abstractmethod
    def get_sensitivity_results(self) -> Optional[pd.DataFrame]:
        """Return sensitivity testing results if available."""
        pass

class ModelExportAdapter:
    """Adapter class to make existing models compatible with ExportableModel interface."""
    
    def __init__(self, model):
        self.model = model
    
    def adapt(self) -> ExportableModel:
        """Convert the model into an ExportableModel implementation."""
        # Implementation will be model-specific
        raise NotImplementedError("Subclasses must implement adapt()")

class ExportStrategy(ABC):
    """Abstract base class for export strategies."""
    
    def __init__(self, format_handler: 'ExportFormatHandler', content_types: Optional[Set[str]] = None):
        """Initialize strategy with format handler and optional content selection.
        
        Args:
            format_handler: Handler for the output format
            content_types: Set of content types to export. If None, exports all content.
                         Valid types are: 'timeseries_data', 'staticStats', 'scenario_testing', 'test_results'
        """
        self.format_handler = format_handler
        self.content_types = content_types or set(EXPORT_CONTENT_TYPES.keys())
        
        # Validate content types
        invalid_types = self.content_types - set(EXPORT_CONTENT_TYPES.keys())
        if invalid_types:
            raise ValueError(f"Invalid content types: {invalid_types}. Valid types are: {list(EXPORT_CONTENT_TYPES.keys())}")
    
    def should_export(self, content_type: str) -> bool:
        """Check if a particular content type should be exported."""
        return content_type in self.content_types
    
    def get_written_files(self) -> Set[Path]:
        """Return the set of files that have been written during export."""
        return getattr(self, '_written_files', set())
    
    @abstractmethod
    def export_timeseries_data(self, model: ExportableModel, output_dir: Path) -> None:
        """Export all time series data (features, actuals, predictions) to CSV."""
        pass
    
    @abstractmethod
    def export_statistics(self, model: ExportableModel, output_dir: Path) -> None:
        """Export model statistics to CSV."""
        pass
    
    @abstractmethod
    def export_scenarios(self, model: ExportableModel, output_dir: Path) -> None:
        """Export scenario results to CSV if available."""
        pass
    
    @abstractmethod
    def export_test_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export comprehensive test results to CSV if available."""
        pass
    
    @abstractmethod
    def save_consolidated_results(self, output_dir: Path) -> None:
        """Save consolidated results from all models to files."""
        pass

class ExportFormatHandler(ABC):
    """Abstract base class for handling different export formats."""
    
    @abstractmethod
    def save_dataframe(self, df: pd.DataFrame, filepath: Path) -> None:
        """Save DataFrame in specific format."""
        pass
    
    @abstractmethod
    def save_dict(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save dictionary in specific format."""
        pass

class CSVFormatHandler(ExportFormatHandler):
    """Handler for CSV format exports."""
    
    def save_dataframe(self, df: pd.DataFrame, filepath: Path) -> None:
        df.to_csv(filepath, index=False)
    
    def save_dict(self, data: Dict[str, Any], filepath: Path) -> None:
        pd.DataFrame([data]).to_csv(filepath, index=False)

class ExportManager:
    """Orchestrates the export process."""
    
    def __init__(self, strategy: ExportStrategy, format_handler: ExportFormatHandler):
        self.strategy = strategy
        self.format_handler = format_handler
    
    def export_models(self, models: List[ExportableModel], output_dir: Path) -> None:
        """Export multiple models to consolidated CSV files.
        
        Args:
            models: List of ExportableModel instances to export
            output_dir: Directory to save the consolidated CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each model
        for model in models:
            try:
                if self.strategy.should_export('timeseries_data'):
                    try:
                        self.strategy.export_timeseries_data(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export timeseries_data for {model.get_model_id()}: {e}")
                
                if self.strategy.should_export('staticStats'):
                    try:
                        self.strategy.export_statistics(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export staticStats for {model.get_model_id()}: {e}")
                
                if self.strategy.should_export('scenario_testing'):
                    try:
                        self.strategy.export_scenarios(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export scenario_testing for {model.get_model_id()}: {e}")
                
                if self.strategy.should_export('test_results'):
                    try:
                        self.strategy.export_test_results(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export test_results for {model.get_model_id()}: {e}")
                
                if self.strategy.should_export('sensitivity_testing'):
                    try:
                        self.strategy.export_sensitivity_results(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export sensitivity_testing for {model.get_model_id()}: {e}")
                        
            except Exception as e:
                print(f"Error: Failed to process model {model.get_model_id()}: {e}")
                continue
        
        # Save consolidated results
        self.strategy.save_consolidated_results(output_dir)
        
        # Print completion message only if files were written
        written_files = self.strategy.get_written_files()
        if written_files:
            files_str = ", ".join(f"'{p.name}'" for p in written_files)
            print(f"\nExport completed. Files written: {files_str}")

class OLSExportStrategy(ExportStrategy):
    """Export strategy for OLS models with optimized performance."""
    
    def __init__(self, format_handler: ExportFormatHandler, content_types: Optional[Set[str]] = None, chunk_size: int = 1000):
        """Initialize the strategy with a format handler and optional content selection.
        
        Args:
            format_handler: Handler for the output format
            content_types: Set of content types to export. If None, exports all content.
            chunk_size: Number of rows to process at a time for large datasets
        """
        super().__init__(format_handler=format_handler, content_types=content_types)
        self.chunk_size = chunk_size
        self._initialize_containers()
    
    def _initialize_containers(self):
        """Initialize data containers with pre-allocated lists."""
        self._timeseries_chunks = []
        self._statistics_chunks = []
        self._scenario_chunks = []
        self._test_results_chunks = []
        self._sensitivity_chunks = []  # New container for sensitivity results
        self._current_chunk_size = 0
        self._written_files = set()  # Track which files have been written
    
    def _write_chunk(self, output_dir: Path, content_type: str):
        """Write a chunk of data to disk with proper file handling and user feedback.
        
        Args:
            output_dir: Directory to write the file to
            content_type: Type of content being written ('timeseries_data', 'staticStats', etc.)
        """
        # Configuration for different content types
        content_config = {
            'timeseries_data': {
                'chunks': self._timeseries_chunks,
                'columns': ['date', 'model', 'value_type', 'value'],
                'filename': 'timeseries_data.csv'
            },
            'staticStats': {
                'chunks': self._statistics_chunks,
                'columns': ['category', 'model', 'type', 'value_type', 'value'],
                'filename': 'staticStats.csv'
            },
            'scenario_testing': {
                'chunks': self._scenario_chunks,
                'columns': ['model', 'scenario_name', 'severity', 'date', 'frequency', 'value_type', 'value'],
                'filename': 'scenario_testing.csv'
            },
            'test_results': {
                'chunks': self._test_results_chunks,
                'columns': ['model', 'test_name', 'test_target', 'test_statistics', 'test_statistic_value'],
                'filename': 'test_results.csv'
            },
            'sensitivity_testing': {
                'chunks': self._sensitivity_chunks,
                'columns': ['model', 'test', 'scenario_name', 'severity', 'variable/parameter', 
                          'shock', 'date', 'frequency', 'value_type', 'value'],
                'filename': 'sensitivity_testing.csv'
            }
        }
        
        config = content_config.get(content_type)
        if not config or not config['chunks']:
            return
            
        chunks = config['chunks']
        columns = config['columns']
        filepath = output_dir / config['filename']
        
        # Skip if we've already written this file
        if filepath in self._written_files:
            chunks.clear()
            self._current_chunk_size = 0
            return
        
        # Combine chunks and ensure column order
        df = pd.concat(chunks, copy=False)
        df = df[columns]
        
        # Handle file overwrite
        file_exists = filepath.exists()
        
        try:
            # Write data (will overwrite existing files)
            df.to_csv(filepath, index=False)
            
            # Print appropriate success message
            if file_exists:
                print(f"\n✓ Successfully overwrote {content_type} file to {filepath}")
            else:
                print(f"\n✓ Successfully wrote {content_type} file to {filepath}")
            
            # Mark this file as written
            self._written_files.add(filepath)
            
        except Exception as e:
            if file_exists:
                print(f"\n✗ Failed to overwrite {content_type} file to {filepath}: {str(e)}")
            else:
                print(f"\n✗ Failed to write {content_type} file to {filepath}: {str(e)}")
        
        # Clear the chunks
        chunks.clear()
        self._current_chunk_size = 0
    
    def export_sensitivity_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export sensitivity testing results with chunking support.
        
        Args:
            model: ExportableModel instance
            output_dir: Directory to save the results
        """
        if not self.should_export('sensitivity_testing'):
            return
        
        # Get sensitivity results
        df = model.get_sensitivity_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._sensitivity_chunks.append(df)
            self._current_chunk_size += len(df)
            
            if self._current_chunk_size >= self.chunk_size:
                self._write_chunk(output_dir, 'sensitivity_testing')
    
    def export_test_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export comprehensive test results with chunking support."""
        if not self.should_export('test_results'):
            return
        
        df = model.get_test_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._test_results_chunks.append(df)
            self._current_chunk_size += len(df)
            
            if self._current_chunk_size >= self.chunk_size:
                self._write_chunk(output_dir, 'test_results')
    
    def export_timeseries_data(self, model: ExportableModel, output_dir: Path) -> None:
        """Export time series data with chunking support."""
        if not self.should_export('timeseries_data'):
            return
        df = model.get_timeseries_data()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._timeseries_chunks.append(df)
            self._current_chunk_size += len(df)
            
            if self._current_chunk_size >= self.chunk_size:
                self._write_chunk(output_dir, 'timeseries_data')
    
    def export_statistics(self, model: ExportableModel, output_dir: Path) -> None:
        """Export model statistics with chunking support."""
        if not self.should_export('staticStats'):
            return
        
        df = model.get_model_statistics()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._statistics_chunks.append(df)
            self._current_chunk_size += len(df)
            
            if self._current_chunk_size >= self.chunk_size:
                self._write_chunk(output_dir, 'staticStats')
    
    def export_scenarios(self, model: ExportableModel, output_dir: Path) -> None:
        """Export scenario results with chunking support.
        
        The output will include:
        - Target variable forecasts
        - Base variable forecasts (if available)
        - Quarter labels for better period identification
        
        Output format:
        - model: Model identifier
        - scenario_name: Scenario set name (e.g., 'EWST_2024')
        - severity: Severity level (e.g., 'base', 'adv', 'sev')
        - date: End of period date
        - quarter_label: Period label (e.g., 'P0', 'P1', 'YY-MM')
        - value_type: 'Target' or 'Base'
        - value: The forecasted value
        """
        if not self.should_export('scenario_testing'):
            return
        
        # Get scenario results
        df = model.get_scenario_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._scenario_chunks.append(df)
            self._current_chunk_size += len(df)
            
            if self._current_chunk_size >= self.chunk_size:
                self._write_chunk(output_dir, 'scenario_testing')
    
    def save_consolidated_results(self, output_dir: Path) -> None:
        """Save any remaining data chunks to files."""
        # Write any remaining chunks
        if self.should_export('timeseries_data'):
            self._write_chunk(output_dir, 'timeseries_data')
        
        if self.should_export('staticStats'):
            self._write_chunk(output_dir, 'staticStats')
        
        if self.should_export('scenario_testing'):
            self._write_chunk(output_dir, 'scenario_testing')
        
        if self.should_export('test_results'):
            self._write_chunk(output_dir, 'test_results')
        
        if self.should_export('sensitivity_testing'):  # Add sensitivity testing export
            self._write_chunk(output_dir, 'sensitivity_testing')
        
        # Reset containers
        self._initialize_containers()

class OLSModelAdapter(ExportableModel):
    """Adapter for OLS models with optimized performance.
    
    This adapter follows the same test data access patterns as the report module
    (specifically ModelReportBase.show_test_tbl()) but converts the raw test results
    into structured export format compatible with OLSExportStrategy.
    
    Key Knowledge Transfer from Report Module:
    - Test retrieval: Uses exact same pattern as ModelReportBase.show_test_tbl()
    - Data access: Direct access to model.testset.all_test_results
    - Iteration: Same for test_name, result in results.items() loop
    - DataFrame handling: Processes same DataFrame structures from test classes
    
    Key Export Enhancement:
    - Three-layer structure: Converts raw test DataFrames to normalized format
    - CSV compatibility: Structured data suitable for export and analysis
    - Automated extraction: Handles threshold/observed/pass extraction automatically
    """
    
    def __init__(self, model, model_id: str):
        """Initialize adapter with an OLS model instance.
        
        Args:
            model: The OLS model instance to adapt
            model_id: The model identifier to use in exports
        """
        self.model = model
        self._model_id = model_id  # Use the provided model_id directly
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id
    
    def get_timeseries_data(self) -> pd.DataFrame:
        """Return all time series data in long format with optimized performance.
        
        Returns a DataFrame with columns:
        - date: Time index
        - model: Model identifier
        - value_type: Sample identifier or variable name
            - 'In-Sample': For in-sample predictions
            - 'Out-of-Sample': For out-of-sample predictions
            - 'Actual': For actual target values
            - 'Residual': For residuals
            - driver names: For feature/driver values
        - value: The actual value
        """
        # Pre-calculate common values
        model_id = self._model_id
        data_list = []
        
        # Combine target data efficiently
        target_data_full = pd.concat([self.model.y_in, self.model.y_out])
        target_data_full = target_data_full.sort_index()
        
        # Prepare base dictionaries for efficiency
        base_dict = {
            'model': model_id,
            'date': target_data_full.index,
            'value': target_data_full.values,
            'value_type': 'Actual'
        }
        
        # Add actual values
        data_list.append(pd.DataFrame(base_dict))
        
        # Add in-sample predictions
        predicted_in = self.model.y_fitted_in
        data_list.append(pd.DataFrame({
            'date': predicted_in.index,
            'model': model_id,
            'value_type': 'In-Sample',
            'value': predicted_in.values
        }))
        
        # Process features (in-sample) efficiently
        feature_data_in = self.model.X_in
        for col in feature_data_in.columns:
            data_list.append(pd.DataFrame({
                'date': feature_data_in.index,
                'model': model_id,
                'value_type': col,
                'value': feature_data_in[col].values
            }))
        
        # Process out-of-sample data if available
        if not self.model.X_out.empty:
            predicted_out = self.model.y_pred_out
            feature_data_out = self.model.X_out
            
            # Add out-of-sample predictions
            data_list.append(pd.DataFrame({
                'date': predicted_out.index,
                'model': model_id,
                'value_type': 'Out-of-Sample',
                'value': predicted_out.values
            }))
            
            # Process out-of-sample features
            for col in feature_data_out.columns:
                data_list.append(pd.DataFrame({
                    'date': feature_data_out.index,
                    'model': model_id,
                    'value_type': col,
                    'value': feature_data_out[col].values
                }))
        
        # Add residuals
        data_list.append(pd.DataFrame({
            'date': self.model.resid.index,
            'model': model_id,
            'value_type': 'Residual',
            'value': self.model.resid.values
        }))
        
        # Combine all data efficiently
        return pd.concat(data_list, ignore_index=True, copy=False)
    
    def get_model_statistics(self) -> pd.DataFrame:
        """Return model statistics and metrics with optimized performance.
        
        Returns a DataFrame with columns:
        - category: 'Goodness of Fit' or 'Model Estimation'
        - model: Model identifier
        - type: Metric type or estimation type
        - value_type: Specific metric or driver name
        - value: The actual value
        """
        model_id = self._model_id
        stats_list = []
        
        # Pre-calculate base dict for efficiency
        base_dict = {
            'model': model_id,
            'category': 'Goodness of Fit',
            'type': 'Model Overall'
        }
        
        # Add overall statistics efficiently
        overall_stats = {
            'R2': self.model.rsquared,
            'Adj R2': self.model.rsquared_adj,
            'F-statistic': self.model.fvalue,
            'F-statistic p-value': self.model.f_pvalue,
            'Log-likelihood': self.model.llf,
            'AIC': self.model.aic,
            'BIC': self.model.bic
        }
        
        # Add DW statistic if available
        if hasattr(self.model, 'dw_stat'):
            overall_stats['DW statistic'] = self.model.dw_stat
        
        # Extend stats_list efficiently
        stats_list.extend([
            {**base_dict, 'value_type': name, 'value': value}
            for name, value in overall_stats.items()
            if value is not None  # Only include non-None values
        ])
        
        # Process test results if available
        if hasattr(self.model, 'testset') and self.model.testset is not None:
            test_dict = {t.name: t for t in self.model.testset.tests}
            
            # Process in-sample error measures
            if 'IS Error Measures' in test_dict:
                is_errors = test_dict['IS Error Measures'].test_result
                stats_list.extend([
                    {
                        'category': 'Goodness of Fit',
                        'model': model_id,
                        'type': 'In-Sample',
                        'value_type': metric,
                        'value': value
                    }
                    for metric, value in is_errors.items()
                    if value is not None
                ])
            
            # Process out-of-sample error measures
            if not self.model.X_out.empty and 'OOS Error Measures' in test_dict:
                oos_errors = test_dict['OOS Error Measures'].test_result
                stats_list.extend([
                    {
                        'category': 'Goodness of Fit',
                        'model': model_id,
                        'type': 'Out-of-Sample',
                        'value_type': metric,
                        'value': value
                    }
                    for metric, value in oos_errors.items()
                    if value is not None
                ])
            
            # Process group F-test results if available
            if 'GroupTest' in test_dict:
                group_test = test_dict['GroupTest']
                if hasattr(group_test, 'test_result') and group_test.test_result is not None:
                    for group_name, result in group_test.test_result.items():
                        if isinstance(result, dict) and 'pvalue' in result:
                            stats_list.append({
                                'category': 'Model Estimation',
                                'model': model_id,
                                'type': 'Group Driver P-value',  # Specific type for group F-test p-values
                                'value_type': group_name,
                                'value': result['pvalue']
                            })
        
        # Process model estimation statistics
        params = self.model.params
        pvalues = self.model.pvalues
        std_errors = self.model.bse
        
        # Ensure conf_int is a DataFrame with proper index
        conf_int_result = self.model.conf_int()
        if isinstance(conf_int_result, np.ndarray):
            conf_int = pd.DataFrame(
                conf_int_result,
                index=params.index,
                columns=[0, 1]
            )
        else:
            conf_int = conf_int_result
        
        for var in params.index:
            stats_list.extend([
                {
                    'category': 'Model Estimation',
                    'model': model_id,
                    'type': 'Coefficient',
                    'value_type': var,
                    'value': params[var]
                },
                {
                    'category': 'Model Estimation',
                    'model': model_id,
                    'type': 'P-value',
                    'value_type': var,
                    'value': pvalues[var]
                },
                {
                    'category': 'Model Estimation',
                    'model': model_id,
                    'type': 'Std Error',
                    'value_type': var,
                    'value': std_errors[var]
                },
                {
                    'category': 'Model Estimation',
                    'model': model_id,
                    'type': 'CI Lower',
                    'value_type': var,
                    'value': conf_int.loc[var, 0]
                },
                {
                    'category': 'Model Estimation',
                    'model': model_id,
                    'type': 'CI Upper',
                    'value_type': var,
                    'value': conf_int.loc[var, 1]
                }
            ])
        
        # Create DataFrame efficiently
        return pd.DataFrame(stats_list)
    
    def get_scenario_results(self) -> Optional[pd.DataFrame]:
        """Return scenario testing results in long format.
        
        Returns a DataFrame with columns:
        - model: string, model_id
        - scenario_name: string (e.g., 'EWST_2024')
        - severity: string (e.g., 'base', 'adv', 'sev', 'p0')
        - date: timestamp
        - frequency: string ['monthly'/'quarterly']
        - value_type: string ['Target'/'Base']
        - value: numerical
        
        The method processes both target and base variable forecasts if available,
        includes scen_p0 baseline data, and includes both monthly and quarterly results.
        """
        if self.model.scen_manager is None:
            return None
        
        model_id = self._model_id
        data_list = []
        
        # Get target variable forecasts (monthly)
        scen_results = self.model.scen_manager.y_scens
        
        # Add scen_p0 data to scenario results if available
        if hasattr(self.model.scen_manager, 'scen_p0') and self.model.scen_manager.scen_p0 is not None:
            scen_p0_data = self.model.scen_manager.scen_p0
            for scen_set in scen_results.keys():
                # Create scen_p0 entry for target variable
                df_data = {
                    'model': model_id,
                    'scenario_name': scen_set,
                    'severity': 'p0',
                    'date': scen_p0_data.index,
                    'value_type': 'Target',
                    'value': scen_p0_data.values,
                    'frequency': 'monthly'
                }
                data_list.append(pd.DataFrame(df_data))
        
        # Process target variable forecasts (monthly)
        for scen_set, scenarios in scen_results.items():
            for scen_name, forecast in scenarios.items():
                if forecast is not None and not forecast.empty:
                    # Create DataFrame for target forecasts (monthly)
                    df_data = {
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'date': forecast.index,
                        'value_type': 'Target',
                        'value': forecast.values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))
        
        # Process target variable quarterly forecasts
        if hasattr(self.model.scen_manager, 'forecast_y_qtr_df'):
            qtr_forecasts = self.model.scen_manager.forecast_y_qtr_df
            for scen_set, qtr_df in qtr_forecasts.items():
                if qtr_df is not None and not qtr_df.empty:
                    # Get scenarios for this scenario set
                    scen_scenarios = scen_results.get(scen_set, {})
                    for scen_name in scen_scenarios.keys():
                        # Check if this scenario has quarterly data
                        if scen_name in qtr_df.columns or f"{scen_set}_{scen_name}" in qtr_df.columns:
                            col_name = scen_name if scen_name in qtr_df.columns else f"{scen_set}_{scen_name}"
                            qtr_forecast = qtr_df[col_name].dropna()
                            
                            if not qtr_forecast.empty:
                                # Create quarterly target data
                                df_data = {
                                    'model': model_id,
                                    'scenario_name': scen_set,
                                    'severity': scen_name,
                                    'date': qtr_forecast.index,
                                    'value_type': 'Target',
                                    'value': qtr_forecast.values,
                                    'frequency': 'quarterly'
                                }
                                data_list.append(pd.DataFrame(df_data))
        
        # Process base variable forecasts (monthly) if available
        if hasattr(self.model.scen_manager, 'y_base_scens'):
            base_results = self.model.scen_manager.y_base_scens
            
            # Add scen_p0 base data if available
            if hasattr(self.model, 'base_predictor') and self.model.base_predictor is not None:
                if hasattr(self.model.scen_manager, 'scen_p0') and self.model.scen_manager.scen_p0 is not None:
                    scen_p0_data = self.model.scen_manager.scen_p0
                    base_p0_values = self.model.base_predictor.predict_base(scen_p0_data, scen_p0_data)
                    
                    for scen_set in base_results.keys():
                        # Create scen_p0 entry for base variable
                        df_data = {
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': 'p0',
                            'date': base_p0_values.index,
                            'value_type': 'Base',
                            'value': base_p0_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))
            
            for scen_set, scenarios in base_results.items():
                for scen_name, forecast in scenarios.items():
                    if forecast is not None and not forecast.empty:
                        # Create DataFrame for base forecasts (monthly)
                        df_data = {
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'date': forecast.index,
                            'value_type': 'Base',
                            'value': forecast.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))
        
        # Process base variable quarterly forecasts
        if hasattr(self.model.scen_manager, 'forecast_y_base_qtr_df'):
            base_qtr_forecasts = self.model.scen_manager.forecast_y_base_qtr_df
            for scen_set, qtr_df in base_qtr_forecasts.items():
                if qtr_df is not None and not qtr_df.empty:
                    # Get scenarios for this scenario set
                    if hasattr(self.model.scen_manager, 'y_base_scens'):
                        base_scenarios = self.model.scen_manager.y_base_scens.get(scen_set, {})
                        for scen_name in base_scenarios.keys():
                            # Check if this scenario has quarterly data
                            if scen_name in qtr_df.columns or f"{scen_set}_{scen_name}" in qtr_df.columns:
                                col_name = scen_name if scen_name in qtr_df.columns else f"{scen_set}_{scen_name}"
                                qtr_forecast = qtr_df[col_name].dropna()
                                
                                if not qtr_forecast.empty:
                                    # Create quarterly base data
                                    df_data = {
                                        'model': model_id,
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'date': qtr_forecast.index,
                                        'value_type': 'Base',
                                        'value': qtr_forecast.values,
                                        'frequency': 'quarterly'
                                    }
                                    data_list.append(pd.DataFrame(df_data))
        
        if not data_list:
            return None
            
        # Combine all data and ensure column order
        result = pd.concat(data_list, ignore_index=True)
        return result[['model', 'scenario_name', 'severity', 'date', 'frequency', 'value_type', 'value']]

    def get_test_results(self) -> Optional[pd.DataFrame]:
        """Return test results in standardized three-layer format for specific test categories.
        
        This function follows the exact same test retrieval pattern as ModelReportBase.show_test_tbl()
        in the report module, but converts only the required test results to a structured three-layer format
        suitable for export and analysis.
        
        INCLUDED TEST CATEGORIES:
        
        Assumption Testing:
        - Stationarity (ADF, KPSS, etc.)
        - Normality (Jarque-Bera, Shapiro-Wilk, etc.)
        - Autocorrelation (Durbin-Watson, Ljung-Box, etc.)
        - Heteroscedasticity (White, Breusch-Pagan, etc.)
        - Multicollinearity (VIF, etc.)
        
        Model Estimation:
        - Coefficient Significance (t-tests, p-values)
        - Coefficient Sign Check (expected vs actual signs)
        - Group Driver F-test (joint significance of variable groups)
        
        EXCLUDED TEST CATEGORIES:
        - Error Measures (RMSE, MAE, MAPE, etc.)
        - Goodness-of-Fit (R², Adj R², AIC, BIC, etc.)
        - Performance Metrics (other fit measures)
        
        CLEAN NAMING APPROACH:
        - test_name: Clean category names without test-statistic suffixes
          Examples: 'Stationarity', 'Normality', 'Multicollinearity'
        - test_target: Clean target names without test-statistic suffixes  
          Examples: 'Residual', 'GDP', 'UNRATE'
        - test_statistics: Specific statistic with test type
          Examples: 'P-value_ADF', 'Statistic_JB', 'VIF Observed'
        
        Returns a DataFrame with columns:
        - model: string, model_id
        - test_name: string (e.g., 'Normality', 'Stationarity', 'Coefficient Significance')
        - test_target: string (e.g., 'Residual', variable name, target variable name)
        - test_statistics: string (e.g., 'Statistic_JB', 'P-value_ADF', 'Pass?', 'VIF Observed')
        - test_statistic_value: numerical (boolean values converted to 1/0)
        
        Each test provides three layers of information:
        1. Expected/criteria/threshold value
        2. Observed/estimated value  
        3. Pass/fail indicator (1/0)
        
        Following report module pattern from ModelReportBase.show_test_tbl():
        ```python
        results = self.model.testset.all_test_results
        for test_name, result in results.items():
            # Process only required test categories...
        ```
        """
  
        # === EXACT REPORT MODULE PATTERN ===
        # Follow ModelReportBase.show_test_tbl() pattern exactly:
        # results = self.model.testset.all_test_results
        # for test_name, result in results.items():
        try:
            # Gather all test results (both active and inactive) - same as report module
            results = self.model.testset.all_test_results
        except Exception as e:
            return None
        
        # Check if test results have data
        if not results or len(results) == 0:
            return None
        
        model_id = self._model_id
        results_list = []
        
        # Process each test result following the same iteration pattern as report module
        # for test_name, result in results.items():
        for test_name, result in results.items():
            initial_count = len(results_list)
            
            # Convert raw test results to three-layer structure based on test type
            # Note: report module just prints result.to_string(), we convert to structured data
            
            # === ASSUMPTION TESTING ===
            
            # Handle Normality Tests (Jarque-Bera, Shapiro-Wilk, etc.)
            if 'Normality' in test_name or 'JB' in test_name or 'Jarque' in test_name:
                self._process_assumption_test(results_list, model_id, result, 'Normality', 'Residual', 'JB')
            
            # Handle Stationarity Tests (ADF, KPSS, etc.)
            elif 'Stationarity' in test_name or 'ADF' in test_name or 'KPSS' in test_name:
                self._process_assumption_test(results_list, model_id, result, 'Stationarity', 'Residual', 'ADF')
            
            # Handle Autocorrelation Tests (Durbin-Watson, Ljung-Box, etc.)
            elif 'Autocorrelation' in test_name or 'DW' in test_name or 'Durbin' in test_name:
                self._process_autocorrelation_test(results_list, model_id, result)
            
            # Handle Heteroscedasticity Tests (White, Breusch-Pagan, etc.)
            elif 'Heteroscedasticity' in test_name or 'White' in test_name or 'Breusch' in test_name:
                self._process_assumption_test(results_list, model_id, result, 'Heteroscedasticity', 'Residual', 'White')
            
            # Handle Multicollinearity Tests (VIF, etc.)
            elif 'Multicollinearity' in test_name or 'VIF' in test_name or 'Collinearity' in test_name:
                self._process_multicollinearity_test(results_list, model_id, result)
            
            # === MODEL ESTIMATION ===
            
            # Handle Coefficient Significance Tests (process before Sign Check to avoid conflicts)
            elif 'Significance' in test_name or 'significance' in test_name:
                self._process_coefficient_significance_test(results_list, model_id, result)
            
            # Handle Sign Check Tests
            elif 'Sign' in test_name or 'sign' in test_name:
                self._process_sign_check_test(results_list, model_id, result)
            
            # Handle Group Driver F-tests (Group Test, F-test, etc.)
            elif 'Group' in test_name or 'F-test' in test_name or 'Joint' in test_name:
                self._process_group_f_test(results_list, model_id, result)
            
            # === EXCLUDED TESTS (documented for clarity) ===
            # Skip Error Measures (RMSE, MAE, etc.) - not included in test results export
            # Skip Goodness-of-Fit measures (R², Adj R², etc.) - not included in test results export
            # Skip other performance metrics - not included in test results export
            else:
                continue
        
        if not results_list:
            return None
            
        return pd.DataFrame(results_list)

    def _process_assumption_test(self, results_list: list, model_id: str, result: pd.DataFrame, 
                                test_name: str, test_target: str, test_type: str):
        """Process assumption tests (Normality, Stationarity, Heteroscedasticity) with three-layer structure."""
        if not isinstance(result, pd.DataFrame):
            return
        
        if result.empty:
            return
        
        added_count = 0
        
        # Process each row in the DataFrame (each test within the test class)
        for test_idx in result.index:
            test_row = result.loc[test_idx]
            
            # Extract statistic value (multiple possible column names)
            statistic_value = None
            for col in ['Statistic', 'F-statistic', 'Test-statistic']:
                if col in test_row and pd.notnull(test_row[col]):
                    statistic_value = float(test_row[col])
                    break
            
            # Extract p-value
            pvalue_value = None
            for col in ['P-value', 'p-value', 'pvalue']:
                if col in test_row and pd.notnull(test_row[col]):
                    pvalue_value = float(test_row[col])
                    break
            
            # Extract pass status
            pass_value = None
            if 'Passed' in test_row and pd.notnull(test_row['Passed']):
                pass_value = float(1.0 if test_row['Passed'] else 0.0)
            
            # Create clean test target name (no test-statistic names)
            # Use clean target name without test index
            clean_test_target = test_target
            
            # Add the three layers of information
            if statistic_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': test_name,
                    'test_target': clean_test_target,
                    'test_statistics': f'Statistic_{test_type}',
                    'test_statistic_value': statistic_value
                })
                added_count += 1
            
            if pvalue_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': test_name,
                    'test_target': clean_test_target,
                    'test_statistics': f'P-value_{test_type}',
                    'test_statistic_value': pvalue_value
                })
                added_count += 1
            
            if pass_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': test_name,
                    'test_target': clean_test_target,
                    'test_statistics': 'Pass?',
                    'test_statistic_value': pass_value
                })

    def _process_autocorrelation_test(self, results_list: list, model_id: str, result):
        """Process autocorrelation tests with three-layer structure."""
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return
            
            # Process each row in the autocorrelation test DataFrame
            for test_idx in result.index:
                test_row = result.loc[test_idx]
                
                # Extract DW statistic or general statistic
                dw_value = None
                if 'Statistic' in test_row and pd.notnull(test_row['Statistic']):
                    dw_value = float(test_row['Statistic'])
                
                # Extract pass status
                pass_value = None
                if 'Passed' in test_row and pd.notnull(test_row['Passed']):
                    pass_value = float(1.0 if test_row['Passed'] else 0.0)
                
                # Extract threshold information if available
                threshold_info = None
                if 'Threshold' in test_row and pd.notnull(test_row['Threshold']):
                    threshold_info = test_row['Threshold']
                
                # Use clean test name without test-statistic names
                clean_test_name = 'Autocorrelation'
                clean_test_target = 'Residual'
                
                # Handle Durbin-Watson test specifically
                if 'Durbin' in test_idx or 'DW' in test_idx or isinstance(threshold_info, (tuple, list)):
                    # Add threshold information with separate upper and lower limits
                    if isinstance(threshold_info, (tuple, list)) and len(threshold_info) >= 2:
                        lower_limit, upper_limit = threshold_info[0], threshold_info[1]
                    else:
                        # Standard DW acceptable range
                        lower_limit, upper_limit = 1.5, 2.5
                    
                    results_list.extend([
                        {
                            'model': model_id,
                            'test_name': clean_test_name,
                            'test_target': clean_test_target,
                            'test_statistics': 'Threshold Lower Limit',
                            'test_statistic_value': float(lower_limit)
                        },
                        {
                            'model': model_id,
                            'test_name': clean_test_name,
                            'test_target': clean_test_target,
                            'test_statistics': 'Threshold Upper Limit',
                            'test_statistic_value': float(upper_limit)
                        }
                    ])
                    
                    if dw_value is not None:
                        results_list.append({
                            'model': model_id,
                            'test_name': clean_test_name,
                            'test_target': clean_test_target,
                            'test_statistics': 'Durbin-Watson Observed',
                            'test_statistic_value': dw_value
                        })
                else:
                    # For other autocorrelation tests, just add the statistic
                    if dw_value is not None:
                        results_list.append({
                            'model': model_id,
                            'test_name': clean_test_name,
                            'test_target': clean_test_target,
                            'test_statistics': 'Statistic Observed',
                            'test_statistic_value': dw_value
                        })
                
                # Add pass status for all autocorrelation tests
                if pass_value is not None:
                    results_list.append({
                        'model': model_id,
                        'test_name': clean_test_name,
                        'test_target': clean_test_target,
                        'test_statistics': 'Pass?',
                        'test_statistic_value': pass_value
                    })
        
        elif isinstance(result, (int, float)):
            # Handle direct DW statistic value (legacy support)
            results_list.extend([
                {
                    'model': model_id,
                    'test_name': 'Autocorrelation',
                    'test_target': 'Residual',
                    'test_statistics': 'Threshold Lower Limit',
                    'test_statistic_value': 1.5
                },
                {
                    'model': model_id,
                    'test_name': 'Autocorrelation',
                    'test_target': 'Residual',
                    'test_statistics': 'Threshold Upper Limit',
                    'test_statistic_value': 2.5
                },
                {
                    'model': model_id,
                    'test_name': 'Autocorrelation',
                    'test_target': 'Residual',
                    'test_statistics': 'Durbin-Watson Observed',
                    'test_statistic_value': float(result)
                }
            ])

    def _process_sign_check_test(self, results_list: list, model_id: str, result):
        """Process sign check tests with three-layer structure."""
        if not isinstance(result, pd.DataFrame):
            return
        
        if result.empty:
            return
        
        # Process each row in the sign check DataFrame (each variable)
        for var_name in result.index:
            test_row = result.loc[var_name]
            
            # Extract expected sign
            expected_sign_value = None
            if 'Expected' in test_row and pd.notnull(test_row['Expected']):
                expected_str = str(test_row['Expected'])
                if expected_str == '+' or 'positive' in expected_str.lower():
                    expected_sign_value = 1.0
                elif expected_str == '-' or 'negative' in expected_str.lower():
                    expected_sign_value = -1.0
                else:
                    expected_sign_value = 0.0
            
            # Extract coefficient value
            coefficient_value = None
            if 'Coefficient' in test_row and pd.notnull(test_row['Coefficient']):
                coefficient_value = float(test_row['Coefficient'])
            
            # Extract pass status
            pass_value = None
            if 'Passed' in test_row and pd.notnull(test_row['Passed']):
                pass_value = float(1.0 if test_row['Passed'] else 0.0)
            
            # Add three layers of information
            if expected_sign_value is not None:
                sign_str = 'positive' if expected_sign_value > 0 else 'negative' if expected_sign_value < 0 else 'zero'
                results_list.append({
                    'model': model_id,
                    'test_name': 'Coefficient Sign Check',
                    'test_target': var_name,
                    'test_statistics': f'Expected Sign ({sign_str})',
                    'test_statistic_value': expected_sign_value
                })
            
            if coefficient_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Coefficient Sign Check',
                    'test_target': var_name,
                    'test_statistics': 'Coefficient Estimated',
                    'test_statistic_value': coefficient_value
                })
            
            if pass_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Coefficient Sign Check',
                    'test_target': var_name,
                    'test_statistics': 'Pass?',
                    'test_statistic_value': pass_value
                })

    def _process_coefficient_significance_test(self, results_list: list, model_id: str, result):
        """Process coefficient significance tests with three-layer structure."""
        if not isinstance(result, pd.DataFrame):
            return
        
        if result.empty:
            return
        
        # Process each row in the coefficient significance DataFrame (each variable)
        for var_name in result.index:
            test_row = result.loc[var_name]
            
            # Standard significance threshold
            threshold_value = 0.05
            
            # Extract p-value
            pvalue_value = None
            for col in ['P-value', 'p-value', 'pvalue']:
                if col in test_row and pd.notnull(test_row[col]):
                    pvalue_value = float(test_row[col])
                    break
            
            # Extract t-statistic if available
            tstat_value = None
            for col in ['T-statistic', 't-statistic', 'tstat', 'Statistic']:
                if col in test_row and pd.notnull(test_row[col]):
                    tstat_value = float(test_row[col])
                    break
            
            # Extract pass status
            pass_value = None
            if 'Passed' in test_row and pd.notnull(test_row['Passed']):
                pass_value = float(1.0 if test_row['Passed'] else 0.0)
            
            # Add three layers of information
            results_list.append({
                'model': model_id,
                'test_name': 'Coefficient Significance',
                'test_target': var_name,
                'test_statistics': 'Significance Threshold',
                'test_statistic_value': threshold_value
            })
            
            if pvalue_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Coefficient Significance',
                    'test_target': var_name,
                    'test_statistics': 'P-value Observed',
                    'test_statistic_value': pvalue_value
                })
            
            if tstat_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Coefficient Significance',
                    'test_target': var_name,
                    'test_statistics': 'T-statistic Observed',
                    'test_statistic_value': tstat_value
                })
            
            if pass_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Coefficient Significance',
                    'test_target': var_name,
                    'test_statistics': 'Pass?',
                    'test_statistic_value': pass_value
                })

    def _process_error_measures_test(self, results_list: list, model_id: str, result, test_name: str):
        """Process error measures with single layer (no threshold/pass concept)."""
        if isinstance(result, pd.DataFrame):
            if result.empty:
                print(f"⚠️  {test_name}: Empty DataFrame")
                return
            
            sample_type = 'In-Sample' if 'IS' in test_name else 'Out-of-Sample' if 'OOS' in test_name else 'Unknown'
            
            # Process each row/column in the error measures DataFrame
            for metric_idx in result.index:
                for col in result.columns:
                    value = result.loc[metric_idx, col]
                    if pd.notnull(value):
                        metric_name = f"{metric_idx}_{col}" if metric_idx != col else str(metric_idx)
                        results_list.append({
                            'model': model_id,
                            'test_name': 'Error Measures',
                            'test_target': f'{sample_type} Prediction',
                            'test_statistics': f'{metric_name} Observed',
                            'test_statistic_value': float(value)
                        })
        
        elif isinstance(result, dict):
            # Legacy support for dictionary format
            sample_type = 'In-Sample' if 'IS' in test_name else 'Out-of-Sample' if 'OOS' in test_name else 'Unknown'
            
            for metric_name, value in result.items():
                if pd.notnull(value):
                    results_list.append({
                        'model': model_id,
                        'test_name': 'Error Measures',
                        'test_target': f'{sample_type} Prediction',
                        'test_statistics': f'{metric_name} Observed',
                        'test_statistic_value': float(value)
                    })
        else:
            print(f"⚠️  {test_name}: Expected DataFrame or dict, got {type(result).__name__}")

    def _process_generic_test(self, results_list: list, model_id: str, result, test_name: str):
        """Process generic tests with simplified structure."""
        added_count = 0
        
        if isinstance(result, pd.DataFrame):
            if result.empty:
                print(f"⚠️  {test_name}: Empty DataFrame")
                return 0
            
            # Process each row/column in the DataFrame
            for row_idx in result.index:
                for col_name in result.columns:
                    value = result.loc[row_idx, col_name]
                    
                    # Initialize numeric_value to None
                    numeric_value = None
                    
                    # Convert value to numeric
                    if isinstance(value, bool):
                        numeric_value = float(1.0 if value else 0.0)
                    elif pd.notnull(value):
                        try:
                            numeric_value = float(value)
                        except (ValueError, TypeError):
                            continue  # Skip non-numeric values
                    
                    # Only add to results if we have a valid numeric_value
                    if numeric_value is not None:
                        # Create meaningful statistic name
                        if str(row_idx) == str(col_name):
                            stat_name = str(row_idx)
                        else:
                            stat_name = f"{row_idx}_{col_name}"
                            
                        results_list.append({
                            'model': model_id,
                            'test_name': test_name,
                            'test_target': str(row_idx) if row_idx != test_name else 'Unknown',
                            'test_statistics': f'{stat_name} Observed',
                            'test_statistic_value': numeric_value
                        })
                        added_count += 1
        
        elif isinstance(result, dict):
            # Legacy support for dictionary format
            for stat_name, value in result.items():
                # Initialize numeric_value to None at the start of each iteration
                numeric_value = None
                
                # Convert boolean to 1/0
                if isinstance(value, bool):
                    numeric_value = float(1.0 if value else 0.0)
                elif pd.notnull(value):
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        continue  # Skip non-numeric values
                
                # Only add to results if we have a valid numeric_value
                if numeric_value is not None:
                    results_list.append({
                        'model': model_id,
                        'test_name': test_name,
                        'test_target': 'Unknown',
                        'test_statistics': f'{stat_name} Observed',
                        'test_statistic_value': numeric_value
                    })
                    added_count += 1
        
        elif isinstance(result, (int, float, bool)):
            # Convert boolean to 1/0
            if isinstance(result, bool):
                numeric_value = float(1.0 if result else 0.0)
            else:
                numeric_value = float(result)
            
            results_list.append({
                'model': model_id,
                'test_name': test_name,
                'test_target': 'Unknown',
                'test_statistics': 'Value Observed',
                'test_statistic_value': numeric_value
            })
            added_count += 1
        
        else:
            print(f"⚠️  Unhandled data type for {test_name}: {type(result)} = {result}")
        
        # Debug: Report if no data was added
        if added_count == 0:
            print(f"⚠️  No data extracted from generic test {test_name} for {model_id}")
            if isinstance(result, pd.DataFrame):
                print(f"   DataFrame shape: {result.shape}, columns: {list(result.columns)}")
            else:
                print(f"   Data type: {type(result)}, content: {result}")
            
        return added_count

    def _process_multicollinearity_test(self, results_list: list, model_id: str, result):
        """Process multicollinearity tests with three-layer structure."""
        if not isinstance(result, pd.DataFrame):
            return
        
        if result.empty:
            return
        
        # Process each row in the multicollinearity test DataFrame (each variable)
        for var_name in result.index:
            test_row = result.loc[var_name]
            
            # Extract VIF value
            vif_value = None
            if 'VIF' in test_row and pd.notnull(test_row['VIF']):
                vif_value = float(test_row['VIF'])
            
            # Extract pass status
            pass_value = None
            if 'Passed' in test_row and pd.notnull(test_row['Passed']):
                pass_value = float(1.0 if test_row['Passed'] else 0.0)
            
            # Add three layers of information
            if vif_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Multicollinearity',
                    'test_target': var_name,
                    'test_statistics': 'VIF Observed',
                    'test_statistic_value': vif_value
                })
            
            if pass_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Multicollinearity',
                    'test_target': var_name,
                    'test_statistics': 'Pass?',
                    'test_statistic_value': pass_value
                })

    def _process_group_f_test(self, results_list: list, model_id: str, result):
        """Process group F-tests with three-layer structure."""
        if not isinstance(result, pd.DataFrame):
            return
        
        if result.empty:
            return
        
        # Process each row in the group F-test DataFrame (each group)
        for group_name in result.index:
            test_row = result.loc[group_name]
            
            # Extract p-value
            pvalue_value = None
            for col in ['P-value', 'p-value', 'pvalue']:
                if col in test_row and pd.notnull(test_row[col]):
                    pvalue_value = float(test_row[col])
                    break
            
            # Extract pass status
            pass_value = None
            if 'Passed' in test_row and pd.notnull(test_row['Passed']):
                pass_value = float(1.0 if test_row['Passed'] else 0.0)
            
            # Add three layers of information
            if pvalue_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Group Driver F-test',
                    'test_target': group_name,
                    'test_statistics': 'P-value Observed',
                    'test_statistic_value': pvalue_value
                })
            
            if pass_value is not None:
                results_list.append({
                    'model': model_id,
                    'test_name': 'Group Driver F-test',
                    'test_target': group_name,
                    'test_statistics': 'Pass?',
                    'test_statistic_value': pass_value
                })

    def get_sensitivity_results(self) -> Optional[pd.DataFrame]:
        """Return sensitivity testing results in long format.
        
        Returns a DataFrame with columns:
        - model: string, model_id
        - test: string ["Input Sensitivity Test", "Parameter Sensitivity Test"]
        - scenario_name: string (e.g., 'EWST_2024')
        - severity: string (e.g., 'base', 'adv', 'sev', 'p0')
        - variable/parameter: string, variable or parameter name being tested (or 'baseline_p0' for scen_p0 data)
        - shock: string, shock level ("-3std", "+1se", "baseline", etc)
        - date: timestamp
        - frequency: string ['monthly'/'quarterly']
        - value_type: string ['Target'/'Base']
        - value: numerical
        """
        if not hasattr(self.model, 'scen_manager') or self.model.scen_manager is None:
            return None

        # Get sensitivity test instance
        sens_test = self.model.scen_manager.sens_test
        if sens_test is None:
            return None

        model_id = self._model_id
        data_list = []

        # Add scen_p0 baseline data for sensitivity tests if available
        if hasattr(self.model.scen_manager, 'scen_p0') and self.model.scen_manager.scen_p0 is not None:
            scen_p0_data = self.model.scen_manager.scen_p0
            
            # Get scenario sets from parameter sensitivity results
            param_shock_df = sens_test.param_shock_df
            for scen_set in param_shock_df.keys():
                for scen_name in param_shock_df[scen_set].keys():
                    # Add scen_p0 for parameter sensitivity baseline
                    df_data = {
                        'model': model_id,
                        'test': 'Parameter Sensitivity Test',
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'variable/parameter': 'baseline_p0',
                        'shock': 'baseline',
                        'date': scen_p0_data.index,
                        'value_type': 'Target',
                        'value': scen_p0_data.values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))
            
            # Get scenario sets from input sensitivity results
            input_shock_df = sens_test.input_shock_df
            for scen_set in input_shock_df.keys():
                for scen_name in input_shock_df[scen_set].keys():
                    # Add scen_p0 for input sensitivity baseline
                    df_data = {
                        'model': model_id,
                        'test': 'Input Sensitivity Test',
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'variable/parameter': 'baseline_p0',
                        'shock': 'baseline',
                        'date': scen_p0_data.index,
                        'value_type': 'Target',
                        'value': scen_p0_data.values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))
            
            # Add base variable scen_p0 data if base predictor is available
            if hasattr(self.model, 'base_predictor') and self.model.base_predictor is not None:
                base_p0_values = self.model.base_predictor.predict_base(scen_p0_data, scen_p0_data)
                
                # Add base p0 for parameter sensitivity
                for scen_set in param_shock_df.keys():
                    for scen_name in param_shock_df[scen_set].keys():
                        df_data = {
                            'model': model_id,
                            'test': 'Parameter Sensitivity Test',
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'variable/parameter': 'baseline_p0',
                            'shock': 'baseline',
                            'date': base_p0_values.index,
                            'value_type': 'Base',
                            'value': base_p0_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))
                
                # Add base p0 for input sensitivity
                for scen_set in input_shock_df.keys():
                    for scen_name in input_shock_df[scen_set].keys():
                        df_data = {
                            'model': model_id,
                            'test': 'Input Sensitivity Test',
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'variable/parameter': 'baseline_p0',
                            'shock': 'baseline',
                            'date': base_p0_values.index,
                            'value_type': 'Base',
                            'value': base_p0_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))

        # Process parameter sensitivity results (monthly)
        param_shock_df = sens_test.param_shock_df
        for scen_set, scen_dict in param_shock_df.items():
            for scen_name, df in scen_dict.items():
                # Get baseline column name
                baseline_col = f"{scen_set}_{scen_name}"
                
                # Process each parameter's shocks (monthly)
                for col in df.columns:
                    if col == baseline_col:
                        continue
                    
                    # Extract parameter name and shock level
                    param_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                    shock = ('+' if '+' in col else '-') + shock
                    
                    # Create DataFrame for target variable results (monthly)
                    df_data = {
                        'model': model_id,
                        'test': 'Parameter Sensitivity Test',
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'variable/parameter': param_name,
                        'shock': shock,
                        'date': df.index,
                        'value_type': 'Target',
                        'value': df[col].values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))

        # Process parameter sensitivity quarterly results
        if hasattr(sens_test, 'param_shock_qtr_df'):
            param_shock_qtr_df = sens_test.param_shock_qtr_df
            for scen_set, scen_dict in param_shock_qtr_df.items():
                for scen_name, qtr_df in scen_dict.items():
                    if qtr_df is not None and not qtr_df.empty:
                        baseline_col = f"{scen_set}_{scen_name}"
                        
                        # Process each parameter's shocks (quarterly)
                        for col in qtr_df.columns:
                            if col == baseline_col:
                                continue
                            
                            # Extract parameter name and shock level
                            param_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                            shock = ('+' if '+' in col else '-') + shock
                            
                            qtr_forecast = qtr_df[col].dropna()
                            if not qtr_forecast.empty:
                                # Create DataFrame for quarterly target results
                                df_data = {
                                    'model': model_id,
                                    'test': 'Parameter Sensitivity Test',
                                    'scenario_name': scen_set,
                                    'severity': scen_name,
                                    'variable/parameter': param_name,
                                    'shock': shock,
                                    'date': qtr_forecast.index,
                                    'value_type': 'Target',
                                    'value': qtr_forecast.values,
                                    'frequency': 'quarterly'
                                }
                                data_list.append(pd.DataFrame(df_data))

        # Process input sensitivity results (monthly)
        input_shock_df = sens_test.input_shock_df
        for scen_set, scen_dict in input_shock_df.items():
            for scen_name, df in scen_dict.items():
                # Get baseline column name
                baseline_col = f"{scen_set}_{scen_name}"
                
                # Process each variable's shocks (monthly)
                for col in df.columns:
                    if col == baseline_col:
                        continue
                    
                    # Extract variable name and shock level
                    var_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                    shock = ('+' if '+' in col else '-') + shock
                    
                    # Create DataFrame for target variable results (monthly)
                    df_data = {
                        'model': model_id,
                        'test': 'Input Sensitivity Test',
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'variable/parameter': var_name,
                        'shock': shock,
                        'date': df.index,
                        'value_type': 'Target',
                        'value': df[col].values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))

        # Process input sensitivity quarterly results
        if hasattr(sens_test, 'input_shock_qtr_df'):
            input_shock_qtr_df = sens_test.input_shock_qtr_df
            for scen_set, scen_dict in input_shock_qtr_df.items():
                for scen_name, qtr_df in scen_dict.items():
                    if qtr_df is not None and not qtr_df.empty:
                        baseline_col = f"{scen_set}_{scen_name}"
                        
                        # Process each variable's shocks (quarterly)
                        for col in qtr_df.columns:
                            if col == baseline_col:
                                continue
                            
                            # Extract variable name and shock level
                            var_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                            shock = ('+' if '+' in col else '-') + shock
                            
                            qtr_forecast = qtr_df[col].dropna()
                            if not qtr_forecast.empty:
                                # Create DataFrame for quarterly target results
                                df_data = {
                                    'model': model_id,
                                    'test': 'Input Sensitivity Test',
                                    'scenario_name': scen_set,
                                    'severity': scen_name,
                                    'variable/parameter': var_name,
                                    'shock': shock,
                                    'date': qtr_forecast.index,
                                    'value_type': 'Target',
                                    'value': qtr_forecast.values,
                                    'frequency': 'quarterly'
                                }
                                data_list.append(pd.DataFrame(df_data))

        # If model has base predictor, add base variable results
        if hasattr(self.model, 'base_predictor') and self.model.base_predictor is not None:
            # Process parameter sensitivity base results (monthly)
            for scen_set, scen_dict in param_shock_df.items():
                for scen_name, df in scen_dict.items():
                    baseline_col = f"{scen_set}_{scen_name}"
                    
                    for col in df.columns:
                        if col == baseline_col:
                            continue
                        
                        param_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                        shock = ('+' if '+' in col else '-') + shock
                        
                        # Convert to base variable
                        base_values = self.model.base_predictor.predict_base(df[col], self.model.dm.scen_p0)
                        
                        # Create DataFrame for base variable results (monthly)
                        df_data = {
                            'model': model_id,
                            'test': 'Parameter Sensitivity Test',
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'variable/parameter': param_name,
                            'shock': shock,
                            'date': base_values.index,
                            'value_type': 'Base',
                            'value': base_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))

            # Process parameter sensitivity base quarterly results
            if hasattr(sens_test, 'param_shock_qtr_df'):
                param_shock_qtr_df = sens_test.param_shock_qtr_df
                for scen_set, scen_dict in param_shock_qtr_df.items():
                    for scen_name, qtr_df in scen_dict.items():
                        if qtr_df is not None and not qtr_df.empty:
                            baseline_col = f"{scen_set}_{scen_name}"
                            
                            for col in qtr_df.columns:
                                if col == baseline_col:
                                    continue
                                
                                param_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                                shock = ('+' if '+' in col else '-') + shock
                                
                                qtr_forecast = qtr_df[col].dropna()
                                if not qtr_forecast.empty:
                                    # Convert to base variable
                                    base_values = self.model.base_predictor.predict_base(qtr_forecast, self.model.dm.scen_p0)
                                    
                                    # Create DataFrame for quarterly base results
                                    df_data = {
                                        'model': model_id,
                                        'test': 'Parameter Sensitivity Test',
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'variable/parameter': param_name,
                                        'shock': shock,
                                        'date': base_values.index,
                                        'value_type': 'Base',
                                        'value': base_values.values,
                                        'frequency': 'quarterly'
                                    }
                                    data_list.append(pd.DataFrame(df_data))

            # Process input sensitivity base results (monthly)
            for scen_set, scen_dict in input_shock_df.items():
                for scen_name, df in scen_dict.items():
                    baseline_col = f"{scen_set}_{scen_name}"
                    
                    for col in df.columns:
                        if col == baseline_col:
                            continue
                        
                        var_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                        shock = ('+' if '+' in col else '-') + shock
                        
                        # Convert to base variable
                        base_values = self.model.base_predictor.predict_base(df[col], self.model.dm.scen_p0)
                        
                        # Create DataFrame for base variable results (monthly)
                        df_data = {
                            'model': model_id,
                            'test': 'Input Sensitivity Test',
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'variable/parameter': var_name,
                            'shock': shock,
                            'date': base_values.index,
                            'value_type': 'Base',
                            'value': base_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))

            # Process input sensitivity base quarterly results
            if hasattr(sens_test, 'input_shock_qtr_df'):
                input_shock_qtr_df = sens_test.input_shock_qtr_df
                for scen_set, scen_dict in input_shock_qtr_df.items():
                    for scen_name, qtr_df in scen_dict.items():
                        if qtr_df is not None and not qtr_df.empty:
                            baseline_col = f"{scen_set}_{scen_name}"
                            
                            for col in qtr_df.columns:
                                if col == baseline_col:
                                    continue
                                
                                var_name, shock = col.rsplit('+', 1) if '+' in col else col.rsplit('-', 1)
                                shock = ('+' if '+' in col else '-') + shock
                                
                                qtr_forecast = qtr_df[col].dropna()
                                if not qtr_forecast.empty:
                                    # Convert to base variable
                                    base_values = self.model.base_predictor.predict_base(qtr_forecast, self.model.dm.scen_p0)
                                    
                                    # Create DataFrame for quarterly base results
                                    df_data = {
                                        'model': model_id,
                                        'test': 'Input Sensitivity Test',
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'variable/parameter': var_name,
                                        'shock': shock,
                                        'date': base_values.index,
                                        'value_type': 'Base',
                                        'value': base_values.values,
                                        'frequency': 'quarterly'
                                    }
                                    data_list.append(pd.DataFrame(df_data))

        if not data_list:
            return None

        # Combine all data and ensure column order
        result = pd.concat(data_list, ignore_index=True)
        return result[['model', 'test', 'scenario_name', 'severity', 'variable/parameter', 
                      'shock', 'date', 'frequency', 'value_type', 'value']] 