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
import logging

# Module logger
logger = logging.getLogger(__name__)

# Shared column schemas
TIMESERIES_COLUMNS = ['date', 'model', 'series_type', 'value_type', 'value']
STATICSTATS_COLUMNS = ['category', 'model', 'type', 'value_type', 'value']
SCENARIO_COLUMNS = ['model', 'scenario_name', 'severity', 'date', 'frequency', 'value_type', 'value']
TEST_RESULTS_COLUMNS = ['model', 'test', 'index', 'metric', 'value']
SENSITIVITY_COLUMNS = ['model', 'test', 'scenario_name', 'severity', 'variable/parameter', 'shock', 'date', 'frequency', 'value_type', 'value']

# Series type constants
SERIES_TYPE_TARGET = 'Target'
SERIES_TYPE_BASE = 'Base'
SERIES_TYPE_RESIDUAL = 'Residual'
SERIES_TYPE_IV = 'Independent Variable'

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
                         Valid types are: 'timeseries_data', 'staticStats', 'scenario_testing', 'test_results', 'sensitivity_testing'
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
    def save_dataframe(self, df: pd.DataFrame, filepath: Path, mode: str = 'w', header: bool = True) -> None:
        """Save DataFrame in specific format with append support."""
        pass
    
    @abstractmethod
    def save_dict(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save dictionary in specific format."""
        pass

class CSVFormatHandler(ExportFormatHandler):
    """Handler for CSV format exports."""
    
    def save_dataframe(self, df: pd.DataFrame, filepath: Path, mode: str = 'w', header: bool = True) -> None:
        df.to_csv(filepath, mode=mode, header=header, index=False)
    
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
        # Per-content chunk counters
        self._chunk_sizes = {
            'timeseries_data': 0,
            'staticStats': 0,
            'scenario_testing': 0,
            'test_results': 0,
            'sensitivity_testing': 0,
        }
        self._written_files = set()  # Track which files have been written
    
    def _write_chunk(self, output_dir: Path, content_type: str, force_write: bool = False):
        """Write a chunk of data to disk with proper file handling and user feedback.
        
        Args:
            output_dir: Directory to write the file to
            content_type: Type of content being written ('timeseries_data', 'staticStats', etc.)
            force_write: If True, forces writing regardless of chunk size
        """
        # Configuration for different content types
        content_config = {
            'timeseries_data': {
                'chunks': self._timeseries_chunks,
                'columns': TIMESERIES_COLUMNS,
                'filename': 'timeseries_data.csv'
            },
            'staticStats': {
                'chunks': self._statistics_chunks,
                'columns': STATICSTATS_COLUMNS,
                'filename': 'staticStats.csv'
            },
            'scenario_testing': {
                'chunks': self._scenario_chunks,
                'columns': SCENARIO_COLUMNS,
                'filename': 'scenario_testing.csv'
            },
            'test_results': {
                'chunks': self._test_results_chunks,
                'columns': TEST_RESULTS_COLUMNS,
                'filename': 'test_results.csv'
            },
            'sensitivity_testing': {
                'chunks': self._sensitivity_chunks,
                'columns': SENSITIVITY_COLUMNS,
                'filename': 'sensitivity_testing.csv'
            }
        }
        
        config = content_config.get(content_type)
        if not config or not config['chunks']:
            return
            
        chunks = config['chunks']
        columns = config['columns']
        filepath = output_dir / config['filename']
        
        # Combine chunks and ensure column order
        df = pd.concat(chunks, copy=False)
        df = df[columns]
        
        # Determine write mode based on file existence
        file_exists = filepath.exists()
        write_mode = 'a' if file_exists else 'w'
        write_header = not file_exists
        
        try:
            # Write data (append if file exists, otherwise create new)
            # Delegate to format handler for flexibility and control of mode/header
            self.format_handler.save_dataframe(df, filepath, mode=write_mode, header=write_header)
            
            # Log appropriate success message only for new files or force_write
            if force_write or not file_exists:
                action = "updated" if file_exists else "wrote"
                logger.info("Successfully %s %s file: %s", action, content_type, filepath)
                self._written_files.add(filepath)
        except Exception as e:
            logger.exception("Failed to write %s file to %s: %s", content_type, filepath, e)
        
        # Clear the chunks and reset per-content chunk size
        chunks.clear()
        self._chunk_sizes[content_type] = 0
    
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
            self._chunk_sizes['sensitivity_testing'] += len(df)
            if self._chunk_sizes['sensitivity_testing'] >= self.chunk_size:
                self._write_chunk(output_dir, 'sensitivity_testing')
    
    def export_test_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export comprehensive test results with chunking support.
        
        Exports test results in long format with columns:
        - model: Model identifier
        - test: Descriptive test name (e.g., 'Residual Autocorrelation')
        - index: Variable name or specific test name (e.g., 'Durbin-Watson')
        - metric: Test metric type (e.g., 'Statistic', 'P-value', 'Passed')
        - value: Numerical value
        """
        if not self.should_export('test_results'):
            return
        
        df = model.get_test_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._test_results_chunks.append(df)
            self._chunk_sizes['test_results'] += len(df)
            if self._chunk_sizes['test_results'] >= self.chunk_size:
                self._write_chunk(output_dir, 'test_results')
    
    def export_timeseries_data(self, model: ExportableModel, output_dir: Path) -> None:
        """Export time series data with chunking support."""
        if not self.should_export('timeseries_data'):
            return
        df = model.get_timeseries_data()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._timeseries_chunks.append(df)
            self._chunk_sizes['timeseries_data'] += len(df)
            if self._chunk_sizes['timeseries_data'] >= self.chunk_size:
                self._write_chunk(output_dir, 'timeseries_data')
    
    def export_statistics(self, model: ExportableModel, output_dir: Path) -> None:
        """Export model statistics with chunking support."""
        if not self.should_export('staticStats'):
            return
        
        df = model.get_model_statistics()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._statistics_chunks.append(df)
            self._chunk_sizes['staticStats'] += len(df)
            if self._chunk_sizes['staticStats'] >= self.chunk_size:
                self._write_chunk(output_dir, 'staticStats')
    
    def export_scenarios(self, model: ExportableModel, output_dir: Path) -> None:
        """Export scenario results with chunking support.
        
        The output will include:
        - Target variable forecasts (monthly frequency only)
        - Base variable forecasts (monthly and quarterly frequencies, if available)
        - All data in datetime format
        
        Output format:
        - model: Model identifier
        - scenario_name: Scenario set name (e.g., 'EWST_2024')
        - severity: Severity level (e.g., 'base', 'adv', 'sev')
        - date: End of period date in datetime format
        - frequency: 'monthly' or 'quarterly'
        - value_type: 'Target' or 'Base'
        - value: The forecasted value
        """
        if not self.should_export('scenario_testing'):
            return
        
        # Get scenario results
        df = model.get_scenario_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._scenario_chunks.append(df)
            self._chunk_sizes['scenario_testing'] += len(df)
            if self._chunk_sizes['scenario_testing'] >= self.chunk_size:
                self._write_chunk(output_dir, 'scenario_testing')
    
    def save_consolidated_results(self, output_dir: Path) -> None:
        """Save any remaining data chunks to files."""
        # Write any remaining chunks with force_write=True to ensure final consolidation
        if self.should_export('timeseries_data'):
            self._write_chunk(output_dir, 'timeseries_data', force_write=True)
        
        if self.should_export('staticStats'):
            self._write_chunk(output_dir, 'staticStats', force_write=True)
        
        if self.should_export('scenario_testing'):
            self._write_chunk(output_dir, 'scenario_testing', force_write=True)
        
        if self.should_export('test_results'):
            self._write_chunk(output_dir, 'test_results', force_write=True)
        
        if self.should_export('sensitivity_testing'):  # Add sensitivity testing export
            self._write_chunk(output_dir, 'sensitivity_testing', force_write=True)
        
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
        - series_type: One of ['Target','Base','Residual','Independent Variable']
        - value_type: Sample identifier or variable name
            - 'In-Sample': For in-sample predictions
            - 'Out-of-Sample': For out-of-sample predictions
            - 'Actual': For actual target values
            - 'Residual': For residuals
            - driver names: For feature/driver values
            - For Base series_type, value_type mimics Target: 'Actual', 'In-Sample', 'Out-of-Sample' (if available)
        - value: The actual value
        """
        model_id = self._model_id
        blocks: List[pd.DataFrame] = []
 
        # Target - Actual
        target_data_full = pd.concat([self.model.y_in, self.model.y_out]).sort_index()
        blocks.append(self._build_ts_block(target_data_full.index, model_id, SERIES_TYPE_TARGET, 'Actual', target_data_full.values))
 
        # Target - In-Sample / Out-of-Sample
        predicted_in = self.model.y_fitted_in
        blocks.append(self._build_ts_block(predicted_in.index, model_id, SERIES_TYPE_TARGET, 'In-Sample', predicted_in.values))
 
        if not self.model.X_out.empty:
            predicted_out = self.model.y_pred_out
            blocks.append(self._build_ts_block(predicted_out.index, model_id, SERIES_TYPE_TARGET, 'Out-of-Sample', predicted_out.values))
 
        # Features (Independent Variables) — vectorized for efficiency
        feature_data_in = self.model.X_in
        if not feature_data_in.empty:
            stacked_in = feature_data_in.stack().reset_index()
            stacked_in.columns = ['date', 'value_type', 'value']
            stacked_in['model'] = model_id
            stacked_in['series_type'] = SERIES_TYPE_IV
            # Reorder columns to schema
            stacked_in = stacked_in[['date', 'model', 'series_type', 'value_type', 'value']]
            blocks.append(stacked_in)
 
        if not self.model.X_out.empty:
            feature_data_out = self.model.X_out
            stacked_out = feature_data_out.stack().reset_index()
            stacked_out.columns = ['date', 'value_type', 'value']
            stacked_out['model'] = model_id
            stacked_out['series_type'] = SERIES_TYPE_IV
            stacked_out = stacked_out[['date', 'model', 'series_type', 'value_type', 'value']]
            blocks.append(stacked_out)
 
        # Residuals (in-sample)
        if hasattr(self.model, 'resid') and self.model.resid is not None:
            resid_in = self.model.resid.dropna()
            if not resid_in.empty:
                blocks.append(
                    self._build_ts_block(
                        resid_in.index,
                        model_id,
                        SERIES_TYPE_RESIDUAL,
                        'Residual',
                        resid_in.values
                    )
                )

        # Residuals (out-of-sample)
        if not self.model.X_out.empty:
            predicted_out = self.model.y_pred_out
            if not predicted_out.empty:
                actual_out = self.model.y_out.reindex(predicted_out.index)
                resid_out = (actual_out - predicted_out).dropna()
                if not resid_out.empty:
                    blocks.append(
                        self._build_ts_block(
                            resid_out.index,
                            model_id,
                            SERIES_TYPE_RESIDUAL,
                            'Residual (Out-of-Sample)',
                            resid_out.values
                        )
                    )
 
        # Base variable series if available
        y_base_full = getattr(self.model, 'y_base_full', None)
        if y_base_full is not None and not y_base_full.empty:
            blocks.append(self._build_ts_block(y_base_full.index, model_id, SERIES_TYPE_BASE, 'Actual', y_base_full.values))
        y_base_fitted_in = getattr(self.model, 'y_base_fitted_in', None)
        if y_base_fitted_in is not None and not y_base_fitted_in.empty:
            blocks.append(self._build_ts_block(y_base_fitted_in.index, model_id, SERIES_TYPE_BASE, 'In-Sample', y_base_fitted_in.values))
        y_base_pred_out = getattr(self.model, 'y_base_pred_out', None)
        if y_base_pred_out is not None and not y_base_pred_out.empty:
            blocks.append(self._build_ts_block(y_base_pred_out.index, model_id, SERIES_TYPE_BASE, 'Out-of-Sample', y_base_pred_out.values))

        # Base residuals (computed if actuals exist)
        if y_base_full is not None and not y_base_full.empty:
            # In-sample residuals for Base: Actual - Fitted_IS
            if y_base_fitted_in is not None and not y_base_fitted_in.empty:
                aligned_actual_in = y_base_full.reindex(y_base_fitted_in.index)
                base_resid_in = (aligned_actual_in - y_base_fitted_in).dropna()
                if not base_resid_in.empty:
                    blocks.append(self._build_ts_block(base_resid_in.index, model_id, SERIES_TYPE_BASE, 'Residual', base_resid_in.values))
            # Out-of-sample residuals for Base: Actual - Pred_OOS (if overlapping actuals exist)
            if y_base_pred_out is not None and not y_base_pred_out.empty:
                aligned_actual_out = y_base_full.reindex(y_base_pred_out.index)
                base_resid_out = (aligned_actual_out - y_base_pred_out).dropna()
                if not base_resid_out.empty:
                    blocks.append(self._build_ts_block(base_resid_out.index, model_id, SERIES_TYPE_BASE, 'Residual', base_resid_out.values))
 
        return pd.concat(blocks, ignore_index=True, copy=False)
    
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
            
            def append_error_measures(df_like: Any, sample_label: str):
                if isinstance(df_like, pd.DataFrame):
                    # Expect index='Metric' and a single 'Value' column
                    value_col = 'Value' if 'Value' in df_like.columns else None
                    if value_col is not None:
                        for metric_name, row in df_like.iterrows():
                            metric_value = row.get(value_col)
                            if pd.notnull(metric_value):
                                stats_list.append({
                        'category': 'Goodness of Fit',
                        'model': model_id,
                                    'type': sample_label,
                                    'value_type': str(metric_name),
                                    'value': float(metric_value)
                                })
                elif isinstance(df_like, pd.Series):
                    # Series keyed by metric name
                    for metric_name, metric_value in df_like.items():
                        if pd.notnull(metric_value):
                            stats_list.append({
                        'category': 'Goodness of Fit',
                        'model': model_id,
                                'type': sample_label,
                                'value_type': str(metric_name),
                                'value': float(metric_value)
                            })
                elif isinstance(df_like, dict):
                    for metric_name, metric_value in df_like.items():
                        if pd.notnull(metric_value):
                            stats_list.append({
                                'category': 'Goodness of Fit',
                                'model': model_id,
                                'type': sample_label,
                                'value_type': str(metric_name),
                                'value': float(metric_value)
                            })
            
            # Prefer explicit IS/OOS error measure tests if available
            if 'IS Error Measures' in test_dict:
                is_errors = test_dict['IS Error Measures'].test_result
                append_error_measures(is_errors, 'In-Sample')
             
            if not getattr(self.model, 'X_out', pd.DataFrame()).empty and 'OOS Error Measures' in test_dict:
                oos_errors = test_dict['OOS Error Measures'].test_result
                append_error_measures(oos_errors, 'Out-of-Sample')
             
            # Fallback: scan all tests for error-measure-shaped tables if explicit names are absent
            # This handles cases where aliases differ but structure matches
            if 'IS Error Measures' not in test_dict or ('OOS Error Measures' not in test_dict and not getattr(self.model, 'X_out', pd.DataFrame()).empty):
                for test in self.model.testset.tests:
                    name_lower = str(test.name).lower()
                    try:
                        tr = test.test_result
                    except Exception:
                        continue
                    if isinstance(tr, pd.DataFrame) and 'Value' in tr.columns and tr.index.name == 'Metric':
                        # Heuristic: treat as error measures table
                        sample_label = 'In-Sample'
                        if 'oos' in name_lower or 'out' in name_lower:
                            sample_label = 'Out-of-Sample'
                        append_error_measures(tr, sample_label)
            
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

        # Add summary statistics for target, base, and driver variables
        def _add_summary_stats(data_series: Optional[pd.Series], var_name: str) -> None:
            if data_series is None:
                return

            clean_series = data_series.dropna()
            if clean_series.empty:
                return

            summary_stats = {
                'Mean': clean_series.mean(),
                'Std': clean_series.std(),
                'Min': clean_series.min(),
                'Max': clean_series.max(),
                'Median': clean_series.median(),
                '25th Percentile': clean_series.quantile(0.25),
                '90th Percentile': clean_series.quantile(0.90),
                '95th Percentile': clean_series.quantile(0.95),
            }

            for stat_name, stat_value in summary_stats.items():
                if pd.notnull(stat_value):
                    stats_list.append({
                        'category': 'Summary Statistics',
                        'model': model_id,
                        'type': stat_name,
                        'value_type': var_name,
                        'value': float(stat_value)
                    })

        # Target variable summary statistics (combine in- and out-of-sample actuals when available)
        target_in = getattr(self.model, 'y_in', pd.Series(dtype=float))
        target_out = getattr(self.model, 'y_out', pd.Series(dtype=float))
        if not target_out.empty:
            target_data = pd.concat([target_in, target_out]).sort_index()
        else:
            target_data = target_in
        _add_summary_stats(target_data, 'Target')

        # Base variable summary statistics (if available)
        base_data = getattr(self.model, 'y_base_full', pd.Series(dtype=float))
        _add_summary_stats(base_data, 'Base')

        # Independent variables summary statistics based on in-sample history
        feature_data_in = getattr(self.model, 'X_in', pd.DataFrame())
        if not feature_data_in.empty:
            for var_name in feature_data_in.columns:
                _add_summary_stats(feature_data_in[var_name], var_name)

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
        
        The method processes both target and base variable forecasts if available:
        - Target variable: Monthly frequency only (quarterly target forecasts deprecated)
        - Base variable: Both monthly and quarterly frequencies
        - Includes scen_p0 baseline data for both target and base variables
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
 
        # Add historical actuals (Target) monthly and quarterly (if applicable)
        target_actual = getattr(self.model, 'y_full', None)
        if target_actual is not None and not target_actual.empty:
            # Monthly actuals per scenario set
            for scen_set in scen_results.keys():
                df_data = {
                    'model': model_id,
                    'scenario_name': scen_set,
                    'severity': 'actual',
                    'date': target_actual.index,
                    'value_type': 'Target',
                    'value': target_actual.values,
                    'frequency': 'monthly'
                }
                data_list.append(pd.DataFrame(df_data))
            # Quarterly actuals aggregated to quarter-end
            actual_q = target_actual.copy()
            actual_q.index = pd.to_datetime(actual_q.index)
            actual_q = actual_q.groupby(pd.Grouper(freq='Q')).mean()
            actual_q.index = actual_q.index.to_period('Q').to_timestamp(how='end').normalize()
            if not actual_q.empty:
                for scen_set in scen_results.keys():
                    df_data = {
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'actual',
                        'date': actual_q.index,
                        'value_type': 'Target',
                        'value': actual_q.values,
                        'frequency': 'quarterly'
                    }
                    data_list.append(pd.DataFrame(df_data))

        # Process target variable quarterly forecasts
        # Target variable quarterly forecasts are no longer available (forecast_y_qtr_df deprecated)
 
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
 
        # Add historical actuals (Base) monthly and quarterly if base actuals exist on model
        base_actual = getattr(self.model, 'y_base_full', None)
        if base_actual is not None and not base_actual.empty:
            for scen_set in scen_results.keys():
                df_data = {
                    'model': model_id,
                    'scenario_name': scen_set,
                    'severity': 'actual',
                    'date': base_actual.index,
                    'value_type': 'Base',
                    'value': base_actual.values,
                    'frequency': 'monthly'
                }
                data_list.append(pd.DataFrame(df_data))
            # Quarterly aggregation
            base_actual_q = base_actual.copy()
            base_actual_q.index = pd.to_datetime(base_actual_q.index)
            base_actual_q = base_actual_q.groupby(pd.Grouper(freq='Q')).mean()
            base_actual_q.index = base_actual_q.index.to_period('Q').to_timestamp(how='end').normalize()
            if not base_actual_q.empty:
                for scen_set in scen_results.keys():
                    df_data = {
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'actual',
                        'date': base_actual_q.index,
                        'value_type': 'Base',
                        'value': base_actual_q.values,
                        'frequency': 'quarterly'
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
        """Return test results in long format.
        
        Returns a DataFrame with columns:
        - model: string, model_id
        - test: string (e.g., 'Residual Autocorrelation', 'Coefficient Significance')
        - index: string (variable name or specific test name like 'Durbin-Watson')
        - metric: string ('Statistic', 'P-value', 'Threshold', 'Passed', etc.)
        - value: numerical
        
        Excludes error measures (RMSE, MAE) and fit measures (R², etc.) from export.
        
        Example output:
        - model='cm1', test='Residual Autocorrelation', index='Durbin-Watson', metric='Statistic', value=2.01
        - model='cm1', test='Residual Autocorrelation', index='Durbin-Watson', metric='Threshold', value=1.5
        - model='cm1', test='Residual Autocorrelation', index='Durbin-Watson', metric='Passed', value=1
        """
        try:
            # Follow the exact same pattern as ModelReportBase.show_test_tbl()
            results = self.model.testset.all_test_results
        except Exception:
            return None
        
        if not results or len(results) == 0:
            return None
        
        model_id = self._model_id
        all_results = []
        
        # Process each test result
        for test_name, result_df in results.items():
            # Skip measure category tests (FitMeasure, ErrorMeasure)
            if any(keyword in test_name for keyword in ['Error Measures', 'Fit Measures']):
                continue
                
            # Map test names to descriptive names
            descriptive_test_name = self._map_test_name(test_name)
            
            # Transform the test result DataFrame to long format
            transformed_results = self._transform_test_to_long_format(
                result_df, model_id, descriptive_test_name
            )
            all_results.extend(transformed_results)
        
        if not all_results:
            return None
            
        return pd.DataFrame(all_results)
    
    def _map_test_name(self, test_name: str) -> str:
        """Map internal test names to descriptive names."""
        test_name_lower = test_name.lower()
        
        # Residual-based tests
        if 'autocorr' in test_name_lower or 'durbin' in test_name_lower:
            return 'Residual Autocorrelation'
        elif 'heteroscedasticity' in test_name_lower or 'het' in test_name_lower:
            return 'Residual Heteroscedasticity'
        elif 'normality' in test_name_lower or 'jarque' in test_name_lower:
            return 'Residual Normality'
        
        # Stationarity tests - determine context
        elif 'stationarity' in test_name_lower:
            if 'resid' in test_name_lower:
                return 'Residual Stationarity'
            elif any(keyword in test_name_lower for keyword in ['x', 'independent', 'input']):
                return 'Independent Variable Stationarity'
            elif any(keyword in test_name_lower for keyword in ['y', 'dependent', 'target']):
                return 'Dependent Variable Stationarity'
            else:
                return 'Stationarity Test'
        
        # Model estimation tests
        elif 'significance' in test_name_lower or 'coef' in test_name_lower:
            return 'Coefficient Significance'
        elif 'sign' in test_name_lower:
            return 'Coefficient Sign Check'
        elif 'group' in test_name_lower or 'f-test' in test_name_lower:
            return 'Group Driver F-test'
        
        # Multicollinearity
        elif 'vif' in test_name_lower or 'collinearity' in test_name_lower:
            return 'Multicollinearity'
        
        # Cointegration
        elif 'coint' in test_name_lower:
            return 'Cointegration'
        
        # Default fallback
        else:
            return test_name
    
    def _transform_test_to_long_format(
        self, 
        test_df: pd.DataFrame, 
        model_id: str, 
        test_name: str
    ) -> List[Dict[str, Any]]:
        """Transform test result DataFrame to long format."""
        if not isinstance(test_df, pd.DataFrame) or test_df.empty:
            return []
        
        results: List[Dict[str, Any]] = []
        
        for index_name, row in test_df.iterrows():
            for column_name, value in row.items():
                # Handle thresholds represented as (lower, upper)
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    lower_val, upper_val = value
                    results.append({
                        'model': model_id,
                        'test': test_name,
                        'index': str(index_name),
                        'metric': f'{column_name}_Lower',
                        'value': float(lower_val) if pd.notnull(lower_val) else None
                    })
                    results.append({
                        'model': model_id,
                        'test': test_name,
                        'index': str(index_name),
                        'metric': f'{column_name}_Upper',
                        'value': float(upper_val) if pd.notnull(upper_val) else None
                    })
                    continue
                
                # Special-case: expected sign mapping
                if test_name == 'Coefficient Sign Check' and column_name == 'Expected':
                    expected_str = str(value).strip().lower()
                    if expected_str in ['+', 'positive']:
                        numeric_value = 1.0
                    elif expected_str in ['-', 'negative']:
                        numeric_value = -1.0
                    else:
                        numeric_value = 0.0
                elif isinstance(value, bool):
                    numeric_value = 1.0 if value else 0.0
                elif pd.isnull(value):
                    numeric_value = None
                else:
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values to ensure 'value' stays numeric
                        numeric_value = None
                
                if numeric_value is not None:
                    results.append({
                        'model': model_id,
                        'test': test_name,
                        'index': str(index_name),
                        'metric': column_name,
                        'value': numeric_value
                    })
        
        return results

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

        if not sens_test.param_names:
            print(
                f"Info: Model {model_id} has no eligible variables for sensitivity testing; "
                "skipping sensitivity export."
            )
            return None

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
                
                # Include baseline (no shock) target series
                if baseline_col in df.columns:
                    df_data = {
                        'model': model_id,
                        'test': 'Parameter Sensitivity Test',
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'variable/parameter': 'no_shock',
                        'shock': 'baseline',
                        'date': df.index,
                        'value_type': 'Target',
                        'value': df[baseline_col].values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))

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
                        
                        # Include baseline (no shock) quarterly target series
                        if baseline_col in qtr_df.columns:
                            qtr_forecast = qtr_df[baseline_col].dropna()
                            if not qtr_forecast.empty:
                                df_data = {
                                    'model': model_id,
                                    'test': 'Parameter Sensitivity Test',
                                    'scenario_name': scen_set,
                                    'severity': scen_name,
                                    'variable/parameter': 'no_shock',
                                    'shock': 'baseline',
                                    'date': qtr_forecast.index,
                                    'value_type': 'Target',
                                    'value': qtr_forecast.values,
                                    'frequency': 'quarterly'
                                }
                                data_list.append(pd.DataFrame(df_data))

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
                
                # Include baseline (no shock) input series target
                if baseline_col in df.columns:
                    df_data = {
                        'model': model_id,
                        'test': 'Input Sensitivity Test',
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'variable/parameter': 'no_shock',
                        'shock': 'baseline',
                        'date': df.index,
                        'value_type': 'Target',
                        'value': df[baseline_col].values,
                        'frequency': 'monthly'
                    }
                    data_list.append(pd.DataFrame(df_data))

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
                        
                        # Include baseline (no shock) quarterly target series
                        if baseline_col in qtr_df.columns:
                            qtr_forecast = qtr_df[baseline_col].dropna()
                            if not qtr_forecast.empty:
                                df_data = {
                                    'model': model_id,
                                    'test': 'Input Sensitivity Test',
                                    'scenario_name': scen_set,
                                    'severity': scen_name,
                                    'variable/parameter': 'no_shock',
                                    'shock': 'baseline',
                                    'date': qtr_forecast.index,
                                    'value_type': 'Target',
                                    'value': qtr_forecast.values,
                                    'frequency': 'quarterly'
                                }
                                data_list.append(pd.DataFrame(df_data))

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
                    
                    # Include baseline (no shock) base conversion for parameter sensitivity monthly
                    if baseline_col in df.columns:
                        base_values = self.model.base_predictor.predict_base(df[baseline_col], self.model.dm.scen_p0)
                        df_data = {
                            'model': model_id,
                            'test': 'Parameter Sensitivity Test',
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'variable/parameter': 'no_shock',
                            'shock': 'baseline',
                            'date': base_values.index,
                            'value_type': 'Base',
                            'value': base_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))

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
                            
                            # Include baseline (no shock) base conversion for quarterly
                            if baseline_col in qtr_df.columns:
                                qtr_forecast = qtr_df[baseline_col].dropna()
                                if not qtr_forecast.empty:
                                    base_values = self.model.base_predictor.predict_base(qtr_forecast, self.model.dm.scen_p0)
                                    df_data = {
                                        'model': model_id,
                                        'test': 'Parameter Sensitivity Test',
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'variable/parameter': 'no_shock',
                                        'shock': 'baseline',
                                        'date': base_values.index,
                                        'value_type': 'Base',
                                        'value': base_values.values,
                                        'frequency': 'quarterly'
                                    }
                                    data_list.append(pd.DataFrame(df_data))

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
                    
                    # Include baseline (no shock) base conversion for input monthly
                    if baseline_col in df.columns:
                        base_values = self.model.base_predictor.predict_base(df[baseline_col], self.model.dm.scen_p0)
                        df_data = {
                            'model': model_id,
                            'test': 'Input Sensitivity Test',
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'variable/parameter': 'no_shock',
                            'shock': 'baseline',
                            'date': base_values.index,
                            'value_type': 'Base',
                            'value': base_values.values,
                            'frequency': 'monthly'
                        }
                        data_list.append(pd.DataFrame(df_data))

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
                            
                            # Include baseline (no shock) base conversion for input quarterly
                            if baseline_col in qtr_df.columns:
                                qtr_forecast = qtr_df[baseline_col].dropna()
                                if not qtr_forecast.empty:
                                    base_values = self.model.base_predictor.predict_base(qtr_forecast, self.model.dm.scen_p0)
                                    df_data = {
                                        'model': model_id,
                                        'test': 'Input Sensitivity Test',
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'variable/parameter': 'no_shock',
                                        'shock': 'baseline',
                                        'date': base_values.index,
                                        'value_type': 'Base',
                                        'value': base_values.values,
                                        'frequency': 'quarterly'
                                    }
                                    data_list.append(pd.DataFrame(df_data))

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

    # Helper to build a standardized time series DataFrame row block
    def _build_ts_block(self, index, model_id: str, series_type: str, value_type: str, values) -> pd.DataFrame:
        return pd.DataFrame({
            'date': index,
            'model': model_id,
            'series_type': series_type,
            'value_type': value_type,
            'value': values
        }) 