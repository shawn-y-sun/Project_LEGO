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
STABILITY_COLUMNS = ['date', 'model', 'test_period', 'value_type', 'value']
STABILITY_STATS_COLUMNS = ['model', 'trial', 'category', 'value_type', 'value']
SCENARIO_STATS_COLUMNS = ['model', 'scenario_name', 'metric', 'period', 'severity', 'value']

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
    'sensitivity_testing': 'Sensitivity testing results for parameters and inputs',
    'stability_testing': 'Walk-forward stability testing results',
    'stability_testing_stats': 'Walk-forward stability testing statistical metrics',
    'scenario_testing_stats': 'Scenario testing statistical metrics for base variables'
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

    @abstractmethod
    def get_stability_results(self) -> Optional[pd.DataFrame]:
        """Return walk-forward stability testing time series in long format."""
        pass

    @abstractmethod
    def get_stability_stats_results(self) -> Optional[pd.DataFrame]:
        """Return walk-forward stability testing statistical metrics."""
        pass

    @abstractmethod
    def get_scenario_stats_results(self) -> Optional[pd.DataFrame]:
        """Return scenario testing statistical metrics for base variables."""
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

    @abstractmethod
    def export_sensitivity_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export sensitivity testing results to CSV if available."""
        pass

    @abstractmethod
    def export_stability_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export walk-forward stability testing results to CSV if available."""
        pass

    @abstractmethod
    def export_stability_stats(self, model: ExportableModel, output_dir: Path) -> None:
        """Export walk-forward stability testing statistical metrics to CSV if available."""
        pass

    @abstractmethod
    def export_scenario_stats(self, model: ExportableModel, output_dir: Path) -> None:
        """Export scenario testing statistical metrics to CSV if available."""
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
                
                if self.strategy.should_export('stability_testing'):
                    try:
                        self.strategy.export_stability_results(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export stability_testing for {model.get_model_id()}: {e}")
                
                if self.strategy.should_export('stability_testing_stats'):
                    try:
                        self.strategy.export_stability_stats(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export stability_testing_stats for {model.get_model_id()}: {e}")
                
                if self.strategy.should_export('scenario_testing_stats'):
                    try:
                        self.strategy.export_scenario_stats(model, output_dir)
                    except Exception as e:
                        print(f"Warning: Failed to export scenario_testing_stats for {model.get_model_id()}: {e}")
                        
            except Exception as e:
                print(f"Error: Failed to process model {model.get_model_id()}: {e}")
                continue
        
        # Save consolidated results
        self.strategy.save_consolidated_results(output_dir)

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
        self._stability_chunks = []  # New container for stability results
        self._stability_stats_chunks = [] # New container for stability stats
        self._scenario_stats_chunks = [] # New container for scenario stats
        # Per-content chunk counters
        self._chunk_sizes = {
            'timeseries_data': 0,
            'staticStats': 0,
            'scenario_testing': 0,
            'test_results': 0,
            'sensitivity_testing': 0,
            'stability_testing': 0,
            'stability_testing_stats': 0,
            'scenario_testing_stats': 0
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
            },
            'stability_testing': {
                'chunks': self._stability_chunks,
                'columns': STABILITY_COLUMNS,
                'filename': 'stability_testing.csv'
            },
            'stability_testing_stats': {
                'chunks': self._stability_stats_chunks,
                'columns': STABILITY_STATS_COLUMNS,
                'filename': 'stability_testing_stats.csv'
            },
            'scenario_testing_stats': {
                'chunks': self._scenario_stats_chunks,
                'columns': SCENARIO_STATS_COLUMNS,
                'filename': 'scenario_testing_stats.csv'
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
            
            # Always track written files, regardless of mode
            self._written_files.add(filepath)
            
            # Log appropriate success message only for new files or force_write
            if force_write or not file_exists:
                action = "updated" if file_exists else "wrote"
                logger.info("Successfully %s %s file: %s", action, content_type, filepath)
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
    
    def export_stability_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export walk-forward stability results with chunking support."""
        if not self.should_export('stability_testing'):
            return
        
        df = model.get_stability_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._stability_chunks.append(df)
            self._chunk_sizes['stability_testing'] += len(df)
            if self._chunk_sizes['stability_testing'] >= self.chunk_size:
                self._write_chunk(output_dir, 'stability_testing')
    
    def export_stability_stats(self, model: ExportableModel, output_dir: Path) -> None:
        """Export walk-forward stability testing statistical metrics with chunking support."""
        if not self.should_export('stability_testing_stats'):
            return
        
        df = model.get_stability_stats_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._stability_stats_chunks.append(df)
            self._chunk_sizes['stability_testing_stats'] += len(df)
            if self._chunk_sizes['stability_testing_stats'] >= self.chunk_size:
                self._write_chunk(output_dir, 'stability_testing_stats')
    
    def export_scenario_stats(self, model: ExportableModel, output_dir: Path) -> None:
        """Export scenario testing statistical metrics with chunking support."""
        if not self.should_export('scenario_testing_stats'):
            return
        
        df = model.get_scenario_stats_results()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._scenario_stats_chunks.append(df)
            self._chunk_sizes['scenario_testing_stats'] += len(df)
            if self._chunk_sizes['scenario_testing_stats'] >= self.chunk_size:
                self._write_chunk(output_dir, 'scenario_testing_stats')
    
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
        
        if self.should_export('sensitivity_testing'):
            self._write_chunk(output_dir, 'sensitivity_testing', force_write=True)
        
        if self.should_export('stability_testing'):
            self._write_chunk(output_dir, 'stability_testing', force_write=True)
        
        if self.should_export('stability_testing_stats'):
            self._write_chunk(output_dir, 'stability_testing_stats', force_write=True)
        
        if self.should_export('scenario_testing_stats'):
            self._write_chunk(output_dir, 'scenario_testing_stats', force_write=True)
        
        # Reset containers but preserve written files tracking
        written_files_backup = self._written_files.copy()
        self._initialize_containers()
        self._written_files = written_files_backup

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
 
        # Residuals (negate statsmodels convention to get fitted - actual)
        blocks.append(self._build_ts_block(self.model.resid.index, model_id, SERIES_TYPE_RESIDUAL, 'Residual', -self.model.resid.values))
 
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
            # In-sample residuals for Base: Fitted_IS - Actual
            if y_base_fitted_in is not None and not y_base_fitted_in.empty:
                aligned_actual_in = y_base_full.reindex(y_base_fitted_in.index)
                base_resid_in = (y_base_fitted_in - aligned_actual_in).dropna()
                if not base_resid_in.empty:
                    blocks.append(self._build_ts_block(base_resid_in.index, model_id, SERIES_TYPE_BASE, 'Residual', base_resid_in.values))
            # Out-of-sample residuals for Base: Pred_OOS - Actual (if overlapping actuals exist)
            if y_base_pred_out is not None and not y_base_pred_out.empty:
                aligned_actual_out = y_base_full.reindex(y_base_pred_out.index)
                base_resid_out = (y_base_pred_out - aligned_actual_out).dropna()
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
        
        # Add VIF values if available
        if hasattr(self.model, 'vif') and self.model.vif is not None:
            vif_values = self.model.vif
            if isinstance(vif_values, pd.Series):
                for var_name, vif_value in vif_values.items():
                    if pd.notnull(vif_value):
                        stats_list.append({
                            'category': 'Model Estimation',
                            'model': model_id,
                            'type': 'VIF',
                            'value_type': var_name,
                            'value': float(vif_value)
                        })
            elif isinstance(vif_values, dict):
                for var_name, vif_value in vif_values.items():
                    if pd.notnull(vif_value):
                        stats_list.append({
                            'category': 'Model Estimation',
                            'model': model_id,
                            'type': 'VIF',
                            'value_type': var_name,
                            'value': float(vif_value)
                        })
        
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
        # Try to get test results from testset
        try:
            if not hasattr(self.model, 'testset') or self.model.testset is None:
                return None
            
            # Access individual test objects directly from testset.tests
            test_objects = getattr(self.model.testset, 'tests', [])
            if not test_objects:
                return None
                
        except Exception:
            return None
        
        model_id = self._model_id
        all_results = []
        
        # Process each test object
        for test_obj in test_objects:
            try:
                # Skip measure category tests (FitMeasure, ErrorMeasure)
                if hasattr(test_obj, 'category') and test_obj.category == 'measure':
                    continue
                    
                # Get test name and map to descriptive name
                test_name = test_obj.name if hasattr(test_obj, 'name') else type(test_obj).__name__
                descriptive_test_name = self._map_test_name(test_name)
                
                # Get test result DataFrame
                if hasattr(test_obj, 'test_result'):
                    result_df = test_obj.test_result
                    
                    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                        # Transform DataFrame: index becomes "index", columns become "metric"
                        transformed_results = self._transform_test_to_long_format(
                            result_df, model_id, descriptive_test_name
                        )
                        all_results.extend(transformed_results)
                    elif isinstance(result_df, pd.Series) and not result_df.empty:
                        # Convert Series to DataFrame format
                        temp_df = pd.DataFrame([result_df.values], columns=result_df.index, index=['Result'])
                        transformed_results = self._transform_test_to_long_format(
                            temp_df, model_id, descriptive_test_name
                        )
                        all_results.extend(transformed_results)
                        
            except Exception:
                continue
        
        if not all_results:
            return None
            
        return pd.DataFrame(all_results)
    
    def _map_test_name(self, test_name: str) -> str:
        """Map internal test names to descriptive names."""
        test_name_lower = test_name.lower()
        
        # Residual-based tests
        if 'autocorr' in test_name_lower or 'durbin' in test_name_lower:
            return 'Residual Autocorrelation'
        elif 'heteroscedasticity' in test_name_lower or 'het' in test_name_lower or 'white' in test_name_lower or 'breusch' in test_name_lower:
            return 'Residual Heteroscedasticity'
        elif 'normality' in test_name_lower or 'jarque' in test_name_lower or 'bera' in test_name_lower:
            return 'Residual Normality'
        
        # Stationarity tests - determine context
        elif 'stationarity' in test_name_lower or 'adf' in test_name_lower or 'unit' in test_name_lower:
            if 'resid' in test_name_lower or 'residual' in test_name_lower:
                return 'Residual Stationarity'
            elif any(keyword in test_name_lower for keyword in ['x', 'independent', 'input', 'feature']):
                return 'Independent Variable Stationarity'
            elif any(keyword in test_name_lower for keyword in ['y', 'dependent', 'target']):
                return 'Dependent Variable Stationarity'
            else:
                return 'Stationarity Test'
        
        # Model estimation tests
        elif 'significance' in test_name_lower or ('coef' in test_name_lower and 'sign' not in test_name_lower):
            return 'Coefficient Significance'
        elif 'sign' in test_name_lower and 'coef' in test_name_lower:
            return 'Coefficient Sign Check'
        elif 'group' in test_name_lower or 'f-test' in test_name_lower or 'ftest' in test_name_lower:
            return 'Group Driver F-test'
        
        # Multicollinearity
        elif 'vif' in test_name_lower or 'collinearity' in test_name_lower or 'multicollinear' in test_name_lower:
            return 'Multicollinearity'
        
        # Cointegration tests - be more specific
        elif 'coint' in test_name_lower or 'cointegration' in test_name_lower:
            if 'engle' in test_name_lower and 'granger' in test_name_lower:
                return 'Engle-Granger Cointegration'
            elif 'johansen' in test_name_lower:
                return 'Johansen Cointegration'
            elif 'phillips' in test_name_lower and 'ouliaris' in test_name_lower:
                return 'Phillips-Ouliaris Cointegration'
            else:
                return 'Cointegration'
        
        # Additional specific tests
        elif 'ljung' in test_name_lower and 'box' in test_name_lower:
            return 'Ljung-Box Test'
        elif 'arch' in test_name_lower:
            return 'ARCH Test'
        elif 'reset' in test_name_lower or 'ramsey' in test_name_lower:
            return 'RESET Test'
        elif 'chow' in test_name_lower:
            return 'Chow Test'
        elif 'cusum' in test_name_lower:
            return 'CUSUM Test'
        
        # Default fallback - clean up the name
        else:
            # Remove common prefixes/suffixes and clean up
            cleaned_name = test_name.replace('_', ' ').replace('-', ' ')
            # Capitalize first letter of each word
            return ' '.join(word.capitalize() for word in cleaned_name.split())
    
    def _transform_test_to_long_format(
        self, 
        test_df: pd.DataFrame, 
        model_id: str, 
        test_name: str
    ) -> List[Dict[str, Any]]:
        """Transform test result DataFrame to long format.
        
        Converts DataFrame rows (index) and columns (metric) to long format where:
        - Each row index becomes an 'index' value  
        - Each column becomes a 'metric' value
        - All values are converted to numeric format
        """
        if not isinstance(test_df, pd.DataFrame) or test_df.empty:
            return []
        
        results: List[Dict[str, Any]] = []
        
        # Handle case where DataFrame has no explicit index names
        if test_df.index.name is None and len(test_df.index) == 1:
            # Single-row DataFrame - use a generic index name
            index_names = ['Test_Result']
        else:
            index_names = test_df.index.tolist()
        
        for i, (index_name, row) in enumerate(test_df.iterrows()):
            # Use the actual index name or fallback to position-based name
            if isinstance(index_name, (int, float)) and test_df.index.name is None:
                display_index = index_names[i] if i < len(index_names) else f"Row_{i}"
            else:
                display_index = str(index_name)
            
            for column_name, value in row.items():
                # Handle thresholds represented as (lower, upper) tuples
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    lower_val, upper_val = value
                    # Create separate entries for lower and upper bounds
                    if pd.notnull(lower_val):
                        try:
                            results.append({
                                'model': model_id,
                                'test': test_name,
                                'index': display_index,
                                'metric': f'{column_name}_Lower',
                                'value': float(lower_val)
                            })
                        except (ValueError, TypeError):
                            pass
                    
                    if pd.notnull(upper_val):
                        try:
                            results.append({
                                'model': model_id,
                                'test': test_name,
                                'index': display_index,
                                'metric': f'{column_name}_Upper',
                                'value': float(upper_val)
                            })
                        except (ValueError, TypeError):
                            pass
                    continue
                
                # Handle nested data structures
                if isinstance(value, (list, tuple)) and len(value) > 2:
                    # Multiple values - create separate entries
                    for j, sub_value in enumerate(value):
                        if pd.notnull(sub_value):
                            try:
                                numeric_value = float(sub_value)
                                results.append({
                                    'model': model_id,
                                    'test': test_name,
                                    'index': display_index,
                                    'metric': f'{column_name}_{j}',
                                    'value': numeric_value
                                })
                            except (ValueError, TypeError):
                                pass
                    continue
                
                # Convert value to appropriate format (preserve strings and numbers)
                final_value = None
                
                if isinstance(value, bool):
                    # Convert boolean to string for clarity
                    final_value = 'True' if value else 'False'
                elif pd.isnull(value):
                    final_value = None
                elif isinstance(value, (int, float)):
                    # Keep numeric values as-is
                    final_value = float(value)
                elif isinstance(value, str):
                    # Keep string values as-is (no forced conversion)
                    final_value = str(value).strip()
                else:
                    # Convert other types to string representation
                    final_value = str(value)
                
                # Add result if we have a valid value (string or numeric)
                if final_value is not None:
                    results.append({
                        'model': model_id,
                        'test': test_name,
                        'index': display_index,
                        'metric': str(column_name),
                        'value': final_value
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

    def get_stability_results(self) -> Optional[pd.DataFrame]:
        """Return walk-forward stability testing results in long format.
        
        Columns:
        - date: timestamp index
        - model: model id
        - test_period: WF label with period (e.g., 'WF1: Mar2003-Mar2024')
        - value_type: 'Actual' | 'In-Sample' | 'Out-of-Sample'
        - value: numeric value
        """
        # Try to build Walk Forward Test via model API
        try:
            wft = self.model.stability_test
        except Exception:
            return None
        
        wf_models = getattr(wft, 'wf_models', None)
        if not isinstance(wf_models, dict) or len(wf_models) == 0:
            return None
        
        def _format_month(dt: pd.Timestamp) -> str:
            return dt.strftime('%b%Y')
        
        def _format_period_label(wf_idx: int, wf_model) -> str:
            # Use each WF model's own in-sample start/end when available
            is_start = getattr(wf_model.dm._internal_loader, 'in_sample_start', None)
            is_end = getattr(wf_model.dm, 'in_sample_end', None)
            if is_start is None or is_end is None:
                return f"WF{wf_idx}"
            # Convert to pandas Timestamp
            is_start = pd.to_datetime(is_start)
            is_end = pd.to_datetime(is_end)
            if getattr(wf_model.dm, 'freq', 'M') == 'M':
                return f"WF{wf_idx}: {_format_month(is_start)}-{_format_month(is_end)}"
            elif getattr(wf_model.dm, 'freq', 'M') == 'Q':
                q_start = f"{is_start.year}-Q{(is_start.month - 1)//3 + 1}"
                q_end = f"{is_end.year}-Q{(is_end.month - 1)//3 + 1}"
                return f"WF{wf_idx}: {q_start}-{q_end}"
            else:
                return f"WF{wf_idx}: {is_start.date()}-{is_end.date()}"
        
        blocks: List[pd.DataFrame] = []
        model_id = self._model_id
        
        # Iterate in the WF order (wf_models preserves insertion order)
        for i, (wf_name, wf_model) in enumerate(wf_models.items(), start=1):
            try:
                label = _format_period_label(i, wf_model)
                # Actual values: from in-sample start to end of OOS if available
                actual_series = wf_model.dm.internal_data[wf_model.target]
                oos_end = wf_model.dm.out_sample_idx.max() if len(getattr(wf_model.dm, 'out_sample_idx', [])) > 0 else wf_model.dm.in_sample_end
                is_start = getattr(wf_model.dm._internal_loader, 'in_sample_start', None)
                if is_start is None:
                    is_start = actual_series.index.min()
                # Build mask and slice
                is_start = pd.to_datetime(is_start)
                oos_end = pd.to_datetime(oos_end)
                actual_slice = actual_series[(actual_series.index >= is_start) & (actual_series.index <= oos_end)]
                if not actual_slice.empty:
                    blocks.append(pd.DataFrame({
                        'date': actual_slice.index,
                        'model': model_id,
                        'test_period': label,
                        'value_type': 'Actual',
                        'value': actual_slice.values
                    }))
                
                # In-Sample fitted
                y_fitted_in = getattr(wf_model, 'y_fitted_in', pd.Series(dtype=float))
                if isinstance(y_fitted_in, pd.Series) and not y_fitted_in.empty:
                    blocks.append(pd.DataFrame({
                        'date': y_fitted_in.index,
                        'model': model_id,
                        'test_period': label,
                        'value_type': 'In-Sample',
                        'value': y_fitted_in.values
                    }))
                
                # Out-of-Sample predicted
                X_out = getattr(wf_model, 'X_out', pd.DataFrame())
                if isinstance(X_out, pd.DataFrame) and not X_out.empty:
                    y_pred_out = getattr(wf_model, 'y_pred_out', pd.Series(dtype=float))
                    if isinstance(y_pred_out, pd.Series) and not y_pred_out.empty:
                        blocks.append(pd.DataFrame({
                            'date': y_pred_out.index,
                            'model': model_id,
                            'test_period': label,
                            'value_type': 'Out-of-Sample',
                            'value': y_pred_out.values
                        }))
            except Exception:
                continue
        
        if not blocks:
            return None
        
        df = pd.concat(blocks, ignore_index=True)
        return df[STABILITY_COLUMNS] 

    def get_stability_stats_results(self) -> Optional[pd.DataFrame]:
        """Return walk-forward stability testing statistical metrics.
        
        Returns a DataFrame with columns:
        - model: model id
        - trial: WF trial identifier (e.g., 'WF1', 'WF2')
        - category: metric category ('P-value', 'Coefficient', 'Coefficient %Change', 'adj R-squared', 'RMSE')
        - value_type: specific metric identifier (variable names for coefficients/p-values, 'In-Sample'/'Out-of-Sample' for performance)
        - value: numerical value
        """
        # Try to build Walk Forward Test via model API
        try:
            wft = self.model.stability_test
        except Exception:
            return None
        
        wf_models = getattr(wft, 'wf_models', None)
        final_model = getattr(wft, 'final_model', None)
        if not isinstance(wf_models, dict) or len(wf_models) == 0 or final_model is None:
            return None
        
        model_id = self._model_id
        stats_list = []

        # Get final model parameters for percentage change calculation
        final_params = final_model.params
        
        for i, (wf_name, wf_model) in enumerate(wf_models.items(), start=1):
            try:
                trial_name = wf_name  # Use 'WF1', 'WF2', etc.
                
                # 1. Coefficients
                if hasattr(wf_model, 'params') and wf_model.params is not None:
                    for var_name, coef_value in wf_model.params.items():
                        if pd.notnull(coef_value):
                            stats_list.append({
                                'model': model_id,
                                'trial': trial_name,
                                'category': 'Coefficient',
                                'value_type': var_name,
                                'value': float(coef_value)
                            })
                
                # 2. P-values
                if hasattr(wf_model, 'pvalues') and wf_model.pvalues is not None:
                    for var_name, p_value in wf_model.pvalues.items():
                        if pd.notnull(p_value):
                            stats_list.append({
                                'model': model_id,
                                'trial': trial_name,
                                'category': 'P-value',
                                'value_type': var_name,
                                'value': float(p_value)
                            })
                
                # 3. Coefficient % Change (vs final model)
                if hasattr(wf_model, 'params') and wf_model.params is not None:
                    for var_name, wf_coef in wf_model.params.items():
                        if var_name in final_params and pd.notnull(wf_coef) and pd.notnull(final_params[var_name]):
                            final_coef = final_params[var_name]
                            if final_coef != 0:
                                pct_change = (wf_coef - final_coef) / final_coef
                                stats_list.append({
                                    'model': model_id,
                                    'trial': trial_name,
                                    'category': 'Coefficient %Change',
                                    'value_type': var_name,
                                    'value': float(pct_change)
                                })
                
                # 4. Adj R-squared (In-Sample)
                if hasattr(wf_model, 'rsquared_adj') and pd.notnull(wf_model.rsquared_adj):
                    stats_list.append({
                        'model': model_id,
                        'trial': trial_name,
                        'category': 'adj R-squared',
                        'value_type': 'In-Sample',
                        'value': float(wf_model.rsquared_adj)
                    })
                
                # 5. RMSE - In-Sample and Out-of-Sample
                # In-Sample RMSE
                if hasattr(wf_model, 'in_perf_measures'):
                    in_measures = wf_model.in_perf_measures
                    if isinstance(in_measures, pd.Series) and 'RMSE' in in_measures:
                        stats_list.append({
                            'model': model_id,
                            'trial': trial_name,
                            'category': 'RMSE',
                            'value_type': 'In-Sample',
                            'value': float(in_measures['RMSE'])
                        })
                
                # Out-of-Sample RMSE
                if hasattr(wf_model, 'out_perf_measures'):
                    out_measures = wf_model.out_perf_measures
                    if isinstance(out_measures, pd.Series) and 'RMSE' in out_measures:
                        stats_list.append({
                            'model': model_id,
                            'trial': trial_name,
                            'category': 'RMSE',
                            'value_type': 'Out-of-Sample',
                            'value': float(out_measures['RMSE'])
                        })
                
            except Exception:
                continue
        
        if not stats_list:
            return None

        return pd.DataFrame(stats_list)

    def get_scenario_stats_results(self) -> Optional[pd.DataFrame]:
        """Return scenario testing statistical metrics for base variables.
        
        Returns a DataFrame with columns:
        - model: model id
        - scenario_name: scenario set name (e.g., 'EWST_2024')
        - metric: metric type (P0, P1, P2, ..., P12, 4Q_CAGR, 9Q_CAGR, 12Q_CAGR, 9Q_Change, 9Q_%Change, %Change_from_Base(at_P9))
        - period: time period identifier (e.g., '2024-Q1', 'P0_to_P4', 'P0_to_P9', 'P0_to_P12')
        - severity: severity level for the metric (e.g., 'base', 'adv', 'sev')
        - value: numerical value
        
        Calculates quarterly statistics for base variables only:
        1. 12-quarter forecast values (P0 to P12)
        2. 4, 9, 12 Quarter CAGR using P0 as starting point
        3. 9 Quarter Change = value at P9 - value at P0
        4. 9 Quarter %Change = 9 Quarter Change / value at P0
        5. %Change from Base(at P9) = stress scenario P9 / baseline scenario P9
        """
        if not hasattr(self.model, 'scen_manager') or self.model.scen_manager is None:
            return None
        
        # Only process base variable quarterly forecasts
        if not hasattr(self.model.scen_manager, 'forecast_y_base_qtr_df'):
            return None
            
        base_qtr_forecasts = self.model.scen_manager.forecast_y_base_qtr_df
        if not base_qtr_forecasts:
            return None
        
        model_id = self._model_id
        stats_list = []
        
        for scen_set, qtr_df in base_qtr_forecasts.items():
            if qtr_df is None or qtr_df.empty:
                continue
                
            # Get scenarios for this scenario set
            if hasattr(self.model.scen_manager, 'y_base_scens'):
                base_scenarios = self.model.scen_manager.y_base_scens.get(scen_set, {})
                
                # Collect data for each severity level (need at least 13 quarters: P0 to P12)
                severity_data = {}
                
                for scen_name in base_scenarios.keys():
                    # Check if this scenario has quarterly data
                    col_name = scen_name if scen_name in qtr_df.columns else f"{scen_set}_{scen_name}"
                    if col_name in qtr_df.columns:
                        qtr_forecast = qtr_df[col_name].dropna()
                        if not qtr_forecast.empty and len(qtr_forecast) >= 13:  # Need P0 to P12 (13 quarters)
                            severity_data[scen_name] = qtr_forecast
                
                if not severity_data:
                    continue
                
                # SECTION 1: 12-Quarter Forecast Values (P0 to P12)
                # Order: P0, P1, P2, ..., P9, P10, P11, P12 (not P1, P10, P11, P12, P2, ...)
                for period_idx in range(13):  # P0 to P12
                    period_name = f"P{period_idx}"
                    
                    for severity, data in severity_data.items():
                        if len(data) > period_idx:
                            # Period shows the actual quarter date
                            quarter_date = data.index[period_idx]
                            period_label = f"{quarter_date.year}-Q{quarter_date.quarter}"
                            
                            stats_list.append({
                                'model': model_id,
                                'scenario_name': scen_set,
                                'metric': period_name,
                                'period': period_label,
                                'severity': severity,
                                'value': float(data.iloc[period_idx])
                            })
                
                # SECTION 2: CAGR Calculations (4Q, 9Q, 12Q) using P0 as starting point
                for severity, data in severity_data.items():
                    data_values = data.values
                    p0_value = data_values[0]  # P0 as starting point
                    
                    # Calculate CAGR for different periods using P0 as base
                    for quarters in [4, 9, 12]:
                        if len(data_values) > quarters and p0_value > 0:  # P0 to P{quarters}
                            end_val = data_values[quarters]  # P{quarters} value
                            
                            if end_val > 0:
                                # CAGR = (P{quarters}/P0)^(1/(quarters/4)) - 1 (annualized)
                                cagr = (end_val / p0_value) ** (4.0 / quarters) - 1
                                
                                stats_list.append({
                                    'model': model_id,
                                    'scenario_name': scen_set,
                                    'metric': f'{quarters}Q_CAGR',
                                    'period': f'P0_to_P{quarters}',
                                    'severity': severity,
                                    'value': float(cagr)
                                })
                
                # SECTION 3: 9Q Change and %Change calculations
                for severity, data in severity_data.items():
                    data_values = data.values
                    
                    if len(data_values) > 9:  # Need P0 to P9
                        p0_value = data_values[0]  # P0
                        p9_value = data_values[9]  # P9
                        
                        # 9 Quarter Change = P9 - P0
                        q9_change = p9_value - p0_value
                        stats_list.append({
                            'model': model_id,
                            'scenario_name': scen_set,
                            'metric': '9Q_Change',
                            'period': 'P0_to_P9',
                            'severity': severity,
                            'value': float(q9_change)
                        })
                        
                        # 9 Quarter %Change = (P9 - P0) / P0
                        if p0_value != 0:
                            q9_pct_change = q9_change / p0_value
                            stats_list.append({
                                'model': model_id,
                                'scenario_name': scen_set,
                                'metric': '9Q_%Change',
                                'period': 'P0_to_P9',
                                'severity': severity,
                                'value': float(q9_pct_change)
                            })
                
                # SECTION 4: %Change from Base (at P9) - stress scenarios vs baseline
                # Find base scenario with flexible pattern matching
                base_data = None
                base_severity_name = None
                
                for severity_name in severity_data.keys():
                    severity_lower = severity_name.lower()
                    if 'base' in severity_lower:
                        base_data = severity_data[severity_name]
                        base_severity_name = severity_name
                        break
                
                # Calculate %Change from Base for stress scenarios
                if base_data is not None and len(base_data) > 9:
                    base_p9_value = base_data.iloc[9]  # Baseline P9 value
                    
                    if base_p9_value != 0:
                        for stress_severity, stress_data in severity_data.items():
                            # Only compare non-base scenarios
                            stress_lower = stress_severity.lower()
                            if (stress_severity != base_severity_name and 
                                'base' not in stress_lower and
                                len(stress_data) > 9):
                                
                                stress_p9_value = stress_data.iloc[9]  # Stress scenario P9 value
                                
                                # %Change from Base = stress_P9 / baseline_P9
                                pct_change_from_base = stress_p9_value / base_p9_value
                                
                                stats_list.append({
                                    'model': model_id,
                                    'scenario_name': scen_set,
                                    'metric': '%Change_from_Base(at_P9)',
                                    'period': 'P9',
                                    'severity': stress_severity,
                                    'value': float(pct_change_from_base)
                                })
        
        if not stats_list:
            return None
        
        # Create DataFrame and sort to ensure proper ordering
        df = pd.DataFrame(stats_list)
        
        # Define custom ordering for metrics
        metric_order = []
        
        # 1. First: P0 to P12 forecast values (in numerical order)
        for i in range(13):
            metric_order.append(f'P{i}')
        
        # 2. Then: CAGR metrics
        metric_order.extend(['4Q_CAGR', '9Q_CAGR', '12Q_CAGR'])
        
        # 3. Finally: Change metrics
        metric_order.extend(['9Q_Change', '9Q_%Change', '%Change_from_Base(at_P9)'])
        
        # Create categorical ordering for proper sorting
        df['metric'] = pd.Categorical(df['metric'], categories=metric_order, ordered=True)
        
        # Sort by scenario_name, metric (in custom order), then severity
        df_sorted = df.sort_values(['scenario_name', 'metric', 'severity']).reset_index(drop=True)
        
        # Convert metric back to string for output
        df_sorted['metric'] = df_sorted['metric'].astype(str)
        
        return df_sorted 