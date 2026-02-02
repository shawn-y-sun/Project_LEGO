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
from typing import Dict, List, Any, Optional, Set, Type, Iterable
import pandas as pd
from pathlib import Path
import numpy as np
import logging
import warnings
from pandas.tseries.frequencies import to_offset

# Module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Shared column schemas
TIMESERIES_COLUMNS = ['date', 'model', 'series_type', 'value_type', 'value']
STATICSTATS_COLUMNS = ['category', 'model', 'type', 'value_type', 'value']
SCENARIO_COLUMNS = ['category', 'model', 'scenario_name', 'severity', 'date', 'frequency', 'value_type', 'value']
TEST_RESULTS_COLUMNS = ['model', 'test', 'index', 'metric', 'value']
SENSITIVITY_COLUMNS = ['model', 'test', 'scenario_name', 'severity', 'variable/parameter', 'shock', 'date', 'frequency', 'value_type', 'value']
STABILITY_COLUMNS = ['date', 'model', 'test_period', 'value_type', 'value']
STABILITY_STATS_COLUMNS = ['model', 'trial', 'category', 'value_type', 'value']
SCENARIO_STATS_COLUMNS = ['model', 'scenario_name', 'metric', 'severity', 'value']
BACKTESTING_COLUMNS = ['date', 'model', 'route', 'value']

# Series type constants
SERIES_TYPE_TARGET = 'Target'
SERIES_TYPE_BASE = 'Base'
SERIES_TYPE_RESIDUAL = 'Residual'
SERIES_TYPE_IV = 'Independent Variable'

# Category constants for scenario testing
CATEGORY_TARGET_FORECAST = 'Target Variable Forecast'
CATEGORY_BASE_FORECAST = 'Base Variable Forecast'
CATEGORY_DRIVER_DATA = 'Driver Scenario Data'

# Value type constants for scenario testing
VALUE_TYPE_TARGET_FORECAST = 'Target Variable Forecast'
VALUE_TYPE_BASE_FORECAST = 'Base Variable Forecast'

# Define available export content types as constants
EXPORT_CONTENT_TYPES = {
    'timeseries_data': 'Combined modeling dataset and fit results',
    'staticStats': 'Model statistics and metrics',
    'scenario_testing': 'Scenario testing results',
    'test_results': 'Comprehensive test results from all tests',
    'sensitivity_testing': 'Sensitivity testing results for parameters and inputs',
    'stability_testing': 'Walk-forward stability testing results',
    'stability_testing_stats': 'Walk-forward stability testing statistical metrics',
    'scenario_testing_stats': 'Scenario testing statistical metrics for base variables',
    'backtesting_results': 'Rolling in-sample backtesting results'
}

# Utility functions for scenario testing
def is_seasonal_dummy(var_name: str) -> bool:
    """
    Check if a variable name represents a seasonal dummy.
    
    Seasonal dummies follow patterns like:
    - M:2, M:3, ..., M:12 (monthly dummies)
    - Q:2, Q:3, Q:4 (quarterly dummies)
    - System columns like 'M', 'Q'
    
    Parameters
    ----------
    var_name : str
        Variable name to check
        
    Returns
    -------
    bool
        True if the variable is a seasonal dummy, False otherwise
    """
    if not isinstance(var_name, str):
        return False
    
    # Check for monthly dummies (M:2 through M:12)
    if var_name.startswith('M:'):
        try:
            month_num = int(var_name.split(':')[1])
            return 2 <= month_num <= 12
        except (ValueError, IndexError):
            return False
    
    # Check for quarterly dummies (Q:2 through Q:4)
    if var_name.startswith('Q:'):
        try:
            quarter_num = int(var_name.split(':')[1])
            return 2 <= quarter_num <= 4
        except (ValueError, IndexError):
            return False
    
    # Check for system columns
    if var_name in ['M', 'Q']:
        return True
    
    return False


def filter_driver_columns(df: pd.DataFrame, target_var: str = None, base_var: str = None) -> List[str]:
    """
    Filter DataFrame columns to get driver variables, excluding seasonal dummies and target/base variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing driver data
    target_var : str, optional
        Target variable name to exclude
    base_var : str, optional
        Base variable name to exclude
        
    Returns
    -------
    List[str]
        List of driver column names after filtering
    """
    if df.empty:
        return []
    
    # Get all columns
    all_columns = df.columns.tolist()
    
    # Filter out seasonal dummies
    driver_columns = [col for col in all_columns if not is_seasonal_dummy(col)]
    
    # Filter out target and base variables if specified
    if target_var and target_var in driver_columns:
        driver_columns.remove(target_var)
    
    if base_var and base_var in driver_columns:
        driver_columns.remove(base_var)
    
    return driver_columns


def infer_model_frequency(freq_value: Any) -> str:
    """Return simplified monthly/quarterly code for assorted frequency labels."""

    if freq_value is None:
        return 'M'

    if isinstance(freq_value, str):
        freq_str = freq_value.strip().lower()
        if not freq_str:
            return 'M'

        if freq_str.startswith('m') or freq_str in {'monthly', 'month', 'months'}:
            return 'M'
        if freq_str.startswith('q') or freq_str in {'quarterly', 'quarter', 'quarters'}:
            return 'Q'

    try:
        offset = to_offset(freq_value)
        if offset is not None:
            name = offset.name.upper()
            if name.startswith('M'):
                return 'M'
            if name.startswith('Q'):
                return 'Q'
    except Exception:
        pass

    return 'M'

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
    def get_backtesting_results(self) -> Optional[pd.DataFrame]:
        """Return rolling in-sample backtesting results in long format."""
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
    def export_backtesting_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export rolling in-sample backtesting results to CSV if available."""
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

    def __init__(
        self,
        strategy: ExportStrategy,
        format_handler: ExportFormatHandler,
    ):
        """Create a new export manager.

        Args:
            strategy: Export strategy that orchestrates how data is exported.
            format_handler: Concrete handler that knows how to persist exported data.
        """
        self.strategy = strategy
        self.format_handler = format_handler

    def _log_warning(self, message: str) -> None:
        """Emit muted warning information via debug logging."""
        logger.debug(message)

    @staticmethod
    def _log_error(message: str) -> None:
        """Emit an error level log message."""
        logger.error(message)
    
    def export_models(self, models: List[ExportableModel], output_dir: Path) -> None:
        """Export multiple models to consolidated CSV files.

        Args:
            models: List of ExportableModel instances to export
            output_dir: Directory to save the consolidated CSV files

        Notes:
            All library warnings are suppressed during the export process to
            avoid noisy output when optional data is missing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Process each model
            for model in models:
                try:
                    if self.strategy.should_export('timeseries_data'):
                        try:
                            self.strategy.export_timeseries_data(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export timeseries_data for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('staticStats'):
                        try:
                            self.strategy.export_statistics(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export staticStats for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('scenario_testing'):
                        try:
                            self.strategy.export_scenarios(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export scenario_testing for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('test_results'):
                        try:
                            self.strategy.export_test_results(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export test_results for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('sensitivity_testing'):
                        try:
                            self.strategy.export_sensitivity_results(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export sensitivity_testing for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('stability_testing'):
                        try:
                            self.strategy.export_stability_results(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export stability_testing for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('stability_testing_stats'):
                        try:
                            self.strategy.export_stability_stats(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export stability_testing_stats for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('scenario_testing_stats'):
                        try:
                            self.strategy.export_scenario_stats(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export scenario_testing_stats for {model.get_model_id()}: {e}"
                            )

                    if self.strategy.should_export('backtesting_results'):
                        try:
                            self.strategy.export_backtesting_results(model, output_dir)
                        except Exception as e:
                            self._log_warning(
                                f"Failed to export backtesting_results for {model.get_model_id()}: {e}"
                            )

                except Exception as e:
                    self._log_error(
                        f"Failed to process model {model.get_model_id()}: {e}"
                    )
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
        self._backtesting_chunks = [] # New container for backtesting results
        # Per-content chunk counters
        self._chunk_sizes = {
            'timeseries_data': 0,
            'staticStats': 0,
            'scenario_testing': 0,
            'test_results': 0,
            'sensitivity_testing': 0,
            'stability_testing': 0,
            'stability_testing_stats': 0,
            'scenario_testing_stats': 0,
            'backtesting_results': 0
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
            },
            'backtesting_results': {
                'chunks': self._backtesting_chunks,
                'columns': BACKTESTING_COLUMNS,
                'filename': 'backtesting_results.csv'
            }
        }
        
        config = content_config.get(content_type)
        if not config:
            return
            
        chunks = config['chunks']
        columns = config['columns']
        filepath = output_dir / config['filename']
        
        # Combine chunks and ensure column order - create empty DataFrame if no chunks
        if chunks:
            df = pd.concat(chunks, copy=False)
            df = df[columns]
        else:
            # Always create empty file with proper column headers
            df = pd.DataFrame(columns=columns)
        
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
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
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
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
            self._test_results_chunks.append(df)
            self._chunk_sizes['test_results'] += len(df)
            if self._chunk_sizes['test_results'] >= self.chunk_size:
                self._write_chunk(output_dir, 'test_results')
    
    def export_timeseries_data(self, model: ExportableModel, output_dir: Path) -> None:
        """Export time series data with chunking support."""
        if not self.should_export('timeseries_data'):
            return
        df = model.get_timeseries_data()
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
            self._timeseries_chunks.append(df)
            self._chunk_sizes['timeseries_data'] += len(df)
            if self._chunk_sizes['timeseries_data'] >= self.chunk_size:
                self._write_chunk(output_dir, 'timeseries_data')
    
    def export_statistics(self, model: ExportableModel, output_dir: Path) -> None:
        """Export model statistics with chunking support."""
        if not self.should_export('staticStats'):
            return
        
        df = model.get_model_statistics()
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
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
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
            self._scenario_chunks.append(df)
            self._chunk_sizes['scenario_testing'] += len(df)
            if self._chunk_sizes['scenario_testing'] >= self.chunk_size:
                self._write_chunk(output_dir, 'scenario_testing')
    
    def export_stability_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export walk-forward stability results with chunking support."""
        if not self.should_export('stability_testing'):
            return
        
        df = model.get_stability_results()
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
            self._stability_chunks.append(df)
            self._chunk_sizes['stability_testing'] += len(df)
            if self._chunk_sizes['stability_testing'] >= self.chunk_size:
                self._write_chunk(output_dir, 'stability_testing')
    
    def export_stability_stats(self, model: ExportableModel, output_dir: Path) -> None:
        """Export walk-forward stability testing statistical metrics with chunking support."""
        if not self.should_export('stability_testing_stats'):
            return
        
        df = model.get_stability_stats_results()
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
            self._stability_stats_chunks.append(df)
            self._chunk_sizes['stability_testing_stats'] += len(df)
            if self._chunk_sizes['stability_testing_stats'] >= self.chunk_size:
                self._write_chunk(output_dir, 'stability_testing_stats')

    def export_backtesting_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export rolling in-sample backtesting results with chunking support."""
        if not self.should_export('backtesting_results'):
            return

        df = model.get_backtesting_results()
        if isinstance(df, pd.DataFrame):
            self._backtesting_chunks.append(df)
            self._chunk_sizes['backtesting_results'] += len(df)
            if self._chunk_sizes['backtesting_results'] >= self.chunk_size:
                self._write_chunk(output_dir, 'backtesting_results')
    
    def export_scenario_stats(self, model: ExportableModel, output_dir: Path) -> None:
        """Export scenario testing statistical metrics with chunking support."""
        if not self.should_export('scenario_testing_stats'):
            return
        
        df = model.get_scenario_stats_results()
        # Always add data to chunks (even if empty) to ensure file creation
        if isinstance(df, pd.DataFrame):
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

        if self.should_export('backtesting_results'):
            self._write_chunk(output_dir, 'backtesting_results', force_write=True)
        
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

        # In-sample error (Model - Actual)
        target_actual_in = getattr(self.model, 'y_in', pd.Series(dtype=float))
        if isinstance(target_actual_in, pd.Series) and not target_actual_in.empty:
            aligned_actual_in = target_actual_in.reindex(predicted_in.index)
            error_in = (predicted_in - aligned_actual_in).dropna()
            if not error_in.empty:
                blocks.append(self._build_ts_block(
                    error_in.index,
                    model_id,
                    SERIES_TYPE_TARGET,
                    'Error(Model-Actual)',
                    error_in.values
                ))

        if not self.model.X_out.empty:
            predicted_out = self.model.y_pred_out
            blocks.append(self._build_ts_block(predicted_out.index, model_id, SERIES_TYPE_TARGET, 'Out-of-Sample', predicted_out.values))

            # Out-of-sample error (Model - Actual)
            target_actual_out = getattr(self.model, 'y_out', pd.Series(dtype=float))
            if isinstance(target_actual_out, pd.Series) and not target_actual_out.empty:
                aligned_actual_out = target_actual_out.reindex(predicted_out.index)
                error_out = (predicted_out - aligned_actual_out).dropna()
                if not error_out.empty:
                    blocks.append(self._build_ts_block(
                        error_out.index,
                        model_id,
                        SERIES_TYPE_TARGET,
                        'Error(Model-Actual)',
                        error_out.values
                    ))
 
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
        resid_series = getattr(self.model, 'resid', pd.Series(dtype=float))
        if isinstance(resid_series, pd.Series) and not resid_series.empty:
            blocks.append(self._build_ts_block(
                resid_series.index,
                model_id,
                SERIES_TYPE_RESIDUAL,
                'Residual',
                -resid_series.values
            ))

        # Out-of-sample residuals when actuals are available
        y_actual_out = getattr(self.model, 'y_out', pd.Series(dtype=float))
        y_pred_out = getattr(self.model, 'y_pred_out', pd.Series(dtype=float))
        if isinstance(y_pred_out, pd.Series) and not y_pred_out.empty and isinstance(y_actual_out, pd.Series):
            aligned_actual_out = y_actual_out.reindex(y_pred_out.index)
            residual_out = (y_pred_out - aligned_actual_out).dropna()
            if not residual_out.empty:
                blocks.append(self._build_ts_block(
                    residual_out.index,
                    model_id,
                    SERIES_TYPE_RESIDUAL,
                    'Residual',
                    residual_out.values
                ))
 
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
        - category: 'Goodness of Fit', 'Model Estimation', or 'Summary Statistics'
        - model: Model identifier
        - type: Metric type or estimation type
        - value_type: Specific metric or variable name
        - value: The actual value
        
        Summary Statistics includes descriptive statistics for:
        - Target variable (reported separately for in-sample and full-sample data)
        - Base variable (if available, split by in-sample and full-sample)
        - All independent variables (split by in-sample and full-sample data)
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

        # Add Partial R2 values if available
        if hasattr(self.model, 'partial_r2') and self.model.partial_r2 is not None:
            partial_r2_values = self.model.partial_r2
            if isinstance(partial_r2_values, pd.Series):
                for var_name, r2_value in partial_r2_values.items():
                    if pd.notnull(r2_value):
                        stats_list.append({
                            'category': 'Model Estimation',
                            'model': model_id,
                            'type': 'Partial R2',
                            'value_type': var_name,
                            'value': float(r2_value)
                        })
            elif isinstance(partial_r2_values, dict):
                for var_name, r2_value in partial_r2_values.items():
                    if pd.notnull(r2_value):
                        stats_list.append({
                            'category': 'Model Estimation',
                            'model': model_id,
                            'type': 'Partial R2',
                            'value_type': var_name,
                            'value': float(r2_value)
                        })
        
        # Add Summary Statistics for all variables
        def _append_summary_stats(data_series: Optional[pd.Series], var_name: str, category: str) -> None:
            """Calculate and append summary statistics for a given series and category."""
            if data_series is None:
                return

            series = data_series.dropna()
            if series.empty:
                return

            summary_stats = {
                'Mean': series.mean(),
                'Std': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Median': series.median(),
                '25th Percentile': series.quantile(0.25),
                '90th Percentile': series.quantile(0.90),
                '95th Percentile': series.quantile(0.95)
            }

            for stat_name, stat_value in summary_stats.items():
                if pd.notnull(stat_value):
                    stats_list.append({
                        'category': category,
                        'model': model_id,
                        'type': stat_name,
                        'value_type': var_name,
                        'value': float(stat_value)
                    })

        def _add_in_full_stats(series_in: Optional[pd.Series], series_full: Optional[pd.Series], var_name: str) -> None:
            """Add both in-sample and full-sample stats for a variable when available."""
            _append_summary_stats(series_in, var_name, 'Summary Statistics(In-Sample)')
            _append_summary_stats(series_full, var_name, 'Summary Statistics(Full-Sample)')

        # Target variable summary statistics
        y_in = getattr(self.model, 'y_in', pd.Series(dtype=float))
        y_full = getattr(self.model, 'y_full', pd.Series(dtype=float))
        _add_in_full_stats(y_in, y_full, 'Target')

        # Base variable summary statistics (if available)
        base_in = getattr(self.model, 'y_base_in', None)
        base_full = getattr(self.model, 'y_base_full', None)
        if base_in is not None or base_full is not None:
            _add_in_full_stats(base_in, base_full, 'Base')

        # Independent variables summary statistics
        feature_data_in = getattr(self.model, 'X_in', pd.DataFrame())
        feature_data_full = getattr(self.model, 'X_full', pd.DataFrame())

        feature_columns: Iterable[str] = []
        if not feature_data_in.empty:
            feature_columns = feature_data_in.columns
        elif not feature_data_full.empty:
            feature_columns = feature_data_full.columns

        for var_name in feature_columns:
            series_in = feature_data_in[var_name] if var_name in feature_data_in.columns else None
            series_full = feature_data_full[var_name] if var_name in feature_data_full.columns else None
            _add_in_full_stats(series_in, series_full, var_name)
        
        # Create DataFrame efficiently
        return pd.DataFrame(stats_list)
    
    def get_scenario_results(self) -> pd.DataFrame:
        """Return scenario testing results in long format with driver data.
        
        Returns a DataFrame with columns:
        - category: string ['Target Variable Forecast', 'Base Variable Forecast', 'Driver Scenario Data']
        - model: string, model_id
        - scenario_name: string (e.g., 'EWST_2024')
        - severity: string (e.g., 'base', 'adv', 'sev', 'p0')
        - date: timestamp
        - frequency: string ['monthly'/'quarterly']
        - value_type: string (specific series identifier)
        - value: numerical
        
        The method processes:
        - Target variable forecasts: Monthly frequency only
        - Base variable forecasts: Both monthly and quarterly frequencies
        - Driver scenario data: Actual transformed model drivers (default P0 to P60 for
          monthly models or P0 to P20 for quarterly models), excludes seasonal dummies
          and intercept terms
        """
        if self.model.scen_manager is None:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=SCENARIO_COLUMNS)
        
        model_id = self._model_id
        data_list = []
        
        # Get target variable forecasts (monthly)
        scen_results = self.model.scen_manager.y_scens
        
        if not scen_results:
            return pd.DataFrame(columns=SCENARIO_COLUMNS)
        
        # Get model frequency to determine driver data frequency
        freq_value = getattr(self.model, 'dm', None)
        if freq_value is not None:
            freq_value = getattr(freq_value, 'freq', None)
        model_freq = infer_model_frequency(freq_value)
        is_monthly = model_freq == 'M'

        qtr_method = getattr(self.model.scen_manager, 'qtr_method', 'mean')

        def aggregate_to_quarterly(series: pd.Series) -> pd.Series:
            series_copy = series.copy()
            series_copy.index = pd.to_datetime(series_copy.index)
            quarterly_grouped = series_copy.groupby(pd.Grouper(freq='Q'))
            if qtr_method == 'mean':
                result = quarterly_grouped.mean()
            elif qtr_method == 'sum':
                try:
                    result = quarterly_grouped.sum(min_count=1)
                except TypeError:
                    result = quarterly_grouped.apply(lambda s: s.sum() if s.notna().any() else np.nan)
            elif qtr_method == 'end':
                result = quarterly_grouped.last()
            else:
                result = quarterly_grouped.mean()
            result.index = result.index.to_period('Q').to_timestamp(how='end').normalize()
            return result
        
        # Get target and base variable names for filtering
        target_var = getattr(self.model, 'target', None)
        base_var = getattr(self.model, 'target_base', None)
        
        # === TARGET VARIABLE FORECASTS ===
        
        # Add scen_p0 data if available
        if hasattr(self.model.scen_manager, 'scen_p0') and self.model.scen_manager.scen_p0 is not None:
            scen_p0_data = self.model.scen_manager.scen_p0
            scen_p0_freq = 'monthly' if is_monthly else 'quarterly'
            for scen_set in scen_results.keys():
                df_data = {
                    'category': CATEGORY_TARGET_FORECAST,
                    'model': model_id,
                    'scenario_name': scen_set,
                    'severity': 'p0',
                    'date': scen_p0_data.index,
                    'frequency': scen_p0_freq,
                    'value_type': VALUE_TYPE_TARGET_FORECAST,
                    'value': scen_p0_data.values
                }
                data_list.append(pd.DataFrame(df_data))
                if is_monthly:
                    scen_p0_q = aggregate_to_quarterly(scen_p0_data)
                    df_data_q = {
                        'category': CATEGORY_TARGET_FORECAST,
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'p0',
                        'date': scen_p0_q.index,
                        'frequency': 'quarterly',
                        'value_type': VALUE_TYPE_TARGET_FORECAST,
                        'value': scen_p0_q.values
                    }
                    data_list.append(pd.DataFrame(df_data_q))

        # Process target variable forecasts
        for scen_set, scenarios in scen_results.items():
            for scen_name, forecast in scenarios.items():
                if forecast is not None and not forecast.empty:
                    freq_label = 'monthly' if is_monthly else 'quarterly'
                    df_data = {
                        'category': CATEGORY_TARGET_FORECAST,
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': scen_name,
                        'date': forecast.index,
                        'frequency': freq_label,
                        'value_type': VALUE_TYPE_TARGET_FORECAST,
                        'value': forecast.values
                    }
                    data_list.append(pd.DataFrame(df_data))
                    if is_monthly:
                        qtr_forecast = aggregate_to_quarterly(forecast)
                        df_data_q = {
                            'category': CATEGORY_TARGET_FORECAST,
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'date': qtr_forecast.index,
                            'frequency': 'quarterly',
                            'value_type': VALUE_TYPE_TARGET_FORECAST,
                            'value': qtr_forecast.values
                        }
                        data_list.append(pd.DataFrame(df_data_q))

        # Add historical actuals (Target)
        target_actual = getattr(self.model, 'y_full', None)
        if target_actual is not None and not target_actual.empty:
            if is_monthly:
                for scen_set in scen_results.keys():
                    df_data = {
                        'category': CATEGORY_TARGET_FORECAST,
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'actual',
                        'date': target_actual.index,
                        'frequency': 'monthly',
                        'value_type': VALUE_TYPE_TARGET_FORECAST,
                        'value': target_actual.values
                    }
                    data_list.append(pd.DataFrame(df_data))

            actual_q = aggregate_to_quarterly(target_actual)
            if not actual_q.empty:
                for scen_set in scen_results.keys():
                    df_data = {
                        'category': CATEGORY_TARGET_FORECAST,
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'actual',
                        'date': actual_q.index,
                        'frequency': 'quarterly',
                        'value_type': VALUE_TYPE_TARGET_FORECAST,
                        'value': actual_q.values
                    }
                    data_list.append(pd.DataFrame(df_data))

        # === BASE VARIABLE FORECASTS ===

        # Process base variable forecasts in original frequency if monthly
        if is_monthly and hasattr(self.model.scen_manager, 'y_base_scens'):
            base_results = self.model.scen_manager.y_base_scens

            # Add scen_p0 base data if available
            if hasattr(self.model, 'base_predictor') and self.model.base_predictor is not None:
                if hasattr(self.model.scen_manager, 'scen_p0') and self.model.scen_manager.scen_p0 is not None:
                    scen_p0_data = self.model.scen_manager.scen_p0
                    base_p0_values = self.model.base_predictor.predict_base(scen_p0_data, scen_p0_data)

                    for scen_set in base_results.keys():
                        df_data = {
                            'category': CATEGORY_BASE_FORECAST,
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': 'p0',
                            'date': base_p0_values.index,
                            'frequency': 'monthly',
                            'value_type': VALUE_TYPE_BASE_FORECAST,
                            'value': base_p0_values.values
                        }
                        data_list.append(pd.DataFrame(df_data))

                        # quarterly aggregate of p0
                        base_p0_q = aggregate_to_quarterly(base_p0_values)
                        df_data_q = {
                            'category': CATEGORY_BASE_FORECAST,
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': 'p0',
                            'date': base_p0_q.index,
                            'frequency': 'quarterly',
                            'value_type': VALUE_TYPE_BASE_FORECAST,
                            'value': base_p0_q.values
                        }
                        data_list.append(pd.DataFrame(df_data_q))

            for scen_set, scenarios in base_results.items():
                for scen_name, forecast in scenarios.items():
                    if forecast is not None and not forecast.empty:
                        df_data = {
                            'category': CATEGORY_BASE_FORECAST,
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'date': forecast.index,
                            'frequency': 'monthly',
                            'value_type': VALUE_TYPE_BASE_FORECAST,
                            'value': forecast.values
                        }
                        data_list.append(pd.DataFrame(df_data))
                        qtr_forecast = aggregate_to_quarterly(forecast)
                        df_data_q = {
                            'category': CATEGORY_BASE_FORECAST,
                            'model': model_id,
                            'scenario_name': scen_set,
                            'severity': scen_name,
                            'date': qtr_forecast.index,
                            'frequency': 'quarterly',
                            'value_type': VALUE_TYPE_BASE_FORECAST,
                            'value': qtr_forecast.values
                        }
                        data_list.append(pd.DataFrame(df_data_q))

        # Add historical actuals (Base)
        base_actual = getattr(self.model, 'y_base_full', None)
        if base_actual is not None and not base_actual.empty:
            if is_monthly:
                for scen_set in scen_results.keys():
                    df_data = {
                        'category': CATEGORY_BASE_FORECAST,
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'actual',
                        'date': base_actual.index,
                        'frequency': 'monthly',
                        'value_type': VALUE_TYPE_BASE_FORECAST,
                        'value': base_actual.values
                    }
                    data_list.append(pd.DataFrame(df_data))

            base_actual_q = aggregate_to_quarterly(base_actual)
            if not base_actual_q.empty:
                for scen_set in scen_results.keys():
                    df_data = {
                        'category': CATEGORY_BASE_FORECAST,
                        'model': model_id,
                        'scenario_name': scen_set,
                        'severity': 'actual',
                        'date': base_actual_q.index,
                        'frequency': 'quarterly',
                        'value_type': VALUE_TYPE_BASE_FORECAST,
                        'value': base_actual_q.values
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
                                        'category': CATEGORY_BASE_FORECAST,
                                        'model': model_id,
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'date': qtr_forecast.index,
                                        'frequency': 'quarterly',
                                        'value_type': VALUE_TYPE_BASE_FORECAST,
                                        'value': qtr_forecast.values
                                    }
                                    data_list.append(pd.DataFrame(df_data))
        
        # === DRIVER SCENARIO DATA ===
        
        # Get actual model drivers (transformed features used in the model)
        model_drivers = get_model_driver_names(self.model)
        
        if model_drivers and hasattr(self.model, 'scen_manager') and self.model.scen_manager is not None:
            for scen_set in scen_results.keys():
                scenarios = scen_results[scen_set]

                for scen_name in scenarios.keys():
                    driver_data = get_scenario_driver_data(
                        self.model, scen_set, scen_name, model_drivers,
                        jump_off_date=None
                    )

                    if driver_data is not None and not driver_data.empty:
                        if is_monthly:
                            for driver_name in driver_data.columns:
                                driver_series = driver_data[driver_name].dropna()
                                if not driver_series.empty:
                                    df_data = {
                                        'category': CATEGORY_DRIVER_DATA,
                                        'model': model_id,
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'date': driver_series.index,
                                        'frequency': 'monthly',
                                        'value_type': driver_name,
                                        'value': driver_series.values
                                    }
                                    data_list.append(pd.DataFrame(df_data))

                            driver_q = driver_data.copy()
                            driver_q.index = pd.to_datetime(driver_q.index)
                            driver_q = driver_q.groupby(pd.Grouper(freq='Q')).mean()
                            driver_q.index = driver_q.index.to_period('Q').to_timestamp(how='end').normalize()
                            for driver_name in driver_q.columns:
                                driver_series = driver_q[driver_name].dropna()
                                if not driver_series.empty:
                                    df_data_q = {
                                        'category': CATEGORY_DRIVER_DATA,
                                        'model': model_id,
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'date': driver_series.index,
                                        'frequency': 'quarterly',
                                        'value_type': driver_name,
                                        'value': driver_series.values
                                    }
                                    data_list.append(pd.DataFrame(df_data_q))
                        else:
                            for driver_name in driver_data.columns:
                                driver_series = driver_data[driver_name].dropna()
                                if not driver_series.empty:
                                    df_data = {
                                        'category': CATEGORY_DRIVER_DATA,
                                        'model': model_id,
                                        'scenario_name': scen_set,
                                        'severity': scen_name,
                                        'date': driver_series.index,
                                        'frequency': 'quarterly',
                                        'value_type': driver_name,
                                        'value': driver_series.values
                                    }
                                    data_list.append(pd.DataFrame(df_data))
        
        if not data_list:
            return pd.DataFrame(columns=SCENARIO_COLUMNS)
            
        # Combine all data and ensure column order
        result = pd.concat(data_list, ignore_index=True)
        return result[SCENARIO_COLUMNS]

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
        """Return sensitivity testing results in long format."""
        if not hasattr(self.model, "scen_manager") or self.model.scen_manager is None:
            return None

        sens_test = self.model.scen_manager.sens_test
        if sens_test is None:
            return None

        results_df = getattr(sens_test, "results_df", None)
        if results_df is None or results_df.empty:
            param_names = getattr(sens_test, "param_names", [])
            if not param_names:
                message = (
                    f"Model {self._model_id} has no eligible variables for sensitivity testing "
                    "(likely dummy-only)."
                )
                logger.info(message)
            return None

        df = results_df.copy()
        df.insert(0, "model", self._model_id)
        return df[["model", "test", "scenario_name", "severity", "variable/parameter",
                   "shock", "date", "frequency", "value_type", "value"]]


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
        """Return walk-forward stability testing results in long format."""
        try:
            wft = self.model.stability_test
        except Exception:
            return None

        results_df = getattr(wft, 'results_df', None)
        if results_df is None or results_df.empty:
            return None

        df = results_df.copy()
        df.insert(0, 'model', self._model_id)
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
        try:
            wft = self.model.stability_test
        except Exception:
            return None

        stats_df = getattr(wft, 'stats_df', None)
        if stats_df is None or stats_df.empty:
            return None

        df = stats_df.copy()
        df.insert(0, 'model', self._model_id)
        return df[STABILITY_STATS_COLUMNS]

    def get_backtesting_results(self) -> Optional[pd.DataFrame]:
        """Return rolling in-sample backtesting results in long format."""
        try:
            backtest = self.model.backtesting_test
        except Exception:
            return None

        results_df = getattr(backtest, 'results_df', None)
        if results_df is None or results_df.empty:
            return None

        df = results_df.copy()
        df.insert(1, 'model', self._model_id)
        return df[BACKTESTING_COLUMNS]

    def get_scenario_stats_results(self) -> Optional[pd.DataFrame]:
        """Return scenario testing statistical metrics for base variables."""
        if not hasattr(self.model, 'scen_manager') or self.model.scen_manager is None:
            return None

        stats_df = getattr(self.model.scen_manager, 'scenario_stats_df', None)
        if stats_df is None or stats_df.empty:
            return None

        df = stats_df.copy()
        df.insert(0, 'model', self._model_id)
        return df[['model', 'scenario_name', 'metric', 'severity', 'value']]

# =============================================================================
# Helper functions for driver scenario data export
# =============================================================================

def get_model_driver_names(model) -> List[str]:
    """
    Get the actual driver feature names used in a fitted model.
    
    This function extracts the column names from the model's feature matrix (X_in),
    excluding seasonal dummies and the intercept term. These represent the actual
    transformed features that are used as drivers in the model.
    
    Parameters
    ----------
    model : ModelBase
        Fitted model instance with X_in attribute
        
    Returns
    -------
    List[str]
        List of actual driver feature names used in the model
    """
    if not hasattr(model, 'X_in') or model.X_in is None or model.X_in.empty:
        return []
    
    # Get all feature column names from the model
    all_features = model.X_in.columns.tolist()
    
    # Filter out intercept and seasonal dummies
    driver_features = []
    for feature_name in all_features:
        # Skip intercept terms
        if feature_name.lower() in ['const', 'intercept', 'c']:
            continue
        # Skip seasonal dummies using existing function
        if is_seasonal_dummy(feature_name):
            continue
        driver_features.append(feature_name)
    
    return driver_features


def get_scenario_driver_data(model, scen_set: str, scen_name: str, driver_names: List[str],
                           jump_off_date=None, periods: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Get transformed driver scenario data for specific scenario and drivers.
    
    This function gets the actual transformed driver data as used by the model,
    filtered to the specified date range (P0 to P{periods}).
    
    Parameters
    ----------
    model : ModelBase
        Fitted model instance with scenario manager
    scen_set : str
        Scenario set name
    scen_name : str
        Scenario name within the set
    driver_names : List[str]
        List of driver feature names to extract
    jump_off_date : pd.Timestamp, optional
        Jump-off date (P0). If None, uses model's scenario manager P0
    periods : int, optional
        Number of periods after P0 to include (P1 to P{periods}). If not provided,
        defaults to 60 periods for monthly models and 20 periods for quarterly models
        (approximately five years of data).
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with driver data columns filtered to date range, or None if not available
    """
    if not hasattr(model, 'scen_manager') or model.scen_manager is None:
        return None
    
    # Get scenario features from scenario manager
    try:
        X_scens = model.scen_manager.X_scens
        if scen_set not in X_scens or scen_name not in X_scens[scen_set]:
            return None
        
        scenario_features = X_scens[scen_set][scen_name]
        
        # Filter to only the requested driver names that exist in the scenario features
        available_drivers = [name for name in driver_names if name in scenario_features.columns]
        if not available_drivers:
            return None
        
        driver_data = scenario_features[available_drivers].copy()
        try:
            driver_data.index = pd.to_datetime(driver_data.index)
        except Exception:
            pass
        driver_data = driver_data.sort_index()
        
        # Apply date filtering (P0 to P{periods})
        if jump_off_date is None:
            jump_off_date = getattr(model.scen_manager, 'P0', None)
        
        if jump_off_date is not None:
            # Convert to pandas timestamp if needed
            jump_off_date = pd.to_datetime(jump_off_date)

            # Determine frequency from model
            model_freq = infer_model_frequency(getattr(getattr(model, 'dm', None), 'freq', None))

            # Resolve default horizon when not explicitly provided
            if periods is None:
                periods = 60 if model_freq == 'M' else 20

            if periods <= 0:
                return driver_data if not driver_data.empty else None

            # Calculate end date to include exactly `periods` data points starting from P0
            horizon = max(periods - 1, 0)
            if model_freq == 'M':
                end_date = jump_off_date + pd.DateOffset(months=horizon)
            else:  # Quarterly
                end_date = jump_off_date + pd.DateOffset(months=horizon * 3)

            # Filter data to P0 to P{periods} range
            mask = (driver_data.index >= jump_off_date) & (driver_data.index <= end_date)
            driver_data = driver_data.loc[mask]

            if periods is not None and len(driver_data) > periods:
                driver_data = driver_data.iloc[:periods]

        return driver_data if not driver_data.empty else None

    except Exception:
        return None
