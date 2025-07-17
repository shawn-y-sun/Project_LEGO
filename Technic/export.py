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
    'test_results': 'Comprehensive test results from all tests'
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
            self.strategy.export_timeseries_data(model, output_dir)
            self.strategy.export_statistics(model, output_dir)
            self.strategy.export_scenarios(model, output_dir)
            self.strategy.export_test_results(model, output_dir)
        
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
                'columns': ['date', 'model', 'scenario', 'severity', 'value'],
                'filename': 'scenario_testing.csv'
            },
            'test_results': {
                'chunks': self._test_results_chunks,
                'columns': ['model', 'test_name', 'test_category', 'metric', 'value'],
                'filename': 'test_results.csv'
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
        should_write = True
        
        if file_exists:
            response = input(f"\nFile {filepath} already exists. Do you want to overwrite it? (y/n): ").lower()
            should_write = response == 'y'
        
        if should_write:
            try:
                # Remove existing file for clean overwrite
                if file_exists:
                    filepath.unlink()
                
                # Write data
                df.to_csv(filepath, index=False)
                
                # Print success message
                action = "Updated" if file_exists else "Added"
                print(f"\n✓ Successfully {action} {content_type} to {filepath}")
                
                # Mark this file as written
                self._written_files.add(filepath)
                
            except Exception as e:
                print(f"\n✗ Failed to write {content_type} to {filepath}: {str(e)}")
        
        # Clear the chunks
        chunks.clear()
        self._current_chunk_size = 0
    
    def export_test_results(self, model: ExportableModel, output_dir: Path) -> None:
        """Export comprehensive test results with chunking support."""
        if not self.should_export('test_results'):
            print("check0")
            return
        
        print("check1")
        df = model.get_test_results()
        print("check2")
        print(df)
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
        """Export scenario results with chunking support and enhanced formatting.
        
        The output will include:
        - Historical data (Actual values)
        - Scenario forecasts with severity levels
        
        Output format:
        - date: End of period date (Monthly: YYYY-MM-DD, Quarterly: YYYYQN)
        - model: Model identifier
        - scenario: Scenario name or 'Actual (Historical)'
        - severity: Severity level or 'Actual (Historical)'
        - value: The actual/forecasted value
        """
        if not self.should_export('scenario_testing'):
            return
        
        # Get scenario results
        df = model.get_scenario_results()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
            
        # Get historical data
        historical_data = model.get_timeseries_data()
        if isinstance(historical_data, pd.DataFrame) and not historical_data.empty:
            # Filter for actual values only
            historical = historical_data[
                (historical_data['value_type'] == 'Actual')
            ].copy()
            
            # Format dates
            dates = pd.to_datetime(historical['date'])
            freq = pd.infer_freq(dates)
            
            if freq and 'Q' in freq:
                # Quarterly data: format as YYYYQN
                formatted_dates = dates.map(lambda x: f"{x.year}Q{x.quarter}")
            else:
                # Monthly or other: keep as period end date
                formatted_dates = dates.map(lambda x: x.strftime('%Y-%m-%d'))
            
            # Prepare historical data in scenario format
            historical = pd.DataFrame({
                'date': formatted_dates,
                'model': historical['model'],
                'scenario': 'Actual (Historical)',
                'severity': 'Actual (Historical)',
                'value': historical['value']
            })
            
            # Add to scenario chunks
            self._scenario_chunks.append(historical)
        
        # Process scenario data
        if not df.empty:
            # Extract severity from scenario name (assuming format: scenarioset_severity)
            df[['scenario', 'severity']] = df['scenario_name'].str.split('_', n=1, expand=True)
            
            # Format dates
            dates = pd.to_datetime(df['date'])
            freq = pd.infer_freq(dates)
            
            if freq and 'Q' in freq:
                # Quarterly data: format as YYYYQN
                formatted_dates = dates.map(lambda x: f"{x.year}Q{x.quarter}")
            else:
                # Monthly or other: keep as period end date
                formatted_dates = dates.map(lambda x: x.strftime('%Y-%m-%d'))
            
            # Update date column with formatted dates
            df['date'] = formatted_dates
            
            # Select and reorder columns
            df = df[['date', 'model', 'scenario', 'severity', 'value']]
            
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
        
        
        # Reset containers
        self._initialize_containers()

class OLSModelAdapter(ExportableModel):
    """Adapter for OLS models with optimized performance."""
    
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
            'type': 'Model Overall'  # Changed from 'Overall' to 'Model Overall' for clarity
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
        
        # Add covariance type
        cov_type = getattr(self.model, 'cov_type', 'non-robust')
        # Map internal cov_type names to more descriptive ones
        cov_type_mapping = {
            'nonrobust': 'non-robust',
            'HC1': 'HC1',
            'HAC': 'HAC'
        }
        cov_type = cov_type_mapping.get(cov_type, cov_type)
        overall_stats['Covariance Type'] = cov_type
        
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
        """Return scenario testing results with optimized performance.
        
        Returns a DataFrame with columns:
        - model: Model identifier
        - date: Timestamp of the forecast
        - scenario_name: Name of the scenario
        - value: The forecasted value
        
        The method accesses scenario results through the model's scen_manager.y_scens property,
        which contains nested forecast results for all scenarios.
        """
        if self.model.scen_manager is None:
            return None
        
        model_id = self._model_id
        data = {
            'model': [],
            'date': [],
            'scenario_name': [],
            'value': []
        }
        
        # Access scenario results through y_scens property
        scen_results = self.model.scen_manager.y_scens
        
        # Process all scenario sets and their scenarios
        for scen_set, scenarios in scen_results.items():
            for scen_name, forecast in scenarios.items():
                if forecast is not None and not forecast.empty:
                    n_points = len(forecast)
                    data['model'].extend([model_id] * n_points)
                    data['date'].extend(forecast.index)
                    data['scenario_name'].extend([f"{scen_set}_{scen_name}"] * n_points)
                    data['value'].extend(forecast.values)
        
        if not data['model']:
            return None
            
        return pd.DataFrame(data)

    def get_test_results(self) -> Optional[pd.DataFrame]:
        """Return test results focusing on key model diagnostics.
        
        Returns a DataFrame with columns:
        - model: Model identifier
        - test_name: Name of the test
        - test_category: Category of the test
        - metric: Name of the specific metric or result
        - value: The actual value
        """
        if not hasattr(self.model, 'testset') or self.model.testset is None:
            print(f"No test results available for model {self._model_id}")
            return None
        
        model_id = self._model_id
        results_list = []
        
        # Get all test results (both active and inactive) from TestSet
        test_results = self.model.testset.all_test_results
        
        # Process each test result
        for test_name, result in test_results.items():
            # Handle DataFrame results
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    for idx in result.index:
                        results_list.append({
                            'model': model_id,
                            'test_name': test_name,
                            'test_category': 'Model Validation',
                            'metric': f"{col}_{idx}",
                            'value': result.loc[idx, col]
                        })
            
            # Handle Series results
            elif isinstance(result, pd.Series):
                for idx, value in result.items():
                    results_list.append({
                        'model': model_id,
                        'test_name': test_name,
                        'test_category': 'Model Validation',
                        'metric': str(idx),
                        'value': value
                    })
            
            # Handle dictionary results
            elif isinstance(result, dict):
                for metric, value in result.items():
                    if pd.notnull(value):  # Only add non-null values
                        results_list.append({
                            'model': model_id,
                            'test_name': test_name,
                            'test_category': 'Model Validation',
                            'metric': str(metric),
                            'value': value
                        })
            
            # Handle scalar results
            elif pd.notnull(result):
                results_list.append({
                    'model': model_id,
                    'test_name': test_name,
                    'test_category': 'Model Validation',
                    'metric': 'value',
                    'value': result
                })
        
        if not results_list:
            print(f"No valid test results found for model {self._model_id}")
            return None
            
        print("\nTest Results List:")
        for result in results_list:
            print(f"- {result['test_category']} | {result['test_name']} | {result['metric']}: {result['value']}")
            
        return pd.DataFrame(results_list) 