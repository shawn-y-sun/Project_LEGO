# TECHNIC/segment.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Type, Dict, List, Optional, Any, Union, Callable, Tuple, Set
from pathlib import Path

from .cm import CM
from .model import ModelBase, OLS
from .template import ExportTemplateBase
from .report import ReportSet
from .search import ModelSearch
from .scenario import ScenManager
from .sensitivity import SensitivityTest
from .stability import WalkForwardTest
from .export import (
    EXPORT_CONTENT_TYPES,
    ExportStrategy,
    ExportFormatHandler,
    OLSExportStrategy,
    CSVFormatHandler,
    OLSModelAdapter,
    ExportManager
)


class Segment:
    """
    Manages a collection of Candidate Models (CM) and their reporting/export.
    
    A Segment represents a logical grouping of related candidate models for a specific
    target variable. It provides functionality for building, managing, analyzing, and 
    exporting these models.

    Parameters
    ----------
    segment_id : str
        Unique identifier for this Segment.
    target : str
        Name of the target variable to be modeled.
    target_base : str, optional
        Name of the base variable of interest (highly recommended if available).
    target_exposure : str, optional
        Name of the exposure variable (required for Ratio model types).
    data_manager : Any
        DataManager instance containing the data to be used.
    model_cls : Type[ModelBase]
        ModelBase subclass to use for model fitting.
    export_template_cls : Optional[Type[ExportTemplateBase]], optional
        Excel export template class for exporting results.
    reportset_cls : Type[ReportSet], default ReportSet
        Class for assembling and displaying model reports.
    search_cls : Type[ModelSearch], default ModelSearch
        Class to use for exhaustive model search.
    scen_cls : Type, optional
        Class to use for scenario management. If None, defaults to ScenManager.

    Attributes
    ----------
    cms : Dict[str, CM]
        Dictionary of candidate models, keyed by their IDs.
    top_cms : List[CM]
        List of top performing models from the last search.
    searcher : Optional[ModelSearch]
        Instance of ModelSearch if a search has been performed.

    Example
    -------
    >>> # Create a segment for GDP forecasting
    >>> segment = Segment(
    ...     segment_id="gdp_models",
    ...     target="gdp_growth",
    ...     data_manager=dm,
    ...     model_cls=LinearModel
    ... )
    >>> 
    >>> # Build a candidate model
    >>> segment.build_cm(
    ...     cm_id="gdp_model_1",
    ...     specs={"variables": ["inflation", "unemployment"]}
    ... )
    >>> 
    >>> # Show reports for all models
    >>> segment.show_report(show_params=True)
    """
    def __init__(
        self,
        segment_id: str,
        target: str,
        model_type: Optional[Any] = None,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None,
        data_manager: Any = None,
        model_cls: Type[ModelBase] = None,
        export_template_cls: Optional[Type[ExportTemplateBase]] = None,
        reportset_cls: Type[ReportSet] = ReportSet,
        search_cls: Type[ModelSearch] = ModelSearch,
        scen_cls: Optional[Type[ScenManager]] = None,
        qtr_method: str = 'mean'
    ):
        self.segment_id = segment_id
        self.target = target
        self.model_type = model_type
        self.target_base = target_base
        self.target_exposure = target_exposure
        self.dm = data_manager
        self.model_cls = model_cls
        self.export_template_cls = export_template_cls
        self.reportset_cls = reportset_cls
        self.search_cls = search_cls
        self.qtr_method = qtr_method
        # Import and set default ScenManager if not provided
        if scen_cls is None:
            self.scen_cls = ScenManager
        else:
            self.scen_cls = scen_cls
        # Will hold the ModelSearch instance once we've run a search
        self.searcher: Optional[ModelSearch] = None
        self.cms: Dict[str, CM] = {}               # existing CMs in this segment
        self.top_cms: List[CM] = []                # placeholder for top models

    def build_cm(
        self,
        cm_id: str,
        specs: Any,
        sample: str = 'both'
    ) -> CM:
        """
        Instantiate and fit a Candidate Model (CM) with given specifications.

        This method creates a new CM instance, fits it to the data according to the
        provided specifications, and stores it in the segment's collection.

        Parameters
        ----------
        cm_id : str
            Unique identifier for this candidate model. Must be unique within
            this segment.
        specs : Any
            Feature specification passed to DataManager. The exact format depends
            on your DataManager implementation, but typically includes:
            - List of variable names
            - Transformation specifications
            - Lag specifications
        sample : str, default 'both'
            Which sample to build the model on:
            - 'in': in-sample only
            - 'full': full sample
            - 'both': both in-sample and full sample (default)

        Returns
        -------
        CM
            The constructed and fitted CM instance.

        Example
        -------
        >>> # Build a simple model with two variables
        >>> cm = segment.build_cm(
        ...     cm_id="model_1",
        ...     specs=["gdp_lag1", "inflation"],
        ...     sample="both"
        ... )
        >>> 
        >>> # Build a model with transformations
        >>> cm = segment.build_cm(
        ...     cm_id="model_2",
        ...     specs={
        ...         "variables": ["gdp", "cpi"],
        ...         "transforms": ["diff", "pct_change"]
        ...     }
        ... )
        """
        cm = CM(
            model_id=cm_id,
            target=self.target,
            model_type=self.model_type,
            target_base=self.target_base,
            target_exposure=self.target_exposure,
            data_manager=self.dm,
            model_cls=self.model_cls,
            scen_cls=self.scen_cls,
            qtr_method=self.qtr_method,
        )
        cm.build(specs, sample=sample)
        self.cms[cm_id] = cm
        return cm
    
    def remove_cm(self, cm_ids: Union[str, List[str]]) -> None:
        """
        Remove one or more candidate models from this segment.

        Parameters
        ----------
        cm_ids : Union[str, List[str]]
            A single model ID or list of model IDs to remove from the segment.
            Non-existent IDs are silently ignored.

        Example
        -------
        >>> # Remove a single model
        >>> segment.remove_cm("model_1")
        >>> 
        >>> # Remove multiple models
        >>> segment.remove_cm(["model_2", "model_3"])
        """
        # allow passing a single string
        if isinstance(cm_ids, str):
            cm_ids = [cm_ids]
        for cm_id in cm_ids:
            if cm_id in self.cms:
                del self.cms[cm_id]

    def show_report(
        self,
        cm_ids: Optional[List[str]] = None,
        report_sample: str = 'in',
        show_out: bool = True,
        show_params: bool = False,
        show_tests: bool = False,
        show_scens: bool = False,
        show_sens: bool = False,
        show_stab: bool = False,
        perf_kwargs: Optional[Dict[str, Any]] = None,
        params_kwargs: Optional[Dict[str, Any]] = None,
        test_kwargs: Optional[Dict[str, Any]] = None,
        scen_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Display consolidated reports for one or multiple Candidate Models.

        This method provides a comprehensive view of model performance, parameters,
        and diagnostic tests using the ReportSet class.

        Parameters
        ----------
        cm_ids : Optional[List[str]], default None
            List of CM IDs to include in the report. If None, reports on all
            models in the segment.
        report_sample : str, default 'in'
            Which sample to use for reporting:
            - 'in': in-sample results
            - 'full': full sample results
        show_out : bool, default True
            Whether to include out-of-sample results.
        show_params : bool, default False
            Whether to include parameter tables.
        show_tests : bool, default False
            Whether to include diagnostic test results.
        show_scens : bool, default False
            Whether to include scenario forecast and variable plots.
        show_sens : bool, default False
            Whether to include sensitivity testing plots for all scenarios.
        show_stab : bool, default False
            Whether to include stability test results for each model.
        perf_kwargs : Optional[Dict[str, Any]], default None
            Additional kwargs for performance display.
        params_kwargs : Optional[Dict[str, Any]], default None
            Additional kwargs for parameter tables.
        test_kwargs : Optional[Dict[str, Any]], default None
            Additional kwargs for test display.
        scen_kwargs : Optional[Dict[str, Any]], default None
            Additional kwargs for scenario plotting.

        Example
        -------
        >>> # Show basic report for all models
        >>> segment.show_report()
        >>> 
        >>> # Detailed report for specific models
        >>> segment.show_report(
        ...     cm_ids=["model_1", "model_2"],
        ...     show_params=True,
        ...     show_tests=True
        ... )
        >>> 
        >>> # Full sample report with custom performance display
        >>> segment.show_report(
        ...     report_sample="full",
        ...     perf_kwargs={"show_rmse": True, "show_mae": True}
        ... )
        """
        perf_kwargs = perf_kwargs or {}
        params_kwargs = params_kwargs or {}
        test_kwargs = test_kwargs or {}
        scen_kwargs = scen_kwargs or {}
        cm_ids = cm_ids or list(self.cms.keys())

        # Print all selected CM IDs and their representations
        print("=== Candidate Models to Report ===")
        for cm_id in cm_ids:
            cm = self.cms[cm_id]
            print(f"- {cm_id}: {cm}")
        print("\n")

        if report_sample not in {'in', 'full'}:
            raise ValueError("report_sample must be 'in' or 'full'")

        # Build mapping of model_id to report instances based on sample
        reports: Dict[str, Any] = {}
        for cm_id in cm_ids:
            cm = self.cms[cm_id]
            if report_sample == 'in':
                rpt = cm.report_in
            else:
                rpt = cm.report_full
            reports[cm_id] = rpt

        # Instantiate ReportSet and delegate display
        rs = self.reportset_cls(reports)
        rs.show_report(
            show_out=show_out,
            show_params=show_params,
            show_tests=show_tests,
            show_scens=show_scens,
            perf_kwargs=perf_kwargs,
            params_kwargs=params_kwargs,
            test_kwargs=test_kwargs,
            scen_kwargs=scen_kwargs
        )
        
        # Sensitivity testing (handled separately since it's not part of ReportSet)
        if show_sens:
            for cm_id in cm_ids:
                cm = self.cms[cm_id]
                # In-sample sensitivity testing
                if cm.model_in is not None and hasattr(cm.model_in, 'scen_manager') and cm.model_in.scen_manager is not None:
                    print(f"\n=== Model: {cm_id} — In-Sample Sensitivity Analysis ===")
                    try:
                        cm.model_in.scen_manager.sens_test.plot_all()
                    except Exception as e:
                        print(f"Error generating in-sample sensitivity plots for {cm_id}: {e}")
                
                # Full-sample sensitivity testing (if report_sample is 'full')
                if report_sample == 'full' and cm.model_full is not None and hasattr(cm.model_full, 'scen_manager') and cm.model_full.scen_manager is not None:
                    print(f"\n=== Model: {cm_id} — Full-Sample Sensitivity Analysis ===")
                    try:
                        cm.model_full.scen_manager.sens_test.plot_all()
                    except Exception as e:
                        print(f"Error generating full-sample sensitivity plots for {cm_id}: {e}")
        
        # Stability testing (handled separately since it's not part of ReportSet)
        if show_stab:
            for cm_id in cm_ids:
                cm = self.cms[cm_id]
                
                # In-sample stability testing
                if cm.model_in is not None:
                    print(f"\n=== Model: {cm_id} — In-Sample Stability Analysis ===")
                    try:
                        cm.model_in.stability_test.show_all()
                    except Exception as e:
                        print(f"Error generating in-sample stability test results for {cm_id}: {e}")
                else:
                    print(f"\n=== Model: {cm_id} — No In-Sample Model Available for Stability Testing ===")
                    print("In-sample model not built. Call build_cm() first.")
                
                # Full-sample stability testing (if report_sample is 'full')
                if report_sample == 'full':
                    if cm.model_full is not None:
                        print(f"\n=== Model: {cm_id} — Full-Sample Stability Analysis ===")
                        try:
                            cm.model_full.stability_test.show_all()
                        except Exception as e:
                            print(f"Error generating full-sample stability test results for {cm_id}: {e}")
                    else:
                        print(f"\n=== Model: {cm_id} — No Full-Sample Model Available for Stability Testing ===")
                        print("Full-sample model not built. Call build_cm() first.")
    
    def plot_vars(
        self,
        vars_list: List[str],
        plot_type: str = 'line',
        sample: str = 'full',
        date_range: Optional[Tuple[str, str]] = None
    ) -> None:
        """
        Create exploratory plots comparing variables and their transformations to the target.

        This method generates all applicable transformations for each variable and creates
        a separate figure for each variable showing all its transformed versions plotted
        against the target. Each subplot includes correlation coefficient in the title.

        Parameters
        ----------
        vars_list : List[str]
            List of variable names to explore. For each variable, all applicable
            transformations will be generated and plotted.
        plot_type : str, default 'line'
            Type of plot to create:
            - 'line': time series plot with dual y-axes
            - 'scatter': scatter plot of variable vs target
        sample : str, default 'full'
            Which sample to use for plotting and correlation calculation:
            - 'in': use in-sample data only
            - 'full': use full sample data (in-sample + out-sample)
        date_range : Tuple[str, str], optional
            Date range for zooming in, e.g., ('2020-05-31', '2022-02-28').
            If provided, plots and correlations will be calculated only for this period.

        Example
        -------
        >>> # Explore basic variables with line plots
        >>> segment.plot_vars(
        ...     vars_list=["GDP", "UNRATE", "CPI"]
        ... )
        >>> # This creates 3 separate figures:
        >>> # Figure 1: GDP and all its transformations vs target
        >>> # Figure 2: UNRATE and all its transformations vs target  
        >>> # Figure 3: CPI and all its transformations vs target
        >>> 
        >>> # Create scatter plots for specific period
        >>> segment.plot_vars(
        ...     vars_list=["GDP", "UNRATE"],
        ...     plot_type="scatter",
        ...     date_range=("2020-01-01", "2022-12-31")
        ... )
        """
        # Generate transformations for each variable (no lags, no periods)
        var_dfs = self.dm.build_search_vars(vars_list, max_lag=0, max_periods=1)
        
        # Get target data based on sample
        if sample == 'in':
            target_idx = self.dm.in_sample_idx
        else:  # sample == 'full'
            target_idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        
        target_series = self.dm.internal_data.loc[target_idx, self.target]
        
        # Apply date range filter to target if specified
        if date_range:
            start_date, end_date = date_range
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            mask = (target_series.index >= start_date) & (target_series.index <= end_date)
            target_series = target_series[mask]

        for var_name, df in var_dfs.items():
            # Apply date range filter to variable data if specified
            if date_range:
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df[mask]
            
            # Align df and target to their common index
            common_idx = df.index.intersection(target_series.index)
            df_aligned = df.loc[common_idx]
            ts_aligned = target_series.loc[common_idx]

            cols = df_aligned.columns.tolist()
            n = len(cols)
            
            # Dynamic column adjustment based on number of transformations
            if n == 1:
                ncols = n  # Use 1 column for 1 transformation
                fig_width = 7
            elif n == 2:
                ncols = n  # Use 2 columns for 2 transformations
                fig_width = 15
            else:
                ncols = 3  # Use 3 columns for 3+ transformations
                fig_width = 15

            nrows = math.ceil(n / ncols)
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(fig_width, 4 * nrows), squeeze=False
                # figsize=(5 * ncols, 4 * nrows), squeeze=False
            )
            
            # Create title with date range info
            title_parts = [f"{var_name} vs. {self.target}"]
            if date_range:
                title_parts.append(f"({start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')})")
            fig.suptitle(" ".join(title_parts), fontsize=14)

            for idx, col in enumerate(cols):
                row, col_idx = divmod(idx, ncols)
                ax = axes[row][col_idx]

                # Calculate correlation
                var_series = df_aligned[col]
                target_series_aligned = ts_aligned
                
                # Remove NaN values for correlation calculation
                combined = pd.concat([var_series, target_series_aligned], axis=1).dropna()
                if len(combined) > 1:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                    corr_text = f"Corr: {corr:.2f}"
                else:
                    corr_text = "Corr: N/A"
                
                # Set subplot title with correlation
                ax.set_title(f"{col} - {corr_text}")

                if plot_type == 'line':
                    # primary vs secondary y-axis
                    line1, = ax.plot(
                        ts_aligned.index,
                        ts_aligned,
                        color='tab:blue',
                        label=self.target,
                        linewidth=2
                    )
                    ax2 = ax.twinx()
                    line2, = ax2.plot(
                        df_aligned.index,
                        df_aligned[col],
                        color='tab:orange',
                        label=col,
                        linewidth=2
                    )
                    ax.legend(handles=[line1, line2], loc='best')

                    # remove all axis labels
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax2.set_xlabel('')
                    ax2.set_ylabel('')

                elif plot_type == 'scatter':
                    ax.scatter(
                        df_aligned[col],
                        ts_aligned,
                        color='dodgerblue'
                    )
                    # remove both axis labels
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    # remove any legend
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

                else:
                    raise ValueError("plot_type must be 'line' or 'scatter'")

            # hide unused subplots
            for i in range(n, nrows * ncols):
                r, c = divmod(i, ncols)
                axes[r][c].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def explore_vars(
        self,
        vars_list: List[str],
        max_lag: int = 3,
        max_periods: int = 12,
        sample: str = 'full',
        plot_type: str = 'line',
        date_range: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        Explore variables by creating plots and returning correlation analysis.
        
        This method consolidates the functionality of plot_vars() and get_corr() methods.
        It generates transformation specifications for variables, creates exploratory plots,
        and returns a DataFrame with correlation rankings.

        Parameters
        ----------
        vars_list : List[str]
            List of variable names to analyze and transform.
        max_lag : int, default 3
            Maximum lag to consider in transformation specifications.
        max_periods : int, default 12
            Maximum number of periods to consider in transformations.
        sample : str, default 'full'
            Which sample to use:
            - 'in': use in-sample data only  
            - 'full': use full sample data (in-sample + out-sample)
        plot_type : str, default 'line'
            Type of plot to create ('line' or 'scatter').
        date_range : Tuple[str, str], optional
            Date range for zooming in, e.g., ('2020-05-31', '2022-02-28').
            If provided, plots and correlations will be calculated only for this period.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['variable', 'corr', 'abs_corr'] sorted by absolute
            correlation in descending order. Contains all possible transformations
            for each variable.

        Example
        -------
        >>> # Basic exploration
        >>> corr_df = segment.explore_vars(
        ...     vars_list=['GDP', 'UNRATE']
        ... )
        >>> print(corr_df.head())
        >>> 
        >>> # Explore specific period with scatter plots
        >>> corr_df = segment.explore_vars(
        ...     vars_list=['GDP', 'UNRATE'],
        ...     plot_type='scatter',
        ...     date_range=('2020-01-01', '2022-12-31')
        ... )
        """
        # First create the plots
        self.plot_vars(
            vars_list=vars_list,
            plot_type=plot_type,
            sample=sample,
            date_range=date_range
        )
        
        # Adjust max_periods for quarterly data if needed
        if self.dm.freq == 'Q' and max_periods <= 4:
            max_periods = 4
        
        # Generate all possible transformations for each variable
        var_dfs = self.dm.build_search_vars(vars_list, max_lag=max_lag, max_periods=max_periods)
        
        # Get target data based on sample
        if sample == 'in':
            target_idx = self.dm.in_sample_idx
        else:  # sample == 'full'
            target_idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        
        target_data = self.dm.internal_data.loc[target_idx, self.target]
        
        # Apply date range filter if specified
        if date_range:
            start_date, end_date = date_range
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            mask = (target_data.index >= start_date) & (target_data.index <= end_date)
            target_data = target_data[mask]
        
        # Calculate correlations for all transformations
        corr_results = []
        
        for var_name, var_df in var_dfs.items():
            # Apply same date range filter to variable data
            if date_range:
                mask = (var_df.index >= start_date) & (var_df.index <= end_date)
                var_df = var_df[mask]
            
            # Align with target data
            common_idx = var_df.index.intersection(target_data.index)
            var_aligned = var_df.loc[common_idx]
            target_aligned = target_data.loc[common_idx]
            
            for col in var_aligned.columns:
                # Calculate correlation, handling NaN values
                combined = pd.concat([var_aligned[col], target_aligned], axis=1).dropna()
                if len(combined) > 1:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                    if pd.isna(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                
                corr_results.append({
                    'variable': col,
                    'corr': corr,
                    'abs_corr': abs(corr)
                })
        
        # Create result DataFrame and sort by absolute correlation
        result_df = pd.DataFrame(corr_results)
        result_df = result_df.sort_values('abs_corr', ascending=False).reset_index(drop=True)
        
        return result_df

    def export(
        self,
        model_ids: Optional[List[str]] = None,
        output_dir: Union[str, Path] = Path.cwd(),
        strategy_cls: Type[ExportStrategy] = OLSExportStrategy,
        format_handler_cls: Type[ExportFormatHandler] = CSVFormatHandler,
        content: Optional[List[str]] = None,
        overwrite: bool = True
    ) -> None:
        """
        Export model results using the specified export strategy and format handler.
        
        Parameters
        ----------
        model_ids : List[str], optional
            List of model IDs to export. If None, exports all models in the segment.
        output_dir : Union[str, Path], default Path.cwd()
            Directory to save exports. By default, uses current working directory.
        strategy_cls : Type[ExportStrategy], default OLSExportStrategy
            Export strategy class to use.
        format_handler_cls : Type[ExportFormatHandler], default CSVFormatHandler
            Format handler class to use.
        content : List[str], optional
            List of content types to export. If None, exports all content types.
            Valid types are:
            - 'timeseries_data': Combined modeling dataset and fit results
            - 'staticStats': Model statistics and metrics
            - 'scenario_testing': Scenario testing results with target and base variables
            - 'sensitivity_testing': Sensitivity testing results for parameters and inputs
            - 'test_results': Comprehensive test results from all tests
        overwrite : bool, default False
            Whether to overwrite existing files. If False and files exist, the operation
            will be cancelled with a warning message.
        
        Example
        -------
        >>> # Export all content for all models to current directory
        >>> segment.export()
        >>> 
        >>> # Export only timeseries data and statistics for specific models to custom directory
        >>> segment.export(
        ...     model_ids=['model1'],
        ...     output_dir='my_exports',
        ...     content=['timeseries_data', 'staticStats']
        ... )
        >>> 
        >>> # Export scenario and sensitivity testing results
        >>> segment.export(
        ...     content=['scenario_testing', 'sensitivity_testing'],
        ...     output_dir='scenario_analysis'
        ... )
        >>> 
        >>> # Export with overwrite enabled to replace existing files
        >>> segment.export(
        ...     output_dir='my_exports',
        ...     overwrite=True
        ... )
        """
        # Convert output_dir to Path object
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing files and handle overwrite logic
        content_types_set = set(content) if content is not None else set(EXPORT_CONTENT_TYPES.keys())
        expected_files = []
        for content_type in content_types_set:
            if content_type == 'timeseries_data':
                expected_files.append(output_dir / 'timeseries_data.csv')
            elif content_type == 'staticStats':
                expected_files.append(output_dir / 'staticStats.csv')
            elif content_type == 'scenario_testing':
                expected_files.append(output_dir / 'scenario_testing.csv')
            elif content_type == 'sensitivity_testing':
                expected_files.append(output_dir / 'sensitivity_testing.csv')
            elif content_type == 'test_results':
                expected_files.append(output_dir / 'test_results.csv')
        
        # Check if any expected files exist
        existing_files = [f for f in expected_files if f.exists()]
        
        if existing_files and not overwrite:
            print(f"\n❌ Export cancelled: The following files already exist in {output_dir}:")
            for file in existing_files:
                print(f"   - {file.name}")
            print(f"\nTo overwrite existing files, use: segment.export(overwrite=True)")
            print("Or choose a different output directory.")
            return
        elif existing_files and overwrite:
            print(f"\n⚠️  Overwrite mode enabled: The following existing files will be replaced:")
            for file in existing_files:
                print(f"   - {file.name}")
            print(f"Files will be overwritten in: {output_dir}")
            
            # Actually delete existing files to ensure clean overwrite
            print("Removing existing files...")
            for file in existing_files:
                try:
                    file.unlink()
                    print(f"   ✓ Removed: {file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {file.name}: {e}")
                    return
        else:
            print(f"\n✓ No existing files detected. Proceeding with export to: {output_dir}")
        
        # Get models to export
        if model_ids is None:
            models_to_export = [(cm_id, cm) for cm_id, cm in self.cms.items()]
        else:
            models_to_export = [
                (model_id, self.cms[model_id])
                for model_id in model_ids 
                if model_id in self.cms
            ]
        
        # Print export start message
        print(f"\nStarting export for segment '{self.segment_id}':")
        print(f"- Target variable: {self.target}")
        if self.target_base:
            print(f"- Base variable: {self.target_base}")
        print(f"- Number of models: {len(models_to_export)}")
        print(f"- Output directory: {output_dir}")
        
        # Validate content types (content_types_set already defined above for overwrite check)
        if content is not None:
            invalid_types = content_types_set - set(EXPORT_CONTENT_TYPES.keys())
            if invalid_types:
                raise ValueError(f"Invalid content types: {invalid_types}. Valid types are: {list(EXPORT_CONTENT_TYPES.keys())}")
            print(f"- Content types to export: {', '.join(content_types_set)}")
        else:
            print("- Content types to export: all")
            # For strategy creation, use all content types when content is None
            content_types_set = None
        print("\nPreparing export...")
        
        # Create format handler and strategy
        format_handler = format_handler_cls()
        strategy = strategy_cls(
            format_handler=format_handler,
            content_types=content_types_set
        )
        
        # Create export manager
        export_manager = ExportManager(
            strategy=strategy,
            format_handler=format_handler
        )
        
        # Create exportable models
        exportable_models = []
        for model_id, cm in models_to_export:
            if isinstance(cm.model_in, OLS):
                adapter = OLSModelAdapter(cm.model_in, model_id + "_in")
                exportable_models.append(adapter)
            if isinstance(cm.model_full, OLS):
                adapter = OLSModelAdapter(cm.model_full, model_id + "_full")
                exportable_models.append(adapter)
        
        # Export models
        export_manager.export_models(exportable_models, output_dir)
        
        # Print final success message
        print(f"\nExport completed successfully for segment '{self.segment_id}'!")
        print(f"Results have been saved to: {output_dir}")
        
        # Print overwrite confirmation if files were overwritten
        if existing_files and overwrite:
            print(f"✓ Successfully overwrote {len(existing_files)} existing file(s).")

    def clear_cms(self) -> None:
        """
        Clear all candidate models from this segment.
        
        This method empties the self.cms dictionary, removing all stored 
        candidate models from the segment. This is useful when you want to 
        start fresh with a new set of models or free up memory.
        
        Note that this operation cannot be undone. Models will need to be 
        rebuilt if needed again.
        
        Example
        -------
        >>> # Build some models
        >>> segment.build_cm("model1", specs1)
        >>> segment.build_cm("model2", specs2)
        >>> print(len(segment.cms))  # 2
        >>> 
        >>> # Clear all models
        >>> segment.clear_cms()
        >>> print(len(segment.cms))  # 0
        >>> 
        >>> # Start fresh with new models
        >>> segment.build_cm("new_model", new_specs)
        """
        self.cms.clear()

    def get_corr(
        self,
        vars_list: List[str],
        max_lag: int = 3,
        max_periods: int = 12,
        sample: str = 'full'
    ) -> pd.DataFrame:
        """
        Rank variables and their transformations by correlation with the target variable.

        This method generates all possible transformations for the specified variables
        and ranks them by their correlation with the target variable. It's useful for
        feature selection and understanding which transformations are most predictive.

        Parameters
        ----------
        vars_list : List[str]
            List of variable names to analyze and transform.
        max_lag : int, default 3
            Maximum lag to consider in transformation specifications.
        max_periods : int, default 12
            Maximum number of periods to consider in transformations.
            For quarterly data, automatically adjusted to 4 if not specified > 4.
        sample : str, default 'full'
            Which sample to use for correlation calculation:
            - 'in': in-sample period
            - 'full': full sample period

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'variable': Variable or transformation name
            - 'corr': Correlation coefficient with target
            - 'abs_corr': Absolute value of correlation
            Sorted by absolute correlation in descending order with reset index.

        Example
        -------
        >>> # Basic correlation ranking
        >>> corr_df = segment.get_corr(['GDP', 'UNRATE', 'CPI'])
        >>> print(corr_df.head())
        >>>
        >>> # With custom parameters for quarterly data
        >>> corr_df = segment.get_corr(
        ...     vars_list=['GDP', 'UNRATE'],
        ...     max_lag=2,
        ...     max_periods=8,  # Will be adjusted to 4 for quarterly data
        ...     sample='in'
        ... )
        >>> 
        >>> # Find top 10 most correlated features
        >>> top_features = corr_df.head(10)['variable'].tolist()
        """
        if sample not in ['in', 'full']:
            raise ValueError("sample must be 'in' or 'full'")

        # Check if internal data is quarterly and adjust max_periods if needed
        try:
            internal_freq = pd.infer_freq(self.dm.internal_data.index)
            if internal_freq and internal_freq.startswith('Q'):
                # For quarterly data, if user didn't specify max_periods > 4, set to 4
                if max_periods <= 4:
                    max_periods = 4
        except (AttributeError, TypeError):
            # If frequency detection fails, use original max_periods
            pass

        # Build all possible transformations for the variables
        var_dfs = self.dm.build_search_vars(
            vars_list,
            max_lag=max_lag,
            max_periods=max_periods
        )

        # Get target variable for the specified sample using index properties
        if sample == 'in':
            target_idx = self.dm.in_sample_idx
        else:  # sample == 'full'
            # Combine in-sample and out-sample indices for full sample
            target_idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        
        target_data = self.dm.internal_data.loc[target_idx, self.target]

        # Calculate correlations for all variables/transformations
        corr_results = []
        
        for var_name, var_df in var_dfs.items():
            for col in var_df.columns:
                # Align the feature and target data
                common_idx = var_df.index.intersection(target_data.index)
                if len(common_idx) == 0:
                    continue
                
                feature_aligned = var_df.loc[common_idx, col]
                target_aligned = target_data.loc[common_idx]
                
                # Calculate correlation, handling NaN values
                with np.errstate(invalid='ignore', divide='ignore'):
                    corr = feature_aligned.corr(target_aligned)
                
                # Skip if correlation is NaN (e.g., constant feature)
                if pd.isna(corr):
                    continue
                
                corr_results.append({
                    'variable': col,
                    'corr': corr,
                    'abs_corr': abs(corr)
                })

        # Create DataFrame and sort by absolute correlation
        result_df = pd.DataFrame(corr_results)
        
        if result_df.empty:
            # Return empty DataFrame with correct columns if no valid correlations
            return pd.DataFrame(columns=['variable', 'corr', 'abs_corr'])
        
        # Sort by absolute correlation in descending order and reset index
        result_df = result_df.sort_values('abs_corr', ascending=False).reset_index(drop=True)
        
        return result_df

    def search_cms(
        self,
        desired_pool: List[Union[str, Any]],
        forced_in: Optional[List[Union[str, Any]]] = None,
        top_n: int = 10,
        sample: str = 'in',
        max_var_num: int = 5,
        max_lag: int = 3,
        max_periods: int = 3,
        category_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None,
        rank_weights: Tuple[float, float, float] = (1, 1, 1),
        test_update_func: Optional[Callable] = None,
        outlier_idx: Optional[List[Any]] = None,
        add_in: bool = True,
        override: bool = False,
        re_rank: bool = True
    ) -> None:
        """
        Run an exhaustive search to find the best performing model specifications.

        This method systematically explores combinations of variables and their
        transformations to identify the most promising model specifications
        based on performance criteria.

        Parameters
        ----------
        desired_pool : List[Union[str, Any]]
            Pool of variables or transformation specifications to consider
            in the search.
        forced_in : Optional[List[Union[str, Any]]], default None
            Variables or specifications that must be included in every model.
            If provided, these are treated as one group.
        top_n : int, default 10
            Number of top performing models to retain.
        sample : str, default 'in'
            Which sample to use for model building:
            - 'in': in-sample only
            - 'full': full sample
        max_var_num : int, default 5
            Maximum number of features allowed in each model.
        max_lag : int, default 3
            Maximum lag to consider in transformation specifications.
        max_periods : int, default 3
            Maximum number of periods to consider in transformations.
            Note: Can be set to 12 for monthly target variables, or 4 for quarterly 
            target variables if user wants to expand the coverage of this exhaustive 
            search for more model options. For monthly data, periods > 3 will 
            automatically include only multiples of 3 (e.g., 6, 9, 12).
        category_limit : int, default 1
            Maximum number of variables from each MEV category per combo.
            Only applies to top-level strings and TSFM instances in desired_pool.
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            Passed to ModelSearch.run_search().
        rank_weights : Tuple[float, float, float], default (1, 1, 1)
            Weights for (Fit Measures, IS Error, OOS Error) when ranking models.
        test_update_func : Optional[Callable], default None
            Optional function to update each CM's test set.
        outlier_idx : Optional[List[Any]], default None
            List of index labels corresponding to outliers to exclude.
        add_in : bool, default True
            If True, add the resulting top CMs to self.cms.
        override : bool, default False
            If True, clear existing cms before adding new ones. Only applies
            when add_in=True.
        re_rank : bool, default True
            If True and add_in=True and override=False, rank new top_cms
            along with pre-existing cms and update model_ids based on ranking.
            If False, simply append new cms with collision-resolved IDs.

        Returns
        -------
        None
            Results are stored in `self.top_cms` and (optionally) `self.cms`.

        Example
        -------
        >>> # Basic search with default parameters
        >>> segment.search_cms(
        ...     desired_pool=["gdp", "inflation", "unemployment"]
        ... )
        >>> top_models = segment.top_cms  # access results
        >>> 
        >>> # Search and override existing models
        >>> segment.search_cms(
        ...     desired_pool=["new_var1", "new_var2"],
        ...     override=True  # clears existing models first
        ... )
        >>> 
        >>> # Search and add without re-ranking
        >>> segment.search_cms(
        ...     desired_pool=["additional_var"],
        ...     re_rank=False  # just append with unique IDs
        ... )
        >>> 
        >>> # Advanced search with re-ranking
        >>> segment.search_cms(
        ...     desired_pool=[
        ...         {"var": "gdp", "transform": ["diff", "pct_change"]},
        ...         {"var": "cpi", "transform": "diff"},
        ...         "unemployment"
        ...     ],
        ...     forced_in=["gdp_lag1"],
        ...     top_n=10,
        ...     max_var_num=3,
        ...     category_limit=2,  # Allow up to 2 variables per category
        ...     rank_weights=(0.5, 1.0, 1.5),  # emphasize OOS performance
        ...     re_rank=True  # re-rank with existing models
        ... )
        """
        # 1) Reuse existing searcher if present, else create & store one
        if self.searcher is None:
            self.searcher = self.search_cls(
                self.dm, 
                self.target, 
                self.model_cls,
                model_type=self.model_type,
                target_base=self.target_base,
                target_exposure=self.target_exposure,
                qtr_method=self.qtr_method
            )
        searcher = self.searcher

        # 2) Run the search (populates searcher.top_cms; no return value)
        searcher.run_search(
            desired_pool=desired_pool,
            forced_in=forced_in or [],
            top_n=top_n,
            sample=sample,
            max_var_num=max_var_num,
            max_lag=max_lag,
            max_periods=max_periods,
            category_limit=category_limit,
            rank_weights=rank_weights,
            test_update_func=test_update_func,
            outlier_idx=outlier_idx,
            exp_sign_map=exp_sign_map
        )

        # 3) Collect the top_n results from the searcher
        self.top_cms = self.searcher.top_cms[:top_n]

        # 4) Optionally add them to this segment's cms
        if add_in:
            if override:
                # Clear existing cms and add new ones with simple IDs
                self.cms.clear()
                for i, cm in enumerate(self.top_cms):
                    cm.model_id = f"cm{i+1}"
                    self.cms[cm.model_id] = cm
            else:
                # Add to existing cms
                if re_rank and self.cms:
                    # Before re-ranking: drop new models that duplicate existing ones by exact formula match
                    existing_formulas = {getattr(cm, 'formula', None) for cm in self.cms.values()}
                    distinct_new = [cm for cm in self.top_cms if getattr(cm, 'formula', None) not in existing_formulas]
                    dup_count = len(self.top_cms) - len(distinct_new)
                    print(f"\nDuplicate check: {dup_count} duplicate model(s) found among new results; {len(distinct_new)} distinct model(s) to consider.")

                    if not distinct_new:
                        print("No distinct new models to add. Skipping re-ranking.")
                        return None

                    # Combine existing with only distinct new for re-ranking
                    all_cms = list(self.cms.values()) + distinct_new
                    
                    # Keep track of newly searched CM object identities for stable tracking
                    newly_searched_obj_ids = {id(cm) for cm in distinct_new}
                    
                    # Temporarily assign unique IDs to handle duplicates during ranking
                    original_ids = {}
                    for i, cm in enumerate(all_cms):
                        original_ids[f"temp_{i}"] = cm
                        cm.model_id = f"temp_{i}"
                    
                    # Re-rank all models together
                    df_ranked = self.searcher.rank_cms(all_cms, sample, rank_weights)

                    # Clear and rebuild cms with new ranking-based IDs
                    self.cms.clear()
                    ordered_temp_ids = df_ranked['model_id'].tolist()

                    # Assign new sequential IDs and track newly searched models' final positions
                    newly_searched_final_positions = []
                    temp_to_new_id: Dict[str, str] = {}

                    # Assign final IDs based on ranking order
                    for i, temp_id in enumerate(ordered_temp_ids):
                        cm = original_ids[temp_id]
                        new_id = f"cm{i+1}"
                        temp_to_new_id[temp_id] = new_id
                        cm.model_id = new_id
                        self.cms[new_id] = cm

                        # Track final position if this was a newly searched CM
                        if id(cm) in newly_searched_obj_ids:
                            newly_searched_final_positions.append(new_id)

                    # Prepare and print updated ranking table with new IDs
                    df_updated = df_ranked.copy()
                    df_updated['model_id'] = df_updated['model_id'].map(temp_to_new_id)
                    print("\n=== Updated Ranked Results ===")
                    print(df_updated.to_string(index=False))

                    # Print positions for newly added CMs
                    if newly_searched_final_positions:
                        order_list = df_updated['model_id'].tolist()
                        pos_map = {mid: (i + 1) for i, mid in enumerate(order_list)}
                        positions_str = ", ".join(f"{mid} (#{pos_map.get(mid, '?')})" for mid in newly_searched_final_positions)
                        print("\nNewly added models ranked at:")
                        print(positions_str)

                    # Print top model formulas for all CMs in current ranking order
                    print("\n=== Top Model Formulas ===")
                    for temp_id in ordered_temp_ids:
                        new_id = temp_to_new_id[temp_id]
                        cm = self.cms[new_id]
                        print(f"{new_id}: {cm.formula}")
                else:
                    # Simply add new cms with collision-resolved IDs (original behavior)
                    for cm in self.top_cms:
                        # Resolve any model_id collisions by appending _2, _3, etc.
                        base_id = cm.model_id
                        new_id = base_id
                        # If there's a collision, find the next available suffix
                        if new_id in self.cms:
                            suffix = 2
                            while f"{base_id}_{suffix}" in self.cms:
                                suffix += 1
                            new_id = f"{base_id}_{suffix}"
                        # Assign the unique ID back to the CM and register it
                        cm.model_id = new_id
                        self.cms[new_id] = cm

        return None
