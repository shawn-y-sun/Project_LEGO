# =============================================================================
# =============================================================================
# module: segment.py
# Purpose: Manage candidate model construction, evaluation, and visualization workflows.
# Key Types/Classes: Segment
# Key Functions: build_cm, plot_vars, explore_vars, export, save_cms, load_cms
# Dependencies: warnings, pandas, numpy, matplotlib.pyplot, math, inspect, copy, typing, pathlib,
#               datetime, shutil, internal modules
# =============================================================================
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import inspect
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import shutil
from copy import deepcopy
from typing import Type, Dict, List, Optional, Any, Union, Callable, Tuple, Set, Sequence
from pathlib import Path

from .cm import CM
from .model import ModelBase, OLS, FixedOLS
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
from .feature import Feature
from .periods import resolve_periods_argument
from .plot import _plot_segmented_series
from .pretest import PreTestSet, FeatureTest
from .persistence import (
    ensure_segment_dirs,
    get_segment_dirs,
    load_cm,
    load_index,
    save_cm,
    save_index,
    generate_search_id,
)
from .transform import TSFM


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
    passed_cms : Dict[str, CM]
        Dictionary of candidate models loaded from persisted passed_cms sets.
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
        self.passed_cms: Dict[str, CM] = {}        # loaded passed CMs
        self.top_cms: List[CM] = []                # placeholder for top models
        self.last_search_id: Optional[str] = None
        self.working_dir: Path = Path.cwd()

    def build_cm(
        self,
        cm_id: str,
        specs: Any,
        sample: str = 'in',
        outlier_idx: Optional[Sequence[Any]] = None
    ) -> CM:
        """
        Build and fit a Candidate Model (CM) for this segment.

        The method creates a CM, fits it with the supplied specifications, and
        keeps the fitted model in the segment registry.

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
        sample : str, default 'in'
            Which sample to build the model on:
            - 'in': in-sample only (default)
            - 'full': full sample
            - 'both': both in-sample and full sample
        outlier_idx : Sequence[Any], optional
            Iterable of row labels to skip when fitting the in-sample model.
            Provide the labels exactly as they appear in the DataFrame index.

        Raises
        ------
        TypeError
            If ``outlier_idx`` is given as a string or any value that cannot be
            iterated over.

        Returns
        -------
        CM
            The constructed and fitted CM instance.

        Example
        -------
        >>> # Build a simple model with two variables
        >>> cm = segment.build_cm(
        ...     cm_id="model_1",
        ...     specs=["gdp_lag1", "inflation"]
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
        >>>
        >>> # Build a model while excluding specific outlier observations
        >>> cm = segment.build_cm(
        ...     cm_id="model_3",
        ...     specs=["gdp", "cpi"],
        ...     outlier_idx=["2020-03-31", "2020-04-30"]
        ... )
        """
        if isinstance(outlier_idx, (str, bytes)):
            raise TypeError(
                "outlier_idx must be a list (or other iterable) of index labels. "
                "Use ['label'] if you need to skip a single observation."
            )

        cleaned_outliers: Optional[List[Any]] = None
        if outlier_idx is not None:
            try:
                cleaned_outliers = list(outlier_idx)
            except TypeError as exc:
                raise TypeError(
                    "outlier_idx must be a list (or other iterable) of index "
                    "labels. Use ['label'] if you need to skip a single "
                    "observation."
                ) from exc

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
        cm.build(specs, sample=sample, outlier_idx=cleaned_outliers)
        self.cms[cm_id] = cm
        return cm

    def _resolve_model_pretestset(self) -> Optional[PreTestSet]:
        """Return a deep copy of the default pre-test bundle for ``model_cls``."""

        if self.model_cls is None:
            return None

        try:
            signature = inspect.signature(self.model_cls.__init__)
        except (TypeError, ValueError):
            return None

        parameter = signature.parameters.get("pretestset")
        if parameter is None or parameter.default is inspect._empty:
            return None

        default_bundle = parameter.default
        if default_bundle is None:
            return None

        return deepcopy(default_bundle)

    def _run_target_pretest(
        self,
        pretestset: Optional[PreTestSet],
        *,
        print_result: bool = True
    ) -> Optional[Any]:
        """Execute the configured target pre-test and optionally print the result.

        Parameters
        ----------
        pretestset : PreTestSet, optional
            The pre-test bundle resolved from the active ``model_cls``. When
            ``None`` or when no target test is present, the method exits early.
        print_result : bool, default True
            Flag indicating whether to print the target pre-test output. Set to
            ``False`` when callers need to suppress console noise (for example
            during exploratory analysis).

        Returns
        -------
        Any, optional
            The raw result returned by the target pre-test implementation. The
            concrete type depends on the configured diagnostics.
        """

        if pretestset is None or pretestset.target_test is None:
            return None

        target_test = pretestset.target_test
        if target_test.dm is None:
            target_test.dm = self.dm
        if target_test.target is None:
            target_test.target = self.target

        try:
            result = target_test.test_filter
        except Exception as exc:
            print(f"Target pre-test raised {type(exc).__name__}: {exc}")
            return None

        description = ""
        if hasattr(result, "attrs"):
            description = result.attrs.get("filter_mode_desc", "") or ""
        if not description and hasattr(result, "filter_mode_desc"):
            description = getattr(result, "filter_mode_desc", "")

        if print_result:
            print("--- Target Pre-Test Result ---")
            if description:
                print(description)
            print(result)
            print("")

        return result

    def _prepare_feature_pretest(
        self,
        pretestset: Optional[PreTestSet],
        target_pretest_result: Optional[Any]
    ) -> Optional[FeatureTest]:
        """Align the feature pre-test with the current data context."""

        if pretestset is None:
            return None

        feature_test = pretestset.feature_test
        if feature_test is None:
            return None

        if feature_test.dm is not self.dm:
            feature_test.dm = self.dm

        if (
            getattr(feature_test, "target_test_result", None) is None
            and target_pretest_result is not None
        ):
            feature_test.target_test_result = target_pretest_result

        return feature_test

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
        vars_list: List[Union[str, Feature]],
        plot_type: str = 'line',
        sample: str = 'full',
        date_range: Optional[Tuple[str, str]] = None,
        outlier_idx: Optional[Sequence[Any]] = None,
        active_idx: Optional[pd.Index] = None
    ) -> None:
        """
        Create exploratory plots comparing variables and their transformations to the target.

        This method generates all applicable transformations for each variable and creates
        a separate figure for each variable showing all its transformed versions plotted
        against the target. Each subplot includes correlation coefficient in the title.

        Parameters
        ----------
        vars_list : List[Union[str, Feature]]
            List of variable names or pre-constructed Feature/TSFM objects to explore.
            For each variable, all applicable transformations will be generated and
            plotted when provided as names. When Feature objects are supplied, the
            method plots those specific transformations without rebuilding the
            broader search grid.
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
        outlier_idx : Sequence[Any], optional
            Iterable of index labels representing observations to exclude from plotting
            and correlation calculations. Labels must match those in the modeling
            DataFrame index. Useful for removing anomalous dates prior to visualization.
        active_idx : pd.Index, optional
            Index labels designating the active subset of observations to retain for
            plotting and correlation calculations. Primarily used for regime-aware
            visualizations to ensure inactive periods are excluded from both axes. When
            provided, the target and variable series are intersected with this index
            after outlier handling and prior to date range filtering.

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
        ...     date_range=("2020-01-01", "2022-12-31"),
        ...     outlier_idx=["2020-03-31"]
        ... )
        """
        # Generate transformations for each variable (no lags, minimal periods)
        var_dfs = self.dm.build_search_vars(vars_list, max_lag=0, periods=[1])

        outlier_labels: Optional[pd.Index]
        if outlier_idx is None:
            outlier_labels = None
        else:
            # NOTE: Ensure fast membership checks and alignment-safe removal.
            outlier_labels = pd.Index(outlier_idx)

        # Get target data based on sample
        if sample == 'in':
            target_idx = self.dm.in_sample_idx
        else:  # sample == 'full'
            target_idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)

        target_series_full = self.dm.internal_data.loc[target_idx, self.target]
        target_series_plot = target_series_full.copy()

        if outlier_labels is not None:
            # NOTE: Retain original index positions in the plotting series so NaNs
            # create visible gaps while still excluding these observations from
            # correlation calculations.
            target_series_plot.loc[target_series_plot.index.isin(outlier_labels)] = np.nan
            target_series = target_series_full.drop(labels=outlier_labels, errors='ignore')
        else:
            target_series = target_series_full

        if active_idx is not None:
            # NOTE: Restrict plotting to the explicitly active index (e.g., regime-on
            # periods) to avoid displaying inactive intervals when regime filters are
            # applied upstream.
            active_idx = pd.Index(active_idx)
            target_series_plot = target_series_plot.loc[target_series_plot.index.intersection(active_idx)]
            target_series = target_series.loc[target_series.index.intersection(active_idx)]

        # Apply date range filter to target if specified
        if date_range:
            start_date, end_date = date_range
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            mask_plot = (target_series_plot.index >= start_date) & (target_series_plot.index <= end_date)
            target_series_plot = target_series_plot[mask_plot]

            mask = (target_series.index >= start_date) & (target_series.index <= end_date)
            target_series = target_series[mask]
        else:
            start_date = end_date = None

        for var_name, df in var_dfs.items():
            df = df.copy()
            df_plot = df.copy()

            if outlier_labels is not None:
                df_plot.loc[df_plot.index.isin(outlier_labels), :] = np.nan
                df = df.drop(index=outlier_labels, errors='ignore')

            if active_idx is not None:
                # NOTE: Keep variable data aligned to the active regime periods so the
                # plotted transformations mirror the filtered target timeline.
                df_plot = df_plot.loc[df_plot.index.intersection(active_idx)]
                df = df.loc[df.index.intersection(active_idx)]

            # Apply date range filter to variable data if specified
            if date_range:
                mask_plot = (df_plot.index >= start_date) & (df_plot.index <= end_date)
                df_plot = df_plot[mask_plot]

                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df[mask]

            # NOTE: Keep variable data constrained to the same timeline as the target.
            # Without this alignment, the secondary axis could include observations
            # outside the selected sample, causing plots to display the full series
            # even when no date range is provided.
            df_plot = df_plot.loc[df_plot.index.intersection(target_series_plot.index)]
            df = df.loc[df.index.intersection(target_series.index)]

            # Align df and target to their common index for correlation computations
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
                    plot_index = target_series_plot.index.union(df_plot.index)
                    target_plot_aligned = target_series_plot.reindex(plot_index)
                    var_plot_aligned = df_plot[col].reindex(plot_index)

                    _plot_segmented_series(
                        ax,
                        target_plot_aligned,
                        color='tab:blue',
                        label=self.target,
                        linewidth=2
                    )
                    ax2 = ax.twinx()
                    _plot_segmented_series(
                        ax2,
                        var_plot_aligned,
                        color='tab:orange',
                        label=col,
                        linewidth=2
                    )

                    # Synchronize legend entries across twin axes for consistency
                    handles_1, labels_1 = ax.get_legend_handles_labels()
                    handles_2, labels_2 = ax2.get_legend_handles_labels()
                    ax.legend(handles=handles_1 + handles_2, labels=labels_1 + labels_2, loc='best')

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
        periods: Optional[Sequence[int]] = None,
        exp_sign_map: Optional[Dict[str, int]] = None,
        regime: Optional[str] = None,
        regime_on: Union[bool, int] = True,
        sample: str = 'full',
        plot_type: str = 'line',
        date_range: Optional[Tuple[str, str]] = None,
        plot: bool = True,
        plot_top: Optional[int] = None,
        pretest: bool = False,
        print_pretest: bool = False,
        outlier_idx: Optional[Sequence[Any]] = None,
        **legacy_kwargs: Any
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
        periods : Sequence[int], optional
            Period configuration forwarded to
            :meth:`DataManager.build_search_vars`. Provide a list of positive
            integers to explicitly control period-based transforms.
            Recommended choices include ``[1, 2, 3, 6, 9, 12]`` for monthly
            data and ``[1, 2, 3, 4]`` for quarterly data. When ``None``
            (default), frequency-aware defaults are applied automatically. The
            deprecated ``max_periods`` keyword is still accepted for backward
            compatibility.
        exp_sign_map : Dict[str, int], optional
            Mapping from variable codes to expected coefficient signs. When
            provided, expected sign metadata is applied to compatible feature
            specifications generated by :meth:`DataManager.build_tsfm_specs` and
            used to filter correlation output to transformations whose observed
            correlation sign matches the expected sign.
        regime : str, optional
            Column name of a regime indicator. When supplied, all transform
            specifications are wrapped in regime-aware features and correlation
            analysis only includes observations where the regime equals
            ``regime_on``.
        regime_on : bool or int, default True
            Active regime value used to filter observations when ``regime`` is
            provided. Must be interpretable as 0 or 1.
        sample : str, default 'full'
            Which sample to use:
            - 'in': use in-sample data only
            - 'full': use full sample data (in-sample + out-sample)
        plot_type : str, default 'line'
            Type of plot to create ('line' or 'scatter').
        date_range : Tuple[str, str], optional
            Date range for zooming in, e.g., ('2020-05-31', '2022-02-28').
            If provided, plots and correlations will be calculated only for this period.
        plot : bool, default True
            Flag indicating whether to generate plots via :meth:`plot_vars` before
            running the correlation analysis. Set to ``False`` to skip plotting when
            only tabular correlations are required.
        plot_top : int, optional
            Positive integer indicating how many of the highest absolute-correlation
            transformations to plot. When provided alongside ``plot=True``, the method
            skips the initial plotting pass, computes correlations first, and then plots
            only the top ``plot_top`` entries using :meth:`plot_vars` by forwarding
            either the variable names or the corresponding :class:`Feature` objects.
        pretest : bool, default False
            When ``True``, execute the target and feature pre-tests defined on
            ``self.model_cls`` (if any) before calculating correlations. Features
            failing validation are omitted from the output.
        print_pretest : bool, default False
            When ``True`` and ``pretest`` is enabled, print a summary of excluded
            features mirroring the :class:`ModelSearch` reporting style.
        outlier_idx : Sequence[Any], optional
            Iterable of index labels representing observations to exclude from both the
            plotting step and correlation analysis. Labels must match those in the
            modeling DataFrame index. Useful for omitting anomalous periods before
            ranking transformations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['variable', 'corr', 'abs_corr'] sorted by absolute
            correlation in descending order. When ``exp_sign_map`` is provided, only
            transformations whose correlation sign matches the mapped expectation
            are retained.

        Raises
        ------
        TypeError
            If unexpected keyword arguments are provided, ``regime`` is not a string,
            ``regime_on`` cannot be interpreted as 0/1, or ``plot_top`` is not an
            integer when supplied.
        ValueError
            If ``regime_on`` is not interpretable as 0/1 or boolean, or ``plot_top``
            is not positive when provided.
        KeyError
            If the specified ``regime`` column cannot be found in either internal or
            MEV data sources.

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
        ...     periods=[1, 3, 6, 12],
        ...     date_range=('2020-01-01', '2022-12-31'),
        ...     outlier_idx=['2020-03-31']
        ... )
        """
        outlier_labels: Optional[pd.Index]
        active_regime_idx: Optional[pd.Index] = None
        if outlier_idx is None:
            outlier_labels = None
        else:
            # NOTE: Preserve user-specified labels for consistent filtering across steps.
            outlier_labels = pd.Index(outlier_idx)

        # First create the plots when requested. This keeps backward compatibility
        # with the original behavior while allowing callers to opt out of plotting.
        # When plot_top is provided, defer plotting until correlations are available.
        if plot and plot_top is None:
            self.plot_vars(
                vars_list=vars_list,
                plot_type=plot_type,
                sample=sample,
                date_range=date_range,
                outlier_idx=outlier_labels
            )

        legacy_max_periods = legacy_kwargs.pop("max_periods", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if plot_top is not None:
            if not isinstance(plot_top, int):
                raise TypeError("plot_top must be provided as a positive integer.")
            if plot_top <= 0:
                raise ValueError("plot_top must be a positive integer when specified.")

        # Respect explicit caller intent for quarterly data by only forcing the
        # default quarterly floor when no custom periods (including legacy
        # ``max_periods``) are supplied. The previous implementation always
        # enforced ``[1, 2, 3, 4]`` which prevented users from narrowing the
        # window to values such as ``[1]``.
        resolved_periods = resolve_periods_argument(
            self.dm.freq,
            periods,
            legacy_max_periods=legacy_max_periods,
            ensure_quarterly_floor=(periods is None and legacy_max_periods is None)
        )

        tsfm_specs = self.dm.build_tsfm_specs(
            vars_list,
            max_lag=max_lag,
            periods=resolved_periods,
            exp_sign_map=exp_sign_map,
            regime=regime,
            regime_on=regime_on
        )

        feature_test: Optional[FeatureTest] = None
        excluded_features: List[str] = []
        excluded_seen: Set[str] = set()
        excluded_sign_variants: List[str] = []
        pretest_cache: Dict[str, bool] = {}
        if pretest:
            model_pretestset = self._resolve_model_pretestset()
            target_pretest_result = self._run_target_pretest(
                model_pretestset,
                print_result=False
            )
            feature_test = self._prepare_feature_pretest(
                model_pretestset,
                target_pretest_result
            )

        # Apply feature pretests on specs before constructing feature dataframes
        filtered_tsfm_specs: Dict[str, List[Union[str, Feature, TSFM]]] = {}
        tsfm_lookup: Dict[str, Union[str, Feature, TSFM]] = {}
        for var_name, tsfms in tsfm_specs.items():
            filtered: List[Union[str, Feature, TSFM]] = []
            for tsfm in tsfms:
                if feature_test is None:
                    filtered.append(tsfm)
                    if isinstance(tsfm, Feature):
                        tsfm_lookup[tsfm.name] = tsfm
                    elif isinstance(tsfm, str):
                        tsfm_lookup[tsfm] = tsfm
                    continue

                cache_key = repr(tsfm)
                if cache_key in pretest_cache:
                    passes_feature = pretest_cache[cache_key]
                else:
                    feature_test.feature = tsfm
                    try:
                        passes_feature = bool(feature_test.test_filter)
                    except Exception as exc:
                        print(
                            "Feature pre-test raised "
                            f"{type(exc).__name__} for {cache_key!r}: {exc}"
                        )
                        passes_feature = True
                    pretest_cache[cache_key] = passes_feature

                if passes_feature:
                    filtered.append(tsfm)
                    if isinstance(tsfm, Feature):
                        tsfm_lookup[tsfm.name] = tsfm
                    elif isinstance(tsfm, str):
                        tsfm_lookup[tsfm] = tsfm
                elif cache_key not in excluded_seen:
                    excluded_features.append(cache_key)
                    excluded_seen.add(cache_key)

            filtered_tsfm_specs[var_name] = filtered

        if feature_test is not None and excluded_features and print_pretest:
            # Surface exclusions as soon as pre-testing completes to aid inspection.
            print("--- Feature Pre-Test Exclusions ---")
            print(
                "Excluded "
                f"{len(excluded_features)} variant(s): "
                + ", ".join(excluded_features)
            )
            print("")

        # Generate all possible transformations for each variable after filtering.
        # We also capture a mapping from the realized column names (which include
        # frequency prefixes and lag suffixes resolved during apply()) back to the
        # originating transform objects. This prevents plot_vars() from rebuilding
        # an untransformed series when a top entry stems from a lagged or
        # frequency-annotated TSFM whose name is finalized only after execution.
        var_dfs: Dict[str, pd.DataFrame] = {}
        for var_name, tsfms in filtered_tsfm_specs.items():
            var_df = self.dm.build_features(tsfms)
            var_dfs[var_name] = var_df

            # Prefer a one-to-one mapping between the provided specs and the
            # resulting columns when sizes align; otherwise, fall back to any
            # explicit output_names defined on the feature object to preserve
            # downstream plotting fidelity.
            if len(tsfms) == len(var_df.columns):
                for tsfm, col in zip(tsfms, var_df.columns):
                    tsfm_lookup[col] = tsfm
            else:
                for tsfm in tsfms:
                    output_names = getattr(tsfm, "output_names", None)
                    if output_names:
                        for col in output_names:
                            tsfm_lookup[col] = tsfm
        
        # Get target data based on sample
        if sample == 'in':
            target_idx = self.dm.in_sample_idx
        else:  # sample == 'full'
            target_idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)

        target_data = self.dm.internal_data.loc[target_idx, self.target]

        if regime is not None:
            if not isinstance(regime, str):
                raise TypeError("regime must be provided as a column name string when set.")

            try:
                normalized_regime_on = int(regime_on)
            except (TypeError, ValueError):
                raise TypeError("regime_on must be a boolean or int interpretable as 0/1.")

            if normalized_regime_on not in (0, 1):
                raise ValueError("regime_on must be interpretable as 0/1 or boolean.")

            if regime in self.dm.internal_data.columns:
                regime_series = self.dm.internal_data[regime]
            elif regime in self.dm.model_mev.columns:
                regime_series = self.dm.model_mev[regime]
            else:
                raise KeyError(
                    f"Regime column '{regime}' not found in internal_data or model_mev."
                )

            aligned_regime = regime_series.reindex(target_data.index)
            active_regime_idx = aligned_regime.index[aligned_regime == normalized_regime_on]
            # Restrict target observations to the active regime before applying other filters.
            target_data = target_data.loc[target_data.index.intersection(active_regime_idx)]

        if outlier_labels is not None:
            target_data = target_data.drop(labels=outlier_labels, errors='ignore')

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
            var_df = var_df.copy()

            expected_sign = None
            if exp_sign_map is not None:
                expected_sign = exp_sign_map.get(var_name)

            if outlier_labels is not None:
                var_df = var_df.drop(index=outlier_labels, errors='ignore')

            if active_regime_idx is not None:
                var_df = var_df.loc[var_df.index.intersection(active_regime_idx)]

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

                # When expected signs are provided, only keep correlations aligned with the expectation.
                if expected_sign in (-1, 0, 1):
                    if expected_sign != 0:
                        corr_sign = int(np.sign(corr)) if corr != 0 else 0
                        if corr_sign != expected_sign:
                            if print_pretest:
                                excluded_sign_variants.append(col)
                            continue

                corr_results.append({
                    'variable': col,
                    'corr': corr,
                    'abs_corr': abs(corr)
                })

        # Create result DataFrame and sort by absolute correlation
        result_df = pd.DataFrame(corr_results)
        result_df = result_df.sort_values('abs_corr', ascending=False).reset_index(drop=True)

        if print_pretest and excluded_sign_variants:
            print("--- Expected Sign Exclusions ---")
            print(
                "Excluded "
                f"{len(excluded_sign_variants)} variant(s): "
                + ", ".join(excluded_sign_variants)
            )
            print("")

        if plot and plot_top is not None and not result_df.empty:
            top_n = min(plot_top, len(result_df))
            top_entries = result_df.head(top_n)['variable'].tolist()
            # NOTE: Preserve Feature objects for transformed variants so plot_vars()
            # can render the exact transformation instead of rebuilding all variants.
            plot_specs: List[Union[str, Feature, TSFM]] = [
                tsfm_lookup.get(var_name, var_name) for var_name in top_entries
            ]

            print(
                "Plotting top "
                f"{top_n} transformation(s) by absolute correlation: "
                + ", ".join(map(str, top_entries))
            )
            self.plot_vars(
                vars_list=plot_specs,
                plot_type=plot_type,
                sample=sample,
                date_range=date_range,
                outlier_idx=outlier_labels,
                active_idx=active_regime_idx
            )

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
            - 'stability_testing': Walk-forward stability testing results
            - 'stability_testing_stats': Walk-forward stability testing statistical metrics
            - 'scenario_testing_stats': Scenario testing statistical metrics for base variables
        overwrite : bool, default True
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
            elif content_type == 'stability_testing':
                expected_files.append(output_dir / 'stability_testing.csv')
            elif content_type == 'stability_testing_stats':
                expected_files.append(output_dir / 'stability_testing_stats.csv')
            elif content_type == 'scenario_testing_stats':
                expected_files.append(output_dir / 'scenario_testing_stats.csv')
        
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
        
        # Track files that existed before export for accurate reporting
        files_existed_before = set(f.name for f in existing_files)
        
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
        
        # Get the files that were actually written during export
        written_files = strategy.get_written_files()
        
        if written_files:
            # Categorize files as overwritten vs newly created
            written_file_names = set(f.name for f in written_files)
            overwritten_files = written_file_names.intersection(files_existed_before)
            new_files = written_file_names - files_existed_before
            
            # Print detailed success message
            print(f"\n✅ Export completed successfully for segment '{self.segment_id}'!")
            print(f"📁 Output directory: {output_dir}")
            
            if new_files:
                print(f"📄 New files created ({len(new_files)}):")
                for file_name in sorted(new_files):
                    print(f"   ✓ {file_name}")
            
            if overwritten_files:
                print(f"🔄 Files overwritten ({len(overwritten_files)}):")
                for file_name in sorted(overwritten_files):
                    print(f"   ✓ {file_name}")
            
            print(f"📊 Total files written: {len(written_files)}")
            
            # Analyze and report on empty files with diagnostic information
            self._analyze_empty_exports(output_dir, models_to_export, content_types_set or set(EXPORT_CONTENT_TYPES.keys()))
            
        else:
            print(f"\n⚠️  Export completed but no files were written.")
            print(f"This may indicate that the selected models had no data to export.")
            print(f"Output directory: {output_dir}")

    def _analyze_empty_exports(self, output_dir: Path, models_to_export: List[Tuple[str, Any]], content_types: Set[str]) -> None:
        """
        Analyze exported files and provide diagnostic information for empty exports.
        
        This method examines each exported file and provides detailed explanations
        for why certain export components might be empty, helping users understand
        potential issues with their models or data.
        
        Parameters
        ----------
        output_dir : Path
            Directory where files were exported
        models_to_export : List[Tuple[str, Any]]
            List of (model_id, cm) tuples that were exported
        content_types : Set[str]
            Set of content types that were exported
        """
        print(f"\n🔍 Analyzing export results for potential issues...")
        
        # File mapping for content types
        file_mapping = {
            'timeseries_data': 'timeseries_data.csv',
            'staticStats': 'staticStats.csv',
            'scenario_testing': 'scenario_testing.csv',
            'sensitivity_testing': 'sensitivity_testing.csv',
            'test_results': 'test_results.csv',
            'stability_testing': 'stability_testing.csv',
            'stability_testing_stats': 'stability_testing_stats.csv',
            'scenario_testing_stats': 'scenario_testing_stats.csv'
        }
        
        empty_files = []
        model_diagnostics = {}
        
        # Check each exported file for emptiness
        for content_type in content_types:
            filename = file_mapping.get(content_type)
            if filename:
                filepath = output_dir / filename
                if filepath.exists():
                    try:
                        df = pd.read_csv(filepath)
                        if df.empty:
                            empty_files.append((content_type, filename))
                    except Exception as e:
                        print(f"   ⚠️  Could not read {filename}: {e}")
        
        if not empty_files:
            print("   ✅ All export files contain data - no issues detected!")
            return
        
        # Provide summary of what was found
        content_types_with_issues = [ct for ct, _ in empty_files]
        print(f"   📊 Summary: {len(empty_files)} out of {len(content_types)} export types are empty")
        print(f"   🔍 Empty types: {', '.join(content_types_with_issues)}")
        
        print(f"\n📋 Found {len(empty_files)} empty export file(s). Analyzing potential causes...")
        
        # Collect model diagnostics
        for model_id, cm in models_to_export:
            model_diagnostics[model_id] = self._diagnose_model_issues(cm)
        
        # Analyze each empty file
        for content_type, filename in empty_files:
            print(f"\n📄 {filename} (0 rows)")
            print(f"   Content Type: {content_type}")
            
            # Provide specific diagnostic information based on content type
            if content_type == 'timeseries_data':
                self._diagnose_timeseries_empty(model_diagnostics)
            elif content_type == 'staticStats':
                self._diagnose_statistics_empty(model_diagnostics)
            elif content_type == 'scenario_testing':
                self._diagnose_scenario_empty(model_diagnostics)
            elif content_type == 'sensitivity_testing':
                self._diagnose_sensitivity_empty(model_diagnostics)
            elif content_type == 'test_results':
                self._diagnose_test_results_empty(model_diagnostics)
            elif content_type == 'stability_testing':
                self._diagnose_stability_empty(model_diagnostics)
            elif content_type == 'stability_testing_stats':
                self._diagnose_stability_stats_empty(model_diagnostics)
            elif content_type == 'scenario_testing_stats':
                self._diagnose_scenario_stats_empty(model_diagnostics)
        
        # Provide general recommendations
        self._provide_general_recommendations(empty_files, model_diagnostics)

    def _diagnose_model_issues(self, cm: Any) -> Dict[str, Any]:
        """
        Diagnose potential issues with a candidate model.
        
        Parameters
        ----------
        cm : Any
            Candidate model to diagnose
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing diagnostic information about the model
        """
        diagnostics = {
            'has_model_in': cm.model_in is not None,
            'has_model_full': cm.model_full is not None,
            'has_base_variable': self.target_base is not None,
            'has_scen_manager': False,
            'has_testset': False,
            'has_stability_test': False,
            'model_type': type(cm.model_in).__name__ if cm.model_in else 'None',
            'fitted_successfully': False,
            'has_data': False,
            'has_scenarios': False,
            'has_sensitivity': False
        }
        
        # Check model_in diagnostics
        if cm.model_in is not None:
            diagnostics['fitted_successfully'] = hasattr(cm.model_in, 'params') and cm.model_in.params is not None
            diagnostics['has_data'] = (hasattr(cm.model_in, 'y_in') and 
                                     cm.model_in.y_in is not None and 
                                     not cm.model_in.y_in.empty)
            diagnostics['has_testset'] = (hasattr(cm.model_in, 'testset') and 
                                        cm.model_in.testset is not None)
            diagnostics['has_scen_manager'] = (hasattr(cm.model_in, 'scen_manager') and 
                                             cm.model_in.scen_manager is not None)
            diagnostics['has_stability_test'] = (hasattr(cm.model_in, 'stability_test') and 
                                               cm.model_in.stability_test is not None)
            
            # Check for scenario and sensitivity data
            if diagnostics['has_scen_manager']:
                scen_mgr = cm.model_in.scen_manager
                diagnostics['has_scenarios'] = (hasattr(scen_mgr, 'y_scens') and 
                                              scen_mgr.y_scens is not None and 
                                              len(scen_mgr.y_scens) > 0)
                diagnostics['has_sensitivity'] = (hasattr(scen_mgr, 'sens_test') and 
                                                scen_mgr.sens_test is not None)
        
        return diagnostics

    def _diagnose_timeseries_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty timeseries data."""
        print("   🔍 Possible causes:")
        
        models_without_data = [mid for mid, diag in model_diagnostics.items() if not diag['has_data']]
        models_not_fitted = [mid for mid, diag in model_diagnostics.items() if not diag['fitted_successfully']]
        
        if models_without_data:
            print(f"   • Models without input data: {', '.join(models_without_data)}")
            print("     → Check if data_manager contains the target variable")
            print("     → Verify in_sample_idx and out_sample_idx are properly set")
        
        if models_not_fitted:
            print(f"   • Models that failed to fit: {', '.join(models_not_fitted)}")
            print("     → Check for model specification errors")
            print("     → Verify feature variables exist in the dataset")
            print("     → Check for multicollinearity or insufficient data")
        
        if not models_without_data and not models_not_fitted:
            print("   • All models appear to have data and fitted successfully")
            print("     → This may be a temporary issue or data processing error")

    def _diagnose_statistics_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty statistics."""
        print("   🔍 Possible causes:")
        
        models_not_fitted = [mid for mid, diag in model_diagnostics.items() if not diag['fitted_successfully']]
        models_without_tests = [mid for mid, diag in model_diagnostics.items() if not diag['has_testset']]
        
        if models_not_fitted:
            print(f"   • Models that failed to fit: {', '.join(models_not_fitted)}")
            print("     → Model fitting failed - no statistical results available")
            print("     → Check model specification and data quality")
        
        if models_without_tests:
            print(f"   • Models without test sets: {', '.join(models_without_tests)}")
            print("     → Test set not initialized - limited statistics available")
            print("     → Consider running model tests to generate complete statistics")

    def _diagnose_scenario_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty scenario testing."""
        print("   🔍 Possible causes:")
        
        models_without_scen_mgr = [mid for mid, diag in model_diagnostics.items() if not diag['has_scen_manager']]
        models_without_scenarios = [mid for mid, diag in model_diagnostics.items() if not diag['has_scenarios']]
        missing_base = not any(diag['has_base_variable'] for diag in model_diagnostics.values())
        
        if models_without_scen_mgr:
            print(f"   • Models without scenario manager: {', '.join(models_without_scen_mgr)}")
            print("     → Scenario manager not initialized")
            print("     → Run scenario forecasting first: model.scen_manager = ScenManager(...)")
        
        if models_without_scenarios:
            print(f"   • Models without scenario data: {', '.join(models_without_scenarios)}")
            print("     → No scenario forecasts available")
            print("     → Check if scenario data exists in data_manager.scen_mevs or scen_internal_data")
        
        if missing_base:
            print("   • No base variable specified for this segment")
            print("     → Base variable forecasts will not be available")
            print("     → Consider setting target_base parameter when creating the segment")

    def _diagnose_sensitivity_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty sensitivity testing."""
        print("   🔍 Possible causes:")
        
        models_without_scen_mgr = [mid for mid, diag in model_diagnostics.items() if not diag['has_scen_manager']]
        models_without_sensitivity = [mid for mid, diag in model_diagnostics.items() if not diag['has_sensitivity']]
        
        if models_without_scen_mgr:
            print(f"   • Models without scenario manager: {', '.join(models_without_scen_mgr)}")
            print("     → Scenario manager required for sensitivity testing")
            print("     → Initialize: model.scen_manager = ScenManager(...)")
        
        if models_without_sensitivity:
            print(f"   • Models without sensitivity tests: {', '.join(models_without_sensitivity)}")
            print("     → Sensitivity tests not run")
            print("     → Run: model.scen_manager.sens_test = SensitivityTest(...)")
            print("     → Then execute: model.scen_manager.sens_test.run_all()")

    def _diagnose_test_results_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty test results."""
        print("   🔍 Possible causes:")
        
        models_without_tests = [mid for mid, diag in model_diagnostics.items() if not diag['has_testset']]
        models_not_fitted = [mid for mid, diag in model_diagnostics.items() if not diag['fitted_successfully']]
        
        if models_without_tests:
            print(f"   • Models without test sets: {', '.join(models_without_tests)}")
            print("     → Test set not initialized")
            print("     → Run model tests to generate test results")
        
        if models_not_fitted:
            print(f"   • Models that failed to fit: {', '.join(models_not_fitted)}")
            print("     → Model fitting failed - no test results available")

    def _diagnose_stability_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty stability testing."""
        print("   🔍 Possible causes:")
        
        models_without_stability = [mid for mid, diag in model_diagnostics.items() if not diag['has_stability_test']]
        
        if models_without_stability:
            print(f"   • Models without stability tests: {', '.join(models_without_stability)}")
            print("     → Walk-forward stability test not run")
            print("     → Run: model.stability_test = WalkForwardTest(...)")
            print("     → Then execute: model.stability_test.run()")
        
        print("   • Stability testing requires sufficient historical data")
        print("     → Check if data_manager has enough periods for walk-forward analysis")

    def _diagnose_stability_stats_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty stability statistics."""
        print("   🔍 Possible causes:")
        
        models_without_stability = [mid for mid, diag in model_diagnostics.items() if not diag['has_stability_test']]
        
        if models_without_stability:
            print(f"   • Models without stability tests: {', '.join(models_without_stability)}")
            print("     → Walk-forward stability test not run")
            print("     → Statistics require completed stability testing first")

    def _diagnose_scenario_stats_empty(self, model_diagnostics: Dict[str, Dict]) -> None:
        """Provide diagnostic information for empty scenario statistics."""
        print("   🔍 Possible causes:")
        
        missing_base = not any(diag['has_base_variable'] for diag in model_diagnostics.values())
        models_without_scenarios = [mid for mid, diag in model_diagnostics.items() if not diag['has_scenarios']]
        
        if missing_base:
            print("   • No base variable specified for this segment")
            print("     → Scenario statistics require a base variable")
            print("     → Set target_base parameter when creating the segment")
        
        if models_without_scenarios:
            print(f"   • Models without scenario data: {', '.join(models_without_scenarios)}")
            print("     → Base variable quarterly forecasts not available")
            print("     → Check if base variable scenario forecasting was completed")

    def _provide_general_recommendations(self, empty_files: List[Tuple[str, str]], model_diagnostics: Dict[str, Dict]) -> None:
        """Provide general recommendations based on empty file analysis."""
        if not empty_files:
            return
        
        print(f"\n💡 General Recommendations:")
        
        # Check overall model health
        total_models = len(model_diagnostics)
        fitted_models = sum(1 for diag in model_diagnostics.values() if diag['fitted_successfully'])
        models_with_data = sum(1 for diag in model_diagnostics.values() if diag['has_data'])
        
        if fitted_models < total_models:
            print(f"   • {total_models - fitted_models}/{total_models} models failed to fit properly")
            print("     → Review model specifications and feature selection")
            print("     → Check for data quality issues or multicollinearity")
        
        if models_with_data < total_models:
            print(f"   • {total_models - models_with_data}/{total_models} models have no input data")
            print("     → Verify data_manager setup and sample period definitions")
        
        # Content-specific recommendations
        empty_content_types = [ct for ct, _ in empty_files]
        
        if any(ct in empty_content_types for ct in ['scenario_testing', 'sensitivity_testing', 'scenario_testing_stats']):
            print("   • Scenario-related exports are empty:")
            print("     → Ensure scenario forecasting is set up and run")
            print("     → Check data_manager.scen_mevs and scen_internal_data")
            print("     → Verify ScenManager initialization and execution")
        
        if 'stability_testing' in empty_content_types or 'stability_testing_stats' in empty_content_types:
            print("   • Stability testing exports are empty:")
            print("     → Run walk-forward stability tests: model.stability_test.run()")
            print("     → Ensure sufficient historical data for multiple periods")
        
        if 'test_results' in empty_content_types:
            print("   • Test results are empty:")
            print("     → Initialize and run model diagnostic tests")
            print("     → Check if TestSet is properly configured")
        
        print(f"\n📚 For detailed troubleshooting, refer to the User Manual or documentation.")

    def add_benchmark_cm(
        self,
        cm_id: str,
        specs: Any,
        fixed_params: Union[Dict[str, float], pd.Series],
        sample: str = 'both',
        coef_map_mode: str = 'auto'
    ) -> CM:
        """
        Add a benchmark CM with fixed, pre-trained coefficients.

        This constructs a `CM` whose underlying model is `FixedOLS`, so the model
        will not estimate coefficients—it computes fitted/predicted values directly
        from the supplied `fixed_params`.

        Coefficient mapping convenience:
        - Users may specify keys using any of the following forms:
          * exact feature column names (e.g., 'GDP_QQDF2_L1')
          * TSFM instances used in specs
          * canonical names without MM/QQ prefixes (e.g., 'GDP_DF2_L1')
          * raw MEV names or internal variables
        - Mapping resolution is handled automatically when building the model.

        Parameters
        ----------
        cm_id : str
            Unique identifier for this candidate model.
        specs : Any
            Feature specifications passed to DataManager for building drivers.
        fixed_params : dict or Series
            Mapping from feature identifier to coefficient. Include 'const' for intercept
            (assumed 0.0 if omitted). Names can be flexible as described above.
        sample : {'in','full','both'}, default 'both'
            Which sample(s) to construct.
        coef_map_mode : {'auto'}, optional
            Reserved for future mapping strategies; currently only 'auto'.

        Returns
        -------
        CM
            The constructed benchmark CM instance (also stored in `self.cms`).
        """
        cm = CM(
            model_id=cm_id,
            target=self.target,
            model_type=self.model_type,
            target_base=self.target_base,
            target_exposure=self.target_exposure,
            data_manager=self.dm,
            model_cls=FixedOLS,
            scen_cls=self.scen_cls,
            qtr_method=self.qtr_method,
        )
        cm.build(
            specs=specs,
            sample=sample,
            model_kwargs={'fixed_params': pd.Series(fixed_params, dtype=float)}
        )
        self.cms[cm_id] = cm
        return cm

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
        periods: Optional[Sequence[int]] = None,
        sample: str = 'full',
        **legacy_kwargs: Any
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
        periods : Sequence[int], optional
            Period configuration forwarded to
            :meth:`DataManager.build_search_vars`. Provide a list of positive
            integers to explicitly control period-based transforms.
            Recommended choices include ``[1, 2, 3, 6, 9, 12]`` for monthly
            data and ``[1, 2, 3, 4]`` for quarterly data. When ``None``
            (default), frequency-aware defaults are applied automatically. The
            deprecated ``max_periods`` keyword is still accepted for backward
            compatibility.
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
        ...     periods=[1, 2, 3, 4],
        ...     sample='in'
        ... )
        >>> 
        >>> # Find top 10 most correlated features
        >>> top_features = corr_df.head(10)['variable'].tolist()
        """
        if sample not in ['in', 'full']:
            raise ValueError("sample must be 'in' or 'full'")

        legacy_max_periods = legacy_kwargs.pop("max_periods", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        resolved_periods = resolve_periods_argument(
            self.dm.freq,
            periods,
            legacy_max_periods=legacy_max_periods,
            ensure_quarterly_floor=(periods is None and legacy_max_periods is None)
        )

        # Build all possible transformations for the variables
        var_dfs = self.dm.build_search_vars(
            vars_list,
            max_lag=max_lag,
            periods=resolved_periods
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
        periods: Optional[Sequence[int]] = None,
        category_limit: int = 1,
        regime_limit: Optional[int] = None,
        exp_sign_map: Optional[Dict[str, int]] = None,
        rank_weights: Tuple[float, float, float] = (1, 1, 1),
        test_update_func: Optional[Callable] = None,
        outlier_idx: Optional[List[Any]] = None,
        add_in: bool = True,
        overwrite: bool = False,
        re_rank: bool = True,
        search_id: Optional[str] = None,
        **legacy_kwargs: Any
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
        periods : Sequence[int], optional
            Period configuration forwarded to :meth:`ModelSearch.run_search`.
            Provide a list of positive integers to explicitly control
            period-based transforms. Recommended choices include
            ``[1, 2, 3, 6, 9, 12]`` for monthly data and ``[1, 2, 3, 4]`` for
            quarterly data. When ``None`` (default), frequency-aware defaults
            are applied automatically. The deprecated ``max_periods`` keyword is
            still accepted for backward compatibility.
        category_limit : int, default 1
            Maximum number of variables from each MEV category per combo.
            Applies to both top-level strings/TSFM instances in ``desired_pool``
            and :class:`RgmVar` entries, evaluated per ``(regime, regime_on)``
            signature so active/inactive variants are constrained separately.
        regime_limit : Optional[int], default None
            Maximum number of :class:`RgmVar` instances from the same regime per
            combo. Applies across the full combination, including forced
            specifications. When ``None`` (default), no limit is applied.
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
        overwrite : bool, default False
            If True, clear existing cms before adding new ones. Only applies
            when add_in=True. The legacy ``override`` keyword is still honored
            for backward compatibility.
        re_rank : bool, default True
            If True and add_in=True and overwrite=False, rank new top_cms
            along with pre-existing cms and update model_ids based on ranking.
            If False, simply append new cms with collision-resolved IDs.
        search_id : str, optional
            Explicit search identifier to use. When omitted, a new identifier
            of the form ``search_<segment_id>_<YYYYMMDD_HHMMSS>`` is generated
            and stored on ``self.last_search_id``.

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
        >>> # Search and overwrite existing models
        >>> segment.search_cms(
        ...     desired_pool=["new_var1", "new_var2"],
        ...     overwrite=True  # clears existing models first
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
        searcher.segment = self

        legacy_overwrite = legacy_kwargs.pop("override", None)
        legacy_max_periods = legacy_kwargs.pop("max_periods", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        effective_overwrite = overwrite if legacy_overwrite is None else bool(legacy_overwrite)

        resolved_periods = resolve_periods_argument(
            self.dm.freq,
            periods,
            legacy_max_periods=legacy_max_periods
        )

        # 2) Run the search (populates searcher.top_cms; no return value)
        active_search_id = search_id or generate_search_id(self.segment_id)
        self.last_search_id = active_search_id
        searcher.run_search(
            desired_pool=desired_pool,
            forced_in=forced_in or [],
            top_n=top_n,
            sample=sample,
            max_var_num=max_var_num,
            max_lag=max_lag,
            periods=resolved_periods,
            category_limit=category_limit,
            regime_limit=regime_limit,
            rank_weights=rank_weights,
            test_update_func=test_update_func,
            outlier_idx=outlier_idx,
            exp_sign_map=exp_sign_map,
            search_id=active_search_id,
            base_dir=self.working_dir,
        )

        # Honor the search_id actually used (may differ when resuming).
        self.last_search_id = getattr(self.searcher, "search_id", active_search_id)

        # 3) Collect the top_n results from the searcher
        self.top_cms = self.searcher.top_cms[:top_n]

        # 4) Optionally add them to this segment's cms
        if add_in:
            if effective_overwrite:
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

    def rerank_cms(
        self,
        rank_weights: Tuple[float, float, float],
        all_passed: bool = True,
        overwrite: bool = False,
        top_n: int = 10,
        sample: str = 'in',
        **legacy_kwargs: Any
    ) -> List[CM]:
        """
        Re-compute rankings for candidate models using new weights.

        Parameters
        ----------
        rank_weights : Tuple[float, float, float]
            Weights for (Fit Measures, IS Error, OOS Error) used during ranking.
        all_passed : bool, default True
            When ``True``, include every CM stored in ``self.passed_cms`` when
            available, otherwise use ``self.searcher.passed_cms``.
            When ``False``, limit the re-ranking to models currently in
            ``self.cms``.
        overwrite : bool, default False
            If ``True``, replace ``self.cms`` and ``self.searcher.top_cms`` with
            the newly ranked ``top_n`` models. The legacy ``override`` keyword
            remains supported for backward compatibility.
        top_n : int, default 10
            Number of models to display and return from the refreshed ranking.
        sample : {'in', 'full'}, default 'in'
            Sample to use when retrieving diagnostics for ranking. Choose
            ``'in'`` for in-sample diagnostics or ``'full'`` for full-sample.

        Returns
        -------
        List[CM]
            The top ``top_n`` candidate models ordered by the refreshed
            composite score.

        Raises
        ------
        RuntimeError
            If no search has been performed or if there are no models to rank.

        Examples
        --------
        >>> # After running search_cms(...)
        >>> refreshed = segment.rerank_cms(
        ...     rank_weights=(0.4, 0.4, 0.2),
        ...     all_passed=True,
        ...     top_n=5
        ... )
        >>> refreshed[0].model_id
        'cm1'
        """
        legacy_overwrite = legacy_kwargs.pop("override", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        effective_overwrite = overwrite if legacy_overwrite is None else bool(legacy_overwrite)

        # Ensure a ModelSearch instance exists so ``rank_cms`` can be reused even
        # when candidate models were loaded directly from disk rather than
        # produced by a prior search run.
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
        self.searcher.segment = self

        # Determine which candidate models participate in the re-ranking.
        if all_passed:
            # Prefer persisted/loaded passed CMs when available; otherwise fall
            # back to any passed CMs retained on the searcher.
            if self.passed_cms:
                candidate_cms = list(self.passed_cms.values())
            else:
                passed_cms = getattr(self.searcher, 'passed_cms', None)
                if not passed_cms:
                    raise RuntimeError(
                        "ModelSearch has no passed candidate models to rerank."
                    )
                candidate_cms = list(passed_cms)
        else:
            if not self.cms:
                raise RuntimeError(
                    "Segment has no candidate models to rerank."
                )
            candidate_cms = list(self.cms.values())

        # Track existing models to identify newly promoted entries when
        # re-ranking across all passed combinations.
        existing_obj_ids: Set[int] = {id(cm) for cm in self.cms.values()}

        # Assign temporary identifiers to avoid collisions during ranking while
        # preserving object identity for later reassignment.
        temp_to_cm: Dict[str, CM] = {}
        for idx, cm in enumerate(candidate_cms):
            temp_id = f"temp_{idx}"
            temp_to_cm[temp_id] = cm
            cm.model_id = temp_id

        df_ranked = self.searcher.rank_cms(candidate_cms, sample, rank_weights)

        ordered_temp_ids = df_ranked['model_id'].tolist()
        ordered_cms = [temp_to_cm[temp_id] for temp_id in ordered_temp_ids]

        # Reassign sequential model identifiers based on refreshed ranking.
        df_updated = df_ranked.copy()
        for idx, cm in enumerate(ordered_cms):
            new_id = f"cm{idx + 1}"
            cm.model_id = new_id
            df_updated.at[idx, 'model_id'] = new_id

        # Derive the refreshed top-N models and update search state to reflect
        # the latest ranking outcome.
        top_limit = min(top_n, len(ordered_cms))
        top_models = ordered_cms[:top_limit]
        self.top_cms = top_models
        self.searcher.top_cms = top_models
        if all_passed:
            self.searcher.passed_cms = ordered_cms
        self.searcher.df_scores = df_updated

        # Persist cms registry depending on overwrite preference.
        if effective_overwrite:
            self.cms.clear()
            for cm in top_models:
                self.cms[cm.model_id] = cm
        else:
            self.cms.clear()
            for cm in ordered_cms:
                self.cms[cm.model_id] = cm

        print("=== Updated Ranked Results ===")
        print(df_updated.head(top_limit).to_string(index=False))

        print(f"\n=== Top {top_limit} Model Formulas ===")
        for cm in top_models:
            print(f"{cm.model_id}: {cm.formula}")

        # Highlight newly promoted models when re-ranking across all passed CMs.
        if all_passed:
            order_list = df_updated['model_id'].tolist()
            pos_map = {mid: (i + 1) for i, mid in enumerate(order_list)}
            newly_added = [
                (cm.model_id, pos_map.get(cm.model_id))
                for cm in top_models
                if id(cm) not in existing_obj_ids
            ]
            if newly_added:
                print("\nNew models entering the top list:")
                formatted = ", ".join(
                    f"{mid} (#{rank})" for mid, rank in newly_added
                )
                print(formatted)

        return top_models

    def _normalize_cm_collection(
        self, collection: Union[Dict[str, CM], List[CM]]
    ) -> Dict[str, CM]:
        """Return a model_id-keyed mapping while validating CMs for persistence."""

        cm_dict: Dict[str, CM] = {}
        for cm in collection.values() if isinstance(collection, dict) else collection:
            model_id = getattr(cm, "model_id", None)
            if not model_id:
                raise ValueError("Each CM must expose a non-empty model_id for persistence.")
            if model_id in cm_dict:
                raise ValueError(f"Duplicate model_id '{model_id}' encountered while saving CMs.")
            cm_dict[model_id] = cm
        return cm_dict

    def _save_cm_entry(
        self,
        cm: CM,
        target_dir: Path,
        created_at: str,
        overwrite: bool,
    ) -> Dict[str, Any]:
        """
        Persist a single CM pickle and return its index entry payload.

        Parameters
        ----------
        cm : CM
            Candidate model to persist.
        target_dir : Path
            Directory in which to store the pickle and related index entry.
        created_at : str
            Timestamp used for the index metadata.
        overwrite : bool
            Forwarded to :func:`save_cm` to control clobbering behavior.

        Returns
        -------
        Dict[str, Any]
            Index payload describing the saved CM.
        """

        model_id = getattr(cm, "model_id", None)
        if not model_id:
            raise ValueError("Each CM must expose a non-empty model_id for persistence.")

        filename = f"{model_id}.pkl"
        save_cm(cm, target_dir / filename, overwrite)
        return {
            "model_id": model_id,
            "filename": filename,
            "segment_id": self.segment_id,
            "created_at": created_at,
        }

    @staticmethod
    def _clear_cm_directory(target_dir: Path) -> None:
        """
        Remove persisted CM artifacts from a CM directory.

        Parameters
        ----------
        target_dir : Path
            Directory whose contents should be removed prior to saving.

        Notes
        -----
        This helper leaves the target directory in place while deleting any
        existing files or nested folders so that subsequent save operations
        write into a clean location when ``overwrite`` is requested.
        """

        if not target_dir.exists():
            return

        # Ensure stale pickles and indexes do not linger when overwrite=True.
        for item in target_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    @staticmethod
    def _latest_search_id(cms_dir: Path) -> Optional[str]:
        """
        Return the most recent search_id tracked within a CMS directory.

        Parameters
        ----------
        cms_dir : Path
            Base ``cms`` directory that may contain ``search_<segment>_*``
            folders and an optional ``search_index.json``.

        Returns
        -------
        Optional[str]
            The newest search identifier based on the timestamp suffix with a
            preference for runs that remain incomplete according to their
            progress metadata. ``None`` is returned when no search directories
            are available.
        """

        index_path = cms_dir / "search_index.json"
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            search_ids = list(index_data.keys())
        except (FileNotFoundError, json.JSONDecodeError):
            # Fall back to scanning directories when index is missing.
            search_ids = [p.name for p in cms_dir.iterdir() if p.is_dir() and p.name.startswith("search_")]

        if not search_ids:
            return None

        def _timestamp_value(search_label: str) -> str:
            # Timestamp component is always the suffix after the final underscore.
            return search_label.rsplit("_", 1)[-1]

        # Prefer the newest run that is still incomplete according to its
        # progress metadata. This allows load_cms() to surface models produced
        # during an active or recently interrupted search instead of defaulting
        # to the last finished run.
        log_dir = cms_dir.parent / "log"
        incomplete_ids = []
        for search_id in search_ids:
            progress_path = log_dir / f"{search_id}.progress"
            try:
                progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            total = progress_data.get("total_combos")
            completed = progress_data.get("completed_combos")
            if isinstance(total, int) and isinstance(completed, int) and completed < total:
                incomplete_ids.append(search_id)

        target_pool = incomplete_ids if incomplete_ids else search_ids
        return max(target_pool, key=_timestamp_value)

    def save_passed_cm(
        self,
        cm: CM,
        *,
        base_dir: Union[str, Path, None] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Save a single passed candidate model to the ``passed_cms`` directory.

        Parameters
        ----------
        cm : CM
            Candidate model to persist. ``cm.model_id`` must be populated.
        base_dir : Union[str, Path], optional
            Base directory under which the capitalized ``Segment`` folder is
            created. Defaults to the current working directory when ``None``.
        overwrite : bool, default True
            When ``True``, replace any existing pickle or index entry for the
            same ``model_id``. When ``False``, raises :class:`FileExistsError`
            if the pickle or index record already exists.

        Returns
        -------
        Path
            Filesystem path to the saved pickle.

        Raises
        ------
        ValueError
            If ``cm`` is missing a ``model_id``.
        FileExistsError
            If ``overwrite`` is ``False`` and a duplicate pickle or index entry
            is detected.

        Examples
        --------
        >>> segment.save_passed_cm(cm)
        PosixPath('.../Segment/my_segment/cms/passed_cms/passed_1.pkl')
        """

        base_path = Path(base_dir) if base_dir is not None else Path.cwd()
        dirs = ensure_segment_dirs(self.segment_id, base_path)
        created_at = datetime.now().isoformat(timespec="seconds")

        entry = self._save_cm_entry(cm, dirs["passed_dir"], created_at, overwrite)

        try:
            existing_entries = load_index(dirs["passed_dir"])
        except FileNotFoundError:
            existing_entries = []

        if not overwrite and any(e["model_id"] == entry["model_id"] for e in existing_entries):
            raise FileExistsError(
                f"Index entry for model_id '{entry['model_id']}' already exists in passed_cms."
            )

        # Replace any pre-existing entry for this model_id before writing.
        updated_entries = [e for e in existing_entries if e["model_id"] != entry["model_id"]]
        updated_entries.append(entry)
        save_index(dirs["passed_dir"], updated_entries, overwrite=True)
        return dirs["passed_dir"] / entry["filename"]

    def save_cms(
        self,
        save_selected: bool = True,
        save_passed: bool = True,
        overwrite: bool = True,
        base_dir: Union[str, Path, None] = None,
        search_id: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Save candidate models for this segment to disk.

        Selected models come from ``self.cms`` and passed models originate from
        ``self.searcher.passed_cms`` when available. Each model is saved under
        ``Segment/<segment_id>/cms/<search_id>/<group>`` using its ``model_id``
        as the filename and indexed via a minimal ``index.json`` file. Both
        ``selected_cms`` and ``passed_cms`` folders are scoped to the same
        ``search_id`` so that artifacts stay grouped by search run.

        Parameters
        ----------
        save_selected : bool, default True
            When ``True``, persist models stored in ``self.cms`` to the
            ``selected_cms`` directory.
        save_passed : bool, default True
            When ``True``, persist models stored in ``self.searcher.passed_cms``
            to the ``passed_cms`` directory.
        overwrite : bool, default True
            When ``True``, existing pickles and indexes in the target directories
            are cleared before saving. When ``False``, a :class:`FileExistsError`
            is raised if a pickle or index already exists.
        base_dir : Union[str, Path], optional
            Base directory under which the ``Segment`` folder is created. When
            ``None``, the current working directory is used.
        search_id : str, optional
            Search identifier whose selected and passed models should be saved
            under ``cms/<search_id>``. When omitted, the most recent search for
            this segment is used; a :class:`RuntimeError` is raised if no search
            metadata can be located.

        Returns
        -------
        Dict[str, Path]
            Mapping with keys ``selected_cms`` and ``passed_cms`` pointing to
            directories that were updated.

        Raises
        ------
        RuntimeError
            If saving passed CMs is requested but no search results are
            available or no ``search_id`` can be resolved.
        ValueError
            If duplicate ``model_id`` values are encountered while preparing
            pickles.
        FileExistsError
            If ``overwrite`` is ``False`` and a target pickle or index file
            already exists.
        """

        # Normalize the base directory so all persisted CMs live under a
        # capitalized "Segment" folder. ``ensure_segment_dirs`` will create the
        # hierarchy when it does not yet exist.
        base_path = Path(base_dir) if base_dir is not None else Path.cwd()
        dirs = ensure_segment_dirs(self.segment_id, base_path)
        search_target_id = (
            search_id
            or self.last_search_id
            or getattr(self.searcher, "search_id", None)
            or self._latest_search_id(dirs["cms_dir"])
        )
        created_at = datetime.now().isoformat(timespec="seconds")
        saved_paths: Dict[str, Path] = {}

        if search_target_id is None:
            raise RuntimeError(
                "A search_id is required to save selected or passed CMs. Run a search "
                "first or supply search_id explicitly."
            )

        search_root = dirs["cms_dir"] / search_target_id
        passed_dir = search_root / "passed_cms"
        selected_dir = search_root / "selected_cms"
        passed_dir.mkdir(parents=True, exist_ok=True)
        selected_dir.mkdir(parents=True, exist_ok=True)

        if save_selected:
            if not self.cms:
                raise RuntimeError("No selected candidate models are available to save.")

            selected_cms = self._normalize_cm_collection(self.cms)
            if overwrite:
                self._clear_cm_directory(selected_dir)
            selected_entries: List[Dict[str, Any]] = []
            for cm in selected_cms.values():
                entry = self._save_cm_entry(cm, selected_dir, created_at, overwrite)
                selected_entries.append(entry)

            save_index(selected_dir, selected_entries, overwrite)
            saved_paths["selected_cms"] = selected_dir

        if save_passed:
            if self.searcher is None or not getattr(self.searcher, "passed_cms", None):
                raise RuntimeError("No passed candidate models available to save from ModelSearch.")

            passed_cms = self._normalize_cm_collection(getattr(self.searcher, "passed_cms"))
            target_dir = passed_dir
            if overwrite:
                self._clear_cm_directory(target_dir)
            passed_entries: List[Dict[str, Any]] = []
            for cm in passed_cms.values():
                entry = self._save_cm_entry(cm, target_dir, created_at, overwrite)
                passed_entries.append(entry)

            save_index(target_dir, passed_entries, overwrite)
            saved_paths["passed_cms"] = target_dir

        return saved_paths

    def load_cms(
        self,
        which: str = "both",
        base_dir: Union[str, Path, None] = None,
        search_id: Optional[str] = None,
    ) -> None:
        """
        Load persisted candidate models and bind them to the current DataManager.

        Parameters
        ----------
        which : {'selected', 'passed', 'both'}, default 'both'
            Controls which sets to load from disk.
        base_dir : Union[str, Path], optional
            Base directory containing the ``Segment`` folder. Defaults to the
            current working directory when ``None``.
        search_id : str, optional
            When provided, loads passed candidate models exclusively from
            ``cms/<search_id>``. When omitted, the most recent search tracked
            for this segment is used if present, otherwise legacy locations are
            scanned.

        Returns
        -------
        None
            This method prints a concise summary of loaded models.

        Notes
        -----
        Prints a brief summary of how many selected and passed models were loaded.
        Both selected and passed models are expected under
        ``Segment/<segment_id>/cms/<search_id>/(selected_cms|passed_cms)`` for
        recent runs while still tolerating legacy top-level folders when no
        search metadata is available.

        Empty CM directories (or missing ``index.json`` files) are tolerated and
        will result in zero loaded models for that group.

        Raises
        ------
        ValueError
            If ``which`` is not one of ``'selected'``, ``'passed'``, or
            ``'both'`` or if the segment lacks a bound DataManager.
        """
        if which not in {"selected", "passed", "both"}:
            raise ValueError("Parameter 'which' must be 'selected', 'passed', or 'both'.")
        if self.dm is None:
            raise ValueError("Segment must have an attached DataManager before loading CMs.")

        # Always resolve directories relative to a capitalized "Segment" root
        # so loading mirrors the persistence convention used in ``save_cms``.
        # ``ensure_segment_dirs`` guarantees the structure exists when invoked
        # in save paths, but loading should still look in the same location.
        base_path = Path(base_dir) if base_dir is not None else Path.cwd()
        dirs = get_segment_dirs(self.segment_id, base_path)
        target_search_id = (
            search_id
            or getattr(self.searcher, "search_id", None)
            or self.last_search_id
            or self._latest_search_id(dirs["cms_dir"])
        )

        if target_search_id is not None:
            self.last_search_id = target_search_id

        def _load_group(target_dir: Path, container: Dict[str, CM]) -> List[Dict[str, Any]]:
            # Gracefully handle absent or empty CM folders by returning zero entries.
            if not target_dir.exists():
                container.clear()
                return []

            try:
                index_entries = load_index(target_dir)
            except FileNotFoundError:
                container.clear()
                return []

            # Reset container to mirror persisted state.
            container.clear()
            for entry in index_entries:
                model_id = entry["model_id"]
                cm_path = target_dir / entry["filename"]
                cm = load_cm(cm_path)
                cm.bind_data_manager(self.dm)
                container[model_id] = cm
            return index_entries

        if which in {"selected", "both"}:
            selected_dir = (
                dirs["cms_dir"] / target_search_id / "selected_cms"
                if target_search_id is not None
                else dirs["selected_dir"]
            )
            _load_group(selected_dir, self.cms)

        if which in {"passed", "both"}:
            passed_dir = (
                dirs["cms_dir"] / target_search_id / "passed_cms"
                if target_search_id is not None
                else dirs["passed_dir"]
            )
            _load_group(passed_dir, self.passed_cms)

        summary = {
            "passed": len(self.passed_cms),
            "selected": len(self.cms),
        }
        print(
            f"Loaded passed_cms={summary['passed']}; "
            f"selected_cms={summary['selected']}."
        )
        if target_search_id:
            print(f"Search artifacts loaded from search_id={target_search_id}.")
        else:
            print("Loaded CMS artifacts from legacy locations (no search_id resolved).")
