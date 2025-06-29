# TECHNIC/segment.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Type, Dict, List, Optional, Any, Union, Callable, Tuple

from .cm import CM
from .model import ModelBase
from .template import ExportTemplateBase
from .report import ReportSet
from .search import ModelSearch  # new import


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
        data_manager: Any,
        model_cls: Type[ModelBase],
        export_template_cls: Optional[Type[ExportTemplateBase]] = None,
        reportset_cls: Type[ReportSet] = ReportSet,
        search_cls: Type[ModelSearch] = ModelSearch  # added parameter
    ):
        self.segment_id = segment_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.export_template_cls = export_template_cls
        self.reportset_cls = reportset_cls
        self.search_cls = search_cls                # store search class
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
            data_manager=self.dm,
            model_cls=self.model_cls,
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
    
    def explore_vars(
        self,
        vars_list: List[str],
        plot_type: str = 'line'
    ) -> None:
        """
        Create exploratory plots comparing variables to the target.

        This method helps visualize relationships between potential predictor
        variables and the target variable, useful for feature selection and
        model specification.

        Parameters
        ----------
        vars_list : List[str]
            List of variable names or transformation specifications to explore.
            Each variable will be plotted against the target.
        plot_type : str, default 'line'
            Type of plot to create:
            - 'line': time series plot with dual y-axes
            - 'scatter': scatter plot of variable vs target

        Example
        -------
        >>> # Explore basic variables with line plots
        >>> segment.explore_vars(
        ...     vars_list=["gdp", "inflation", "unemployment"]
        ... )
        >>> 
        >>> # Create scatter plots for transformed variables
        >>> segment.explore_vars(
        ...     vars_list=[
        ...         {"var": "gdp", "transform": "pct_change"},
        ...         {"var": "cpi", "transform": "diff"}
        ...     ],
        ...     plot_type="scatter"
        ... )
        """
        var_dfs = self.dm.build_search_vars(vars_list)
        target_series = self.dm.internal_data[self.target]

        for var_name, df in var_dfs.items():
            # Align df and target to their common index
            common_idx = df.index.intersection(target_series.index)
            df_aligned = df.loc[common_idx]
            ts_aligned = target_series.loc[common_idx]

            cols = df_aligned.columns.tolist()
            n = len(cols)
            ncols = 3
            nrows = math.ceil(n / ncols)
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(5 * ncols, 4 * nrows), squeeze=False
            )
            fig.suptitle(f"{var_name} vs. {self.target}", fontsize=16)

            for idx, col in enumerate(cols):
                row, col_idx = divmod(idx, ncols)
                ax = axes[row][col_idx]

                # set subplot title to the variable name
                ax.set_title(col)

                if plot_type == 'line':
                    # primary vs secondary y-axis
                    line1, = ax.plot(
                        ts_aligned.index,
                        ts_aligned,
                        color='tab:blue',
                        label=self.target
                    )
                    ax2 = ax.twinx()
                    line2, = ax2.plot(
                        df_aligned.index,
                        df_aligned[col],
                        color='tab:orange',
                        label=col
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

    def export(
        self,
        output_map: Dict[str, str],
        *args,
        **kwargs
    ) -> None:
        """
        Export segment results to Excel files using the template class.

        This method uses the configured export template class to write model
        results, parameters, and diagnostics to Excel files in a standardized
        format.

        Parameters
        ----------
        output_map : Dict[str, str]
            Mapping from template file paths to output file paths.
            Example: {"template.xlsx": "results.xlsx"}
        *args, **kwargs
            Additional arguments passed to the export template constructor.

        Raises
        ------
        ValueError
            If no export_template_cls was provided during segment initialization.

        Example
        -------
        >>> # Basic export using default template
        >>> segment.export({
        ...     "template.xlsx": "model_results.xlsx"
        ... })
        >>> 
        >>> # Export with custom settings
        >>> segment.export(
        ...     {
        ...         "template.xlsx": "detailed_results.xlsx",
        ...         "summary.xlsx": "summary_results.xlsx"
        ...     },
        ...     include_plots=True,
        ...     sheet_name="Model Results"
        ... )
        """
        if not self.export_template_cls:
            raise ValueError("No export_template_cls provided for exporting.")
        exporter = self.export_template_cls(self.cms, *args, **kwargs)
        exporter.export(output_map)

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
        rank_weights: Tuple[float, float, float] = (1, 1, 1),
        test_update_func: Optional[Callable] = None,
        outlier_idx: Optional[List[Any]] = None,
        add_in: bool = True,
        override: bool = False,
        re_rank: bool = True
    ) -> List[CM]:
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
        List[CM]
            List of the top_n best performing CM instances.

        Example
        -------
        >>> # Basic search with default parameters
        >>> top_models = segment.search_cms(
        ...     desired_pool=["gdp", "inflation", "unemployment"]
        ... )
        >>> 
        >>> # Search and override existing models
        >>> top_models = segment.search_cms(
        ...     desired_pool=["new_var1", "new_var2"],
        ...     override=True  # clears existing models first
        ... )
        >>> 
        >>> # Search and add without re-ranking
        >>> top_models = segment.search_cms(
        ...     desired_pool=["additional_var"],
        ...     re_rank=False  # just append with unique IDs
        ... )
        >>> 
        >>> # Advanced search with re-ranking
        >>> top_models = segment.search_cms(
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
            self.searcher = self.search_cls(self.dm, self.target, self.model_cls)
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
            outlier_idx=outlier_idx
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
                    # Combine new top_cms with existing cms for re-ranking
                    all_cms = list(self.cms.values()) + self.top_cms
                    
                    # Keep track of newly searched CM objects for final position tracking
                    newly_searched_cms = set(self.top_cms)
                    
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
                    
                    # Assign final IDs based on ranking order
                    for i, temp_id in enumerate(ordered_temp_ids):
                        cm = original_ids[temp_id]
                        new_id = f"cm{i+1}"
                        cm.model_id = new_id
                        self.cms[new_id] = cm
                        
                        # Track final position if this was a newly searched CM
                        if cm in newly_searched_cms:
                            newly_searched_final_positions.append(new_id)
                    
                    print(f"\n=== Re-ranked All Models ===")
                    print(f"Total models after re-ranking: {len(self.cms)}")
                    print(f"Newly searched models ranked at: {', '.join(newly_searched_final_positions)}")
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

        return self.top_cms
