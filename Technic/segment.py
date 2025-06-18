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

    :param segment_id: Unique identifier for this Segment.
    :param target: Name of the target variable.
    :param data_manager: DataManager instance.
    :param model_cls: ModelBase subclass to use for model fitting.
    :param export_template_cls: Optional Excel export template class.
    :param reportset_cls: Class for assembling report sets.
    :param search_cls: Class to use for exhaustive model search (default: ModelSearch).
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
        Instantiate and fit a CM for the given cm_id and specs.

        :param cm_id: Unique identifier for this candidate model.
        :param specs: Feature specification passed to DataManager.
        :param sample: Which sample to build ('in', 'full', 'both').
        :return: The constructed and fitted CM instance.
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
        Remove candidate model(s) with the given ID(s) from this segment.

        :param cm_ids: a single model ID or list of model IDs to remove
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
        Display consolidated reports for one or multiple CMs using ReportSet.

        :param cm_ids: List of CM IDs to include (defaults to all in the segment).
        :param report_sample: Which sample report to use: 'in' or 'full' (default 'in').
        :param show_out: Whether to include out-of-sample results.
        :param show_params: Whether to include parameter tables.
        :param show_tests: Whether to include diagnostic test results.
        :param show_scens: Whether to include scenario forecast and variable plots.
        :param perf_kwargs: Additional kwargs for performance display.
        :param params_kwargs: Additional kwargs for parameter tables.
        :param test_kwargs: Additional kwargs for test display.
        :param scen_kwargs: Additional kwargs for scenario plotting.
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
        For each variable in vars_list, build transformed DataFrames via DataManager,
        then plot each feature column against the target variable.

        :param vars_list: list of variable names (or TSFM specs) to explore
        :param plot_type: 'line' for time-series or 'scatter' for scatter plot
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
        Export segment results via the provided ExportTemplateBase subclass.

        Parameters
        ----------
        output_map : Dict[str, str]
            Mapping from template file paths to output file paths.
        *args, **kwargs : passed to the export template constructor.
        """
        if not self.export_template_cls:
            raise ValueError("No export_template_cls provided for exporting.")
        exporter = self.export_template_cls(self.cms, *args, **kwargs)
        exporter.export(output_map)


    def search_cms(
        self,
        desired_pool: List[Union[str, Any]],
        forced_in: Optional[List[Union[str, Any]]] = None,
        top_n: int = 5,
        sample: str = 'in',
        max_var_num: int = 5,
        max_lag: int = 3,
        max_periods: int = 3,
        rank_weights: Tuple[float, float, float] = (1, 1, 1),
        test_update_func: Optional[Callable] = None,
        outlier_idx: Optional[List[Any]] = None,
        add_in: bool = True
    ) -> List[CM]:
        """
        Run an exhaustive search over feature-spec combinations.

        :param desired_pool: List of variables or TSFM specs to consider.
        :param forced_in:  List of vars/TSFMs always included (treated as one group if provided).
        :param top_n:      Number of top models to retain based on ranking.
        :param sample:     Which sample to build ('in' or 'full').
        :param max_var_num: Maximum number of features per model.
        :param max_lag:    Max lag to consider in TSFM specs.
        :param max_periods:Max periods to consider in TSFM specs.
        :param rank_weights:
                             Weights for (Fit Measures, IS Error, OOS Error) when ranking.
        :param test_update_func:
                             Optional function to update each CM’s testset.
        :param outlier_index:
                            List of index labels (e.g. timestamps or keys) corresponding to outliers
        :param add_in:     If True, add the resulting top CMs into `self.cms`.
        :return:           List of the top_n passing CM instances.
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
            rank_weights=rank_weights,
            test_update_func=test_update_func,
            outlier_idx=outlier_idx
        )

        # 3) Collect the top_n results from the searcher
        self.top_cms = self.searcher.top_cms[:top_n]

        # 4) Optionally add them to this segment’s cms
        if add_in:
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
