# TECHNIC/segment.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Type, Dict, List, Optional, Any

from .cm import CM
from .model import ModelBase
from .template import ExportTemplateBase
from .report import ReportSet


class Segment:
    """
    Manages a collection of Candidate Models (CM) and their reporting/export.
    """
    def __init__(
        self,
        segment_id: str,
        target: str,
        data_manager: Any,
        model_cls: Type[ModelBase],
        export_template_cls: Optional[Type[ExportTemplateBase]] = None,
        reportset_cls: Type[ReportSet] = ReportSet
    ):
        self.segment_id = segment_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.export_template_cls = export_template_cls
        self.reportset_cls = reportset_cls
        self.cms: Dict[str, CM] = {}

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
    
    def remove_cm(self, cm_ids: List[str]) -> None:
        """
        Remove candidate models with the given IDs from this segment.

        :param cm_ids: list of model IDs to remove from self.cms
        """
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
        perf_kwargs: Optional[Dict[str, Any]] = None,
        params_kwargs: Optional[Dict[str, Any]] = None,
        test_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Display consolidated reports for one or multiple CMs using ReportSet.

        :param cm_ids: List of CM IDs to include (defaults to all in the segment).
        :param report_sample: Which sample report to use: 'in' or 'full' (default 'in').
        :param show_out: Whether to include out-of-sample results.
        :param show_params: Whether to include parameter tables.
        :param show_tests: Whether to include diagnostic test results.
        :param perf_kwargs: Additional kwargs for performance display.
        :param params_kwargs: Additional kwargs for parameter tables.
        :param test_kwargs: Additional kwargs for test display.
        """
        perf_kwargs = perf_kwargs or {}
        params_kwargs = params_kwargs or {}
        test_kwargs = test_kwargs or {}
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
            perf_kwargs=perf_kwargs,
            params_kwargs=params_kwargs,
            test_kwargs=test_kwargs
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
