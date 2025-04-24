# TECHNIC/segment.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from typing import (
    Type, Dict, List, Union, Optional, Any
)
from .datamgr import DataManager
from .cm import CM
from .model import ModelBase
from .measure import MeasureBase
from .transform import TSFM
from .report import ModelReportBase, SegmentReportBase
from .template import ExportTemplateBase

class Segment:
    """
    Group of CMs (candidate models) for a single product segment.

    Parameters:
    ----------
    segment_id: unique identifier for the segment
    target: name of the dependent variable used by all CMs
    data_manager: DataManager instance with all data loaded
    model_cls: subclass of ModelBase (e.g. OLS)
    measure_cls: subclass of MeasureBase (e.g. OLSMeasures)
    report_cls: optional subclass of ModelReportBase (e.g. OLSReport)
    export_template_cls: optional subclass of ExportTemplateBase to handle exports
    """
    def __init__(
        self,
        segment_id: str,
        target: str,
        data_manager: DataManager,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
        report_cls: Optional[Type[ModelReportBase]] = None,
        export_template_cls: Optional[Type[ExportTemplateBase]] = None,
    ):
        self.segment_id = segment_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.measure_cls = measure_cls
        self.report_cls = report_cls
        self.export_template_cls = export_template_cls
        self.cms: Dict[str, CM] = {}

    def build_cm(
        self,
        cm_id: str,
        specs: List[Union[str, Dict[str, TSFM]]]
    ) -> CM:
        """
        Create and build a new CM for this segment.

        - cm_id: unique identifier for this CM
        - specs: list of var names or {var: TSFM} dicts (nested lists OK)
        """
        cm = CM(
            model_id=cm_id,
            data_manager=self.dm,
            model_cls=self.model_cls,
            measure_cls=self.measure_cls,
            target=self.target
        )
        cm.build(specs)
        self.cms[cm_id] = cm
        return cm
    
    def show_report(
        self,
        show_full: bool = False,
        show_tests: bool = False,
        perf_kwargs: Dict[str, Any] = None,
        test_kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Delegate to the configured SegmentReportBase subclass to display
        segment-level performance and tests, mirroring CM.show_report.

        Parameters
        ----------
        show_full : bool
            If True, includes full-sample performance and out-of-sample tables/plots.
        show_tests : bool
            If True, includes diagnostic test tables/plots.
        perf_kwargs : dict, optional
            Keyword args for performance plots/tables.
        test_kwargs : dict, optional
            Keyword args for diagnostic test plots/tables.
        """
        if self.report_cls is None:
            raise ValueError("No report class specified for this segment.")
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # Instantiate the segment report
        seg_report = self.report_cls(self.cms)
        # Delegate to its show_report
        seg_report.show_report(
            show_out=show_full,
            show_tests=show_tests,
            perf_kwargs=perf_kwargs,
            test_kwargs=test_kwargs
        )
    
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