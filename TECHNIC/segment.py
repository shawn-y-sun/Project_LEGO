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
from .transform import TSFM
from .report import ModelReportBase, SegmentReportBase
from .template import ExportTemplateBase

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
        testset_cls: Optional[Type[Any]] = None,
        report_cls: Optional[Type[Any]] = None,
        export_template_cls: Optional[Type[Any]] = None
    ):
        self.segment_id = segment_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.testset_cls = testset_cls
        self.report_cls = report_cls
        self.export_template_cls = export_template_cls
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
            testset_cls=self.testset_cls,
            report_cls=self.report_cls
        )
        cm.build(specs, sample=sample)
        self.cms[cm_id] = cm
        return cm

    def show_report(
        self,
        cm_ids: Optional[List[str]] = None,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: Optional[Dict[str, Any]] = None,
        test_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Display reports for one or multiple CMs.

        :param cm_ids: List of CM IDs to show (defaults to all in the segment).
        :param show_out: Whether to include out-of-sample results.
        :param show_tests: Whether to include diagnostic test results.
        :param perf_kwargs: Additional kwargs for performance display.
        :param test_kwargs: Additional kwargs for test display.
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}
        cm_ids = cm_ids or list(self.cms.keys())

        for cm_id in cm_ids:
            cm = self.cms[cm_id]
            print(f"--- Segment {self.segment_id}: Report for CM '{cm_id}' ---")
            cm.show_report(
                show_out=show_out,
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