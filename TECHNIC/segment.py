# TECHNIC/segment.py

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

class Segment:
    """
    Group of CMs (candidate models) for a single product segment.

    Parameters:
      segment_id:   unique identifier for the segment
      target:       name of the dependent variable used by all CMs
      data_manager: DataManager instance with all data loaded
      model_cls:    subclass of ModelBase (e.g. OLS)
      measure_cls:  subclass of MeasureBase (e.g. OLSMeasures)
      report_cls:   optional subclass of ModelReportBase (e.g. OLSReport)
    """
    def __init__(
        self,
        segment_id: str,
        target: str,
        data_manager: DataManager,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
        report_cls: Optional[Type[SegmentReportBase]] = None,
    ):
        self.segment_id  = segment_id
        self.target      = target
        self.dm          = data_manager
        self.model_cls   = model_cls
        self.measure_cls = measure_cls
        self.report_cls  = report_cls  # should be a SegmentReportBase subclass
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
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate segment-level reports: in-sample performance, optional full-sample performance,
        and optional diagnostic tests.

        Parameters
        ----------
        show_full : bool
            If True, include full-sample performance plots.
        show_tests : bool
            If True, include diagnostic test plots.
        **kwargs : Any
            Additional keyword arguments passed to plotting methods.

        Returns
        -------
        Dict[str, Any]
            Dictionary of figures: 'in_sample', optionally 'full_sample', 'tests'.
        """
        if self.report_cls is None:
            raise ValueError("No report class specified for this segment.")
        # Instantiate segment report
        seg_report = self.report_cls(self.cms)
        figs: Dict[str, Any] = {}
        # In-sample performance
        figs['in_sample'] = seg_report.plot_perf(**kwargs)
        # Full-sample performance
        if show_full:
            figs['full_sample'] = seg_report.plot_full_perf(**kwargs)
        # Diagnostic tests
        if show_tests:
            figs['tests'] = seg_report.plot_tests(**kwargs)
        return figs