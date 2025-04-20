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
from .report import ModelReportBase

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
        report_cls: Optional[Type[ModelReportBase]] = None,
    ):
        self.segment_id  = segment_id
        self.target      = target
        self.dm          = data_manager
        self.model_cls   = model_cls
        self.measure_cls = measure_cls
        self.report_cls  = report_cls
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
            report_cls=self.report_cls,
            target=self.target
        )
        cm.build(specs)
        self.cms[cm_id] = cm
        return cm

    def compare_perf(self, sample: str = 'in') -> pd.DataFrame:
        """
        Return a DataFrame comparing perf metrics of all CMs.

        - sample: 'in' or 'full'
        """
        records = []
        for cm_id, cm in self.cms.items():
            perf = (cm.measure_in.performance_measures if sample == 'in'
                    else cm.measure_full.performance_measures)
            perf['cm_id'] = cm_id
            records.append(perf)
        df = pd.DataFrame(records).set_index('cm_id')
        return df

    def compare_tests(self, sample: str = 'in') -> pd.DataFrame:
        """
        Return a DataFrame comparing testing metrics of all CMs.

        - sample: 'in' or 'full'
        """
        records = []
        for cm_id, cm in self.cms.items():
            tests = (cm.measure_in.testing_measures if sample == 'in'
                     else cm.measure_full.testing_measures)
            tests['cm_id'] = cm_id
            records.append(tests)
        df = pd.json_normalize(records).set_index('cm_id')
        return df
