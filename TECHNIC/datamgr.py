# TECHNIC/datamgr.py
import pandas as pd
from typing import List, Callable, Optional, Dict

from data.internal import InternalDataLoader
from data.mev import MEVLoader


class DataManager:
    """
    Manage and combine internal and MEV data for modeling.

    - Loads and stores internal and MEV data via their loaders or directly via parameters.
    - Interpolates MEV tables to match internal data frequency.
    - Applies transformations to MEV tables.
    - Selects specified drivers from both data sources.
    """
    def __init__(
        self,
        internal_loader: Optional[InternalDataLoader] = None,
        mev_loader: Optional[MEVLoader] = None,
        internal_source: Optional[str] = None,
        internal_df: Optional[pd.DataFrame] = None,
        internal_date_col: Optional[str] = None,
        internal_start: Optional[str] = None,
        internal_end: Optional[str] = None,
        internal_freq: str = 'M',
        model_workbook: Optional[str] = None,
        model_sheet: Optional[str] = None,
        scenario_workbooks: Optional[List[str]] = None,
        scenario_sheets: Optional[Dict[str, str]] = None,
    ):
        if internal_loader is None:
            internal_loader = InternalDataLoader(
                source=internal_source,
                df=internal_df,
                date_col=internal_date_col,
                start=internal_start,
                end=internal_end,
                freq=internal_freq,
            )
        internal_loader.load()
        self._internal_loader = internal_loader
        self.internal_data = internal_loader.internal_data

        if mev_loader is None:
            if model_workbook is None or model_sheet is None:
                raise ValueError("model_workbook and model_sheet must be provided if mev_loader is not.")
            mev_loader = MEVLoader(
                model_workbook=model_workbook,
                model_sheet=model_sheet,
                scenario_workbooks=scenario_workbooks,
                scenario_sheets=scenario_sheets,
            )
        mev_loader.load()
        self._mev_loader = mev_loader
        self.model_mev = mev_loader.model_mev
        self.scenario_mevs = mev_loader.scenario_mevs

    def interpolate_mevs(self, freq: str = 'M'):
        current_freq = pd.infer_freq(self.internal_data.index)
        if current_freq != freq:
            raise ValueError(f"Internal data frequency is not {freq}")
        target_idx = self.internal_data.index.normalize()
        self.model_mev = self._interpolate_df(self.model_mev, target_idx)
        for scen, df in self.scenario_mevs.items():
            self.scenario_mevs[scen] = self._interpolate_df(df, target_idx)

    def apply_to_mevs(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        self._mev_loader.apply_to_all(func)
        self.model_mev = self._mev_loader.model_mev
        self.scenario_mevs = self._mev_loader.scenario_mevs

    def get_drivers(self, driver_names: List[str]) -> pd.DataFrame:
        parts = []
        int_cols = [c for c in driver_names if c in self.internal_data.columns]
        if int_cols:
            parts.append(self.internal_data[int_cols])
        mev_cols = [c for c in driver_names if c in self.model_mev.columns]
        if mev_cols:
            parts.append(self.model_mev[mev_cols])
        if not parts:
            raise ValueError("No driver names found in internal or MEV data.")
        result = pd.concat(parts, axis=1)
        result.index = result.index.normalize()
        return result

    @staticmethod
    def _interpolate_df(df: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
        df_interp = df.copy()
        df_interp.index = pd.to_datetime(df_interp.index).normalize()
        df_interp = df_interp.reindex(target_idx)
        df_interp = df_interp.interpolate(method='cubic')
        df_interp.index = df_interp.index.normalize()
        return df_interp

    @property
    def source(self) -> Optional[str]:
        return self._internal_loader.source

    @property
    def raw_df(self) -> Optional[pd.DataFrame]:
        return self._internal_loader.raw_df

    @property
    def date_col(self) -> Optional[str]:
        return self._internal_loader.date_col

    @property
    def start(self) -> Optional[str]:
        return self._internal_loader.start

    @property
    def end(self) -> Optional[str]:
        return self._internal_loader.end

    @property
    def freq(self) -> str:
        return self._internal_loader.freq

    @property
    def model_workbook(self) -> str:
        return self._mev_loader.model_workbook

    @property
    def model_sheet(self) -> str:
        return self._mev_loader.model_sheet

    @property
    def scenario_workbooks(self) -> List[str]:
        return self._mev_loader.scenario_workbooks

    @property
    def scenario_sheets(self) -> Dict[str, str]:
        return self._mev_loader.scenario_sheets

    @property
    def model_map(self) -> Dict[str, str]:
        return self._mev_loader.model_map

    @property
    def scenario_maps(self) -> Dict[str, Dict[str, str]]:
        return self._mev_loader.scenario_maps