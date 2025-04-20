# TECHNIC/datamgr.py

import pandas as pd
from typing import (
    List, Callable, Optional, Dict, Any
)

from .internal import InternalDataLoader
from .mev import MEVLoader
from .transform import TSFM

class DataManager:
    """
    Manage and combine internal and MEV data for modeling.

    - Builds or accepts InternalDataLoader and MEVLoader.
    - Interpolates MEV tables to match internal data frequency.
    - Applies arbitrary transforms to MEV tables.
    - Builds independent-variable DataFrames from specs.
    - Supports separate in-sample split for modeling (in_sample_end)
      and scenario testing (scen_in_sample_end).
    """
    def __init__(
        self,
        # Internal loader inputs
        internal_loader: Optional[InternalDataLoader] = None,
        internal_source: Optional[str]               = None,
        internal_df: Optional[pd.DataFrame]          = None,
        internal_date_col: Optional[str]             = None,
        internal_start: Optional[str]                = None,
        internal_end: Optional[str]                  = None,
        internal_freq: str                           = 'M',
        # MEV loader inputs
        mev_loader: Optional[MEVLoader]              = None,
        model_workbook: Optional[str]                = None,
        model_sheet: Optional[str]                   = None,
        scen_workbooks: Optional[List[str]]          = None,
        scen_sheets: Optional[Dict[str,str]]         = None,
        # Modeling in-sample cutoff
        in_sample_end: Optional[str]                 = None,
        # Scenario-testing in-sample cutoff
        scen_in_sample_end: Optional[str]            = None,
    ):
        # Internal data
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
        self.internal_data    = internal_loader.internal_data

        # MEV data
        if mev_loader is None:
            if model_workbook is None or model_sheet is None:
                raise ValueError(
                    "model_workbook and model_sheet required if mev_loader not provided"
                )
            mev_loader = MEVLoader(
                model_workbook=model_workbook,
                model_sheet=model_sheet,
                scenario_workbooks=scen_workbooks,
                scenario_sheets=scen_sheets,
            )
        mev_loader.load()
        self._mev_loader = mev_loader
        self.model_mev   = mev_loader.model_mev

        # Cutoff dates (stored but not auto‐split)
        self.in_sample_end      = (
            pd.to_datetime(in_sample_end).normalize()
            if in_sample_end else None
        )
        self.scen_in_sample_end = (
            pd.to_datetime(scen_in_sample_end).normalize()
            if scen_in_sample_end else None
        )

    # Modeling in‑sample/out‑of‑sample splits
    @property
    def internal_in(self) -> pd.DataFrame:
        if self.in_sample_end is None:
            return self.internal_data
        return self.internal_data.loc[: self.in_sample_end]

    @property
    def internal_out(self) -> pd.DataFrame:
        if self.in_sample_end is None:
            return pd.DataFrame()
        return self.internal_data.loc[self.in_sample_end + pd.Timedelta(days=1) :]

    @property
    def model_in(self) -> pd.DataFrame:
        if self.in_sample_end is None:
            return self.model_mev
        return self.model_mev.loc[: self.in_sample_end]

    @property
    def model_out(self) -> pd.DataFrame:
        if self.in_sample_end is None:
            return pd.DataFrame()
        return self.model_mev.loc[self.in_sample_end + pd.Timedelta(days=1) :]

    # Scenario MEVs, trimmed by scen_in_sample_end
    @property
    def scen_mevs(self) -> Dict[str, pd.DataFrame]:
        raw = self._mev_loader.scenario_mevs
        if self.scen_in_sample_end is None:
            return raw
        cutoff = self.scen_in_sample_end
        return {name: df.loc[:cutoff] for name, df in raw.items()}

    def interpolate_mevs(self, freq: str = 'M'):
        current_freq = pd.infer_freq(self.internal_data.index)
        if current_freq != freq:
            raise ValueError(f"Internal data frequency is not {freq}")
        target_idx     = self.internal_data.index.normalize()
        self.model_mev = self._interpolate_df(self.model_mev, target_idx)

    def apply_to_mevs(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        self._mev_loader.apply_to_all(func)
        self.model_mev = self._mev_loader.model_mev

    def build_indep_vars(
        self,
        specs: Any,
        internal_df: Optional[pd.DataFrame] = None,
        mev_df: Optional[pd.DataFrame]      = None
    ) -> pd.DataFrame:
        internal = internal_df or self.internal_data
        mev      = mev_df      or self.model_mev

        def _flatten(items):
            for it in items:
                if isinstance(it, list):
                    yield from _flatten(it)
                else:
                    yield it

        flat_specs = list(_flatten(specs))
        names, transforms = [], {}
        for itm in flat_specs:
            if isinstance(itm, str):
                names.append(itm)
            elif isinstance(itm, dict):
                transforms.update(itm)
            else:
                raise ValueError(f"Invalid spec element: {itm}")

        for var, tsfm in transforms.items():
            if var in internal.columns:
                target = internal
            elif var in mev.columns:
                target = mev
            else:
                raise KeyError(f"Variable '{var}' not found.")
            series      = target[var]
            transformed = tsfm.transform_fn(series)
            col         = transformed.shift(tsfm.max_lag) if tsfm.max_lag > 0 else transformed
            name        = f"{var}_{tsfm.suffix}"
            target[name] = col

        final_cols = names + [f"{v}_{tsfm.suffix}" for v, tsfm in transforms.items()]
        pieces = []
        for col in final_cols:
            if col in internal.columns:
                pieces.append(internal[col])
            elif col in mev.columns:
                pieces.append(mev[col])
            else:
                raise KeyError(f"Column '{col}' not found after transformation.")
        X = pd.concat(pieces, axis=1)
        X.index = X.index.normalize()
        return X

    @staticmethod
    def _interpolate_df(df: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        df2 = df2.reindex(target_idx)
        df2 = df2.interpolate(method='cubic')
        df2.index = df2.index.normalize()
        return df2

    # Delegated loader properties
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
    def scen_workbooks(self) -> List[str]:
        return self._mev_loader.scenario_workbooks

    @property
    def scen_sheets(self) -> Dict[str,str]:
        return self._mev_loader.scenario_sheets

    @property
    def model_map(self) -> Dict[str,str]:
        return self._mev_loader.model_map

    @property
    def scen_maps(self) -> Dict[str,Dict[str,str]]:
        return self._mev_loader.scenario_maps