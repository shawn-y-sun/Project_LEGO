# TECHNIC/datamgr.py
import os
from pathlib import Path
import pandas as pd
import warnings
import yaml
from typing import Any, Dict, List, Optional, Callable, Union

from .internal import InternalDataLoader
from .mev import MEVLoader
from .transform import TSFM
from . import transform as transform_module

# Determine support directory relative to this module file using pathlib
_BASE_DIR = Path(__file__).resolve().parent
_SUPPORT_DIR = _BASE_DIR / 'support'
_MEV_TYPE_XLSX_PATH = _SUPPORT_DIR / 'mev_type.xlsx'
_TYPE_TSFM_YAML_PATH = _SUPPORT_DIR / 'type_tsfm.yaml'


# Load and validate MEV type mapping
if not os.path.exists(_MEV_TYPE_XLSX_PATH):
    raise FileNotFoundError(f"MEV type mapping file not found: {_MEV_TYPE_XLSX_PATH}")
_mev_type_df = pd.read_excel(_MEV_TYPE_XLSX_PATH)
required_cols = {'mev_code', 'type'}
if not required_cols.issubset(_mev_type_df.columns):
    raise ValueError(f"Expected columns {required_cols} in {_MEV_TYPE_XLSX_PATH}, got {_mev_type_df.columns.tolist()}")
MEV_TYPE_MAP: Dict[str, str] = dict(zip(_mev_type_df['mev_code'], _mev_type_df['type']))

# Load and validate transform specifications
if not os.path.exists(_TYPE_TSFM_YAML_PATH):
    raise FileNotFoundError(f"Transform specification file not found: {_TYPE_TSFM_YAML_PATH}")
with open(_TYPE_TSFM_YAML_PATH) as _f:
    _tf_spec = yaml.safe_load(_f)
if 'transforms' not in _tf_spec or not isinstance(_tf_spec['transforms'], dict):
    raise KeyError(f"Key 'transforms' missing or invalid format in {_TYPE_TSFM_YAML_PATH}")
TYPE_TSFM_MAP: Dict[str, List[str]] = _tf_spec['transforms']

warnings.simplefilter(action="ignore", category=FutureWarning)

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
        model_mev_source: Optional[Dict[str, str]]   = None,
        scen_mevs_source: Optional[Dict[str, Dict[str, str]]] = None,
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
            if model_mev_source is None or scen_mevs_source is None:
                raise ValueError(
                    "model_mev_source and scen_mevs_source required if mev_loader not provided"
                )
            mev_loader = MEVLoader(
                model_mev_source=model_mev_source,
                scen_mevs_source=scen_mevs_source,
            )
        mev_loader.load()
        self._mev_loader = mev_loader

        # Cutoff dates (stored but not auto-split)
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

    def apply_to_mevs(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        self._mev_loader.apply_to_all(func)

    def build_indep_vars(
        self,
        specs: Any,
        internal_df: Optional[pd.DataFrame] = None,
        mev_df: Optional[pd.DataFrame]      = None
    ) -> pd.DataFrame:
        """
        Build independent-variable DataFrame from specs, applying TSFM transforms.

        :param specs: list of feature names (str) or TSFM instances.
        :param internal_df: override for internal data.
        :param mev_df: override for MEV data.
        """
        internal = internal_df or self.internal_data
        mev      = mev_df      or self.model_mev

        def _flatten(items):
            for it in items:
                if isinstance(it, list):
                    yield from _flatten(it)
                else:
                    yield it

        flat_specs = list(_flatten(specs))
        raw_names: List[str] = []
        transformers: List[TSFM] = []

        for itm in flat_specs:
            if isinstance(itm, str):
                raw_names.append(itm)
            elif isinstance(itm, TSFM):
                transformers.append(itm)
            else:
                raise ValueError(f"Invalid spec element: {itm!r}")

        pieces = []

        # 1) add raw features
        for name in raw_names:
            if name in internal.columns:
                pieces.append(internal[name])
            elif name in mev.columns:
                pieces.append(mev[name])
            else:
                raise KeyError(f"Column '{name}' not found in internal or MEV data.")

        # 2) apply each TSFM
        for tsfm in transformers:
            # pick up the series
            if tsfm.feature is not None:
                series = tsfm.feature
            else:
                fn = tsfm.feature_name
                if fn in internal.columns:
                    series = internal[fn]
                elif fn in mev.columns:
                    series = mev[fn]
                else:
                    raise KeyError(f"Variable '{fn}' not found for transformation.")
                # inject it so apply_transform() works
                tsfm.feature = series

            # apply transform + lag
            col = tsfm.apply()
            pieces.append(col)

        # concat and normalize
        X = pd.concat(pieces, axis=1)
        X.index = X.index.normalize()
        return X

    def build_search_vars(
        self,
        specs: List[Union[str, TSFM]],
        mev_type_map: Dict[str, str] = MEV_TYPE_MAP,
        type_tsfm_map: Dict[str, List[str]] = TYPE_TSFM_MAP
    ) -> Dict[str, pd.DataFrame]:
        """
        Build DataFrames for each variable based on specs.
        Returns a dict mapping variable name to its DataFrame of raw and transformed features.
        Warns if any variable has no type mapping and builds raw variable only in that case.
        """
        var_df_map: Dict[str, pd.DataFrame] = {}
        missing_vars: List[str] = []

        for spec in specs:
            # Determine TSFM list and variable name
            if isinstance(spec, TSFM):
                var_name = spec.feature_name
                tsfms = [spec]
            elif isinstance(spec, str):
                var_name = spec
                var_type = mev_type_map.get(spec)
                if var_type is None:
                    missing_vars.append(spec)
                    # No type mapping → build raw variable only
                    tsfms = [spec]
                else:
                    tf_names = type_tsfm_map.get(var_type)
                    if not tf_names:
                        raise KeyError(f"No transforms for type '{var_type}'.")
                    tsfms = [TSFM(spec, getattr(transform_module, name)) for name in tf_names]
            else:
                raise ValueError(f"Invalid spec: {spec!r}")

            # Build DataFrame for this variable
            var_df_map[var_name] = self.build_indep_vars(tsfms)

        if missing_vars:
            warnings.warn(
                f"No type mapping found for variables: {', '.join(missing_vars)}. "
                "Building raw variables only for those.",
                UserWarning
            )

        return var_df_map
    
    @staticmethod
    def _interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate quarterly MEV DataFrames to match monthly frequency.
        If df has quarterly frequency, reindex from its own min to max with monthly freq
        and cubic-interpolate; otherwise return original df.
        """
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        freq_mev = pd.infer_freq(df2.index)
        # Only interpolate Q -> M
        if freq_mev and freq_mev.startswith('Q'):
            start = df2.index.min()
            end = df2.index.max()
            target_idx = pd.date_range(start=start, end=end, freq='M')
            df2 = df2.reindex(target_idx).astype(float)
            df2 = df2.interpolate(method='cubic')
            df2.index = df2.index.normalize()
            return df2
        return df

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
    def model_mev(self) -> pd.DataFrame:
        """
        Model MEV DataFrame, interpolated to match monthly frequency when needed.
        """
        df = self._mev_loader.model_mev
        return self._interpolate_df(df)

    @property
    def model_map(self) -> Dict[str, str]:
        return self._mev_loader.model_map

    @property
    def scen_mevs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Scenario MEVs interpolated to match monthly frequency.
        Returns nested dict: workbook_key -> {scenario: DataFrame}.
        """
        raw = self._mev_loader.scen_mevs
        interpolated: Dict[str, Dict[str, pd.DataFrame]] = {}
        for key, df_dict in raw.items():
            interp_dict: Dict[str, pd.DataFrame] = {}
            for scen, df in df_dict.items():
                interp_dict[scen] = self._interpolate_df(df)
            interpolated[key] = interp_dict
        return interpolated

    @property
    def scen_maps(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return self._mev_loader.scen_maps