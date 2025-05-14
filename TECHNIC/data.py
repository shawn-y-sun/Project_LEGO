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
from .condition import CondVar
import inspect
import functools

# Determine support directory relative to this module file using pathlib
_BASE_DIR = Path(__file__).resolve().parent
_SUPPORT_DIR = _BASE_DIR / 'support'
_MEV_TYPE_XLSX_PATH = _SUPPORT_DIR / 'mev_type.xlsx'
_TYPE_TSFM_YAML_PATH = _SUPPORT_DIR / 'type_tsfm.yaml'

# Load and validate variable type mapping
if not os.path.exists(_MEV_TYPE_XLSX_PATH):
    raise FileNotFoundError(f"Variable type mapping file not found: {_MEV_TYPE_XLSX_PATH}")
_var_type_df = pd.read_excel(_MEV_TYPE_XLSX_PATH)
required_cols = {'mev_code', 'type'}
if not required_cols.issubset(_var_type_df.columns):
    raise ValueError(f"Expected columns {required_cols} in {_MEV_TYPE_XLSX_PATH}, got {_var_type_df.columns.tolist()}")
VAR_TYPE_MAP: Dict[str, str] = dict(zip(_var_type_df['mev_code'], _var_type_df['type']))

# Load and validate transform specifications
if not os.path.exists(_TYPE_TSFM_YAML_PATH):
    raise FileNotFoundError(f"Transform specification file not found: {_TYPE_TSFM_YAML_PATH}")
with open(_TYPE_TSFM_YAML_PATH) as _f:
    _tf_spec = yaml.safe_load(_f)
if 'transforms' not in _tf_spec or not isinstance(_tf_spec['transforms'], dict):
    raise KeyError(f"Key 'transforms' missing or invalid format in {_TYPE_TSFM_YAML_PATH}")
TYPE_TSFM_MAP: Dict[str, List[str]] = _tf_spec['transforms']

warnings.simplefilter(action="ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# DataManager class
# ----------------------------------------------------------------------------

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
        # Modeling in-sample cutoffs
        in_sample_start: Optional[str]               = None,
        in_sample_end: Optional[str]                 = None,
        # Scenario-testing in-sample cutoff
        scen_in_sample_end: Optional[str]            = None,
        # Optional override maps
        var_type_map: Optional[Dict[str, str]]       = None,
        type_tsfm_map: Optional[Dict[str, List[str]]] = None,
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

        # In-sample range
        self.in_sample_start = (
            pd.to_datetime(in_sample_start).normalize()
            if in_sample_start else None
        )
        self.in_sample_end   = (
            pd.to_datetime(in_sample_end).normalize()
            if in_sample_end else None
        )
        # Scenario-testing in-sample cutoff
        self.scen_in_sample_end = (
            pd.to_datetime(scen_in_sample_end).normalize()
            if scen_in_sample_end else None
        )

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

        # Variable type and transform mappings
        self.var_type_map = var_type_map.copy() if var_type_map is not None else VAR_TYPE_MAP.copy()
        self.type_tsfm_map = type_tsfm_map.copy() if type_tsfm_map is not None else TYPE_TSFM_MAP.copy()

        # Interpolate MEV to match freq
        self._model_mev_data = self._interpolate_df(self._mev_loader.model_mev)
        raw_scen = self._mev_loader.scen_mevs
        self._scen_mevs_data: Dict[str, Dict[str, pd.DataFrame]] = {
            key: {scen: self._interpolate_df(df) for scen, df in df_dict.items()}
            for key, df_dict in raw_scen.items()
        }

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
    
    def update_var_type_map(self, new_map: Dict[str, str]) -> None:
        """Merge new variable-type mappings into existing var_type_map."""
        self.var_type_map.update(new_map)

    def update_type_tsfm_map(self, new_map: Dict[str, List[str]]) -> None:
        """Merge new transform-type mappings into existing type_tsfm_map."""
        self.type_tsfm_map.update(new_map)

    def update_mev_maps(
        self,
        new_map: Dict[str, str]
    ) -> None:
        """Merge a set of MEV code→name mappings into both model_map and all scenario maps."""
        # Update model_map
        self._mev_loader._model_map.update(new_map)
        # Update each scenario mapping under each workbook key
        for wb_key, scen_map in self._mev_loader._scen_maps.items():
            scen_map.update(new_map)

    def build_indep_vars(
        self,
        specs: Any,
        internal_df: Optional[pd.DataFrame] = None,
        mev_df: Optional[pd.DataFrame]      = None
    ) -> pd.DataFrame:
        """
        Build independent-variable DataFrame from specs, applying TSFM transforms and CondVar.

        :param specs: list of feature names (str), TSFM instances, or CondVar instances.
        :param internal_df: override for internal data.
        :param mev_df: override for MEV data.
        """
        internal = internal_df if internal_df is not None else self.internal_data
        mev      = mev_df      if mev_df      is not None else self.model_mev

        def _flatten(items):
            for it in items:
                if isinstance(it, list):
                    yield from _flatten(it)
                else:
                    yield it

        flat_specs = list(_flatten(specs))
        pieces: List[pd.Series] = []

        for spec in flat_specs:
            # Conditional variable
            if isinstance(spec, CondVar):
                # set main_series if needed
                if spec.main_series is None:
                    if spec.main_name in internal.columns:
                        spec.main_series = internal[spec.main_name]
                    elif spec.main_name in mev.columns:
                        spec.main_series = mev[spec.main_name]
                    else:
                        raise KeyError(f"CondVar main_var '{spec.main_name}' not found")
                # set cond_var series list
                cond_series_list = []
                for cv in spec.cond_var:
                    if isinstance(cv, pd.Series):
                        cond_series_list.append(cv)
                    elif isinstance(cv, str):
                        if cv in internal.columns:
                            cond_series_list.append(internal[cv])
                        elif cv in mev.columns:
                            cond_series_list.append(mev[cv])
                        else:
                            raise KeyError(f"CondVar cond_var '{cv}' not found")
                    else:
                        raise TypeError("`cond_var` must be a column name or pandas Series")
                spec.cond_var = cond_series_list
                # apply and collect
                pieces.append(spec.apply())
            # Time-series feature transform
            elif isinstance(spec, TSFM):
                # pick up the series
                if spec.feature is not None:
                    series = spec.feature
                else:
                    fn = spec.feature_name
                    if fn in internal.columns:
                        series = internal[fn]
                    elif fn in mev.columns:
                        series = mev[fn]
                    else:
                        raise KeyError(f"Variable '{fn}' not found for transformation.")
                    spec.feature = series
                pieces.append(spec.apply())
            # Raw feature
            elif isinstance(spec, str):
                if spec in internal.columns:
                    pieces.append(internal[spec])
                elif spec in mev.columns:
                    pieces.append(mev[spec])
                else:
                    raise KeyError(f"Column '{spec}' not found in internal or MEV data.")
            else:
                raise ValueError(f"Invalid spec element: {spec!r}")

        # concat and normalize
        X = pd.concat(pieces, axis=1)
        X.index = X.index.normalize()
        return X

    def build_tsfm_specs(
        self,
        specs: List[Union[str, TSFM]],
        var_type_map: Optional[Dict[str, str]] = None,
        type_tsfm_map: Optional[Dict[str, List[str]]] = None,
        max_lag: int = 0,
        max_periods: int = 1
    ) -> Dict[str, List[Union[str, TSFM]]]:
        """
        Generate TSFM specification lists for each variable.
        Returns a mapping { var_name: [TSFM instance, ... ] } or raw string specs if unmapped.

        :param specs: list of variable names or TSFM instances to construct specs for
        :param var_type_map: optional dict mapping variable code to its type; defaults to self.var_type_map
        :param type_tsfm_map: optional dict mapping variable type to list of transform fn names; defaults to self.type_tsfm_map
        :param max_lag: non-negative int, generate TSFM entries for lags 0 through max_lag
        :param max_periods: positive int, for transforms accepting 'periods', generate TSFM entries for periods 1 through max_periods
        """
        if max_lag < 0:
            raise ValueError("max_lag must be >= 0")
        if max_periods < 1:
            raise ValueError("max_periods must be >= 1")

        vt_map = var_type_map or self.var_type_map
        tf_map = type_tsfm_map or self.type_tsfm_map
        specs_map: Dict[str, List[Union[str, TSFM]]] = {}
        missing: List[str] = []

        for spec in specs:
            if isinstance(spec, TSFM):
                specs_map[spec.feature_name] = [spec]

            elif isinstance(spec, str):
                var_name = spec
                vtype = vt_map.get(spec)
                if vtype is None:
                    missing.append(spec)
                    specs_map[var_name] = [spec]
                else:
                    fnames = tf_map.get(vtype, [])
                    tsfms: List[Union[str, TSFM]] = []
                    for name in fnames:
                        fn = getattr(transform_module, name, None)
                        if not callable(fn):
                            continue
                        sig = inspect.signature(fn)
                        pvals = list(range(1, max_periods+1)) if 'periods' in sig.parameters else [None]
                        for p in pvals:
                            base_fn = functools.partial(fn, periods=p) if p else fn
                            for lag in range(max_lag+1):
                                tsfms.append(TSFM(spec, base_fn, lag))
                    specs_map[var_name] = tsfms
            else:
                raise ValueError(f"Invalid spec: {spec!r}")

        if missing:
            warnings.warn(
                f"No type mapping for variables: {missing!r}, using raw-only", UserWarning
            )
        return specs_map

    def build_search_vars(
        self,
        specs: List[Union[str, TSFM]],
        var_type_map: Optional[Dict[str, str]] = None,
        type_tsfm_map: Optional[Dict[str, List[str]]] = None,
        max_lag: int = 0,
        periods: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Build DataFrame for each variable by:
          1) generating TSFM specs via build_tsfm_specs
          2) passing each spec-list to build_indep_vars
        """
        tsfm_specs = self.build_tsfm_specs(
            specs,
            var_type_map=var_type_map,
            type_tsfm_map=type_tsfm_map,
            max_lag=max_lag,
            periods=periods
        )
        var_df_map: Dict[str, pd.DataFrame] = {}
        for var, tsfms in tsfm_specs.items():
            var_df_map[var] = self.build_indep_vars(tsfms)
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
    
    def apply_to_mevs(
        self,
        fn: Callable[
            [pd.DataFrame, pd.DataFrame],
            pd.DataFrame
        ]
    ) -> None:
        '''
        Apply a two-arg function to all MEV tables (model and scenarios).

        The function signature is:
            df_mev = fn(df_mev, internal_df)

        It should return a DataFrame (possibly the same df_mev mutated) with new features.
        These returned columns will be merged back into each MEV table.
        '''
        internal_df = self.internal_data

        def _apply_mev(mev_df: pd.DataFrame):
            result = fn(mev_df.copy(), internal_df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"apply_to_mevs: fn must return a DataFrame, got {type(result)}")
            for col in result.columns:
                mev_df[col] = result[col].reindex(mev_df.index)

        # Apply to main MEV
        _apply_mev(self._model_mev_data)

        # Apply to each scenario
        for wb_key, scen_map in self._scen_mevs_data.items():
            for scen_name, df in scen_map.items():
                _apply_mev(df)
    
    def apply_to_internal(
        self,
        fn: Callable[
            [pd.DataFrame],
            Optional[Union[pd.Series, pd.DataFrame]]
        ]
    ) -> None:
        '''
        Apply a one-arg feature-engineering function to:
          • self.internal_data

        The function signature is:
            ret = fn(internal_df)

        - If ret is None, assume fn performed in-place mutations on internal_df.
        - If ret is a Series or DataFrame, merge its columns back into internal_data only.
        '''
        internal_df = self.internal_data

        ret = fn(internal_df)
        if ret is None:
            return
        if isinstance(ret, pd.Series):
            internal_df[ret.name] = ret.reindex(internal_df.index)
        elif isinstance(ret, pd.DataFrame):
            for col in ret.columns:
                internal_df[col] = ret[col].reindex(internal_df.index)
        else:
            raise TypeError(
                f"apply_to_internal(): fn must return None, Series or DataFrame, got {type(ret)}"
            )

    @property
    def model_mev(self) -> pd.DataFrame:
        """
        Cached model MEV DataFrame.
        """
        return self._model_mev_data

    @property
    def model_map(self) -> Dict[str, str]:
        return self._mev_loader.model_map

    @property
    def scen_mevs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Cached scenario MEV DataFrames.
        """
        return self._scen_mevs_data

    @property
    def scen_maps(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return self._mev_loader.scen_maps