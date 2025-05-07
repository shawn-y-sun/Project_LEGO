# tech/featureBuilder.py

import os
import itertools
import functools
from typing import List, Union, Dict
from pathlib import Path
import pandas as pd
import yaml
from .transform import TSFM
from . import transform as tf
import warnings

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

# Patch TSFM repr for readability
def _tsfm_repr(self):
    """Custom repr handling partial transform functions."""
    fn = self.transform_fn
    # Handle functools.partial
    if isinstance(fn, functools.partial):
        fname = fn.func.__name__
    else:
        fname = getattr(fn, '__name__', 'transform')
    return f"TSFM(variable='{self.feature_name}', transform_fn='{fname}', max_lag={self.max_lag})"

TSFM.__repr__ = _tsfm_repr

class FeatureBuilder:
    """
    Builds feature combinations based on MEV types and YAML-driven transforms.
    """
    def __init__(
        self,
        max_var_num: int,
        forced_in: List[Union[str, TSFM]],
        driver_pool: List[str],
        desired_pool: List[str],
        max_lag: int = 2
    ):
        # Core parameters
        self.max_var_num = max_var_num
        self.driver_pool = driver_pool
        self.desired_pool = set(desired_pool)
        self.max_lag = max_lag
        self.lags = range(max_lag + 1)

        # Global config maps
        self.mev_type_map = MEV_TYPE_MAP
        self.type_tsfm_map = TYPE_TSFM_MAP

        # Build forced options: one TSFM choice per forced MEV
        self.forced_mevs: List[str] = []
        self.forced_options: Dict[str, List[TSFM]] = {}
        for item in forced_in:
            if isinstance(item, TSFM):
                name = item.feature_name
                self.forced_mevs.append(name)
                self.forced_options[name] = [item]
            else:
                name = item
                self.forced_mevs.append(name)
                self.forced_options[name] = self._build_tsfm_list(name)

        # Build optional pools for driver_pool
        self.options: Dict[str, List[TSFM]] = {}
        for mev in self.driver_pool:
            if mev not in self.forced_mevs:
                self.options[mev] = self._build_tsfm_list(mev)

    def _build_tsfm_list(self, mev: str) -> List[TSFM]:
        """
        Create TSFM instances for a given MEV based on its type and transforms.
        """
        transforms = self.type_tsfm_map.get(self.mev_type_map.get(mev, ''), [])
        tsfms: List[TSFM] = []
        for tkey in transforms:
            fn = getattr(tf, tkey, None)
            if fn is None:
                continue
            if tkey == 'LV':
                for lag in self.lags:
                    tsfms.append(TSFM(mev, fn, lag))
            elif tkey in ('DF', 'GR'):
                for lag in self.lags:
                    periods = lag if lag > 0 else 1
                    part = functools.partial(fn, periods=periods)
                    tsfms.append(TSFM(mev, part, 0))
            else:
                for window in self.lags:
                    part = functools.partial(fn, window=window)
                    tsfms.append(TSFM(mev, part, 0))
        return tsfms

    def generate_combinations(self) -> List[List[TSFM]]:
        """
        Enumerate all valid TSFM feature combinations.
        """
        combos: List[List[TSFM]] = []
        # Iterate over each forced combination choice
        forced_lists = [self.forced_options[m] for m in self.forced_mevs]
        for forced_choice in itertools.product(*forced_lists):
            forced_list = list(forced_choice)
            base_count = len(forced_list)
            extra_max = self.max_var_num - base_count

            # Determine available optional MEVs
            avail = [m for m in self.driver_pool if m not in self.forced_mevs]

            # Enforce desired presence
            forced_names = set(self.forced_mevs)
            required = None
            if not (forced_names & self.desired_pool):
                required = set(avail) & self.desired_pool

            # Enumerate optional subsets
            for r in range(extra_max + 1):
                for subset in itertools.combinations(avail, r):
                    if required and not (set(subset) & required):
                        continue
                    # Cartesian product of chosen option transforms
                    pools = [self.options[m] for m in subset]
                    for prod in itertools.product(*pools):
                        combos.append(forced_list + list(prod))
        return combos

    def __repr__(self) -> str:
        return (
            f"<FeatureBuilder max_var_num={self.max_var_num}, "
            f"forced={self.forced_mevs}, "
            f"desired={list(self.desired_pool)}, max_lag={self.max_lag}>"
        )

