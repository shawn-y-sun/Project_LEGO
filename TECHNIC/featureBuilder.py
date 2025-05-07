# tech/featureBuilder.py
"""
FeatureBuilder: generate TSFM feature combinations based on MEV types and YAML-driven transforms.

Usage:
    from tech.featureBuilder import FeatureBuilder
    fb = FeatureBuilder(
        max_var_num=3,
        forced_in=['GDP'],
        driver_pool=['GDP', 'Unemp'],
        desired_pool=['Unemp'],
        max_lag=1
    )
    combos = fb.generate_combinations()
"""
import itertools
import functools
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import yaml
from .transform import TSFM
from . import transform as tf

# -----------------------------------------------------------------------------
# Configuration file paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MEV_TYPE_CSV = BASE_DIR / 'support' / 'mev_type.csv'
TSFM_YAML = BASE_DIR / 'support' / 'type_tsfm.yaml'

# -----------------------------------------------------------------------------
# Load MEV type mapping and transformation mapping 
# -----------------------------------------------------------------------------
if not MEV_TYPE_CSV.exists():
    raise FileNotFoundError(f"Missing MEV type map: {MEV_TYPE_CSV}")
_mev_df = pd.read_csv(MEV_TYPE_CSV)
_REQUIRED = {'mev_code', 'type'}
if not _REQUIRED.issubset(_mev_df.columns):
    raise ValueError(f"mev_type.csv must contain {_REQUIRED}")
MEV_TYPE_MAP: Dict[str, str] = dict(zip(_mev_df['mev_code'], _mev_df['type']))

if not TSFM_YAML.exists():
    raise FileNotFoundError(f"Missing transform spec: {TSFM_YAML}")
_yaml = yaml.safe_load(TSFM_YAML.read_text())
TRANSFORM_MAP = _yaml.get('transforms')
if not isinstance(TRANSFORM_MAP, dict):
    raise ValueError("type_tsfm.yaml must define a 'transforms' mapping")

# -----------------------------------------------------------------------------
# Patch TSFM.__repr__ to handle functools.partial
# -----------------------------------------------------------------------------
def _tsfm_repr(self):
    fn = self.transform_fn
    if isinstance(fn, functools.partial):
        name = fn.func.__name__
    else:
        name = getattr(fn, '__name__', 'transform')
    return f"TSFM(variable='{self.feature_name}', transform_fn='{name}', max_lag={self.max_lag})"

TSFM.__repr__ = _tsfm_repr

# -----------------------------------------------------------------------------
# FeatureBuilder Class
# -----------------------------------------------------------------------------
class FeatureBuilder:
    """
    Generate valid feature combinations as lists of TSFM instances.
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
        self.transform_map = TRANSFORM_MAP

        # Build forced transforms: one choice per forced MEV
        self.forced_mevs: List[str] = []
        self.forced_options: Dict[str, List[TSFM]] = {}
        for item in forced_in:
            if isinstance(item, TSFM):
                key = item.feature_name
                self.forced_options[key] = [item]
            else:
                key = item
                self.forced_options[key] = self._gen_tsfms(key)
            self.forced_mevs.append(key)

        # Build optional transforms for driver_pool
        self.options: Dict[str, List[TSFM]] = {
            mev: self._gen_tsfms(mev)
            for mev in self.driver_pool
            if mev not in self.forced_mevs
        }

    def _gen_tsfms(self, mev: str) -> List[TSFM]:
        """
        Generate TSFM instances for a given MEV across all transforms and lags.
        """
        mtype = self.mev_type_map.get(mev, '')
        transforms = self.transform_map.get(mtype, [])
        tsfms: List[TSFM] = []
        for tkey in transforms:
            fn = getattr(tf, tkey, None)
            if not fn:
                continue
            for lag in self.lags:
                # Wrap partials for DF/GR and default window transforms
                if tkey == 'LV':
                    transform_fn = fn
                elif tkey in ('DF', 'GR'):
                    transform_fn = functools.partial(fn, periods=1)
                else:
                    transform_fn = functools.partial(fn, window=1)
                tsfms.append(TSFM(mev, transform_fn, lag))
        return tsfms

    def generate_combinations(self) -> List[List[TSFM]]:
        """
        Enumerate all valid TSFM feature combinations.
        """
        combos: List[List[TSFM]] = []
        # Cartesian product over forced options
        forced_lists = [self.forced_options[m] for m in self.forced_mevs]
        for forced_choice in itertools.product(*forced_lists):
            forced_list = list(forced_choice)
            remaining = self.max_var_num - len(forced_list)

            # Available optional MEVs
            avail = [m for m in self.driver_pool if m not in self.forced_mevs]

            # Enforce desired presence if not already in forced
            required = None
            if not set(self.forced_mevs) & self.desired_pool:
                required = set(avail) & self.desired_pool

            # Choose up to 'remaining' optional features
            for r in range(remaining + 1):
                for subset in itertools.combinations(avail, r):
                    if required and not (set(subset) & required):
                        continue
                    # Combine each chosen optional transform
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
