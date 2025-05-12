# tech/featureBuilder.py
"""
FeatureBuilder: generate TSFM feature combinations based on MEV types and YAML-driven transforms.

Usage:
    from tech.featureBuilder import FeatureBuilder
    fb = FeatureBuilder(
        max_var_num=3,
        forced_in=['GDP'],
        driver_pool=['GDP','Unemp'],
        desired_pool=['Unemp', ('A','B')],  # supports single or pair groups
        max_lag=1
    )
    combos = fb.generate_combinations()
"""
import itertools
import functools
from pathlib import Path
from typing import List, Dict, Union, Tuple

import pandas as pd
import yaml
from .transform import TSFM
from . import transform as tf

# ----------------------------------------------------------------------------
# Configuration: load MEV types and transform specs
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MEV_TYPE_CSV = BASE_DIR / 'support' / 'mev_type.csv'
TSFM_YAML    = BASE_DIR / 'support' / 'type_tsfm.yaml'

# Load MEV type map
if not MEV_TYPE_CSV.exists():
    raise FileNotFoundError(f"Missing MEV type map: {MEV_TYPE_CSV}")
_mev_df = pd.read_csv(MEV_TYPE_CSV)
_REQUIRED = {'mev_code', 'type'}
if not _REQUIRED.issubset(_mev_df.columns):
    raise ValueError(f"mev_type.csv must contain {_REQUIRED}")
MEV_TYPE_MAP: Dict[str, str] = dict(zip(_mev_df['mev_code'], _mev_df['type']))

# Load transform spec map
if not TSFM_YAML.exists():
    raise FileNotFoundError(f"Missing transform spec: {TSFM_YAML}")
_yaml = yaml.safe_load(TSFM_YAML.read_text())
TRANSFORM_MAP = _yaml.get('transforms')
if not isinstance(TRANSFORM_MAP, dict):
    raise ValueError("type_tsfm.yaml must define a 'transforms' mapping")

# Patch TSFM repr to handle functools.partial

def _tsfm_repr(self):
    fn = self.transform_fn
    if isinstance(fn, functools.partial):
        name = fn.func.__name__
    else:
        name = getattr(fn, '__name__', 'transform')
    return f"TSFM(variable='{self.feature_name}', transform_fn='{name}', max_lag={self.max_lag})"

TSFM.__repr__ = _tsfm_repr

# ----------------------------------------------------------------------------
# FeatureBuilder class
# ----------------------------------------------------------------------------
class FeatureBuilder:
    """
    Build TSFM feature combinations.

    Parameters:
      max_var_num:    max total features per combo (including forced)
      forced_in:      list of MEV codes or TSFM instances to force in (one per MEV)
      driver_pool:    all MEV codes available for optional features
      desired_pool:   list of MEV codes or tuples/groups that must appear together
      max_lag:        maximum lag to apply to each transform
    """
    def __init__(
        self,
        max_var_num: int=3,
        forced_in: List[Union[str, TSFM]] = [],
        driver_pool: List[str] = [],
        desired_pool: List[Union[str, Tuple[str, ...]]] = [],
        max_lag: int = 2
    ):
        # Ensure required parameters
        if not driver_pool:
            raise ValueError("'driver_pool' must be provided as a non-empty list of Independent Variables.")
        # Core parameters
        # Core parameters
        self.max_var_num = max_var_num
        self.driver_pool = driver_pool
        self.max_lag = max_lag
        self.lags = range(max_lag + 1)

        # Defensive checks
        import warnings
      
        # Configuration maps
        self.mev_type_map = MEV_TYPE_MAP
        self.transform_map = TRANSFORM_MAP

        # Normalize desired_pool entries into sets
        self.desired_pool: List[set] = []
        for d in desired_pool:
            if isinstance(d, set):
                grp = d
            elif isinstance(d, (list, tuple)):
                grp = set(d)
            else:
                grp = {d}
            self.desired_pool.append(grp)

        # Build forced options: each MEV gets a list of TSFM instances
        self.forced_mevs: List[str] = []
        self.forced_options: Dict[str, List[TSFM]] = {}
        for item in forced_in:
            if isinstance(item, TSFM):
                key = item.feature_name
                opts = [item]
            else:
                key = item
                opts = self._gen_tsfms(key)
            self.forced_mevs.append(key)
            self.forced_options[key] = opts

        # Build optional options for non-forced MEVs
        self.options: Dict[str, List[TSFM]] = {
            mev: self._gen_tsfms(mev)
            for mev in self.driver_pool
            if mev not in self.forced_mevs
        }

    def _gen_tsfms(self, mev: str) -> List[TSFM]:
        """
        Generate TSFM instances for a given MEV across transforms and lags.
        If MEV type missing, defaults to only ['LV'] at lag 0.
        """
        mtype = self.mev_type_map.get(mev, '')
        transforms = self.transform_map.get(mtype)
        # Default to identity only if no transforms
        if not transforms:
            return [TSFM(mev, tf.LV, 0)]
        tsfms: List[TSFM] = []
        for tkey in transforms:
            fn = getattr(tf, tkey, None)
            if not fn:
                continue
            for lag in self.lags:
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
        Enumerate valid TSFM feature combinations:
          - one forced transform per forced MEV
          - optional features up to max_var_num total
          - if desired_pool non-empty, at least one group must appear fully
          - enforce group cohesion for desired groups
        """
        combos: List[List[TSFM]] = []
        # Cartesian product of forced choices
        forced_lists = [self.forced_options[mev] for mev in self.forced_mevs]
        for forced_choice in itertools.product(*forced_lists):
            base = list(forced_choice)
            base_names = {ts.feature_name for ts in base}
            remaining = self.max_var_num - len(base)
            avail = [m for m in self.driver_pool if m not in self.forced_mevs]

            # Setup desired-group logic
            if not self.desired_pool:
                def satisfied(names: set) -> bool: return True
                def cohesive(names: set) -> bool: return True
                need_group = False
            else:
                def satisfied(names: set) -> bool:
                    return any(g.issubset(names) for g in self.desired_pool)
                def cohesive(names: set) -> bool:
                    return all(not (names & g) or g.issubset(names) for g in self.desired_pool)
                need_group = not satisfied(base_names)

            # Enumerate optional subsets
            for r in range(remaining + 1):
                for subset in itertools.combinations(avail, r):
                    names = base_names | set(subset)
                    if not cohesive(names) or (need_group and not satisfied(names)):
                        continue
                    pools = [self.options[m] for m in subset]
                    for prod in itertools.product(*pools):
                        combos.append(base + list(prod))
        return combos

    def __repr__(self) -> str:
        return (
            f"<FeatureBuilder max_var_num={self.max_var_num}, "
            f"forced={self.forced_mevs}, "
            f"desired={self.desired_pool}, max_lag={self.max_lag}>"
        )
