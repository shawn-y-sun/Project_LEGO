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
from typing import List, Dict, Union, Tuple

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

# Patch TSFM repr
def _tsfm_repr(self):
    fn = self.transform_fn
    name = fn.func.__name__ if isinstance(fn, functools.partial) else getattr(fn, '__name__', 'transform')
    return f"TSFM(variable='{self.feature_name}', transform_fn='{name}', max_lag={self.max_lag})"
TSFM.__repr__ = _tsfm_repr


# ----------------------------------------------------------------------------
# FeatureBuilder class
# ----------------------------------------------------------------------------
class FeatureBuilder:
    """
    Build feature combinations. Supports desired_pool entries that are single MEVs or tuples
    (pairs) which must appear together in any combination.
    """
    def __init__(
        self,
        max_var_num: int,
        forced_in: List[Union[str, TSFM]],
        driver_pool: List[str],
        desired_pool: List[Union[str, Tuple[str, ...]]],
        max_lag: int = 2
    ):
        # Core settings
        self.max_var_num   = max_var_num
        self.driver_pool   = driver_pool
        self.max_lag       = max_lag
        self.lags          = range(max_lag + 1)

        # Normalize desired groups: each entry to a set
        self.desired_groups: List[set] = []
        for d in desired_pool:
            grp = set(d) if isinstance(d, (list, tuple)) else {d}
            self.desired_groups.append(grp)
        self.desired_groups = [g for g in self.desired_groups if g]

        # Build forced options: choose exactly one TSFM per forced MEV
        self.forced_mevs = []  # list of MEV names
        self.forced_options: Dict[str, List[TSFM]] = {}
        for item in forced_in:
            if isinstance(item, TSFM):
                name = item.feature_name
                self.forced_options[name] = [item]
            else:
                name = item
                self.forced_options[name] = self._gen_tsfms(name)
            self.forced_mevs.append(name)

        # Build optional options for non-forced MEVs
        self.options: Dict[str, List[TSFM]] = {
            m: self._gen_tsfms(m)
            for m in driver_pool if m not in self.forced_mevs
        }

    def _gen_tsfms(self, mev: str) -> List[TSFM]:
        """
        Generate TSFM instances for MEV across all transforms and lags.
        """
        mtype = MEV_TYPE_MAP.get(mev, '')
        transforms = TRANSFORM_MAP.get(mtype, [])
        out: List[TSFM] = []
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
                out.append(TSFM(mev, transform_fn, lag))
        return out

    def generate_combinations(self) -> List[List[TSFM]]:
        """
        Enumerate valid feature combos respecting:
          - one choice per forced MEV
          - at most max_var_num total
          - desired_groups: at least one group fully present
          - group cohesion: if any member of group appears, all must appear
        """
        combos: List[List[TSFM]] = []
        # all forced selections
        forced_lists = [self.forced_options[m] for m in self.forced_mevs]
        for forced_choice in itertools.product(*forced_lists):
            base = list(forced_choice)
            base_names = {ts.feature_name for ts in base}
            remaining = self.max_var_num - len(base)
            # optional MEVs
            avail = [m for m in self.driver_pool if m not in self.forced_mevs]

            # for desired: check if any group already satisfied
            groups = self.desired_groups
            def group_satisfied(selected_names: set) -> bool:
                return any(g.issubset(selected_names) for g in groups)
            # cohesion check
            def group_cohesive(selected_names: set) -> bool:
                return all((not (selected_names & g) or g.issubset(selected_names)) for g in groups)

            # must satisfy desired: at least one group
            need_groups = not group_satisfied(base_names)

            # choose optionals
            for r in range(remaining + 1):
                for subset in itertools.combinations(avail, r):
                    names = base_names.union(subset)
                    # enforce group cohesion
                    if not group_cohesive(names):
                        continue
                    # enforce desired
                    if need_groups and not group_satisfied(names):
                        continue
                    # combine transforms
                    pools = [self.options[m] for m in subset]
                    for prod in itertools.product(*pools):
                        combos.append(base + list(prod))
        return combos

    def __repr__(self) -> str:
        return (
            f"<FeatureBuilder max={self.max_var_num}, forced={self.forced_mevs}, "
            f"desired={self.desired_groups}, lag={self.max_lag}>"
        )
