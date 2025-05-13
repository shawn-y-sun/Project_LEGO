# tech/featureBuilder.py
"""
FeatureBuilder: generate TSFM feature combinations based solely on forced_in and desired_pool.

- `forced_in`: list of MEV codes or TSFM instances; exactly one transform per MEV.
- `desired_pool`: non-empty list of MEV codes or TSFM instances; must include at least one MEV; combinations pick 1–N desired features.
- `max_var_num`: upper bound on total features (forced + desired).
- `max_lag`: maximum lag to apply to each transform.

Usage:
    from tech.featureBuilder import FeatureBuilder

    fb = FeatureBuilder(
        max_var_num=3,
        forced_in=['GDP'],
        desired_pool=['Unemp', ('A','B')],
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
# Load configuration: MEV types and transform specs
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MEV_TYPE_CSV = BASE_DIR / 'support' / 'mev_type.csv'
TSFM_YAML    = BASE_DIR / 'support' / 'type_tsfm.yaml'

if not MEV_TYPE_CSV.exists():
    raise FileNotFoundError(f"Missing MEV type map: {MEV_TYPE_CSV}")
mev_df = pd.read_csv(MEV_TYPE_CSV)
_required = {'mev_code', 'type'}
if not _required.issubset(mev_df.columns):
    raise ValueError(f"mev_type.csv must contain {_required}")
MEV_TYPE_MAP: Dict[str, str] = dict(zip(mev_df['mev_code'], mev_df['type']))

if not TSFM_YAML.exists():
    raise FileNotFoundError(f"Missing transform spec: {TSFM_YAML}")
yaml_cfg = yaml.safe_load(TSFM_YAML.read_text())
TRANSFORM_MAP = yaml_cfg.get('transforms')
if not isinstance(TRANSFORM_MAP, dict):
    raise ValueError("type_tsfm.yaml must define a 'transforms' mapping")

# Patch TSFM repr for clarity: show feature, transform_fn name (no quotes), max_lag

def _tsfm_repr(self):
    fn = self.transform_fn
    if isinstance(fn, functools.partial):
        fn_name = fn.func.__name__
    else:
        fn_name = getattr(fn, '__name__', 'transform')
    # feature_name is string, enclose in quotes
    return f"TSFM(feature='{self.feature_name}', transform_fn={fn_name}, max_lag={self.max_lag})"

TSFM.__repr__ = _tsfm_repr

# ----------------------------------------------------------------------------
# FeatureBuilder class
# ----------------------------------------------------------------------------
class FeatureBuilder:
    """
    Generate TSFM feature combinations using only `forced_in` and `desired_pool`.

    - `forced_in`: list of MEV codes, TSFM instances, or sets thereof; each group is all-in or all-out.
    - `desired_pool`: non-empty list of MEV codes, TSFM instances, or sets thereof; combinations must include at least one desired group.
    - `max_var_num`: maximum total features (forced + desired) per combo.
    - `max_lag`: maximum lag to apply to transforms.
    """
    def __init__(
        self,
        max_var_num: int,
        forced_in: List[Union[str, TSFM, Tuple[Union[str, TSFM], ...]]],
        desired_pool: List[Union[str, TSFM, Tuple[Union[str, TSFM], ...]]],
        max_lag: int = 2
    ):
        if not desired_pool:
            raise ValueError("'desired_pool' must be provided and non-empty.")
        self.max_var_num = max_var_num
        self.max_lag = max_lag
        self.lags = range(max_lag + 1)
        # Config maps
        self.mev_type_map = MEV_TYPE_MAP
        self.transform_map = TRANSFORM_MAP

        # Helper to normalize group items
        def normalize(item):
            if isinstance(item, TSFM):
                return (item,)
            if isinstance(item, str):
                return (item,)
            if isinstance(item, (list, tuple, set)):
                return tuple(item)
            raise TypeError(f"Invalid group item: {item}")

        # Forced groups
        self.forced_groups: List[Tuple[str, ...]] = []
        self.forced_options: Dict[Tuple[str, ...], List[List[TSFM]]] = {}
        for raw in forced_in:
            group = normalize(raw)
            member_lists = []
            keys = []
            for m in group:
                if isinstance(m, TSFM):
                    member_lists.append([m])
                    keys.append(m.feature_name)
                else:
                    keys.append(m)
                    member_lists.append(self._gen_tsfms(m))
            group_key = tuple(keys)
            self.forced_groups.append(group_key)
            # Instances per group: pick one TSFM per member
            instances = [list(combo) for combo in itertools.product(*member_lists)]
            self.forced_options[group_key] = instances

        # Desired groups
        self.desired_groups: List[Tuple[str, ...]] = []
        self.desired_options: Dict[Tuple[str, ...], List[List[TSFM]]] = {}
        for raw in desired_pool:
            group = normalize(raw)
            member_lists = []
            keys = []
            for m in group:
                if isinstance(m, TSFM):
                    member_lists.append([m])
                    keys.append(m.feature_name)
                else:
                    keys.append(m)
                    member_lists.append(self._gen_tsfms(m))
            group_key = tuple(keys)
            self.desired_groups.append(group_key)
            instances = [list(combo) for combo in itertools.product(*member_lists)]
            self.desired_options[group_key] = instances

    def _gen_tsfms(self, mev: str) -> List[TSFM]:
        """
        Generate TSFM instances for a given MEV across transforms and lags.
        Defaults to only ['LV'] at lag=0 if MEV type missing.
        """
        mtype = self.mev_type_map.get(mev, '')
        transforms = self.transform_map.get(mtype)
        if not transforms:
            return [TSFM(mev, tf.LV, 0)]
        result: List[TSFM] = []
        for tkey in transforms:
            fn = getattr(tf, tkey, None)
            if not fn:
                continue
            for lag in self.lags:
                if tkey == 'LV':
                    fn_call = fn
                elif tkey in ('DF', 'GR'):
                    fn_call = functools.partial(fn, periods=1)
                else:
                    fn_call = functools.partial(fn, window=1)
                result.append(TSFM(mev, fn_call, lag))
        return result

    def generate_combinations(self) -> List[List[TSFM]]:
        """
        Build valid combos:
         - One TSFM per forced group.
         - Then up to N desired groups (N = max_var_num - forced_count), including 0.
        """
        combos: List[List[TSFM]] = []
        forced_lists = [self.forced_options[g] for g in self.forced_groups]
        forced_iter = itertools.product(*forced_lists) if forced_lists else [()]

        for forced_choice in forced_iter:
            # flatten forced TSFMs
            forced_flat = []
            for group_inst in forced_choice:
                forced_flat.extend(group_inst)
            used = len(forced_flat)
            max_des = self.max_var_num - used
            # allow 0 desired (just forced)
            # generate forced-only combos
            combos.append(list(forced_flat))
            # now include desired combos
            for r in range(1, max_des + 1):
                for subset in itertools.combinations(self.desired_groups, r):
                    pools = [self.desired_options[g] for g in subset]
                    for prod in itertools.product(*pools):
                        desired_flat = []
                        for group_inst in prod:
                            desired_flat.extend(group_inst)
                        if used + len(desired_flat) <= self.max_var_num:
                            combos.append(forced_flat + desired_flat)
        return combos

    def __repr__(self) -> str:
        return (
            f"<FeatureBuilder max_var_num={self.max_var_num}, "
            f"forced_groups={self.forced_groups}, "
            f"desired_groups={self.desired_groups}, max_lag={self.max_lag}>"
        )
