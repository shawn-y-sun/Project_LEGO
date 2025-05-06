# tech/featureBuilder.py

import itertools
from typing import List, Union, Dict
import pandas as pd
from .transform import TSFM
from . import transform as tf

# Patch TSFM.__repr__ for readability
TSFM.__repr__ = lambda self: f"TSFM(variable='{self.feature_name}', transform_fn='{self.transform_fn.__name__}', max_lag={self.max_lag})"

class FeatureBuilder:
    """
    Build combinations of TSFM instances for macroeconomic variables (MEVs).

    Parameters
    ----------
    mev_transMap : dict[str, str]
        Maps MEV name to key: 'growthrate', 'diff', or 'level'.
    freq : str
        Frequency code ('Q','M','Y', etc.) for dynamic transform selection.
    max_var_num : int
        Max features per combination (including forced-in).
    forced_in : list[str or TSFM]
        MEV names or TSFM instances forced into every combo.
    driver_pool : list[str]
        All MEV names for optional features.
    desired_pool : list[str]
        MEV names; combos must include at least one.
    max_lag : int, optional
        Lags applied to transforms (0..max_lag). Default: 2.
    """
    def __init__(
        self,
        mev_transMap: Dict[str, str],
        freq: str,
        max_var_num: int,
        forced_in: List[Union[str, TSFM]],
        driver_pool: List[str],
        desired_pool: List[str],
        max_lag: int = 2
    ):
        self.mev_transMap = mev_transMap or {}
        self.freq = str(freq).upper()[0]
        self.max_var_num = max_var_num
        self.driver_pool = driver_pool or []
        self.desired_pool = set(desired_pool or [])
        self.max_lag = max_lag
        self.lags = range(max_lag + 1)

        # Map transform keys to dynamic function names
        code_map = {
            'growthrate': f"{self.freq}{self.freq}GR",
            'diff':        f"{self.freq}{self.freq}",
            'level':       'LV'
        }
        self.fn_map: Dict[str, callable] = {}
        for key, code in code_map.items():
            fn = getattr(tf, code, None)
            if fn is None:
                raise ValueError(f"Transform '{code}' not found in transform.py for freq '{self.freq}'")
            self.fn_map[key] = fn

        self._build_forced(forced_in)
        self._build_options()

    def _build_forced(self, forced_in: List[Union[str, TSFM]]) -> None:
        forced_list: List[TSFM] = []
        for item in forced_in:
            if isinstance(item, TSFM):
                forced_list.append(item)
            else:
                ts = TSFM(item, self.fn_map['level'], 0)
                forced_list.append(ts)
        self.forced = forced_list

    def _build_options(self) -> None:
        options: Dict[str, List[TSFM]] = {}
        for mev in self.driver_pool:
            key = self.mev_transMap.get(mev)
            fn = self.fn_map.get(key)
            if fn:
                options[mev] = [TSFM(mev, fn, lag) for lag in self.lags]
        self.options = options

    def generate_combinations(self) -> List[List[TSFM]]:
        forced = self.forced
        base_count = len(forced)
        extra_max = self.max_var_num - base_count
        forced_names = {ts.feature_name for ts in forced}
        avail = [m for m in self.driver_pool if m not in forced_names]

        required = None
        if not (forced_names & self.desired_pool):
            required = set(avail) & self.desired_pool

        combos: List[List[TSFM]] = []
        for r in range(extra_max + 1):
            for subset in itertools.combinations(avail, r):
                if required and not (set(subset) & required):
                    continue
                for prod in itertools.product(*(self.options[m] for m in subset)):
                    combos.append(forced + list(prod))
        return combos

    def __repr__(self) -> str:
        return (
            f"<FeatureBuilder freq={self.freq}, max_var_num={self.max_var_num}, "
            f"forced={[ts.feature_name for ts in self.forced]}, "
            f"desired={list(self.desired_pool)}, max_lag={self.max_lag}>"
        )
