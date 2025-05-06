import itertools
from typing import List, Union, Dict
import pandas as pd
import transform as tf

# Alias TSFM for convenience
TSFM = tf.TSFM

# Patch TSFM.__repr__ for readability
TSFM.__repr__ = lambda self: f"TSFM('{self.feature_name}', '{self.name}')"

class FeatureBuilder:
    """
    Build combinations of TSFM instances for macroeconomic variables (MEVs).

    Parameters
    ----------
    mev_transMap : dict[str, str]
        Maps MEV name to transformation key ('growthrate', 'diff', 'level').
    freq : str
        Frequency code (first letter, e.g. 'Q','M','Y') for dynamic transform selection.
    max_var_num : int
        Maximum number of features in any combination (including forced-in features).
    forced_in : list[str or TSFM]
        MEV names or TSFM instances to include in every combination.
    driver_pool : list[str]
        All MEV names available for optional features.
    desired_pool : list[str]
        MEV names that each combination must include at least one of.
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
        # Core settings
        self.mev_transMap = mev_transMap or {}
        self.freq = str(freq).upper()[0]
        self.max_var_num = max_var_num
        self.driver_pool = driver_pool or []
        self.desired_pool = set(desired_pool or [])
        self.max_lag = max_lag
        self.lags = range(max_lag + 1)

        # Resolve transform functions based on frequency
        code_map = {
            'growthrate': f"{self.freq}{self.freq}GR",
            'diff':        f"{self.freq}{self.freq}",
            'level':       'LV'
        }
        self.fn_map: Dict[str, callable] = {}
        for key, code in code_map.items():
            fn = getattr(tf, code, None)
            if fn is None:
                raise ValueError(f"Transform '{code}' not found for freq '{self.freq}' in transform.py")
            self.fn_map[key] = fn

        # Build forced and optional TSFM pools
        self._build_forced(forced_in)
        self._build_options()

    def _build_forced(self, forced_in: List[Union[str, TSFM]]) -> None:
        """Convert forced_in entries to TSFM instances."""
        forced_list: List[TSFM] = []
        for item in forced_in:
            if isinstance(item, TSFM):
                forced_list.append(item)
            else:
                ts = TSFM(feature=item, transform_fn=self.fn_map['level'], max_lag=0)
                forced_list.append(ts)
        self.forced = forced_list

    def _build_options(self) -> None:
        """Create TSFM options for each MEV in driver_pool."""
        options: Dict[str, List[TSFM]] = {}
        for mev in self.driver_pool:
            key = self.mev_transMap.get(mev)
            fn = self.fn_map.get(key)
            if fn:
                options[mev] = [TSFM(feature=mev, transform_fn=fn, max_lag=lag) for lag in self.lags]
        self.options = options

    def generate_combinations(self) -> List[List[TSFM]]:
        """Enumerate valid combinations of TSFM instances."""
        base_count = len(self.forced)
        extra_max = self.max_var_num - base_count
        forced_names = {ts.feature_name for ts in self.forced}
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
                    combos.append(self.forced + list(prod))
        return combos

    def __repr__(self) -> str:
        return (
            f"<FeatureBuilder freq={self.freq}, max_var_num={self.max_var_num}, "
            f"forced={[ts.feature_name for ts in self.forced]}, "
            f"desired={list(self.desired_pool)}, max_lag={self.max_lag}>"
        )