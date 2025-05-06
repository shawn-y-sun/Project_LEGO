# featureBuilder.py

import itertools
from typing import List, Union, Dict
import itertools
from typing import List, Union, Dict
import pandas as pd
from .transform import TSFM
from . import transform as tf
# Patch TSFM.__repr__ for readability
TSFM.__repr__ = lambda self: f"TSFM('{self.feature_name}', '{self.name}')"

class featureBuilder:
    """
    Build combinations of TSFM instances for macroeconomic variables (MEVs).

    Parameters
    ----------
    mev_transMap : dict[str, str]
        Maps MEV name to transform key: 'growthrate', 'diff', or 'level'.
    freq : str
        Frequency code ('Q', 'M', 'Y', etc.) to determine function names dynamically.
    max_var_num : int
        Maximum number of features per combination (including forced-in).
    forced_in : list[str or TSFM]
        MEV names or TSFM instances to always include in combos.
    driver_pool : list[str]
        All MEV names available for optional features.
    desired_pool : list[str]
        MEV names; combos must include at least one.
    max_lag : int, optional
        Maximum lag to apply for each transform (0..max_lag). Default: 2.
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

        # Map keys to dynamic function names
        code_map = {
            'growthrate': f"{self.freq}{self.freq}GR",
            'diff':        f"{self.freq}{self.freq}",
            'level':       'LV'
        }
        # Resolve functions from transform module
        self.fn_map: Dict[str, callable] = {}
        for key, code in code_map.items():
            fn = getattr(pd.Series, code, None)  # skip pandas methods
            fn = globals().get(code, None) if fn is None else fn
            fn = locals().get(code, None) if fn is None else fn
            fn = globals().get(code) or getattr(pd.Series, code, None) or getattr(self, code, None)
            # Actually import transform functions
            transform_module = __import__('tech.transform', fromlist=[code])
            fn = getattr(transform_module, code, None)
            if fn is None:
                raise ValueError(f"Transform '{code}' not found in tech.transform for freq '{self.freq}'")
            self.fn_map[key] = fn

        # Build forced and optional pools
        self._build_forced(forced_in)
        self._build_options()

    def _build_forced(self, forced_in: List[Union[str, TSFM]]) -> None:
        """Convert forced_in entries to TSFM instances, preserving TSFM if given."""
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
        """Enumerate all valid feature combinations as lists of TSFM instances."""
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


# Example usage when running as script
if __name__ == '__main__':
    # Sample quarterly data
    dates = pd.date_range('2020-01-01', periods=4, freq='Q')
    df = pd.DataFrame({
        'GDP':   [1000, 1050, 1100, 1150],
        'Unemp': [5.0, 4.8, 4.7, 4.5]
    }, index=dates)

    # Transform map and builder
    mev_map = {'GDP': 'growthrate', 'Unemp': 'diff'}
    fb = FeatureBuilder(
        mev_transMap=mev_map,
        freq='Q',
        max_var_num=2,
        forced_in=['GDP'],
        driver_pool=['GDP', 'Unemp'],
        desired_pool=['Unemp'],
        max_lag=1
    )
    print("Sample TSFM combinations:")
    for combo in fb.generate_combinations():
        print(combo)
    print(f"Total combos: {len(fb.generate_combinations())}")
