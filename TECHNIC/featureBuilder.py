# TECHNIC/featureBuilder.py


import itertools
import QT
from QT import TSFM
import QT.transform as tf

# Patch TSFM.__repr__ to display MEV and suffix
TSFM.__repr__ = lambda self: f"TSFM('{self.mev}', '{self.suffix}')"

class FeatureBuilder:
    """
    Efficiently build combinations of TSFM instances for macroeconomic variables (MEVs).

    Attributes
    ----------
    mev_transMap : dict[str, str]
        Maps each MEV to a transform key ('growthrate', 'diff', 'level').
    freq : str
        Frequency code (first letter, e.g. 'Q', 'M', 'Y') to select transform functions.
    max_var_num : int
        Max features per combination (including forced-in items).
    forced : list[TSFM]
        TSFM instances or MEV names forced into every combo (converted to TSFM).
    options : dict[str, list[TSFM]]
        Available TSFM options per MEV for optional selection.
    desired_pool : set[str]
        MEVs that must appear at least once in each combo.
    """
    def __init__(
        self,
        mev_transMap: dict[str, str],
        freq: str,
        max_var_num: int,
        forced_in: list,
        driver_pool: list[str],
        desired_pool: list[str],
        max_lag: int = 2
    ):
        # Core parameters
        self.mev_transMap = mev_transMap or {}
        self.freq = str(freq).upper()[0]
        self.max_var_num = max_var_num
        self.driver_pool = driver_pool or []
        self.desired_pool = set(desired_pool or [])
        self.max_lag = max_lag
        self.lags = range(max_lag + 1)

        # Resolve transformation functions dynamically
        code_map = {
            'growthrate': f"{self.freq}{self.freq}GR",
            'diff': f"{self.freq}{self.freq}",
            'level': 'LV'
        }
        self.fn_map = {}
        for key, code in code_map.items():
            fn = getattr(tf, code, None)
            if not fn:
                raise ValueError(f"Missing transform function '{code}' for frequency '{self.freq}'")
            self.fn_map[key] = fn

        # Process forced-in MEVs and build optional TSFM pools
        self._build_forced(forced_in)
        self._build_options()

    def _build_forced(self, forced_in):
        """Convert forced_in entries to TSFM instances with .mev set."""
        forced = []
        for item in forced_in:
            if isinstance(item, TSFM):
                if not hasattr(item, 'mev') or not item.mev:
                    raise ValueError("Forced-in TSFM must have 'mev' attribute set.")
                forced.append(item)
            else:
                tsfm = TSFM(transform_fn=self.fn_map['level'], max_lag=0)
                tsfm.mev = item
                forced.append(tsfm)
        self.forced = forced

    def _build_options(self):
        """Build a dict of TSFM option lists for each MEV in driver_pool."""
        opts = {}
        for mev, trans_key in self.mev_transMap.items():
            fn = self.fn_map.get(trans_key)
            if not fn or mev not in self.driver_pool:
                continue
            # instantiate one TSFM per lag
            opts[mev] = [self._make_tsfm(mev, fn, lag) for lag in self.lags]
        self.options = opts

    @staticmethod
    def _make_tsfm(mev, fn, lag):
        """Helper: create TSFM with .mev set."""
        tsfm = TSFM(transform_fn=fn, max_lag=lag)
        tsfm.mev = mev
        return tsfm

    def generate_combinations(self) -> list[list[TSFM]]:
        """
        Generate all valid feature combinations as lists of TSFM instances.

        Returns
        -------
        combos : list of list of TSFM
            Each combo respects max_var_num, includes forced TSFMs,
            and contains at least one MEV from desired_pool.
        """
        base = len(self.forced)
        extra_max = self.max_var_num - base
        forced_mevs = {t.mev for t in self.forced}
        avail_mevs = [m for m in self.driver_pool if m not in forced_mevs]

        # Determine if we need to enforce desired presence
        required = None
        if not (forced_mevs & self.desired_pool):
            required = set(avail_mevs) & self.desired_pool

        combos = []
        # Iterate over possible counts of optional MEVs
        for r in range(extra_max + 1):
            for subset in itertools.combinations(avail_mevs, r):
                # Skip if subset cannot satisfy desired requirement
                if required and not required.intersection(subset):
                    continue
                # Cartesian product of TSFM options for the subset
                for prod in itertools.product(*(self.options.get(m, []) for m in subset)):
                    combos.append(self.forced + list(prod))
        return combos

    def __repr__(self):
        return (
            f"<FeatureBuilder freq={self.freq}, max_var_num={self.max_var_num}, "
            f"forced={[t.mev for t in self.forced]}, desired={list(self.desired_pool)}, "
            f"max_lag={self.max_lag}>"
        )
