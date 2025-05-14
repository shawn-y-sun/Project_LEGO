from typing import List, Union, Type
import itertools

from .data import DataManager
from .transform import TSFM
from .model import ModelBase

class ModelSearch:
    """
    Generate and manage model feature-spec combinations for CM.build().

    :param dm: DataManager instance for building features
    :param target: Name of the target variable (y)
    :param model_cls: ModelBase subclass to be used by CM
    """
    def __init__(
        self,
        dm: DataManager,
        target: str,
        model_cls: Type[ModelBase]
    ):
        self.dm = dm
        self.target = target
        self.model_cls = model_cls
        # store all spec combinations
        self.all_specs: List[List[Union[str, TSFM]]] = []

    def build_all_specs(
        self,
        forced_in: List[Union[str, TSFM]],
        desired_pool: List[Union[str, TSFM]],
        max_var_num: int,
        max_lag: int = 0,
        max_periods: int = 1
    ) -> List[List[Union[str, TSFM]]]:
        """
        Build all valid feature-spec lists combining forced and desired groups.

        :param forced_in: list of vars (or TSFM) always included
        :param desired_pool: list of vars (or TSFM) to choose from
        :param max_var_num: maximum total number of features
        :param max_lag: passed to build_tsfm_specs
        :param max_periods: passed to build_tsfm_specs
        """
        # 1) Generate TSFM specs for forced and desired
        forced_map = self.dm.build_tsfm_specs(
            forced_in, max_lag=max_lag, max_periods=max_periods
        )
        desired_map = self.dm.build_tsfm_specs(
            desired_pool, max_lag=max_lag, max_periods=max_periods
        )

        # 2) All combos of forced specs (one per variable)
        forced_keys = list(forced_map.keys())
        forced_lists = [forced_map[k] for k in forced_keys]
        forced_combos = [list(c) for c in itertools.product(*forced_lists)]

        # 3) All valid combos of desired specs
        desired_keys = list(desired_map.keys())
        desired_combos: List[List[Union[str, TSFM]]] = []
        for r in range(1, min(len(desired_keys), max_var_num) + 1):
            for subset in itertools.combinations(desired_keys, r):
                lists = [desired_map[k] for k in subset]
                for prod in itertools.product(*lists):
                    desired_combos.append(list(prod))

        # 4) Combine forced and desired, respecting max_var_num
        combos: List[List[Union[str, TSFM]]] = []
        for f in forced_combos:
            # forced-only
            if len(f) <= max_var_num:
                combos.append(f)
            for d in desired_combos:
                if len(f) + len(d) <= max_var_num:
                    combos.append(f + d)

        # Cache and return
        self.all_specs = combos
        return combos
