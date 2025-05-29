from typing import List, Union, Tuple, Type, Any, Optional, Callable
import itertools
import time
import datetime

import pandas as pd
from tqdm import tqdm

from .data import DataManager
from .feature import Feature
from .transform import TSFM
from .model import ModelBase
from .cm import CM


class ModelSearch:
    """
    Generate and manage model feature-spec combinations and assessments for CM.build().

    Parameters
    ----------
    dm : DataManager
        DataManager instance for feature building and TSFM expansion.
    target : str
        Name of the response variable/target for modeling.
    model_cls : Type[ModelBase]
        The ModelBase subclass to instantiate and fit within CM.
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
        self.all_specs: List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]] = []
        self.passed_cms: List[CM] = []
        self.failed_info: List[Tuple[List[Any], List[str]]] = []
        self.df_scores: Optional[pd.DataFrame] = None
        self.top_cms: List[CM] = []

    def build_spec_combos(
        self,
        forced_in: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
        desired_pool: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
        max_var_num: int,
        max_lag: int = 0,
        max_periods: int = 1
    ) -> List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]]:
        """
        Build valid spec combos combining static forced_in with all subsets of desired_pool.

        Parameters
        ----------
        forced_in : list
            Items or tuple-groups always included (preserved as-is).
        desired_pool : list
            Items or tuple-groups to choose subsets from and expand.
        max_var_num : int
            Max total specs per combo.
        max_lag : int
            Max lag for string TSFM expansion.
        max_periods : int
            Max periods for string TSFM expansion.

        Returns
        -------
        combos : list of spec lists
            Each combo is a list including str, TSFM, Feature, or tuple elements.
        """
        def expand(item):
            if isinstance(item, tuple):
                return [[item]]
            if isinstance(item, (TSFM, Feature)):
                return [[item]]
            if isinstance(item, str):
                tsfm_map = self.dm.build_tsfm_specs([item], max_lag=max_lag, max_periods=max_periods)
                variants = tsfm_map.get(item, [item])
                return [[spec] for spec in variants]
            raise ValueError(f"Invalid spec item: {item!r}")

        # If forced_in is None, we treat it as “no forced specs”;
        # otherwise copy the provided list
        if forced_in is None:
            forced_specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]] = []
            include_forced = False
        else:
            forced_specs = forced_in.copy()
            include_forced = True
        
        desired_combos: List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]] = []
        n = len(desired_pool)
        for r in range(1, n + 1):
            for subset in itertools.combinations(desired_pool, r):
                expansions = [expand(x) for x in subset]
                for combo in itertools.product(*expansions):
                    specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]] = []
                    for elem in combo:
                        specs.extend(elem)
                    desired_combos.append(specs)

        combos: List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]] = []
        # only append the pure-forced combo if the user really passed forced_in
        if include_forced and len(forced_specs) <= max_var_num:
            combos.append(forced_specs.copy())
        for d in desired_combos:
            if len(forced_specs) + len(d) <= max_var_num:
                # forced_specs may be empty, so this yields just d if no forced_in
                combos.append(forced_specs + d)
 
        # ──────────── drop any “empty” spec‐lists ────────────
        combos = [c for c in combos if c]
 
        self.all_specs = combos
        return combos

    def assess_spec(
        self,
        model_id: str,
        specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
        sample: str = 'in',
        test_update_func: Optional[Callable[[ModelBase], dict]] = None
    ) -> Union[CM, Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]]:
        """
        Build and assess a single spec combo via CM.build(), reload tests, and TestSet.filter_pass().

        Parameters
        ----------
        model_id : str
            Unique identifier for this candidate model.
        specs : list
            Feature-specs for CM.build (str, TSFM, Feature, or tuple grouping).
        sample : {'in','full'}
            Which sample to fit and test; do not use 'both'.
        test_update_func : callable, optional
            Function to update/regenerate the testset for the fitted model.

        Returns
        -------
        CM
            The fitted CM instance if all active tests pass.
        (specs, failed_tests)
            Tuple of the input specs and list of failed test names if any test fails.
        """
        if sample not in {'in', 'full'}:
            raise ValueError("`sample` must be either 'in' or 'full'.")

        # Build the candidate model
        cm = CM(model_id=model_id, target=self.target, model_cls=self.model_cls)
        cm.build(specs, sample=sample, data_manager=self.dm)
        mdl = cm.model_in if sample == 'in' else cm.model_full

        # Reload testset, applying update if provided
        mdl.load_testset(test_update_func=test_update_func)

        # Run filtering on updated testset
        passed, failed = mdl.testset.filter_pass()
        if passed:
            return cm
        return specs, failed

    def filter_specs(
        self,
        model_id_prefix: str = 'cm',
        sample: str = 'in',
        test_update_func: Optional[Callable[[ModelBase], dict]] = None
    ) -> Tuple[List[CM], List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]]]:
        """
        Assess all built spec combos and separate passed and failed results,
        using multithreading and a single progress bar update.

        Parameters
        ----------
        model_id_prefix : str, default 'cm'
            Prefix for auto-generated model IDs (appended with index).
        sample : {'in','full'}
            Sample to use for all assessments (default 'in').
        test_update_func : callable, optional
            Function to update/regenerate the testset for each model.

        Returns
        -------
        passed_cms : list of CM
            CM instances that passed all active tests.
        failed_info : list of (specs, failed_tests)
            Spec combos and test names for combos that failed.
        """
        passed_cms: List[CM] = []
        failed_info: List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]] = []
        total = len(self.all_specs)
        start_time = time.time()
 
        # Sequential assessment with one tqdm bar
        bar = tqdm(total=total, desc="Filtering Specs")
        for i, specs in enumerate(self.all_specs):
            model_id = f"{model_id_prefix}{i}"
            result = self.assess_spec(model_id, specs, sample, test_update_func)
            if isinstance(result, CM):
                passed_cms.append(result)
            else:
                failed_info.append(result)
 
            # ETA update
            processed = bar.n + 1
            elapsed = time.time() - start_time
            progress = processed / total if total > 0 else 1
            if progress > 0:
                est_total = elapsed / progress
                rem = est_total - elapsed
                finish_dt = datetime.datetime.now() + datetime.timedelta(seconds=rem)
                eta = finish_dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                eta = ''
            bar.set_postfix(estimated_finish=eta)
            bar.update(1)
        bar.close()
        return passed_cms, failed_info

    @staticmethod
    def rank_cms(
        cms_list: List[CM],
        sample: str = 'in',
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> pd.DataFrame:
        """
        Rank CM instances by composite score from fit, IS error, and OOS error measures on the specified sample.

        Parameters
        ----------
        cms_list : list of CM
            CM instances with loaded testsets.
        sample : {'in','full'}
            Which fitted model to use (in-sample vs full-sample tests).
        weights : tuple of floats
            Weights for (fit, IS error, OOS error). If 'full' and no OOS test,
            only the other two categories are used.

        Returns
        -------
        DataFrame
            Sorted by composite_score descending. Includes fit_score, is_err_score,
            optional oos_err_score, and composite_score.
        """
        if sample not in {'in', 'full'}:
            raise ValueError("sample must be 'in' or 'full'.")
        records = []
        for cm in cms_list:
            mdl = cm.model_in if sample == 'in' else cm.model_full
            testdict = {t.name: t for t in mdl.testset.tests}
            fit_sr = testdict['Fit Measures'].test_result
            is_err_sr = testdict['IS Error Measures'].test_result
            oos_err_sr = testdict.get('OOS Error Measures')
            rec = {'model_id': cm.model_id}
            for nm, val in fit_sr.items(): rec[f'fit_{nm}'] = val
            for nm, val in is_err_sr.items(): rec[f'is_err_{nm}'] = val
            if oos_err_sr is not None:
                for nm, val in oos_err_sr.test_result.items(): rec[f'oos_err_{nm}'] = val
            records.append(rec)

        df = pd.DataFrame(records)
        norm_df = pd.DataFrame({'model_id': df['model_id']})
        def norm(series, invert=False):
            mn, mx = series.min(), series.max()
            return pd.Series(0.5, index=series.index) if mx <= mn else (1 - ((series - mn) / (mx - mn))) if invert else ((series - mn) / (mx - mn))
        
        fit_cols = [c for c in df if c.startswith('fit_')]
        is_err_cols = [c for c in df if c.startswith('is_err_')]
        oos_err_cols = [c for c in df if c.startswith('oos_err_')]
        for c in fit_cols: norm_df[c] = norm(df[c], invert=False)
        for c in is_err_cols: norm_df[c] = norm(df[c], invert=True)
        for c in oos_err_cols: norm_df[c] = norm(df[c], invert=True)
        df_scores = pd.DataFrame({'model_id': df['model_id']})
        df_scores['fit_score'] = norm_df[fit_cols].mean(axis=1)
        df_scores['is_err_score'] = norm_df[is_err_cols].mean(axis=1)
        if oos_err_cols: df_scores['oos_err_score'] = norm_df[oos_err_cols].mean(axis=1)
        w_fit, w_is, w_oos = weights
        if not oos_err_cols:
            total_w = w_fit + w_is
            df_scores['composite_score'] = (w_fit * df_scores['fit_score'] + w_is * df_scores['is_err_score']) / total_w
        else:
            total_w = w_fit + w_is + w_oos
            df_scores['composite_score'] = (w_fit * df_scores['fit_score'] + w_is * df_scores['is_err_score'] + w_oos * df_scores['oos_err_score']) / total_w
        return df_scores.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    def run_search(
        self,
        desired_pool: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
        forced_in: Optional[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]] = None,
        top_n: int = 10,
        sample: str = 'in',
        max_var_num: int = 5,
        max_lag: int = 3,
        max_periods: int = 3,
        rank_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        test_update_func: Optional[Callable[[ModelBase], dict]] = None
    ) -> List[CM]:
        """
        Execute full search pipeline: build specs, filter, rank, and select top_n models.

        Steps
        -----
        1. Print configuration summary.
        2. Build spec combinations via build_spec_combos.
        3. Print number of generated combos.
        4. Assess and filter combos via filter_specs (printing test info for first combo).
        5. Rank passed models via rank_cms and retain top_n.

        Returns
        -------
        top_models : list of CM
            The top_n CM instances sorted by composite score.
        """
        forced = forced_in or []
        # 1. Configuration
        print("=== ModelSearch Configuration ===")
        print(f"Target          : {self.target}")
        print(f"Model class     : {self.model_cls.__name__}")
        print(f"Desired pool    : {desired_pool}")
        print(f"Forced in       : {forced}")
        print(f"Sample          : {sample}\n"
              f"Max var num     : {max_var_num}\n"
              f"Max lag         : {max_lag}\n"
              f"Max periods     : {max_periods}\n"
              f"Top N           : {top_n}\n"
              f"Rank weights    : {rank_weights}\n"
              f"Test update func: {test_update_func}")
        print("==================================\n")

        # 2. Build specs
        combos = self.build_spec_combos(forced, desired_pool, max_var_num, max_lag, max_periods)
        print(f"Built {len(combos)} spec combinations.\n")

        # 3. Example test info
        if combos:
            print("--- Example TestSet Info ---")
            # Always build a fresh CM to display test info regardless of pass/fail
            cm_example = CM(model_id="init_1", target=self.target, model_cls=self.model_cls)
            cm_example.build(combos[0], sample=sample, data_manager=self.dm)
            mdl_example = cm_example.model_in if sample == 'in' else cm_example.model_full
            mdl_example.load_testset(test_update_func=test_update_func)
            mdl_example.testset.print_test_info()
            print()

        # 4) Filter specs
        passed, failed = self.filter_specs(
            sample=sample,
            test_update_func=test_update_func
        )
        self.passed_cms = passed
        self.failed_info = failed
        # Early exit if nothing passed
        if not self.passed_cms:
            print("\n⚠️  No candidate models passed the filter tests. Search terminated.\n")
            return
        print(f"Passed {len(passed)} combos; Failed {len(failed)} combos.\n")

        # 5. Rank models
        df = ModelSearch.rank_cms(passed, sample, rank_weights)
        # Identify and store top cms
        ordered_ids = df['model_id'].tolist()
        top_ids = ordered_ids[:top_n]
        self.top_cms = [next(cm for cm in passed if cm.model_id == mid) for mid in top_ids]

        # Rename model_ids and update df_scores
        new_ids = [f"cm{i+1}" for i in range(len(self.top_cms))]
        for cm, new_id in zip(self.top_cms, new_ids):
            cm.model_id = new_id
        df_updated = df.copy()
        for idx, new_id in enumerate(new_ids):
            df_updated.at[idx, 'model_id'] = new_id
        self.df_scores = df_updated

        # Print updated rankings
        print("=== Updated Ranked Results ===")
        print(df_updated.head(top_n).to_string(index=False))

        # Print top model formulas
        print(f"\n=== Top {top_n} Model Formulas ===")
        for cm in self.top_cms:
            print(f"{cm.model_id}: {cm.formula}")
        # No return; results are stored in self
