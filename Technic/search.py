from typing import List, Union, Tuple, Type, Any, Optional, Callable, Dict
import itertools
import time
import datetime
from collections import defaultdict
import warnings
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import pandas as pd
from tqdm import tqdm

from .data import DataManager
from .feature import Feature, DumVar
from .transform import TSFM
from .model import ModelBase
from .cm import CM


def _sort_specs_with_dummies_first(spec_list: List[Any]) -> List[Any]:
    """
    Sort spec list so quarterly and monthly dummy variables come first.
    
    Priority order:
    1. DumVar('Q') - Quarterly dummies first
    2. DumVar('M') - Monthly dummies second  
    3. Everything else - maintain relative order
    
    Parameters
    ----------
    spec_list : List[Any]
        List of specification items to sort
        
    Returns
    -------
    List[Any]
        Sorted list with dummy variables first
    """
    def get_dummy_priority(spec):
        """Get priority for sorting. Lower numbers come first."""
        try:
            # Check if this is a DumVar instance
            if isinstance(spec, DumVar):
                # Check the variable name to determine type
                var_name = str(spec.var).upper()
                if var_name == 'Q':
                    return 0  # Quarterly first
                elif var_name == 'M':
                    return 1  # Monthly second
                else:
                    return 2  # Other dummy variables
            else:
                return 3  # Non-dummy features
        except:
            return 3  # Safe fallback for any errors
    
    # Sort by priority, keeping relative order for items with same priority
    return sorted(spec_list, key=lambda x: (get_dummy_priority(x), str(x)))


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
        self.error_log: List[Tuple[List[Any], str, str]] = []
        self.df_scores: Optional[pd.DataFrame] = None
        self.top_cms: List[CM] = []
    
    def build_spec_combos(
        self,
        forced_in: Optional[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]],
        desired_pool: List[Union[str, TSFM, Feature, Tuple[Any, ...], set]],
        max_var_num: int,
        max_lag: int = 3,
        max_periods: int = 3,
        category_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None
    ) -> List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]]:
        """
        Build all valid feature-spec combos:
        - If forced_in is provided, include those specs in every combo.
        - desired_pool can contain:
            * str, TSFM, Feature, tuple, or set.
          * tuple: items stay grouped together.
          * set: treated as a pool where exactly one must be selected.
        - Strings at top-level are expanded into TSFM variants via DataManager.
        - Respects max_var_num (total features per combo).
        - Respects category_limit (max variables from each MEV category per combo).

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
        category_limit : int, default 1
            Max variables from each MEV category per combo. Only applies to 
            top-level strings and TSFM instances in desired_pool; other Feature 
            instances or items in nested structures are not subject to this constraint.
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            Passed to DataManager.build_tsfm_specs() for string expansion.

        Returns
        -------
        combos : list of spec lists
            Each combo is a list including str, TSFM, Feature, or tuple elements.
        """
        # Handle forced_in being optional
        forced_specs = (forced_in or []).copy()

        # Step 1: Build raw combos from desired_pool with category constraints
        # Separate constrained and unconstrained items
        constrained_items = []  # top-level strings and TSFM instances
        unconstrained_items = []  # everything else (sets, tuples, other Features)

        for item in desired_pool:
            if isinstance(item, (str, TSFM)):
                constrained_items.append(item)
            else:
                unconstrained_items.append(item)

        # Group constrained items by MEV category
        category_groups = defaultdict(list)
        uncategorized_constrained = []

        if constrained_items:
            mev_map = self.dm.var_map
            for item in constrained_items:
                # Get variable name to look up category
                if isinstance(item, str):
                    var_name = item
                elif isinstance(item, TSFM):
                    var_name = item.var
                else:
                    # This shouldn't happen based on isinstance check above
                    uncategorized_constrained.append(item)
                    continue
                
                # Look up category in MEV map
                var_info = mev_map.get(var_name, {})
                category = var_info.get('category')
                
                if category:
                    category_groups[category].append(item)
                else:
                    uncategorized_constrained.append(item)

        # Create pools for combination generation
        pools: List[List[Any]] = []

        # Add category-based pools (each category becomes one pool of possible combinations)
        for category, items in category_groups.items():
            category_pool = []
            # Generate all possible subsets of size 1 to category_limit
            for r in range(1, min(len(items), category_limit) + 1):
                for combo in itertools.combinations(items, r):
                    category_pool.append(list(combo))
            if category_pool:
                pools.append(category_pool)

        # Add uncategorized constrained items as individual pools
        for item in uncategorized_constrained:
            pools.append([item])

        # Handle unconstrained items (existing logic)
        for item in unconstrained_items:
            if isinstance(item, set):
                # Flatten nested sets into a single pool of choices
                flat: set = set()
                def _flatten(s):
                    for el in s:
                        if isinstance(el, set):
                            _flatten(el)
                        else:
                            flat.add(el)
                _flatten(item)
                pools.append(list(flat))
            else:
                pools.append([item])

        # Generate combinations by choosing from pools
        raw_combos: List[List[Any]] = []
        if pools:
            m = len(pools)
            for r in range(1, m + 1):
                for indices in itertools.combinations(range(m), r):
                    selected_pools = [pools[i] for i in indices]
                    for choice in itertools.product(*selected_pools):
                        # Flatten choice (some elements might be lists from category pools)
                        flat_combo = []
                        for item in choice:
                            if isinstance(item, list):
                                flat_combo.extend(item)
                            else:
                                flat_combo.append(item)
                        raw_combos.append(flat_combo)

        # Step 2: Prepend forced_in-only combo if any forced specs exist
        combos: List[List[Any]] = []
        if forced_specs and len(forced_specs) <= max_var_num:
            combos.append(forced_specs.copy())

        # Mix forced with each raw combo within max_var_num
        for rc in raw_combos:
            if len(forced_specs) + len(rc) <= max_var_num:
                combos.append(forced_specs + rc)

        # Step 3: Expand only top-level strings into TSFM variants
        # Gather unique strings to expand
        top_strings = {spec for combo in combos for spec in combo if isinstance(spec, str)}
        
        # Apply special period logic for monthly data
        # When internal_data is monthly, periods > 3 should only include multiples of 3
        effective_max_periods = max_periods
        if hasattr(self.dm, 'internal_data') and hasattr(self.dm.internal_data, 'index'):
            try:
                freq = pd.infer_freq(self.dm.internal_data.index)
                if freq and freq.startswith('M'):  # Monthly frequency (M, MS, etc.)
                    if max_periods > 3:
                        # Create periods list: (1, 2, 3, 6, 9, 12, ...) up to max_periods
                        periods_list = [1, 2, 3]
                        for p in range(6, max_periods + 1, 3):  # multiples of 3 starting from 6
                            periods_list.append(p)
                        effective_max_periods = periods_list
            except (AttributeError, TypeError):
                # If frequency detection fails, use original max_periods
                pass
        
        # Suppress warnings during TSFM spec building
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsfm_map = self.dm.build_tsfm_specs(
                list(top_strings), max_lag=max_lag, max_periods=effective_max_periods,
                exp_sign_map=exp_sign_map
            )

        expanded: List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]] = []
        for combo in combos:
            variant_lists: List[List[Any]] = []
            for spec in combo:
                if isinstance(spec, str):
                    variant_lists.append(tsfm_map.get(spec, [spec]))
                else:
                    variant_lists.append([spec])
            # Cartesian product over variant lists
            for prod in itertools.product(*variant_lists):
                # Sort each spec list so quarterly and monthly dummies come first
                sorted_prod = _sort_specs_with_dummies_first(list(prod))
                expanded.append(sorted_prod)

        self.all_specs = expanded
        return expanded

    def assess_spec(
        self,
        model_id: str,
        specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
        sample: str = 'in',
        test_update_func: Optional[Callable[[ModelBase], dict]] = None,
        outlier_idx: Optional[List[Any]] = None
    ) -> Union[CM, Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str], Dict[str, Dict[str, str]]]]:
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
        outlier_idx : List[Any], optional
            List of index labels (e.g. timestamps or keys) corresponding to outlier
            records to remove from the in-sample data. If provided and `build_in`
            is True, each label must exist within the in-sample period; otherwise,
            a ValueError is raised.

        Returns
        -------
        CM
            The fitted CM instance if all active tests pass.
        (specs, failed_tests, test_info)
            Tuple of the input specs, list of failed test names, and test info dict if any test fails.
        """
        if sample not in {'in', 'full'}:
            raise ValueError("`sample` must be either 'in' or 'full'.")

        # Build the candidate model
        cm = CM(model_id=model_id, target=self.target, model_cls=self.model_cls, data_manager=self.dm)
        cm.build(specs, sample=sample, outlier_idx=outlier_idx)
        mdl = cm.model_in if sample == 'in' else cm.model_full

        # Reload testset, applying update if provided
        mdl.load_testset(test_update_func=test_update_func)

        # Run filtering on updated testset
        passed, failed = mdl.testset.filter_pass()
        if passed:
            return cm
        return specs, failed, mdl.testset.filter_test_info

    def filter_specs(
        self,
        model_id_prefix: str = 'cm',
        sample: str = 'in',
        test_update_func: Optional[Callable[[ModelBase], dict]] = None,
        outlier_idx: Optional[List[Any]] = None
    ) -> Tuple[List[CM], List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]], List[Tuple[List[Any], str, str]]]:
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
        outlier_idx : List[Any], optional
            List of index labels (e.g. timestamps or keys) corresponding to outlier
            records to remove from the in-sample data. If provided and `build_in`
            is True, each label must exist within the in-sample period; otherwise,
            a ValueError is raised.

        Returns
        -------
        passed_cms : list of CM
            CM instances that passed all active tests.
        failed_info : list of (specs, failed_tests)
            Spec combos and test names for combos that failed.
        error_log : list of (specs, error_type, error_message)
            Spec combos that raised errors during assessment.
        """
        # Suppress all warnings and output for the entire filtering process
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            passed_cms: List[CM] = []
            failed_info: List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]] = []
            error_log: List[Tuple[List[Any], str, str]] = []

            # Test info tracking
            seen_test_names: set = set()
            test_info_header_printed = False
            batch_filter_test_infos: Dict[str, Dict[str, str]] = {}  # Collect all filter_test_info in current batch
            
            total = len(self.all_specs)
            start_time = time.time()
            
            # Print initial empty line for spacing
            print("")
            
            for i, specs in enumerate(self.all_specs):
                model_id = f"{model_id_prefix}{i}"
                try:
                    # Suppress all output during assessment
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        result = self.assess_spec(
                            model_id,
                            specs,
                            sample,
                            test_update_func,
                            outlier_idx=outlier_idx
                        )
                    
                    if isinstance(result, CM):
                        passed_cms.append(result)
                        # Get filter test info from successful CM
                        mdl = result.model_in if sample == 'in' else result.model_full
                        filter_test_info = mdl.testset.filter_test_info
                    else:
                        # Extract specs, failed_tests, and filter_test_info from failed result
                        specs_failed, failed_tests, filter_test_info = result
                        failed_info.append((specs_failed, failed_tests))
                    
                    # Update batch filter_test_info (will overwrite duplicates)
                    batch_filter_test_infos.update(filter_test_info)
                    
                    # Determine batch size based on progress
                    batch_size = 100 if i < 10000 else 10000
                    
                    # Process batch based on dynamic batch size or on first run
                    if (i + 1) % batch_size == 0 or i == 0:
                        # Get all test names from the current batch
                        batch_test_names = set(batch_filter_test_infos.keys())
                        
                        # Find new test names not seen before
                        new_test_names = batch_test_names - seen_test_names
                        seen_test_names.update(new_test_names)
                        
                        if new_test_names:
                            # Clear current line completely
                            print("\r" + " " * 120, end="\r")
                            
                            # Print header and empty lines only for the first batch
                            if not test_info_header_printed:
                                print("--- Active Tests of Filtering ---")
                                test_info_header_printed = True
                            
                            # Print test info lines seamlessly
                            for test_name in sorted(new_test_names):
                                if test_name in batch_filter_test_infos:
                                    test_info = batch_filter_test_infos[test_name]
                                    print(f"- {test_name}: filter_mode: {test_info['filter_mode']} | desc: {test_info['desc']}")
                            
                            # Print empty line after test info
                            # print("")  # Empty line after test info
                        
                        # Clear batch memory for next batch
                        batch_filter_test_infos = {}
                            
                except Exception as e:
                    error_log.append((specs, type(e).__name__, str(e)))
     
                # Progress and ETA update (only every 10 iterations to reduce interference)
                if (i + 1) % 10 == 0 or i == 0 or i == len(self.all_specs) - 1:
                    processed = i + 1
                    elapsed = time.time() - start_time
                    progress = processed / total if total > 0 else 1
                    if progress > 0:
                        est_total = elapsed / progress
                        rem = est_total - elapsed
                        finish_dt = datetime.datetime.now() + datetime.timedelta(seconds=rem)
                        eta = finish_dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        eta = ''
                        
                    # Update progress display
                    progress_pct = int(progress * 100)
                    bar_width = 30
                    filled_width = int(bar_width * progress)
                    bar = '█' * filled_width + '-' * (bar_width - filled_width)
                    processed_count = f"{processed}/{total}"
                    speed = processed / elapsed if elapsed > 0 else 0
                    progress_line = f"Filtering Specs: {progress_pct:3d}%|{bar}| {processed_count} [{elapsed:,.0f}s, {speed:.2f}it/s, estimated_finish={eta}]"
                    print(f"\r{progress_line}", end='', flush=True)
            
            # Final newline after progress bar
            print("")
            return passed_cms, failed_info, error_log

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
        max_var_num: int = 10,
        max_lag: int = 3,
        max_periods: int = 3,
        category_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None,
        rank_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        test_update_func: Optional[Callable[[ModelBase], dict]] = None,
        outlier_idx: Optional[List[Any]] = None
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

        Parameters
        ----------
        desired_pool : list
            Pool of variables or transformation specifications to consider.
        forced_in : list, optional
            Variables or specifications that must be included in every model.
        top_n : int, default 10
            Number of top performing models to retain.
        sample : str, default 'in'
            Which sample to use for model building ('in' or 'full').
        max_var_num : int, default 10
            Maximum number of features allowed in each model.
        max_lag : int, default 3
            Maximum lag to consider in transformation specifications.
        max_periods : int, default 3
            Maximum number of periods to consider in transformations.
        category_limit : int, default 1
            Maximum number of variables from each MEV category per combo.
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            Passed to build_spec_combos() and ultimately to DataManager.build_tsfm_specs().
        rank_weights : tuple, default (1.0, 1.0, 1.0)
            Weights for (Fit Measures, IS Error, OOS Error) when ranking models.
        test_update_func : callable, optional
            Optional function to update each CM's test set.
        outlier_idx : list, optional
            List of index labels corresponding to outliers to exclude.

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
              f"Category limit  : {category_limit}\n"
              f"Exp sign map    : {exp_sign_map}\n"
              f"Top N           : {top_n}\n"
              f"Rank weights    : {rank_weights}\n"
              f"Test update func: {test_update_func}\n"
              f"Outlier idx     : {outlier_idx}\n")
        print("==================================\n")

        # 2. Build specs
        combos = self.build_spec_combos(forced, desired_pool, max_var_num, max_lag, max_periods, category_limit, exp_sign_map)
        print(f"Built {len(combos)} spec combinations.\n")

        # 3) Filter specs
        passed, failed, errors = self.filter_specs(
            sample=sample,
            test_update_func=test_update_func,
            outlier_idx=outlier_idx
        )
        # Print empty line after test info
        print("")  # Empty line after test info
        
        self.passed_cms = passed
        self.failed_info = failed
        self.error_log = errors
        # Early exit if nothing passed
        if not self.passed_cms:
            print("\n⚠️  No candidate models passed the filter tests. Search terminated.\n")
            return
        print(f"Passed {len(passed)} combos; Failed {len(failed)} combos; {len(errors)} errors.\n")

        # 4. Rank models
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
    
    def analyze_failures(self) -> None:
        """
        Analyze self.failed_info and print a summary of failed spec combos.

        1. Print total number of failed spec combinations.
        2. Count how many times each test appears across all failures,
           then display a DataFrame of test names and counts (descending).
        3. For the top 5 most frequent failed tests, identify which 3 spec
           elements are most commonly present in combos that failed that test.
           
        Assumes self.failed_info is a list of tuples:
            (specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
             failed_tests: List[str])
        """
        import pandas as pd
        from collections import Counter
        
        # 1) Total number of failed spec combinations
        total_failed = len(self.failed_info)
        print(f"\n=== Failed Spec Combinations Analysis ===")
        print(f"Total failed spec combos: {total_failed}\n")
        
        if total_failed == 0:
            print("No failures to analyze.")
            return
        
        # 2) Count occurrences of each failed test
        # ------------------------------------------
        # Collect all failed test names into a single list
        all_failed_tests = []
        for combo, failed_tests in self.failed_info:
            all_failed_tests.extend(failed_tests)
        
        # Count how many times each test appears
        test_counter = Counter(all_failed_tests)
        
        # Create a DataFrame for clear display
        df_test_counts = (
            pd.DataFrame.from_records(
                list(test_counter.items()),
                columns=["Test Name", "Failure Count"]
            )
            .sort_values(by="Failure Count", ascending=False)
            .reset_index(drop=True)
        )
        
        print("1) Failure counts by test:")
        print(df_test_counts.to_string(index=False))
        print("\n")
        
        # 3) Analyze top 5 most frequent failed tests
        # --------------------------------------------
        top_tests = df_test_counts["Test Name"].head(5).tolist()
        print("2) Top 5 most frequent failed tests and their common spec elements:\n")
        
        for test_name in top_tests:
            # Gather specs for combos that failed this particular test
            related_specs = [
                combo
                for combo, failed_tests in self.failed_info
                if test_name in failed_tests
            ]
            # Flatten the list of spec combos into individual elements
            element_counter = Counter()
            for combo in related_specs:
                for element in combo:
                    # Convert each spec element to string for consistent counting
                    element_counter[str(element)] += 1
            
            # Find the top 3 most common spec elements for this failure
            top_elements = element_counter.most_common(3)
            
            # Print results in a structured way
            print(f"  Test: {test_name}")
            print(f"    Number of combos that failed this test: {len(related_specs)}")
            print("    Top 3 spec elements contributing to this failure:")
            for elem, count in top_elements:
                print(f"      • {elem}  (appeared in {count} combos)")
            print("")  # Blank line between tests

    def analyze_errors(self) -> None:
        """
        Analyze self.error_log and print a summary of spec combinations that raised errors.

        Assumes self.error_log is a list of tuples:
            (specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
             error_type: str,
             error_message: str)

        Steps:
        1. Print total number of spec combos with errors.
        2. Count how many times each error_type appears, then display a DataFrame
           of error types and counts (descending).
        3. For each error_type, list the top 10 most common error_messages with counts,
           and for each (error_type, error_message) pair, identify the 3 most common
           spec elements associated with those combos.
        """
        import pandas as pd
        from collections import Counter, defaultdict

        # 1) Total spec combos that had errors
        total_errors = len(self.error_log)
        print(f"\n=== Error Log Analysis ===")
        print(f"Total spec combos with errors: {total_errors}\n")

        if total_errors == 0:
            print("No errors to analyze.")
            return

        # 2) Count occurrences of each error_type
        # ----------------------------------------
        error_types = [err_type for _, err_type, _ in self.error_log]
        type_counter = Counter(error_types)

        # Build DataFrame for error_type counts
        df_type_counts = (
            pd.DataFrame.from_records(
                list(type_counter.items()),
                columns=["Error Type", "Occurrence Count"]
            )
            .sort_values(by="Occurrence Count", ascending=False)
            .reset_index(drop=True)
        )

        print("1) Error types and their occurrence counts:")
        print(df_type_counts.to_string(index=False))
        print("\n")

        # 3) For each error_type, analyze top messages and associated spec elements
        # ----------------------------------------------------------------------------
        # Organize error entries by type
        errors_by_type = defaultdict(list)
        for specs, err_type, err_msg in self.error_log:
            errors_by_type[err_type].append((specs, err_msg))

        print("2) Detailed breakdown for each error type:\n")

        # Iterate through each error_type in descending order of frequency
        for err_type, count in df_type_counts.itertuples(index=False, name=None):
            print(f"Error Type: {err_type} (Total occurrences: {count})")

            # Collect all messages for this error_type
            messages = [err_msg for _, err_msg in errors_by_type[err_type]]
            msg_counter = Counter(messages)

            # Prepare DataFrame of top 10 error messages
            top_msgs = msg_counter.most_common(10)
            df_msg_counts = (
                pd.DataFrame.from_records(
                    top_msgs,
                    columns=["Error Message", "Count"]
                )
                .reset_index(drop=True)
            )

            print("  a) Top 10 error messages for this type:")
            print(df_msg_counts.to_string(index=False))
            print("")

            # For each top error message, find 3 most common spec elements
            for err_msg, msg_count in top_msgs:
                # Gather spec lists where this err_type and err_msg co-occur
                related_specs = [
                    specs
                    for specs, msg in errors_by_type[err_type]
                    if msg == err_msg
                ]

                # Flatten and count individual spec elements
                element_counter = Counter()
                for combo in related_specs:
                    for elem in combo:
                        element_counter[str(elem)] += 1

                top_elements = element_counter.most_common(3)

                print(f"    Error Message: \"{err_msg}\" (Count: {msg_count})")
                print(f"      Top 3 spec elements associated with this message:")
                for elem, elem_count in top_elements:
                    print(f"        • {elem}  (in {elem_count} combos)")
                print("")

            print("------------------------------------------------------------\n")