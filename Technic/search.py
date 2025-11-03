from typing import List, Union, Tuple, Type, Any, Optional, Callable, Dict, Sequence, Set
import itertools
from collections import defaultdict
import warnings
import sys
import os
import copy
import time
import multiprocessing
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass


@dataclass
class _ProcessWorkerContext:
    """State shared with forked worker processes for spec evaluation."""

    search: "ModelSearch"
    sample: str
    test_update_func: Optional[Callable[[ModelBase], dict]]
    outlier_idx: Optional[List[Any]]
    capture_timings: bool
    parallel_debug_enabled: bool


_PROCESS_WORKER_CONTEXT: Optional[_ProcessWorkerContext] = None


def _set_process_worker_context(context: _ProcessWorkerContext) -> None:
    global _PROCESS_WORKER_CONTEXT
    _PROCESS_WORKER_CONTEXT = context


def _clear_process_worker_context() -> None:
    global _PROCESS_WORKER_CONTEXT
    _PROCESS_WORKER_CONTEXT = None


def _evaluate_spec_in_process(args: Tuple[int, str]) -> Dict[str, Any]:
    """Worker entry point used by ``ProcessPoolExecutor``."""

    if _PROCESS_WORKER_CONTEXT is None:
        raise RuntimeError("Process worker context was not initialised.")

    context = _PROCESS_WORKER_CONTEXT
    search = context.search
    idx, model_id = args

    specs_template = search.all_specs[idx]
    specs_local = search._clone_spec_list(specs_template)

    timings: Optional[Dict[str, float]] = {} if context.capture_timings else None
    start_total = time.perf_counter() if timings is not None else None
    start_wall = time.time() if context.parallel_debug_enabled else None

    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            outcome = search.assess_spec(
                model_id,
                specs_local,
                context.sample,
                context.test_update_func,
                outlier_idx=context.outlier_idx,
                timing=timings,
            )
    except Exception as exc:
        if timings is not None and start_total is not None:
            timings["total"] = time.perf_counter() - start_total
        end_wall = time.time() if context.parallel_debug_enabled else None
        return {
            "index": idx,
            "status": "error",
            "filter_info": {},
            "timing": timings,
            "error": (type(exc).__name__, str(exc)),
            "debug": {
                "start": start_wall,
                "end": end_wall,
                "pid": os.getpid(),
            }
            if context.parallel_debug_enabled
            else None,
        }

    if timings is not None and start_total is not None:
        timings["total"] = time.perf_counter() - start_total

    end_wall = time.time() if context.parallel_debug_enabled else None

    if isinstance(outcome, CM):
        mdl = outcome.model_in if context.sample == "in" else outcome.model_full
        filter_info = dict(mdl.testset.filter_test_info or {})
        return {
            "index": idx,
            "status": "passed",
            "filter_info": filter_info,
            "timing": timings,
            "debug": {
                "start": start_wall,
                "end": end_wall,
                "pid": os.getpid(),
            }
            if context.parallel_debug_enabled
            else None,
        }

    specs_failed, failed_tests, filter_info = outcome
    return {
        "index": idx,
        "status": "failed",
        "failed_tests": failed_tests,
        "filter_info": dict(filter_info or {}),
        "timing": timings,
        "debug": {
            "start": start_wall,
            "end": end_wall,
            "pid": os.getpid(),
        }
        if context.parallel_debug_enabled
        else None,
    }

import pandas as pd
from tqdm import tqdm

from .data import DataManager
from .feature import Feature, DumVar
from .transform import TSFM
from .model import ModelBase
from .cm import CM
from .periods import default_periods_for_freq, resolve_periods_argument


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


def _summarize_parallel_events(
    events: List[Tuple[float, str]],
    max_active: int,
    actors: Set[str],
    cpu_samples: List[float],
    actor_label: str = "workers",
) -> str:
    """Create a human-readable summary of parallel execution diagnostics."""

    if not events:
        return ""

    events_sorted = sorted(events, key=lambda item: item[0])
    first_ts = events_sorted[0][0]
    last_ts = events_sorted[-1][0]
    active = 0
    last_time = first_ts
    overlap = 0.0

    for ts, kind in events_sorted:
        if ts > last_time and active > 1:
            overlap += ts - last_time
        if kind == "start":
            active += 1
        else:
            active -= 1
        last_time = ts

    total_elapsed = max(0.0, last_ts - first_ts)
    summary_lines = [
        "--- Parallel debug summary ---",
        f"{actor_label} used: {len(actors)} | max concurrent tasks: {max_active}",
        f"elapsed (first start -> last end): {total_elapsed:.3f}s",
        f"time with >1 active task: {overlap:.3f}s",
    ]

    if cpu_samples:
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        peak_cpu = max(cpu_samples)
        summary_lines.append(
            f"process CPU% avg {avg_cpu:.1f} | peak {peak_cpu:.1f} (psutil sampling)"
        )
    else:
        summary_lines.append(
            "process CPU%: n/a (psutil not installed or sampling disabled)"
        )

    return "\n".join(summary_lines)


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
        model_cls: Type[ModelBase],
        model_type: Optional[Any] = None,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None,
        qtr_method: str = 'mean'
    ):
        self.dm = dm
        self.target = target
        self.model_cls = model_cls
        self.model_type = model_type
        self.target_base = target_base
        self.target_exposure = target_exposure
        self.qtr_method = qtr_method
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
        periods: Optional[Sequence[int]] = None,
        category_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None,
        **legacy_kwargs: Any
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
        periods : Sequence[int], optional
            Period configuration forwarded to :meth:`DataManager.build_tsfm_specs`.
            Provide a list of positive integers to explicitly control
            period-based transforms. Recommended choices include
            ``[1, 2, 3, 6, 9, 12]`` for monthly data and ``[1, 2, 3, 4]`` for
            quarterly data. When ``None`` (default), frequency-aware defaults
            are applied automatically. The deprecated ``max_periods`` keyword is
            still accepted for backward compatibility.
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
        forced_specs_template = forced_in or []

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
        if forced_specs_template and len(forced_specs_template) <= max_var_num:
            combos.append(self._clone_spec_list(forced_specs_template))

        # Mix forced with each raw combo within max_var_num
        for rc in raw_combos:
            if len(forced_specs_template) + len(rc) <= max_var_num:
                combo = self._clone_spec_list(forced_specs_template)
                combo.extend(self._clone_spec_list(rc))
                combos.append(combo)

        # Step 3: Expand only top-level strings into TSFM variants
        # Gather unique strings to expand
        top_strings = {spec for combo in combos for spec in combo if isinstance(spec, str)}
        
        legacy_max_periods = legacy_kwargs.pop("max_periods", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        freq = getattr(self.dm, "freq", None)
        resolved_periods = resolve_periods_argument(
            freq,
            periods,
            legacy_max_periods=legacy_max_periods
        )

        # Suppress warnings during TSFM spec building
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsfm_map = self.dm.build_tsfm_specs(
                list(top_strings),
                max_lag=max_lag,
                periods=resolved_periods,
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
                expanded.append(self._clone_spec_list(sorted_prod))

        self.all_specs = expanded
        return expanded
    
        def assess_spec(
            self,
            model_id: str,
            specs: List[Union[str, TSFM, Feature, Tuple[Any, ...]]],
            sample: str = 'in',
            test_update_func: Optional[Callable[[ModelBase], dict]] = None,
            outlier_idx: Optional[List[Any]] = None,
            timing: Optional[Dict[str, float]] = None
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
            timing : dict, optional
                Mutable dictionary that will be populated with timing information when
                provided. Keys include ``build`` (time spent in ``CM.build``),
                ``testset`` (time building/updating the test set), ``filter`` (time
                evaluating ``TestSet.filter_pass``), and ``total`` (overall elapsed time
                measured by the caller).
    
            Returns
            -------
            CM
                The fitted CM instance if all active tests pass.
            (specs, failed_tests, test_info)
                Tuple of the input specs, list of failed test names, and test info dict if any test fails.
            """
            if sample not in {'in', 'full'}:
                raise ValueError("`sample` must be either 'in' or 'full'.")
    
            t_build_start = time.perf_counter() if timing is not None else None
    
            # Build the candidate model
            cm = CM(
                model_id=model_id,
                target=self.target,
                model_type=self.model_type,
                target_base=self.target_base,
                target_exposure=self.target_exposure,
                model_cls=self.model_cls,
                data_manager=self.dm,
                qtr_method=self.qtr_method
            )
            cm.build(specs, sample=sample, outlier_idx=outlier_idx)
            if timing is not None and t_build_start is not None:
                timing['build'] = time.perf_counter() - t_build_start
            mdl = cm.model_in if sample == 'in' else cm.model_full
    
            # Reload testset, applying update if provided
            t_testset_start = time.perf_counter() if timing is not None else None
            mdl.load_testset(test_update_func=test_update_func)
            if timing is not None and t_testset_start is not None:
                timing['testset'] = time.perf_counter() - t_testset_start
    
            # Run filtering on updated testset (fast mode to short-circuit on first failure)
            t_filter_start = time.perf_counter() if timing is not None else None
            passed, failed = mdl.testset.filter_pass(fast_filter=True)
            if timing is not None and t_filter_start is not None:
                timing['filter'] = time.perf_counter() - t_filter_start
            if passed:
                return cm
            return specs, failed, mdl.testset.filter_test_info
    
        def filter_specs(
            self,
            model_id_prefix: str = 'cm',
            sample: str = 'in',
            test_update_func: Optional[Callable[[ModelBase], dict]] = None,
            outlier_idx: Optional[List[Any]] = None,
            parallel: bool = False,
            max_workers: Optional[int] = None
        ) -> Tuple[List[CM], List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]], List[Tuple[List[Any], str, str]]]:
            """
            Assess all built spec combos and separate passed and failed results,
            using multiprocessing and a single progress bar update.
    
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
                try:
                    total = len(self.all_specs)
                    progress_bar = tqdm(
                        total=total,
                        desc="Filtering Specs",
                        unit="spec",
                        leave=False,
                        dynamic_ncols=True,
                        file=sys.stdout,
                    )
                    progress_start = time.perf_counter()
        
                    def _update_progress_postfix(processed: int) -> None:
                        """Update tqdm postfix with an estimated finish timestamp."""
        
                        if total == 0:
                            progress_bar.set_postfix(estimated_finish="n/a")
                            return
                        if processed <= 0:
                            progress_bar.set_postfix(estimated_finish="calculating")
                            return
                        elapsed = time.perf_counter() - progress_start
                        if elapsed <= 0:
                            progress_bar.set_postfix(estimated_finish="calculating")
                            return
                        remaining = max(0.0, (elapsed / processed) * (total - processed))
                        finish_time = datetime.now() + timedelta(seconds=remaining)
                        progress_bar.set_postfix(
                            estimated_finish=finish_time.strftime("%Y-%m-%d %H:%M:%S")
                        )
        
                    _update_progress_postfix(0)
        
                    capture_timings = os.getenv("LEGO_SEARCH_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}
                    parallel_debug_enabled = os.getenv("LEGO_SEARCH_PARALLEL_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        
                    def _finalize_results(records: List[Dict[str, Any]]) -> Tuple[List[CM], List[Tuple[List[Any], List[str]]], List[Tuple[List[Any], str, str]]]:
                        passed: List[CM] = []
                        failed: List[Tuple[List[Any], List[str]]] = []
                        errors: List[Tuple[List[Any], str, str]] = []
                        seen_test_names: Set[str] = set()
                        header_printed = False
        
                        for rec in sorted(records, key=lambda r: r["index"]):
                            status = rec["status"]
                            payload = rec["payload"]
                            filter_info = rec.get("filter_info") or {}
        
                            if status == "passed":
                                passed.append(payload)
                            elif status == "failed":
                                failed.append(payload)
                            elif status == "error":
                                errors.append(payload)
                                continue
        
                            new_test_names = set(filter_info.keys()) - seen_test_names
                            if new_test_names:
                                if not header_printed:
                                    print("--- Active Tests of Filtering ---")
                                    header_printed = True
                                for test_name in sorted(new_test_names):
                                    info = filter_info[test_name]
                                    print(f"- {test_name}: filter_mode: {info['filter_mode']} | desc: {info['desc']}")
                                seen_test_names.update(new_test_names)
        
                        return passed, failed, errors
        
                    def _print_timing_summary(records: List[Dict[str, Any]]) -> None:
                        if not capture_timings:
                            return
                        phase_totals: Dict[str, List[float]] = defaultdict(list)
                        for rec in records:
                            timing_info = rec.get("timing")
                            if not timing_info:
                                continue
                            for phase, value in timing_info.items():
                                if value is not None:
                                    phase_totals[phase].append(value)
                        if not phase_totals:
                            return
                        print("\n--- Timing diagnostics (seconds) ---")
                        for phase in ("total", "build", "testset", "filter"):
                            vals = phase_totals.get(phase)
                            if not vals:
                                continue
                            avg = sum(vals) / len(vals)
                            mx = max(vals)
                            print(f"{phase:>8}: avg {avg:.3f} | max {mx:.3f} | n={len(vals)}")
                        print("")
        
                    # Print initial empty line for spacing
                    print("")
        
                    records: List[Dict[str, Any]] = []
        
                    use_parallel = parallel and total > 1

                    if parallel_debug_enabled and (not parallel or total <= 1):
                        reason = "parallel flag disabled" if not parallel else "only one spec to evaluate"
                        print(f"[parallel-debug] {reason}; running serial path.")

                    mp_context = None
                    if use_parallel:
                        try:
                            mp_context = multiprocessing.get_context("fork")
                        except ValueError:
                            print(
                                "[parallel] 'fork' start method unavailable; falling back to serial execution."
                            )
                            use_parallel = False

                    if use_parallel:
                        unique_specs, feature_columns = self._materialize_parallel_feature_pool()
                        if unique_specs:
                            print(
                                f"[parallel] Created feature cache from {unique_specs} unique spec elements "
                                f"covering {feature_columns} column(s)."
                            )
                        else:
                            print("[parallel] No feature materialization needed (specs empty).")
                        cache_info = self.dm.prepare_parallel_feature_caches()
                        print(
                            "[parallel] DataManager caches ready: "
                            f"internal_cols={cache_info['internal_cols']} | "
                            f"mev_cols={cache_info['mev_cols']}"
                        )
                        self.dm.lock_parallel_read_only()
                        dm_locked = True
                        print("[parallel] DataManager locked for read-only process execution.")
        
                        if max_workers and max_workers > 0:
                            worker_count = max(1, min(max_workers, total))
                        else:
                            default_workers = os.cpu_count() or 1
                            worker_count = max(1, min(default_workers, total))

                        assignments: List[int] = []
                        if worker_count > 0:
                            base = total // worker_count
                            remainder = total % worker_count
                            assignments = [base + (1 if idx < remainder else 0) for idx in range(worker_count)]
                        print(f"[parallel] Launching {worker_count} worker processes for {total} specs.")
                        if assignments:
                            print(f"[parallel] Process spec allocation: {assignments}")

                        processed = 0
                        raw_records: List[Dict[str, Any]] = []
                        _set_process_worker_context(
                            _ProcessWorkerContext(
                                search=self,
                                sample=sample,
                                test_update_func=test_update_func,
                                outlier_idx=outlier_idx,
                                capture_timings=capture_timings,
                                parallel_debug_enabled=parallel_debug_enabled,
                            )
                        )
                        executor_kwargs: Dict[str, Any] = {"max_workers": worker_count}
                        if mp_context is not None:
                            executor_kwargs["mp_context"] = mp_context
                        try:
                            with ProcessPoolExecutor(**executor_kwargs) as executor:
                                futures = [
                                    executor.submit(
                                        _evaluate_spec_in_process, (i, f"{model_id_prefix}{i}")
                                    )
                                    for i in range(total)
                                ]
                                for future in as_completed(futures):
                                    record = future.result()
                                    raw_records.append(record)
                                    processed += 1
                                    progress_bar.update(1)
                                    _update_progress_postfix(processed)
                        finally:
                            _clear_process_worker_context()

                        if parallel_debug_enabled:
                            debug_events: List[Tuple[float, str]] = []
                            actor_ids: Set[str] = set()
                            for rec in raw_records:
                                dbg = rec.get("debug") if isinstance(rec, dict) else None
                                if not dbg:
                                    continue
                                start = dbg.get("start")
                                end = dbg.get("end")
                                pid = dbg.get("pid")
                                if start is None or end is None:
                                    continue
                                debug_events.append((start, "start"))
                                debug_events.append((end, "end"))
                                if pid is not None:
                                    actor_ids.add(str(pid))
                            if debug_events:
                                debug_events.sort(key=lambda item: item[0])
                                active = 0
                                debug_max_active = 0
                                for _, kind in debug_events:
                                    if kind == "start":
                                        active += 1
                                        debug_max_active = max(debug_max_active, active)
                                    else:
                                        active -= 1
                                summary = _summarize_parallel_events(
                                    debug_events,
                                    debug_max_active,
                                    actor_ids,
                                    [],
                                    actor_label="processes",
                                )
                                print(summary)
                            else:
                                print("[parallel-debug] No debug timing data returned from workers.")

                        records = []
                        for rec in raw_records:
                            idx = rec["index"]
                            status = rec["status"]
                            timing_info = rec.get("timing")
                            filter_info = rec.get("filter_info") or {}
                            debug_payload = rec.get("debug")
                            if status == "passed":
                                specs_for_build = self._clone_spec_list(self.all_specs[idx])
                                model_id = f"{model_id_prefix}{idx}"
                                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                                    cm_result = self.assess_spec(
                                        model_id,
                                        specs_for_build,
                                        sample,
                                        test_update_func,
                                        outlier_idx=outlier_idx,
                                        timing=None,
                                    )
                                if not isinstance(cm_result, CM):
                                    # Should not happen; treat as failure fallback.
                                    specs_failed, failed_tests, filter_info_secondary = cm_result
                                    records.append(
                                        {
                                            "index": idx,
                                            "status": "failed",
                                            "payload": (
                                                specs_failed,
                                                failed_tests,
                                            ),
                                            "filter_info": filter_info_secondary,
                                            "timing": timing_info,
                                            "debug": debug_payload,
                                        }
                                    )
                                    continue
                                mdl = cm_result.model_in if sample == "in" else cm_result.model_full
                                effective_filter = filter_info or dict(mdl.testset.filter_test_info or {})
                                records.append(
                                    {
                                        "index": idx,
                                        "status": "passed",
                                        "payload": cm_result,
                                        "filter_info": effective_filter,
                                        "timing": timing_info,
                                        "debug": debug_payload,
                                    }
                                )
                            elif status == "failed":
                                specs_failed = self._clone_spec_list(self.all_specs[idx])
                                failed_tests = rec.get("failed_tests") or []
                                records.append(
                                    {
                                        "index": idx,
                                        "status": "failed",
                                        "payload": (specs_failed, failed_tests),
                                        "filter_info": filter_info,
                                        "timing": timing_info,
                                        "debug": debug_payload,
                                    }
                                )
                            elif status == "error":
                                specs_error = self._clone_spec_list(self.all_specs[idx])
                                err_type, err_msg = rec.get("error", ("Error", "Unknown error"))
                                records.append(
                                    {
                                        "index": idx,
                                        "status": "error",
                                        "payload": (specs_error, err_type, err_msg),
                                        "filter_info": filter_info,
                                        "timing": timing_info,
                                        "debug": debug_payload,
                                    }
                                )

                        progress_bar.close()
                        results = _finalize_results(records)
                        _print_timing_summary(records)
                        return results
        
                    # Serial fallback
                    processed_serial = 0
                    for i, specs in enumerate(self.all_specs):
                        model_id = f"{model_id_prefix}{i}"
                        specs_for_build = self._clone_spec_list(specs)
                        timings: Optional[Dict[str, float]] = {} if capture_timings else None
                        start = time.perf_counter() if timings is not None else None
                        try:
                            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                                result = self.assess_spec(
                                    model_id,
                                    specs_for_build,
                                    sample,
                                    test_update_func,
                                    outlier_idx=outlier_idx,
                                    timing=timings
                                )
                            if timings is not None and start is not None:
                                timings['total'] = time.perf_counter() - start
                            if isinstance(result, CM):
                                mdl = result.model_in if sample == 'in' else result.model_full
                                records.append(
                                    {
                                        "index": i,
                                        "status": "passed",
                                        "payload": result,
                                        "filter_info": dict(mdl.testset.filter_test_info or {}),
                                        "timing": timings,
                                    }
                                )
                            else:
                                specs_failed, failed_tests, filter_info = result
                                records.append(
                                    {
                                        "index": i,
                                        "status": "failed",
                                        "payload": (specs_failed, failed_tests),
                                        "filter_info": dict(filter_info or {}),
                                        "timing": timings,
                                    }
                                )
                        except Exception as e:
                            if timings is not None and start is not None:
                                timings['total'] = time.perf_counter() - start
                            records.append(
                                {
                                    "index": i,
                                    "status": "error",
                                    "payload": (specs_for_build, type(e).__name__, str(e)),
                                    "filter_info": {},
                                    "timing": timings,
                                }
                            )
        
                        processed_serial += 1
                        progress_bar.update(1)
                        _update_progress_postfix(processed_serial)
        
                    progress_bar.close()
                    results = _finalize_results(records)
                    _print_timing_summary(records)
                    return results
        
                finally:
                    if dm_locked:
                        try:
                            self.dm.unlock_parallel_read_only()
                            print("[parallel] DataManager read-only lock released.")
                        except Exception as exc:
                            print(f"[parallel] Warning: failed to release DataManager lock ({exc}).")
    @staticmethod
    def _clone_spec_item(item: Any) -> Any:
        if isinstance(item, Feature):
            return copy.deepcopy(item)
        if isinstance(item, tuple):
            return tuple(ModelSearch._clone_spec_item(sub) for sub in item)
        if isinstance(item, list):
            return [ModelSearch._clone_spec_item(sub) for sub in item]
        return item

    @classmethod
    def _clone_spec_list(cls, specs: Sequence[Any]) -> List[Any]:
        return [cls._clone_spec_item(spec) for spec in specs]

    def _materialize_parallel_feature_pool(self) -> Tuple[int, int]:
        """Materialize a consolidated feature DataFrame covering all unique specs.

        Returns
        -------
        tuple
            (number of unique spec elements, number of resulting feature columns).
        """

        if not self.all_specs:
            return 0, 0

        unique_specs: List[Any] = []
        seen_strings: Set[str] = set()
        seen_features: Set[str] = set()

        def _collect(item: Any) -> None:
            if isinstance(item, Feature):
                key = repr(item)
                if key not in seen_features:
                    seen_features.add(key)
                    unique_specs.append(copy.deepcopy(item))
            elif isinstance(item, str):
                if item not in seen_strings:
                    seen_strings.add(item)
                    unique_specs.append(item)
            elif isinstance(item, (list, tuple, set)):
                for sub in item:
                    _collect(sub)

        for combo in self.all_specs:
            for spec in combo:
                _collect(spec)

        if not unique_specs:
            return 0, 0

        try:
            feature_df = self.dm.build_features(unique_specs)
        except Exception as exc:
            print(f"[parallel] Warning: unable to build consolidated feature cache ({exc}).")
            return len(unique_specs), 0

        column_count = feature_df.shape[1]
        return len(unique_specs), column_count

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

        def _flatten_test_result(obj: Any) -> pd.Series:
            """
            Convert a test_result (Series or DataFrame) into a flat Series of numeric values.
            - If DataFrame, prefer 'Value' column; otherwise, the first numeric column.
            - Returns empty Series if nothing usable.
            """
            if obj is None:
                return pd.Series(dtype=float)
            if isinstance(obj, pd.Series):
                return obj.astype(float)
            if isinstance(obj, pd.DataFrame):
                if 'Value' in obj.columns:
                    return obj['Value'].astype(float)
                numeric_cols = obj.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    return obj[numeric_cols[0]].astype(float)
            return pd.Series(dtype=float)
        for cm in cms_list:
            mdl = cm.model_in if sample == 'in' else cm.model_full
            testdict = {t.name: t for t in mdl.testset.tests}
            fit_sr = _flatten_test_result(testdict['Fit Measures'].test_result) if 'Fit Measures' in testdict else pd.Series(dtype=float)
            is_err_sr = _flatten_test_result(testdict['IS Error Measures'].test_result) if 'IS Error Measures' in testdict else pd.Series(dtype=float)
            oos_err_obj = testdict.get('OOS Error Measures')
            oos_err_sr = _flatten_test_result(oos_err_obj.test_result) if oos_err_obj is not None else pd.Series(dtype=float)
            rec = {'model_id': cm.model_id}
            for nm, val in fit_sr.items():
                rec[f'fit_{nm}'] = float(val)
            for nm, val in is_err_sr.items():
                rec[f'is_err_{nm}'] = float(val)
            if not oos_err_sr.empty:
                for nm, val in oos_err_sr.items():
                    rec[f'oos_err_{nm}'] = float(val)
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
        periods: Optional[Sequence[int]] = None,
        category_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None,
        rank_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        test_update_func: Optional[Callable[[ModelBase], dict]] = None,
        outlier_idx: Optional[List[Any]] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        **legacy_kwargs: Any
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
        periods : Sequence[int], optional
            Period configuration forwarded to :meth:`DataManager.build_tsfm_specs`.
            Provide a list of positive integers to explicitly control
            period-based transforms. Recommended choices include
            ``[1, 2, 3, 6, 9, 12]`` for monthly data and ``[1, 2, 3, 4]`` for
            quarterly data. When ``None`` (default), frequency-aware defaults
            are applied automatically. The deprecated ``max_periods`` keyword is
            still accepted for backward compatibility.
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
        parallel : bool, default False
            Enable process-based assessment of candidate specs. Set to True to
            evaluate combinations concurrently while assuming read-only access to
            the shared DataManager caches. Each worker runs in its own Python
            interpreter, avoiding the GIL, but requires the ``fork`` start method
            so the pre-built caches can be inherited efficiently.
        max_workers : int, optional
            Maximum number of worker processes used when ``parallel`` is True.
            ``None`` lets the executor default to ``os.cpu_count()``; non-positive
            values are ignored.

        Returns
        -------
        top_models : list of CM
            The top_n CM instances sorted by composite score.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forced = forced_in or []

            legacy_max_periods = legacy_kwargs.pop("max_periods", None)
            if legacy_kwargs:
                unexpected = ", ".join(sorted(legacy_kwargs.keys()))
                raise TypeError(f"Unexpected keyword arguments: {unexpected}")

            freq = getattr(self.dm, "freq", None)
            resolved_periods = resolve_periods_argument(
                freq,
                periods,
                legacy_max_periods=legacy_max_periods
            )
            if resolved_periods is None:
                periods_summary = default_periods_for_freq(freq)
            else:
                periods_summary = resolved_periods

            # 1. Configuration
            print("=== ModelSearch Configuration ===")
            print(f"Target          : {self.target}")
            print(f"Model class     : {self.model_cls.__name__}")
            print(f"Desired pool    : {desired_pool}")
            print(f"Forced in       : {forced}")
            print(f"Sample          : {sample}\n"
                  f"Max var num     : {max_var_num}\n"
                  f"Max lag         : {max_lag}\n"
                  f"Periods         : {periods_summary}\n"
                  f"Category limit  : {category_limit}\n"
                  f"Exp sign map    : {exp_sign_map}\n"
                  f"Top N           : {top_n}\n"
                  f"Rank weights    : {rank_weights}\n"
                  f"Test update func: {test_update_func}\n"
                  f"Outlier idx     : {outlier_idx}\n")
            print("==================================\n")
        
            # Warn about interpolated MEV variables within the candidate pool
            def _flatten(items: Any) -> List[Union[str, TSFM]]:
                flat: List[Union[str, TSFM]] = []
                for it in items:
                    if isinstance(it, (str, TSFM)):
                        flat.append(it)
                    elif isinstance(it, (list, tuple, set)):
                        flat.extend(_flatten(it))
                return flat

            vars_to_check = _flatten(forced + desired_pool)
            interp_df = self.dm.interpolated_vars(vars_to_check)
            if interp_df is not None:
                print(interp_df.to_string(index=False))
                print("")
            
            # 2. Build specs
            combos = self.build_spec_combos(
                forced,
                desired_pool,
                max_var_num,
                max_lag,
                periods=resolved_periods,
                category_limit=category_limit,
                exp_sign_map=exp_sign_map,
            )
            print(f"Built {len(combos)} spec combinations.\n")

            # 3) Filter specs
            passed, failed, errors = self.filter_specs(
                sample=sample,
                test_update_func=test_update_func,
                outlier_idx=outlier_idx,
                parallel=parallel,
                max_workers=max_workers
            )
            # Print empty line after test info
            print("")  # Empty line after test info
        
            self.passed_cms = passed
            self.failed_info = failed
            self.error_log = errors
            # Early exit if nothing passed
            if not self.passed_cms:
                print(f"Passed {len(passed)} combos; Failed {len(failed)} combos; {len(errors)} errors.\n")
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
