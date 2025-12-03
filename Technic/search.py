# =============================================================================
# module: search.py
# Purpose: Provide model search utilities for generating and ranking CM specs.
# Key Types/Classes: ModelSearch
# Key Functions: _sort_specs_with_dummies_first
# Dependencies: itertools, datetime, pandas, tqdm, logging, .pretest.PreTestSet
# =============================================================================

from typing import List, Union, Tuple, Type, Any, Optional, Callable, Dict, Sequence, Set
import itertools
import time
import datetime
import inspect
import json
from copy import deepcopy
from collections import defaultdict
import warnings
import sys
import os
import logging
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path


LOGGER = logging.getLogger(__name__)

import pandas as pd
from tqdm import tqdm

from .data import DataManager
from .feature import Feature, DumVar
from .transform import TSFM
from .model import ModelBase
from .pretest import PreTestSet, FeatureTest
from .cm import CM
from .periods import default_periods_for_freq, resolve_periods_argument
from .regime import RgmVar
from .persistence import (
    ensure_segment_dirs,
    sanitize_segment_id,
    generate_search_id,
    get_search_paths,
    load_index,
    load_cm,
    save_cm,
    save_index,
)


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
    model_type : Any, optional
        Optional model type metadata forwarded to downstream builders.
    target_base : str, optional
        Base variable associated with the modeling target.
    target_exposure : str, optional
        Exposure variable when ratio models are in use.
    qtr_method : str, default 'mean'
        Aggregation method for quarterly transformations.
    progress_log_interval_sec : float, optional
        Minimum elapsed seconds between heartbeat progress logs emitted during
        lengthy ``filter_specs`` runs. Defaults to 5 minutes.
    """

    def __init__(
        self,
        dm: DataManager,
        target: str,
        model_cls: Type[ModelBase],
        model_type: Optional[Any] = None,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None,
        qtr_method: str = 'mean',
        progress_log_interval_sec: float = 5 * 60
    ):
        self.dm = dm
        self.target = target
        self.model_cls = model_cls
        self.model_type = model_type
        self.target_base = target_base
        self.target_exposure = target_exposure
        self.qtr_method = qtr_method
        self.progress_log_interval_sec = progress_log_interval_sec
        self.segment: Optional[Any] = None
        self.all_specs: List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]] = []
        self.passed_cms: List[CM] = []
        self.failed_info: List[Tuple[List[Any], List[str]]] = []
        self.error_log: List[Tuple[List[Any], str, str]] = []
        self.df_scores: Optional[pd.DataFrame] = None
        self.top_cms: List[CM] = []
        self.model_pretestset: Optional[PreTestSet] = None
        self.target_pretest_result: Optional[Any] = None
        self.current_search_config_raw: Optional[Dict[str, Any]] = None
        self.current_search_config: Optional[Dict[str, Any]] = None
        self.search_id: Optional[str] = None
        self.total_combos: int = 0
        self.completed_combos: int = 0

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        Convert potentially non-serializable objects into JSON-safe forms.

        Parameters
        ----------
        obj : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON-serializable representation.
        """

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if callable(obj):
            module = getattr(obj, "__module__", None)
            name = getattr(obj, "__name__", None)
            if module and name:
                return f"{module}.{name}"
            return repr(obj)

        if isinstance(obj, dict):
            return {ModelSearch._make_serializable(k): ModelSearch._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [ModelSearch._make_serializable(v) for v in obj]

        return repr(obj)

    @staticmethod
    def _configs_equivalent(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> bool:
        """
        Compare two search configurations ignoring metadata-only keys.

        Parameters
        ----------
        config_a : dict
            First configuration mapping.
        config_b : dict
            Second configuration mapping.

        Returns
        -------
        bool
            ``True`` when configurations match on meaningful search parameters.
        """

        ignored_keys = {"search_id", "timestamp", "total_combos"}
        def _filtered(config: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in config.items() if k not in ignored_keys}

        return _filtered(config_a) == _filtered(config_b)

    @staticmethod
    def _latest_search_config(
        segment_id: str, base_dir: Path
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Return the most recent search configuration for a segment.

        Parameters
        ----------
        segment_id : str
            Segment identifier to inspect.
        base_dir : pathlib.Path
            Working directory that contains the capitalized ``Segment`` root.

        Returns
        -------
        tuple[str, dict] or None
            ``(search_id, config)`` for the most recent search when available;
            otherwise ``None``.
        """

        dirs = ensure_segment_dirs(segment_id, base_dir)
        index_path = dirs["cms_dir"] / "search_index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        prefix = f"search_{sanitize_segment_id(segment_id)}_"

        def _timestamp_value(search_label: str) -> str:
            return search_label.rsplit("_", 1)[-1]

        matching_ids = [sid for sid in index if sid.startswith(prefix)]
        if not matching_ids:
            return None

        latest_id = max(matching_ids, key=_timestamp_value)
        return latest_id, index.get(latest_id)

    @staticmethod
    def _find_resume_candidate(
        segment_id: str,
        base_dir: Path,
        prospective_config: Dict[str, Any],
        total_combos: int,
    ) -> Optional[Tuple[str, Dict[str, Any], int]]:
        """
        Locate the newest compatible search configuration for potential resume.

        Parameters
        ----------
        segment_id : str
            Identifier for the active segment.
        base_dir : pathlib.Path
            Working directory that houses the ``Segment`` folder.
        prospective_config : dict
            Search parameters prepared for the current invocation.
        total_combos : int
            Number of combinations calculated for the current search.

        Returns
        -------
        tuple[str, dict, int] or None
            ``(search_id, config, completed_combos)`` for the most recent
            matching search when available. Returns ``None`` when no compatible
            search is found. Preference is given to the newest *incomplete*
            search based on progress metadata; a finished run is only returned
            when no interrupted counterpart exists.
        """

        dirs = ensure_segment_dirs(segment_id, base_dir)
        index_path = dirs["cms_dir"] / "search_index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        prefix = f"search_{sanitize_segment_id(segment_id)}_"

        def _timestamp_value(search_label: str) -> str:
            return search_label.rsplit("_", 1)[-1]

        # Scan all matching search IDs in reverse chronological order so the
        # newest compatible run is selected. Only incomplete runs are eligible
        # for resume; fully finished searches should not trigger prompts for
        # continuation.
        matching_ids = [sid for sid in index if sid.startswith(prefix)]
        matching_ids.sort(key=_timestamp_value, reverse=True)

        for search_id in matching_ids:
            config_body = index.get(search_id) or {}
            if not ModelSearch._configs_equivalent(config_body, prospective_config):
                continue
            if config_body.get("total_combos") != total_combos:
                continue

            progress_path = dirs["log_dir"] / f"{search_id}.progress"
            progress_info = ModelSearch._read_progress(progress_path)
            completed = int(progress_info.get("completed_combos", 0)) if progress_info else 0
            if progress_info and progress_info.get("total_combos") != total_combos:
                continue

            # Only incomplete runs can be resumed.
            if completed < total_combos:
                return search_id, config_body, completed

        # No interrupted run available for these parameters.
        return None

    @staticmethod
    def _load_passed_cms_from_dir(target_dir: Path, dm: Optional[DataManager]) -> List[CM]:
        """
        Load persisted passed candidate models from disk.

        Parameters
        ----------
        target_dir : pathlib.Path
            Directory containing the ``index.json`` file and CM pickles.
        dm : DataManager, optional
            Data manager to bind to each loaded CM if provided.

        Returns
        -------
        List[CM]
            Loaded candidate models; empty when none are present.
        """

        try:
            index_entries = load_index(target_dir)
        except FileNotFoundError:
            return []

        restored: List[CM] = []
        for entry in index_entries:
            cm_path = target_dir / entry["filename"]
            try:
                cm = load_cm(cm_path)
                if dm is not None:
                    cm.bind_data_manager(dm)
                restored.append(cm)
            except Exception:  # pragma: no cover - corrupted pickles handled gracefully
                continue
        return restored

    @staticmethod
    def _read_progress(progress_path: Path) -> Optional[Dict[str, int]]:
        """Read a progress file if present."""

        if not progress_path.exists():
            return None
        try:
            with progress_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _write_progress(progress_path: Path, total_combos: int, completed_combos: int) -> None:
        """Persist progress to disk overwriting any prior file."""

        with progress_path.open("w", encoding="utf-8") as handle:
            json.dump({"total_combos": total_combos, "completed_combos": completed_combos}, handle, indent=2)

    def _resolve_model_pretestset(self) -> Optional[PreTestSet]:
        """Return a deep-copied default pretest bundle for ``self.model_cls``.

        Returns
        -------
        Optional[PreTestSet]
            Deep copy of the default ``pretestset`` argument defined on the
            model class ``__init__`` signature. ``None`` when the class does not
            expose a pretest bundle or the default value is empty.
        """

        try:
            signature = inspect.signature(self.model_cls.__init__)
        except (TypeError, ValueError):
            return None

        parameter = signature.parameters.get("pretestset")
        if parameter is None or parameter.default is inspect._empty:
            return None

        default_bundle = parameter.default
        if default_bundle is None:
            return None

        # Deep copy to avoid mutating shared constants such as
        # ``ppnr_ols_pretestset`` while populating runtime dependencies.
        return deepcopy(default_bundle)

    def _propagate_target_context(self, target_result: Optional[bool]) -> None:
        """Cache and broadcast the target pre-test outcome to dependents.

        Parameters
        ----------
        target_result : bool, optional
            Outcome of the target-level pre-test. ``True`` preserves the
            default expectation that downstream features should remain
            stationary, while ``False`` inverts the expectation for
            feature-level checks. ``None`` clears any previously cached
            outcome.

        Notes
        -----
        The payload always includes both ``"target_result"`` and
        ``"target_test_result"`` so context routing rules can select the key
        appropriate for each configured pre-test.
        """

        self.target_pretest_result = None if target_result is None else bool(target_result)

        if self.model_pretestset is None:
            return

        context_payload = {
            "target_result": self.target_pretest_result,
            "target_test_result": self.target_pretest_result,
        }
        self.model_pretestset.propagate_context(context_payload)

    def _prepare_feature_pretest(self) -> Optional[FeatureTest]:
        """Return the active feature pre-test with the correct data context.

        Returns
        -------
        Optional[FeatureTest]
            The :class:`FeatureTest` instance configured on ``self.model_cls``
            after aligning it with the current :class:`DataManager`. ``None``
            is returned when no feature-level validation has been supplied.
        """

        if self.model_pretestset is None:
            self.model_pretestset = self._resolve_model_pretestset()

        if self.model_pretestset is None:
            return None

        if self.target_pretest_result is not None:
            self._propagate_target_context(self.target_pretest_result)

        feature_test = self.model_pretestset.feature_test
        if feature_test is None:
            return None

        # Always ensure the feature test operates on the search DataManager so
        # module-level singletons stay in sync with runtime data.
        if feature_test.dm is not self.dm:
            feature_test.dm = self.dm

        return feature_test

    def _log_progress_heartbeat(
        self,
        *,
        log_file: str,
        segment_id: str,
        start_ts: float,
        last_log_ts: float,
        evaluated_count: int,
        passed_cms: List[CM],
        failed_info: List[Tuple[List[Any], List[str]]],
        error_log: List[Tuple[List[Any], str, str]],
        force: bool = False,
    ) -> float:
        """Log filtering progress to stdout logger and a persistent log file.

        Parameters
        ----------
        log_file : str
            Destination path for the heartbeat entry.
        segment_id : str
            Identifier for the active segment.
        start_ts : float
            Start timestamp (``time.time()``) of the filtering run.
        last_log_ts : float
            Timestamp of the previous heartbeat.
        evaluated_count : int
            Number of specs evaluated so far.
        passed_cms : List[CM]
            Successfully passed candidate models.
        failed_info : List[Tuple[List[Any], List[str]]]
            Failed specifications captured during filtering.
        error_log : List[Tuple[List[Any], str, str]]
            Specs that raised errors during filtering.
        force : bool, default False
            When True, emit a log regardless of the configured interval.

        Returns
        -------
        float
            Updated ``last_log_ts`` timestamp.
        """

        now_ts = time.time()
        interval = self.progress_log_interval_sec or 0

        # Respect disabled heartbeat configuration unless explicitly forced.
        if not force and (interval <= 0 or (now_ts - last_log_ts) < interval):
            return last_log_ts

        elapsed_min = (now_ts - start_ts) / 60 if start_ts else 0.0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"[{timestamp_str}] segment={segment_id}, elapsed={elapsed_min:.1f} min, "
            f"evaluated={evaluated_count}, passed={len(passed_cms)}, errors={len(error_log)}, "
            f"failures={len(failed_info)}"
        )

        LOGGER.info(message)
        with open(log_file, "a", encoding="utf-8") as log_handle:
            log_handle.write(message + "\n")

        return now_ts

    def build_spec_combos(
        self,
        forced_in: Optional[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]],
        desired_pool: List[Union[str, TSFM, Feature, Tuple[Any, ...], set]],
        max_var_num: int,
        max_lag: int = 3,
        periods: Optional[Sequence[int]] = None,
        category_limit: int = 1,
        regime_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None,
        **legacy_kwargs: Any
    ) -> List[List[Union[str, TSFM, Feature, Tuple[Any, ...]]]]:
        """
        Build all valid feature-spec combos:
        - If forced_in is provided, include those specs in every combo.
        - desired_pool can contain:
            * str, TSFM, Feature, tuple, or set.
          * tuple: items stay grouped together.
          * set: treated as a pool where exactly one must be selected. When a
            set contains :class:`RgmVar` instances, combinations of distinct
            regimes are also enumerated (respecting ``regime_limit``) so
            multi-regime variants are not excluded prematurely.
        - Duplicate entries in desired_pool are removed before processing to
          prevent redundant combination generation.
        - Strings at top-level are expanded into TSFM variants via DataManager.
        - Feature-level pre-tests (when configured) prune invalid candidates
          before combination enumeration.
        - Respects max_var_num (total features per combo).
        - Respects category_limit (max variables from each MEV category per combo).
        - Excludes combinations containing a :class:`TSFM` and :class:`RgmVar`
          referencing the same base variable.

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
        regime_limit : int, default 1
            Maximum number of :class:`RgmVar` instances sharing the same
            ``(regime, regime_on)`` signature per combo. Applies across the full
            combination, including forced specifications.
            Maximum number of :class:`RgmVar` instances from the same regime per
            combo. Applies across the full combination, including forced
            specifications.
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            Passed to DataManager.build_tsfm_specs() for string expansion.

        Returns
        -------
        combos : list of spec lists
            Each combo is a list including str, TSFM, Feature, or tuple elements.
        """
        if not isinstance(regime_limit, int):
            raise TypeError("regime_limit must be provided as an integer.")

        if regime_limit < 1:
            raise ValueError("regime_limit must be a positive integer.")

        # Run feature-level pretests later in the pipeline so TSFM expansions
        # can be evaluated directly before enumeration.
        feature_test = self._prepare_feature_pretest()
        pretest_cache: Dict[str, bool] = {}
        excluded_variant_labels: List[str] = []
        excluded_variant_seen: Set[str] = set()
        excluded_group_labels: List[str] = []
        excluded_group_seen: Set[str] = set()

        def _has_tsfm_regime_conflict(items: Sequence[Any]) -> bool:
            """Return ``True`` when a TSFM and RgmVar share the same variable."""

            tsfm_vars: Set[str] = set()
            rgm_vars: Set[str] = set()

            def _collect(obj: Any, *, in_regime: bool = False) -> None:
                # Track TSFM variables directly and those nested inside groups.
                if isinstance(obj, TSFM):
                    if not in_regime and obj.var is not None:
                        tsfm_vars.add(str(obj.var))
                elif isinstance(obj, RgmVar):
                    # Regime-wrapped transforms should only set ``rgm_vars`` so
                    # they conflict with standalone TSFMs of the same base
                    # variable without double-counting as direct TSFMs.
                    base_var = getattr(obj, "var", None)
                    if base_var is None and getattr(obj, "var_feature", None) is not None:
                        base_var = obj.var_feature.var

                    if base_var is not None:
                        rgm_vars.add(str(base_var))

                    _collect(getattr(obj, "var_feature", None), in_regime=True)
                elif isinstance(obj, (list, tuple, set)):
                    for el in obj:
                        _collect(el, in_regime=in_regime)

            _collect(items)
            return bool(tsfm_vars & rgm_vars)

        def _passes_feature(candidate: Any) -> bool:
            """Return ``True`` when ``candidate`` satisfies the feature pre-test."""

            if feature_test is None:
                return True

            if isinstance(candidate, tuple):
                cache_key = f"tuple:{repr(candidate)}"
                if cache_key in pretest_cache:
                    passes = pretest_cache[cache_key]
                else:
                    passes = True
                    for element in candidate:
                        if not _passes_feature(element):
                            passes = False
                            break
                    pretest_cache[cache_key] = passes
                if not passes:
                    key = repr(candidate)
                    if key not in excluded_group_seen:
                        excluded_group_labels.append(key)
                        excluded_group_seen.add(key)
                return passes

            cache_key = repr(candidate)
            # Only TSFM objects undergo feature pre-testing; all other feature
            # representations are allowed to pass without evaluation.
            if not isinstance(candidate, TSFM):
                pretest_cache[cache_key] = True
                return True
            if cache_key in pretest_cache:
                passes = pretest_cache[cache_key]
            else:
                feature_test.feature = candidate
                try:
                    result = feature_test.test_filter
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(
                        "Feature pre-test raised "
                        f"{type(exc).__name__} for {candidate!r}: {exc}"
                    )
                    passes = True
                else:
                    try:
                        passes = bool(result)
                    except Exception:  # pragma: no cover - unexpected truthiness
                        passes = True
                pretest_cache[cache_key] = passes

            if not passes:
                if cache_key not in excluded_variant_seen:
                    excluded_variant_labels.append(cache_key)
                    excluded_variant_seen.add(cache_key)
            return passes

        # Handle forced_in being optional
        forced_specs = (forced_in or []).copy()

        # Remove duplicates from desired_pool to avoid repeated combinations
        # when users inadvertently supply the same variable more than once.
        seen_signatures: Set[str] = set()
        unique_desired_pool: List[Union[str, TSFM, Feature, Tuple[Any, ...], set]] = []

        def _dedup_signature(item: Any) -> str:
            """Create a deterministic signature for deduplication."""

            # Using repr ensures we can handle unhashable inputs such as sets and tuples
            # while keeping ordering stable for repeated objects.
            return f"{type(item).__name__}:{repr(item)}"

        for pool_item in desired_pool:
            signature = _dedup_signature(pool_item)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_desired_pool.append(pool_item)

        # Step 1: Build raw combos from desired_pool with category constraints
        # Separate constrained and unconstrained items
        constrained_items = []  # top-level strings and TSFM instances
        unconstrained_items = []  # everything else (sets, tuples, other Features)

        for item in unique_desired_pool:
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
                # When a set contains regime variables, allow combinations of
                # distinct regimes to co-occur by generating subset options.
                # Filtering by ``regime_limit`` later in the pipeline still
                # enforces per-regime caps, but this expansion ensures we do
                # not artificially limit combos to a single RgmVar.
                if any(isinstance(el, RgmVar) for el in flat):
                    subset_pool: List[List[Any]] = []
                    flat_list = list(flat)
                    for r in range(1, len(flat_list) + 1):
                        for combo in itertools.combinations(flat_list, r):
                            subset_pool.append(list(combo))
                    pools.append(subset_pool)
                else:
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

        def _regime_counts(items: Sequence[Any]) -> Dict[Tuple[str, int], int]:
            """Return counts of regime occurrences for any nested :class:`RgmVar`.

            Regime uniqueness accounts for the activation flag (``on``/``regime_on``)
            so that variants targeting the same regime column with different active
            states are treated as distinct. This enables combinations that include
            both active and inactive variants of the same regime indicator.

            Parameters
            ----------
            items : Sequence[Any]
                Collection of specification elements to inspect.

            Returns
            -------
            Dict[Tuple[str, int], int]
                Mapping of ``(regime, regime_on)`` signatures to counts of
                :class:`RgmVar` entries.
            """

            counts: Dict[Tuple[str, int], int] = defaultdict(int)

            def _collect(obj: Any) -> None:
                if isinstance(obj, RgmVar):
                    # Include activation flag so on/off variants coexist.
                    counts[(obj.regime, getattr(obj, "on", 1))] += 1
                elif isinstance(obj, (list, tuple, set)):
                    for el in obj:
                        _collect(el)

            _collect(items)
            return counts

        # Enforce per-regime limits across forced and desired combinations
        filtered_combos: List[List[Any]] = []
        for combo in combos:
            regime_counts = _regime_counts(combo)
            if any(count > regime_limit for count in regime_counts.values()):
                continue
            filtered_combos.append(combo)

        combos = filtered_combos

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
            combo_invalid = False
            for spec in combo:
                if isinstance(spec, str):
                    variants = tsfm_map.get(spec, [spec])
                else:
                    variants = [spec]

                if feature_test is not None:
                    filtered_variants = [
                        variant for variant in variants if _passes_feature(variant)
                    ]
                    if not filtered_variants:
                        if spec in forced_specs:
                            raise ValueError(
                                "Forced specification failed the configured feature pre-test: "
                                f"{spec!r}"
                            )
                        combo_invalid = True
                        break
                    variants = filtered_variants

                variant_lists.append(variants)

            if combo_invalid:
                continue

            # Cartesian product over variant lists
            for prod in itertools.product(*variant_lists):
                # Sort each spec list so quarterly and monthly dummies come first
                sorted_prod = _sort_specs_with_dummies_first(list(prod))
                # Skip combos that mix regime-aware and standard transforms of the same var
                if _has_tsfm_regime_conflict(sorted_prod):
                    continue
                expanded.append(sorted_prod)

        if (feature_test is not None) and (excluded_variant_labels or excluded_group_labels):
            print("--- Feature Pre-Test Exclusions ---")
            if excluded_variant_labels:
                print(
                    "Excluded "
                    f"{len(excluded_variant_labels)} variant(s): "
                    + ", ".join(excluded_variant_labels)
                )
            if excluded_group_labels:
                print(
                    "Removed "
                    f"{len(excluded_group_labels)} grouped candidate(s): "
                    + ", ".join(excluded_group_labels)
                )
            print("")

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
        mdl = cm.model_in if sample == 'in' else cm.model_full

        # Reload testset, applying update if provided
        mdl.load_testset(test_update_func=test_update_func)

        # Run filtering on updated testset (fast mode to short-circuit on first failure)
        passed, failed = mdl.testset.filter_pass(fast_filter=True)
        if passed:
            return cm
        return specs, failed, mdl.testset.filter_test_info

    def filter_specs(
        self,
        sample: str = 'in',
        test_update_func: Optional[Callable[[ModelBase], dict]] = None,
        outlier_idx: Optional[List[Any]] = None,
        log: bool = True,
        start_index: int = 0,
        total_count: Optional[int] = None,
        log_file_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        passed_cms_dir: Optional[Path] = None,
        initial_passed: Optional[List[CM]] = None,
    ) -> Tuple[List[CM], List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]], List[Tuple[List[Any], str, str]]]:
        """
        Assess all built spec combos and separate passed and failed results,
        using multithreading and a single progress bar update.

        Parameters
        ----------
        sample : {'in','full'}
            Sample to use for all assessments (default 'in').
        test_update_func : callable, optional
            Function to update/regenerate the testset for each model.
        outlier_idx : List[Any], optional
            List of index labels (e.g. timestamps or keys) corresponding to outlier
            records to remove from the in-sample data. If provided and `build_in`
            is True, each label must exist within the in-sample period; otherwise,
            a ValueError is raised.
        log : bool, default True
            When True, emit heartbeat logs to ``INFO`` and persist them to the
            segment ``log`` directory under the capitalized ``Segment`` root. A
            notice describing the log destination is printed before filtering
            begins.
        start_index : int, default 0
            Number of spec combinations already completed in a resumed search.
            This value feeds progress reporting but the provided ``all_specs``
            sequence will still be evaluated in full.
        total_count : int, optional
            Total number of combinations in the full search. When provided, the
            progress indicator uses this denominator rather than the length of
            the current ``all_specs`` slice.
        log_file_path : pathlib.Path, optional
            Explicit path for progress logging. When omitted, a timestamped
            search-specific log file is created automatically.
        progress_callback : callable, optional
            Invoked after each spec is evaluated with the current completed
            combination count. Useful for writing incremental progress files.
        passed_cms_dir : pathlib.Path, optional
            Destination directory for persisting passed candidate models under
            ``cms/<search_id>/passed_cms``. When omitted, passed models are
            not persisted to disk during filtering.
        initial_passed : list of CM, optional
            Previously passed models to seed the run with when resuming an
            interrupted search. These models are included in progress updates
            and will be re-indexed alongside newly evaluated specifications.

        Notes
        -----
        When ``log`` is True, emits heartbeat progress logs at ``INFO`` level
        and to a segment-scoped log file during lengthy runs. Heartbeats respect
        ``progress_log_interval_sec`` configured on initialization; set to a
        non-positive number to disable periodic logging. A final summary is
        always appended at completion. Candidate models are labeled
        ``temp_<index>`` during assessment and renamed sequentially to
        ``passed_<n>`` upon successfully passing all tests. Passed models are
        immediately persisted under ``Segment/<segment_id>/cms/<search_id>/passed_cms``
        when a :class:`Segment` context is attached, mirroring
        :meth:`Segment.save_cms` conventions.

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
            
            passed_cms: List[CM] = list(initial_passed) if initial_passed else []
            failed_info: List[Tuple[List[Union[str, TSFM, Feature, Tuple[Any, ...]]], List[str]]] = []
            error_log: List[Tuple[List[Any], str, str]] = []

            # Test info tracking
            seen_test_names: set = set()
            test_info_header_printed = False
            batch_filter_test_infos: Dict[str, Dict[str, str]] = {}  # Collect all filter_test_info in current batch

            total = total_count if total_count is not None else len(self.all_specs)
            start_time = time.time()
            last_log_ts = start_time

            # Resolve segment-scoped log destination and ensure folder readiness.
            segment_obj = getattr(self, "segment", None)
            segment_id = getattr(segment_obj, "segment_id", "unknown_segment")
            working_root = Path(getattr(segment_obj, "working_dir", Path.cwd()))
            segment_dirs: Optional[Dict[str, Path]] = None
            if segment_obj is not None:
                # Guarantee the capitalized Segment root exists when a Segment
                # context is attached so persisted artifacts follow the expected
                # directory casing.
                segment_dirs = ensure_segment_dirs(str(segment_id), working_root)

            # Preserve the capitalized "Segment" directory as the root for
            # per-segment artifacts and write logs to a lowercase ``log``
            # subfolder when optional logging is enabled.
            log_dir: Optional[Path] = None
            if log and log_file_path is None:
                if segment_dirs is None:
                    segment_dirs = ensure_segment_dirs(str(segment_id), working_root)
                log_dir = Path(segment_dirs["segment_dir"]) / "log"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = log_dir / f"search_{segment_id}_{log_ts}.log"

            if log and log_file_path is not None:
                log_dir = log_file_path.parent
                log_dir.mkdir(parents=True, exist_ok=True)
                print(f"Log will be saved to: {log_file_path}")

            # Print initial empty line for spacing
            print("")

            for idx, specs in enumerate(self.all_specs, start=start_index):
                # Use temporary identifiers during assessment to avoid reusing
                # final passed_* labels before models pass all tests.
                model_id = f"temp_{idx}"
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
                        # Promote passed models to sequential passed_* IDs so
                        # the persisted identifier reflects success ordering.
                        result.model_id = f"passed_{len(passed_cms) + 1}"
                        passed_cms.append(result)
                        # Persist passed CMs immediately so long-running runs do
                        # not lose progress if interrupted. Prefer a
                        # search-scoped directory when provided and maintain an
                        # up-to-date index so Segment.load_cms() can observe
                        # in-flight results.
                        target_dir = passed_cms_dir
                        # When no per-search destination is provided, skip disk
                        # persistence to avoid recreating legacy top-level
                        # folders. This keeps all artifacts scoped under the
                        # active search_id structure.
                        if target_dir is not None:
                            try:
                                target_dir.mkdir(parents=True, exist_ok=True)
                                created_at = datetime.datetime.now().isoformat(timespec="seconds")
                                existing_entries: List[Dict[str, Any]] = []
                                try:
                                    existing_entries = load_index(target_dir)
                                except FileNotFoundError:
                                    existing_entries = []

                                if segment_obj is not None:
                                    entry = segment_obj._save_cm_entry(
                                        result,
                                        target_dir,
                                        created_at,
                                        overwrite=True,
                                    )
                                else:
                                    save_cm(
                                        result,
                                        target_dir / f"{result.model_id}.pkl",
                                        overwrite=True,
                                    )
                                    entry = {
                                        "model_id": result.model_id,
                                        "filename": f"{result.model_id}.pkl",
                                        "segment_id": segment_id,
                                        "created_at": created_at,
                                    }

                                existing_entries = [
                                    e for e in existing_entries if e.get("model_id") != entry["model_id"]
                                ]
                                existing_entries.append(entry)
                                save_index(target_dir, existing_entries, overwrite=True)
                            except Exception as exc:  # pragma: no cover - best-effort persistence
                                LOGGER.warning(
                                    "Unable to persist passed CM %s: %s",
                                    result.model_id,
                                    exc,
                                )
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
                    relative_i = idx - start_index
                    batch_size = 100 if relative_i < 10000 else 10000
                    
                    # Process batch based on dynamic batch size or on first run
                    if (relative_i + 1) % batch_size == 0 or relative_i == 0:
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

                # Emit periodic heartbeat logs to track long-running searches.
                processed_count = idx + 1
                if log and log_file_path:
                    last_log_ts = self._log_progress_heartbeat(
                        log_file=str(log_file_path),
                        segment_id=str(segment_id),
                        start_ts=start_time,
                        last_log_ts=last_log_ts,
                        evaluated_count=processed_count,
                        passed_cms=passed_cms,
                        failed_info=failed_info,
                        error_log=error_log,
                    )

                if progress_callback is not None:
                    progress_callback(processed_count)

                # Progress and ETA update (only every 10 iterations to reduce interference)
                if (relative_i + 1) % 10 == 0 or relative_i == 0 or relative_i == len(self.all_specs) - 1:
                    processed = processed_count
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
                    processed_count_str = f"{processed}/{total}"
                    speed = processed / elapsed if elapsed > 0 else 0
                    progress_line = f"Filtering Specs: {progress_pct:3d}%|{bar}| {processed_count_str} [{elapsed:,.0f}s, {speed:.2f}it/s, estimated_finish={eta}]"
                    print(f"\r{progress_line}", end='', flush=True)
            
            # Final newline after progress bar
            print("")
            if log and log_file_path:
                self._log_progress_heartbeat(
                    log_file=str(log_file_path),
                    segment_id=str(segment_id),
                    start_ts=start_time,
                    last_log_ts=last_log_ts,
                    evaluated_count=total,
                    passed_cms=passed_cms,
                    failed_info=failed_info,
                    error_log=error_log,
                    force=True,
                )
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
        regime_limit: int = 1,
        exp_sign_map: Optional[Dict[str, int]] = None,
        rank_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        test_update_func: Optional[Callable[[ModelBase], dict]] = None,
        outlier_idx: Optional[List[Any]] = None,
        search_id: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
        **legacy_kwargs: Any
    ) -> List[CM]:
        """
        Execute full search pipeline: build specs, filter, rank, and select top_n models.

        Steps
        -----
        1. Print configuration summary.
        2. Run the model class' target pre-test when defined.
        3. Build spec combinations via build_spec_combos.
        4. Print number of generated combos.
        5. Assess and filter combos via filter_specs (printing test info for first combo).
        6. Rank passed models via rank_cms and retain top_n.

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
        regime_limit : int, default 1
            Maximum number of :class:`RgmVar` instances sharing the same
            ``(regime, regime_on)`` signature per combo. Applies across the full
            combination, including forced specifications.
        exp_sign_map : Optional[Dict[str, int]], default=None
            Dictionary mapping MEV codes to expected coefficient signs for TSFM instances.
            Passed to build_spec_combos() and ultimately to DataManager.build_tsfm_specs().
        rank_weights : tuple, default (1.0, 1.0, 1.0)
            Weights for (Fit Measures, IS Error, OOS Error) when ranking models.
        test_update_func : callable, optional
            Optional function to update each CM's test set.
        outlier_idx : list, optional
            List of index labels corresponding to outliers to exclude.
        search_id : str, optional
            Identifier for this search run. When omitted, a new value of the
            form ``search_<segment_id>_<YYYYMMDD_HHMMSS>`` is generated.
        base_dir : str or pathlib.Path, optional
            Working directory for search artifacts. Defaults to the current
            working directory.

        Returns
        -------
        top_models : list of CM
            The top_n CM instances sorted by composite score.

        Notes
        -----
        When parameters and total combinations match the most recent search for
        the same segment, the method prompts to resume the previous run using
        its ``search_id``. On resume, previously passed models are loaded from
        ``cms/<search_id>/passed_cms``, combinations recorded as completed in
        ``log/<search_id>.progress`` are skipped, and progress continues in the
        same per-search log and progress files. New runs always generate a
        ``search_<segment_id>_<YYYYMMDD_HHMMSS>`` identifier.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forced = forced_in or []

            # Reset cached target outcome to avoid bleeding across runs when
            # ``run_search`` is invoked multiple times on the same instance.
            self.target_pretest_result = None

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

            working_root = Path(base_dir) if base_dir is not None else Path.cwd()
            segment_label = getattr(self.segment, "segment_id", "unknown_segment")
            sanitized_segment = sanitize_segment_id(segment_label)
            effective_search_id = search_id or generate_search_id(sanitized_segment)
            self.search_id = effective_search_id

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
                  f"Regime limit    : {regime_limit}\n"
                  f"Exp sign map    : {exp_sign_map}\n"
                  f"Top N           : {top_n}\n"
                  f"Rank weights    : {rank_weights}\n"
                  f"Test update func: {test_update_func}\n"
                  f"Outlier idx     : {outlier_idx}\n")
            print("==================================\n")

            search_config_raw = {
                "search_id": None,
                "search_id": effective_search_id,
                "segment_id": segment_label,
                "target": self.target,
                "model_cls": self.model_cls.__name__,
                "model_type": self.model_type,
                "target_base": self.target_base,
                "target_exposure": self.target_exposure,
                "max_var_num": max_var_num,
                "desired_pool": desired_pool,
                "forced_in": forced,
                "sample": sample,
                "max_lag": max_lag,
                "periods": periods_summary,
                "category_limit": category_limit,
                "regime_limit": regime_limit,
                "exp_sign_map": exp_sign_map,
                "rank_weights": rank_weights,
                "test_update_func": test_update_func,
                "outlier_idx": outlier_idx,
            }

            # Execute optional target-level pretests before heavy computations.
            self.model_pretestset = self._resolve_model_pretestset()
            target_test_result: Optional[Any] = None
            if (
                self.model_pretestset is not None
                and self.model_pretestset.target_test is not None
            ):
                target_test = self.model_pretestset.target_test
                if target_test.dm is None:
                    target_test.dm = self.dm
                if target_test.target is None:
                    target_test.target = self.target

                try:
                    # Build once so the filter decision and the tabular results
                    # originate from the same evaluation run.
                    target_testset = target_test.testset
                    target_passed, _ = target_testset.filter_pass()
                    target_test_result = target_passed
                except Exception as exc:
                    print(
                        "Target pre-test raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    description = ""
                    if hasattr(target_test_result, "attrs"):
                        description = target_test_result.attrs.get(
                            "filter_mode_desc",
                            ""
                        ) or ""
                    if not description and hasattr(target_test_result, "filter_mode_desc"):
                        description = getattr(
                            target_test_result,
                            "filter_mode_desc",
                            ""
                        )

                    print("--- Target Pre-Test Result ---")
                    if description:
                        print(description)
                    # Leverage the aggregated view so we always display
                    # the full set of tabular results produced during the
                    # evaluation pass.
                    for name, result in target_testset.all_test_results.items():
                        print(f"{name} Test Result:\n{result}\n")
                    print(f"Target filter passed: {target_test_result}\n")

                if self.model_pretestset is not None:
                    self._propagate_target_context(target_test_result)
            else:
                self.target_pretest_result = None

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
                regime_limit=regime_limit,
                exp_sign_map=exp_sign_map,
            )
            print(f"Built {len(combos)} spec combinations.\n")

            self.total_combos = len(combos)
            self.completed_combos = 0
            search_config_raw["total_combos"] = self.total_combos

            # Determine whether a prior run should be resumed.
            resume_from = 0
            initial_passed: List[CM] = []
            resume_paths: Optional[Dict[str, Path]] = None
            prospective_config = self._make_serializable(search_config_raw)
            resume_candidate = self._find_resume_candidate(
                segment_label, working_root, prospective_config, self.total_combos
            )
            if resume_candidate is not None:
                latest_search_id, _, prior_completed = resume_candidate
                answer = input(
                    "A previous search with identical parameters was found:\n"
                    f"  search_id = {latest_search_id}\n"
                    f"  total_combos = {self.total_combos}\n"
                    "It appears that run may have been interrupted.\n"
                    "Do you want to continue from where it stopped? [y/N]: "
                ).strip()
                if answer.lower() == "y":
                    resume_paths = get_search_paths(segment_label, latest_search_id, working_root)
                    progress_info = self._read_progress(resume_paths["progress_file"])
                    if (
                        progress_info is not None
                        and progress_info.get("total_combos") == self.total_combos
                    ):
                        resume_from = int(progress_info.get("completed_combos", 0))
                        initial_passed = self._load_passed_cms_from_dir(
                            resume_paths["passed_cms_dir"], self.dm
                        )
                    else:
                        # Fall back to prior completed count recorded via index when
                        # progress metadata is missing.
                        resume_from = prior_completed
                        if resume_from:
                            initial_passed = self._load_passed_cms_from_dir(
                                resume_paths["passed_cms_dir"], self.dm
                            )

                    if resume_from:
                        effective_search_id = latest_search_id
                        self.search_id = effective_search_id
                    else:
                        resume_paths = None

            if resume_paths is None:
                effective_search_id = search_id or generate_search_id(sanitized_segment)
                self.search_id = effective_search_id
                resume_paths = get_search_paths(segment_label, effective_search_id, working_root)

            search_paths = resume_paths
            search_paths["search_cms_dir"].mkdir(parents=True, exist_ok=True)
            search_paths["log_dir"].mkdir(parents=True, exist_ok=True)

            search_config_raw["search_id"] = effective_search_id
            search_config_serializable = self._make_serializable(search_config_raw)
            self.current_search_config_raw = search_config_raw
            self.current_search_config = search_config_serializable

            with search_paths["log_file"].open("a", encoding="utf-8") as log_handle:
                log_handle.write("=" * 80 + "\n")
                log_handle.write(
                    f"Search started at {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                )
                log_handle.write(f"search_id: {effective_search_id}\n")
                if resume_from:
                    log_handle.write(f"Resuming from combo {resume_from + 1}.\n")
                log_handle.write("Configuration:\n")
                log_handle.write(json.dumps(search_config_serializable, indent=2))
                log_handle.write("\n")

            index_path = search_paths["cms_root"] / "search_index.json"
            try:
                existing_index = json.loads(index_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                existing_index = {}
            except json.JSONDecodeError:
                existing_index = {}
            existing_index[effective_search_id] = search_config_serializable
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(existing_index, indent=2, sort_keys=True), encoding="utf-8")

            search_paths["config_file"].parent.mkdir(parents=True, exist_ok=True)
            search_paths["config_file"].write_text(
                json.dumps(search_config_serializable, indent=2),
                encoding="utf-8",
            )

            self.completed_combos = resume_from

            def _progress_callback(completed: int) -> None:
                self.completed_combos = completed
                if completed % 1000 == 0 or completed == self.total_combos:
                    self._write_progress(
                        search_paths["progress_file"],
                        self.total_combos,
                        completed,
                    )

            self._write_progress(search_paths["progress_file"], self.total_combos, resume_from)

            # Skip combinations already evaluated when resuming.
            if resume_from:
                self.all_specs = combos[resume_from:]
            else:
                self.all_specs = combos
            self._write_progress(search_paths["progress_file"], self.total_combos, 0)

            # 3) Filter specs
            passed, failed, errors = self.filter_specs(
                sample=sample,
                test_update_func=test_update_func,
                outlier_idx=outlier_idx,
                start_index=resume_from,
                total_count=self.total_combos,
                log_file_path=search_paths["log_file"],
                progress_callback=_progress_callback,
                passed_cms_dir=search_paths["passed_cms_dir"],
                initial_passed=initial_passed,
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
