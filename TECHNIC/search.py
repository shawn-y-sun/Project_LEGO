from typing import Any, List, Tuple, Dict
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .segment import Segment


def _process_spec(args: Tuple[int, List[Any], str, bool, Segment]) -> Tuple[bool, Any, Any, List[str]]:
    """
    Build a CM, run its tests, and return (passed, cm, report, failed_test_names).
    """
    idx, specs, fit_on, fast_test, segment = args
    cm_id = f"cm{idx}"
    cm = segment.build_cm(cm_id, specs=specs, sample=fit_on)
    tests = cm.tests_in if fit_on == "in" else cm.tests_full
    # Correct positional to keyword call to avoid bool iterable error
    result = tests.search_pass(fast_test=fast_test)
    if isinstance(result, (tuple, list)):
        passed, failed_tests = result
    else:
        passed = result
        failed_tests = []
    # Collect only test class names
    failed_names = [test.__class__.__name__ for test in (failed_tests or [])]
    report = cm.report_in if fit_on == "in" else cm.report_full
    return passed, cm, report, failed_names


class ExhaustiveSearch:
    """
    Two-phase exhaustive search over feature combinations:
      1) Parallel filtering via ThreadPoolExecutor
      2) Pre-select top_n by combined performance (perf_all)
      3) Reorder that subset by mode ('simplicity','in_sample_perf','out_sample_perf','all')
    """

    def __init__(
        self,
        segment: Segment,
        feature_combinations: List[List[Any]],
        fit_on: str = "in",
        fast_test: bool = True,
        top_n: int = 10,
        metric_weights: Dict[str, float] = None,
        larger_better: List[str] = None,
        by: str = "simplicity",
    ):
        # Clear existing CMs
        self.segment = segment
        self.segment.cms.clear()
        self.feature_combinations = feature_combinations
        self.fit_on = fit_on
        self.fast_test = fast_test
        self.top_n = top_n
        self.metric_weights = metric_weights or {}
        self.larger_better = set(larger_better) if larger_better else {"r2", "adj_r2"}
        allowed = {"simplicity", "in_sample_perf", "out_sample_perf", "all"}
        if by not in allowed:
            raise ValueError(f"Invalid mode '{by}'. Choose from {allowed}.")
        self.by = by
        # Only store names of failed tests
        self.failures: List[Dict[str, List[str]]] = []

    def run(self) -> List[Any]:
        """
        Execute filtering then ranking; return the final list of CMs.
        """
        survivors = self._filter_models()
        return self._rank_models(survivors) if survivors else []

    def _filter_models(self) -> List[Tuple[Any, Any]]:
        """
        Parallel CM build & test; collect passing CMs and record failed test names.
        """
        survivors: List[Tuple[Any, Any]] = []
        args_list = [
            (i, specs, self.fit_on, self.fast_test, self.segment)
            for i, specs in enumerate(self.feature_combinations) if specs
        ]
        workers = os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_spec, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering CMs"):
                passed, cm, report, failed_names = future.result()
                if not passed:
                    self.failures.append({repr(cm): failed_names})
                    self.segment.cms.pop(cm.model_id, None)
                else:
                    self.segment.cms[cm.model_id] = cm
                    survivors.append((cm, report))
        return survivors

    def _rank_models(self, survivors: List[Tuple[Any, Any]]) -> List[Any]:
        """
        Two-phase ranking:
          1) Pre-select top_n by combined performance
          2) Reorder that subset by `by` mode
        """
        seg = self.segment
        # Gather raw performance metrics
        raw = {
            cm.model_id: {
                **{f"in_{k}": v for k, v in rpt.in_perf_measures.items()},
                **{f"out_{k}": v for k, v in rpt.out_perf_measures.items()},
            }
            for cm, rpt in survivors
        }
        keys = list(next(iter(raw.values())))
        # Orient metrics: invert errors
        oriented = {
            mid: {k: raw[mid][k] if k in self.larger_better else -raw[mid][k] for k in keys}
            for mid in raw
        }
        # Min-max scaling
        scaled: Dict[str, Dict[str, float]] = {mid: {} for mid in oriented}
        for k in keys:
            vals = [oriented[mid][k] for mid in oriented]
            mn, mx = min(vals), max(vals)
            for mid in oriented:
                scaled[mid][k] = (oriented[mid][k] - mn) / (mx - mn) if mx > mn else 1.0
        # Combined performance score
        perf_all = {mid: sum(scaled[mid].values()) / len(scaled[mid]) for mid in scaled}
        # Pre-select top_n
        top_ids = sorted(perf_all, key=perf_all.get, reverse=True)[: self.top_n]
        for mid in list(seg.cms.keys()):
            if mid not in top_ids:
                seg.cms.pop(mid, None)

        # Compute in/out prefix scores
        def compute_score(prefix: str) -> Dict[str, float]:
            return {
                mid: (
                    sum(scaled[mid][k] * self.metric_weights.get(k, 1.0)
                        for k in keys if k.startswith(prefix))
                    / sum(self.metric_weights.get(k, 1.0)
                          for k in keys if k.startswith(prefix))
                ) if any(k.startswith(prefix) for k in keys) else 0.0
                for mid in top_ids
            }
        perf_in = compute_score("in_")
        perf_out = compute_score("out_")
        # Simplicity scoring within top_ids
        simp_scores = {}
        for cm, _ in survivors:
            if cm.model_id in top_ids:
                costs = [1 if getattr(f, 'transform', None) is None else 2 for f in cm.specs]
                simp_scores[cm.model_id] = len(costs) * (sum(costs) / len(costs))
        mn_s, mx_s = min(simp_scores.values()), max(simp_scores.values())
        simp_scores = {
            mid: ((mx_s - simp_scores[mid]) / (mx_s - mn_s)) if mx_s > mn_s else 1.0
            for mid in simp_scores
        }
        # Final ordering
        if self.by == "simplicity":
            final_ids = sorted(top_ids, key=lambda m: simp_scores[m], reverse=True)
        elif self.by == "in_sample_perf":
            final_ids = sorted(top_ids, key=lambda m: perf_in[m], reverse=True)
        elif self.by == "out_sample_perf":
            final_ids = sorted(top_ids, key=lambda m: perf_out[m], reverse=True)
        else:
            final_ids = top_ids
        # Relabel and rebuild segment.cms
        new_cms: Dict[str, Any] = {}
        final_cms: List[Any] = []
        for i, mid in enumerate(tqdm(final_ids, desc="Finalizing Top CMs"), start=1):
            cm = seg.cms[mid]
            new_id = f"cm{i}"
            cm.model_id = new_id
            new_cms[new_id] = cm
            final_cms.append(cm)
        # Replace old cache with renamed CMs
        seg.cms = new_cms
        return final_cms
