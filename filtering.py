# TECHNIC/filtering.py

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union

import pandas as pd
import statsmodels.api as sm

from .cm import CM
from .testing import OLSTesting, ScenarioTesting
from .measure import OLS_Measures


class ModelFilterBase(ABC):
    """
    Abstract base class for model filtering logic.

    Subclasses should implement filter_rules method that applies
    various screening tests and returns a dict of filter results.
    run_filters executes filter_rules and stores results.
    """
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ):
        self.model = model
        self.X = X
        self.y = y

    @abstractmethod
    def filter_rules(self) -> Dict[str, Any]:
        """
        Apply all filter functions and return a dict of results.
        """
        pass

    def run_filters(self) -> Dict[str, Any]:
        """
        Execute filter_rules and store results for later retrieval.
        """
        results = self.filter_rules()
        self.filter_results = results
        return results


class OLSFiltering(ModelFilterBase):
    """
    OLS-specific filtering rules leveraging CM instance:
      1. Max p-value of coefficients
      2. Model F-statistic p-value
      3. Group-wise F-tests
      4. VIF calculation
      5. Coefficient sign consistency
      6. Majority normality tests pass
      7. Majority stationarity tests pass
      8. Scenario ranking using ScenarioTesting results
      9. Coefficient sign stability via pseudo-OOS re-estimation
    """
    def __init__(
        self,
        cm: CM,
        pvalue_thresh: float                = 0.05,
        vif_thresh: float                   = 10.0,
        f_pvalue_thresh: float              = 0.05,
        group_map: Optional[Dict[str, List[str]]] = None,
        normality_thresh: float             = 0.05,
        stationarity_thresh: float          = 0.05,
        forecast_horizons: Optional[List[int]]    = None
    ):
        super().__init__(cm.model_in, cm.X_in, cm.y_in)
        self.cm                   = cm
        self.pvalue_thresh        = pvalue_thresh
        self.vif_thresh           = vif_thresh
        self.f_pvalue_thresh      = f_pvalue_thresh
        self.group_map            = group_map or {}
        # retrieve expected signs from DataManager's scenario_maps
        self.expected_signs       = cm.dm.scen_maps.get('expected_signs', {})
        self.normality_thresh     = normality_thresh
        self.stationarity_thresh  = stationarity_thresh
        # Determine forecast horizons based on data frequency if not specified
        if forecast_horizons is not None:
            self.forecast_horizons = forecast_horizons
        else:
            freq = cm.dm.freq
            if freq == 'M':
                self.forecast_horizons = [3, 6, 12]
            else:
                self.forecast_horizons = [1, 2, 3]

    def filter_rules(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # 1. Max p-value excluding grouped variables
        meas = OLS_Measures(self.model, self.X, self.y)
        param_meas = meas.param_measures
        exclude = [v for grp in self.group_map.values() for v in grp]
        pvals = {var: stats['pvalue'] for var, stats in param_meas.items()
                 if var != 'const' and var not in exclude}
        max_p = max(pvals.values()) if pvals else None
        results['max_pvalue']    = float(max_p) if max_p is not None else None
        results['pvalue_pass']    = (max_p is not None and max_p <= self.pvalue_thresh)
        results['pvalue_details'] = pvals

        # 2. Model F-statistic p-value
        f_p = float(getattr(self.model, 'f_pvalue', math.nan))
        results['f_pvalue']      = f_p
        results['f_pvalue_pass'] = f_p <= self.f_pvalue_thresh

        # 3. Group-wise F-tests
        for name, vars_ in self.group_map.items():
            if vars_:
                constraints = [f"{v} = 0" for v in vars_]
                p = float(self.model.f_test(constraints).pvalue)
                results[f'{name}_pvalue'] = p
                results[f'{name}_pass']   = p <= self.pvalue_thresh

        # 4. VIF calculation
        vifs = {var: stats['vif'] for var, stats in param_meas.items() if var != 'const'}
        max_v = max(vifs.values()) if vifs else None
        results['max_vif']    = float(max_v) if max_v is not None else None
        results['vif_pass']   = (max_v is not None and max_v <= self.vif_thresh)
        results['vif_details']= vifs

        # 5. Coefficient sign consistency
        violations = {}
        for var, exp in self.expected_signs.items():
            coef = param_meas.get(var, {}).get('coef', 0)
            if exp * coef < 0:
                violations[var] = coef
        results['sign_violations'] = violations
        results['sign_pass']       = not bool(violations)

        # 6 & 7. Normality & Stationarity tests
        ols_test = OLSTesting(self.cm)
        stats    = ols_test.test_residuals()
        # Normality
        norm_keys = ['jb_pvalue','shapiro_pvalue','kstest_pvalue','cvm_pvalue']
        norm_pvs  = [stats.get(k, math.nan) for k in norm_keys]
        pass_norm = sum(p > self.normality_thresh for p in norm_pvs)
        results['normality_pass']     = pass_norm >= math.ceil(len(norm_pvs)/2)
        results['normality_details']  = dict(zip(norm_keys, norm_pvs))
        # Stationarity
        stat_keys = ['adf_pvalue','kpss_pvalue','pp_pvalue','za_pvalue','dfgls_pvalue']
        stat_pvs  = [stats.get(k, math.nan) for k in stat_keys]
        pass_stat = sum(
            (p > self.stationarity_thresh if k == 'kpss_pvalue' else p < self.stationarity_thresh)
            for k, p in zip(stat_keys, stat_pvs)
        )
        results['stationarity_pass']     = pass_stat >= math.ceil(len(stat_pvs)/2)
        results['stationarity_details']   = dict(zip(stat_keys, stat_pvs))

        # 8. Scenario ranking using ScenarioTesting results
        scen_test = ScenarioTesting(
            self.cm,
            model_type=self.cm.model_type[0] if self.cm.model_type else None
        )
        raw, conv = scen_test.test_scenarios()
        scenario_metrics: Dict[str, Dict[str, float]] = {}
        for name, series in conv.items():
            metrics = {f'end_{h}m': float(series.iloc[min(h - 1, len(series) - 1)])
                       for h in self.forecast_horizons}
            metrics['cum_sum'] = float(series.iloc[:min(max(self.forecast_horizons), len(series))].sum())
            scenario_metrics[name] = metrics
        severity = ['base','adverse','severely_adverse']
        order_pass = all(
            scenario_metrics.get(severity[i], {}).get(metric, float('-inf')) >
            scenario_metrics.get(severity[i+1], {}).get(metric, float('inf'))
            for metric in next(iter(scenario_metrics.values()))
            for i in range(len(severity)-1)
        )
        results['scenario_ranking_pass']     = order_pass
        results['scenario_ranking_details']  = scenario_metrics

        # 9. Coefficient sign stability via pseudo-OOS re-estimation
        poos_test = OLSTesting(
            self.cm,
            pseudo_vars=list(self.expected_signs.keys()),
            pseudo_horizons=[12, 24, 36]
        )
        poos_res    = poos_test.test_pseudo_oos_robustness()
        poos_coeffs = poos_res.get('poos_coefficients', {})
        poos_viol   = {}
        for var, exp in self.expected_signs.items():
            for months, coeffs in poos_coeffs.items():
                coef = coeffs.get(var, 0)
                if exp * coef < 0:
                    poos_viol.setdefault(var, []).append((months, coef))
        results['poos_sign_violations'] = poos_viol
        results['poos_sign_pass']       = not bool(poos_viol)

        return results

    def show_filter_results(
        self,
        as_df: bool = True
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Display pass/fail summary for each filtering rule,
        including overall pass/fail and human-readable details.
        """
        results = getattr(self, 'filter_results', None)
        if results is None:
            results = self.run_filters()
        pass_dict = {key: results[key] for key in results if key.endswith('_pass')}
        overall = all(pass_dict.values())
        details = []
        # human-readable failure descriptions
        if not pass_dict.get('pvalue_pass', True):
            pvals = results.get('pvalue_details', {})
            for var, pv in pvals.items():
                if pv > self.pvalue_thresh:
                    details.append(
                        f"The p-value of {var} coefficient estimate is {pv:.3f}, "
                        f"which is greater than threshold {self.pvalue_thresh}."
                    )
        if not pass_dict.get('poos_sign_pass', True):
            vio = results.get('poos_sign_violations', {})
            for var, lst in vio.items():
                for months, coef in lst:
                    details.append(
                        f"In pseudo-out-of-sample test (horizon={months} months), "
                        f"variable {var} coefficient sign ({coef:.3f}) is not as expected."
                    )
        # build summary
        summary = [{
            'rule': 'overall',
            'pass': overall,
            'details': None
        }]
        summary.extend({
            'rule': key[:-5],
            'pass': val,
            'details': '; '.join(details) if details and key in ['pvalue_pass','poos_sign_pass'] else None
        } for key, val in pass_dict.items())
        df = pd.DataFrame(summary)
        if as_df:
            return df
        return {'overall_pass': overall, 'rule_pass': pass_dict, 'failure_reasons': details}
