# TECHNIC/testing.py

from abc import ABC
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from scipy.stats import shapiro, kstest, anderson, cramervonmises
from statsmodels.tsa.stattools import (
    adfuller,
    kpss,
    phillips_perron,
    zivot_andrews,
    variance_ratio,
    dfgls
)
from .cm import CM


class ModelTestBase(ABC):
    """
    Abstract base class for model testing using a CM instance.

    Subclasses implement any number of test_* methods.
    run_all_tests discovers and runs them, aggregating results.
    """
    def __init__(self, cm: CM):
        self.cm = cm
        # use in-sample model by default
        self.model = cm.model_in
        self.X = cm.X_in
        self.y = cm.y_in
        self.X_out = cm.X_out
        self.y_out = cm.y_out
        self.y_pred_out = (
            cm.model_in.predict(cm.X_out)
            if not cm.X_out.empty else None
        )

    def run_all_tests(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for attr in dir(self):
            if attr.startswith('test_'):
                method = getattr(self, attr)
                if callable(method):
                    try:
                        out = method()
                        if not out:
                            continue
                        if attr == 'test_residuals':
                            results.update(out)
                        else:
                            results[attr] = out
                    except NotImplementedError:
                        continue
        return results


class OLSTesting(ModelTestBase):
    """
    OLS-specific tests for regression assumption checking and coefficient stability.
    """
    def __init__(
        self,
        cm: CM,
        pseudo_vars: Optional[List[str]] = None,
        pseudo_horizons: Optional[List[int]] = None
    ):
        super().__init__(cm)
        self.pseudo_vars = pseudo_vars or []
        self.pseudo_horizons = pseudo_horizons or []

    def test_residuals(self) -> Dict[str, Any]:
        resid = self.model.resid
        tests: Dict[str, Any] = {}
        # Normality tests
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        tests['jb_stat'] = float(jb_stat)
        tests['jb_pvalue'] = float(jb_p)
        sw_stat, sw_p = shapiro(resid)
        tests['shapiro_stat'] = float(sw_stat)
        tests['shapiro_pvalue'] = float(sw_p)
        ad_stat, ad_crit, _ = anderson(resid)
        tests['anderson_stat'] = float(ad_stat)
        tests['anderson_critical'] = ad_crit.tolist()
        ks_stat, ks_p = kstest((resid - resid.mean())/resid.std(ddof=1), 'norm')
        tests['kstest_stat'] = float(ks_stat)
        tests['kstest_pvalue'] = float(ks_p)
        cvm = cramervonmises((resid - resid.mean())/resid.std(ddof=1), 'norm')
        tests['cvm_stat'] = float(cvm.statistic)
        tests['cvm_pvalue'] = float(cvm.pvalue)
        # Heteroscedasticity
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, self.X)
        tests['bp_stat'] = float(bp_stat)
        tests['bp_pvalue'] = float(bp_p)
        # Autocorrelation
        tests['durbin_watson'] = float(durbin_watson(resid))
        bg_stat, bg_p, _, _ = acorr_breusch_godfrey(self.model, nlags=1)
        tests['bg_stat'] = float(bg_stat)
        tests['bg_pvalue'] = float(bg_p)
        # Stationarity tests
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(resid)
        tests['adf_stat'] = float(adf_stat)
        tests['adf_pvalue'] = float(adf_p)
        tests['adf_critical'] = adf_crit
        kpss_stat, kpss_p, _, _ = kpss(resid, nlags='auto')
        tests['kpss_stat'] = float(kpss_stat)
        tests['kpss_pvalue'] = float(kpss_p)
        pp_stat, pp_p, _, _ = phillips_perron(resid)
        tests['pp_stat'] = float(pp_stat)
        tests['pp_pvalue'] = float(pp_p)
        za_stat, za_p, za_crit = zivot_andrews(resid)
        tests['za_stat'] = float(za_stat)
        tests['za_pvalue'] = float(za_p)
        tests['za_critical'] = za_crit
        vr_stat = variance_ratio(resid)
        tests['variance_ratio'] = float(vr_stat)
        dfgls_stat, dfgls_p = dfgls(resid)
        tests['dfgls_stat'] = float(dfgls_stat)
        tests['dfgls_pvalue'] = float(dfgls_p)
        self.residual_tests = tests
        return tests

    def test_pseudo_oos_robustness(self) -> Dict[str, Any]:
        if not self.pseudo_vars or not self.pseudo_horizons:
            raise NotImplementedError
        orig = self.model.params
        metrics, details, coeffs = [], {}, {}
        for m in self.pseudo_horizons:
            cutoff = self.X.index.max() - pd.DateOffset(months=m)
            X_trunc = self.X.loc[:cutoff]
            y_trunc = self.y.loc[:cutoff]
            res = sm.OLS(y_trunc, sm.add_constant(X_trunc)).fit()
            devs, cf = [], {}
            for v in self.pseudo_vars:
                o = orig.get(v, np.nan)
                n = res.params.get(v, np.nan)
                cf[v] = float(n)
                if o != 0 and not np.isnan(n): devs.append(abs((n - o)/o))
            details[m] = float(np.mean(devs)) if devs else np.nan
            coeffs[m] = cf
            metrics.append(details[m])
        rob = float(np.nanmax(metrics))
        self.pseudo_oos_robustness = rob
        self.pseudo_oos_details = details
        self.pseudo_oos_coefficients = coeffs
        return {'poos_robustness': rob, 'poos_details': details, 'poos_coefficients': coeffs}


class SensitivityTesting(ModelTestBase):
    """
    Sensitivity testing for model robustness:
      - Parameter Sensitivity: +-1 standard error shocks per coefficient.
      - Input Sensitivity: multiple-std shocks per input variable.
    """
    def __init__(self, cm: CM):
        super().__init__(cm)

    def test_parameter_sensitivity(self) -> Dict[str, Dict[str, pd.Series]]:
        p = getattr(self.model, 'params', None)
        s = getattr(self.model, 'bse', None)
        if p is None or s is None:
            raise NotImplementedError
        Xc = sm.add_constant(self.X)
        out = {}
        for v in p.index:
            se = s.get(v, 0)
            pu, pdn = p.copy(), p.copy()
            pu[v] += se; pdn[v] -= se
            out[v] = {'plus': Xc.dot(pu), 'minus': Xc.dot(pdn)}
        self.parameter_sensitivity_tests = out
        return out

    def test_input_sensitivity(self) -> Dict[str, Dict[int, Dict[str, pd.Series]]]:
        st = self.X.std()
        out = {}
        for c in self.X.columns:
            tmp = {}
            for m in [1, 2, 3]:
                Xu, Xd = self.X.copy(), self.X.copy()
                Xu[c] += m * st[c]; Xd[c] -= m * st[c]
                tmp[m] = {'plus': self.model.predict(Xu), 'minus': self.model.predict(Xd)}
            out[c] = tmp
        self.input_sensitivity_tests = out
        return out


class ScenarioTesting(ModelTestBase):
    """
    Scenario testing applies forecast scenarios to the CM instance.
    Returns both raw and level-converted projections.
    """
    def __init__(self, cm: CM, model_type: Optional[str] = None):
        super().__init__(cm)
        # retrieve scenarios from DataManager
        self.scenarios: Dict[str, pd.DataFrame] = cm.dm.scen_mevs
        self.model_type = model_type

    def _convert_to_level(self, raw: pd.Series) -> pd.Series:
        cutoff = self.cm.dm.in_sample_end
        last = self.cm.dm.internal_data[self.cm.level].loc[cutoff]
        if self.model_type == 'GrowthRate':
            res = []
            prev = last
            for g in raw:
                prev = prev * (1 + g)
                res.append(prev)
            return pd.Series(res, index=raw.index)
        if self.model_type == 'RateDifference':
            res = []
            prev = last
            for r in raw:
                prev = prev + r
                res.append(prev)
            return pd.Series(res, index=raw.index)
        # Level
        return raw

    def test_scenarios(self) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        raw, conv = {}, {}
        for n, Xs in self.scenarios.items():
            try:
                pr = self.model.predict(Xs)
            except:
                pr = self.model.predict(sm.add_constant(Xs))
            raw[n] = pr
            conv[n] = self._convert_to_level(pr)
        self.scenario_tests = conv
        return raw, conv
