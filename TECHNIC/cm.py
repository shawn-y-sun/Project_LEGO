# TECHNIC/cm.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from typing import Type, List, Dict, Any, Optional, Union

from .internal import InternalDataLoader
from .mev import MEVLoader
from .model import ModelBase
from .measure import MeasureBase
from .transform import TSFM
from .report import scenario_rank_test  # assuming this exists

# Advanced statistical tests
from scipy.stats import shapiro, kstest, cramervonmises
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
# Unit-root and variance-ratio tests
from arch.unitroot import PhillipsPerron, VarianceRatio, DFGLS

class CM:
    """
    Candidate Model wrapper.

    Manages in-sample, out-of-sample, full-sample fitting,
    scenario analysis, pseudo-out-of-sample checks,
    parameter/input sensitivity, filtering and reporting.
    """
    def __init__(
        self,
        model_id: str,
        target: str,
        data_manager: Any,
        model_cls: Type[ModelBase],
        measure_cls: Type[MeasureBase],
        report_cls: Optional[Type] = None
    ):
        self.model_id = model_id
        self.target = target
        self.dm = data_manager
        self.model_cls = model_cls
        self.measure_cls = measure_cls
        self.report_cls = report_cls
        # placeholders
        self.X = self.y = None
        self.X_in = self.y_in = None
        self.X_out = self.y_out = None
        self.X_full = self.y_full = None
        self.model_in = self.model_full = None
        self.measure_in = self.measure_full = None
        self.report_in = self.report_full = None

    def build(self, specs: List[Union[str, Dict[str, TSFM]]]) -> None:
        """
        Build features/target, validate, split, fit models,
        compute measures and instantiate reports.
        """
        X = self.dm.build_indep_vars(specs)
        y = self.dm.internal_data[self.target].copy()
        idx = self.dm.internal_data.index
        X = X.reindex(idx).astype(float)
        y = y.reindex(idx).astype(float)
        self.X, self.y = X, y

        # data validation
        nan_cols, inf_cols = [], []
        for col in X.columns:
            s = X[col]
            if s.isna().any(): nan_cols.append(col)
            elif is_numeric_dtype(s) and not np.isfinite(s.dropna()).all(): inf_cols.append(col)
        if nan_cols or inf_cols or y.isna().any() or (
            is_numeric_dtype(y) and not np.isfinite(y.dropna()).all()
        ):
            msgs = []
            if nan_cols: msgs.append(f"X contains NaNs: {nan_cols}")
            if inf_cols: msgs.append(f"X contains infinite: {inf_cols}")
            if y.isna().any(): msgs.append("y contains NaNs")
            if is_numeric_dtype(y) and not np.isfinite(y.dropna()).all(): msgs.append("y contains infinite values")
            raise ValueError("Data validation error: " + "; ".join(msgs))

        # split
        cutoff = self.dm.in_sample_end
        if cutoff is not None:
            self.X_in, self.y_in = X.loc[:cutoff], y.loc[:cutoff]
            self.X_out, self.y_out = (
                X.loc[cutoff + pd.Timedelta(days=1):],
                y.loc[cutoff + pd.Timedelta(days=1):]
            )
        else:
            self.X_in, self.y_in = X, y
            self.X_out, self.y_out = pd.DataFrame(), pd.Series(dtype=float)
        self.X_full, self.y_full = X, y

        # fit models
        self.model_in = self.model_cls(self.X_in, self.y_in).fit()
        self.model_full = self.model_cls(self.X_full, self.y_full).fit()

        # compute measures
        self.measure_in = self.measure_cls(
            self.model_in, self.X_in, self.y_in,
            X_out=self.X_out, y_out=self.y_out,
            y_pred_out=(self.model_in.predict(self.X_out) if not self.X_out.empty else None)
        )
        self.measure_full = self.measure_cls(self.model_full, self.X_full, self.y_full)

        # instantiate reports
        if self.report_cls:
            self.report_in = self.report_cls(self.measure_in)
            self.report_full = self.report_cls(self.measure_full)

    def scenario_test(self, *args, **kwargs) -> Any:
        """
        Scenario testing via external scenario_rank_test function.
        """
        return scenario_rank_test(self, *args, **kwargs)

    def _roll_refit(
        self,
        variables: List[str],
        horizons: List[int],
        model_cls: Optional[Type[ModelBase]] = None
    ) -> Dict[int, pd.Series]:
        """
        Helper to refit the model on rolling cutoffs and
        return parameter series for each horizon.
        """
        ModelCls = model_cls or self.model_cls
        last = self.X_in.index.max()
        param_records: Dict[int, pd.Series] = {}
        for m in horizons:
            cutoff = last - pd.DateOffset(months=m)
            X_sub = self.X_in.loc[:cutoff]
            y_sub = self.y_in.loc[:cutoff]
            sub_res = ModelCls(X_sub, y_sub).fit()
            param_records[m] = sub_res.params
        return param_records

    def pseudo_oos_robustness(
        self,
        roll_horizons: List[int],
        model_cls: Optional[Type[ModelBase]] = None
    ) -> Dict[str, Any]:
        """
        Compute POoS_robustness metric over roll_horizons (months),
        leveraging the _roll_refit helper.
        """
        baseline = self.model_full.params
        records = self._roll_refit(
            variables=list(baseline.index),
            horizons=roll_horizons,
            model_cls=model_cls
        )
        deviations: Dict[int, float] = {}
        for m, params in records.items():
            deviations[m] = ((params - baseline) / baseline).abs().mean()
        robustness = max(deviations.values()) if deviations else np.nan
        return {'POoS_robustness': robustness, 'deviations': deviations}

    def parameter_sensitivity(
        self,
        shock_se: float = 1.0
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        One-at-a-time shock of coefficients by +/- shock_se*std_error.
        Returns projections for each var and direction.
        """
        base_params = self.model_full.params
        std_err = self.model_full.bse
        results: Dict[str, Dict[str, pd.Series]] = {}
        X_full = self.X_full.assign(const=1)
        for var in base_params.index.drop('const', errors='ignore'):
            se = std_err.get(var, 0) * shock_se
            for direction, mult in [('plus', 1), ('minus', -1)]:
                new_params = base_params.copy()
                new_params[var] += mult * se
                preds = X_full[new_params.index].dot(new_params)
                results.setdefault(var, {})[direction] = preds
        return results

    def input_sensitivity(
        self,
        shock_levels: List[int] = [1, 2, 3]
    ) -> Dict[str, Dict[int, pd.Series]]:
        """
        Shock each input by +/- shock_levels * its std dev and project target.
        Returns projections for each var and shock magnitude.
        """
        X_full = self.X_full.copy()
        results: Dict[str, Dict[int, pd.Series]] = {}
        for var in X_full.columns:
            sd = self.X_in[var].std()
            for level in shock_levels:
                for sign in [1, -1]:
                    X_shocked = X_full.copy()
                    X_shocked[var] += sign * level * sd
                    preds = self.model_full.predict(X_shocked)
                    key = level if sign == 1 else -level
                    results.setdefault(var, {})[key] = preds
        return results

    def filter(
        self,
        significance_threshold: float = 0.05,
        vif_threshold: float = 10.0,
        sign_map: Optional[Dict[str, int]] = None,
        group_drivers: Optional[Dict[str, List[str]]] = None,
        mev_sign_flip_vars: Optional[List[str]] = None,
        mev_flip_periods: Optional[List[int]] = None,
        mev_model_cls: Optional[Type[ModelBase]] = None
    ) -> Dict[str, Any]:
        """
        Execute all filtering tests on the in-sample model.
        Returns a dict with structured results for each filter category.
        """
        if self.measure_in is None:
            raise RuntimeError("Call build() before filter().")
        res = self.model_in
        pth, vif_th = significance_threshold, vif_threshold
        outcomes: Dict[str, Any] = {}
        resid = getattr(res, 'resid', pd.Series(res.results.resid))

        # Normality
        normals = {
            'jb': jarque_bera(resid)[1],
            'shapiro': shapiro(resid)[1],
            'ks': kstest(resid, 'norm', args=(resid.mean(), resid.std()))[1],
            'cvm': cramervonmises(resid, 'norm').pvalue
        }
        normal_pass = sum(p > pth for p in normals.values()) > len(normals)/2
        outcomes['normality'] = {
            'pass': normal_pass,
            'p_values': normals,
            'failed': [name for name,p in normals.items() if p <= pth]
        }

        # Stationarity
        stats = {
            'adf': adfuller(resid)[1],
            'kpss': kpss(resid, nlags='auto')[1],
            'za': zivot_andrews(resid).pvalue,
            'pp': PhillipsPerron(resid).pvalue,
            'vr': VarianceRatio(resid).p_value,
            'dfgls': DFGLS(resid).pvalue
        }
        stat_flags = [stats['adf']<pth, stats['kpss']>pth, stats['za']<pth,
                      stats['pp']<pth, stats['vr']<pth, stats['dfgls']<pth]
        stationary_pass = sum(stat_flags) > len(stat_flags)/2
        outcomes['stationarity'] = {
            'pass': stationary_pass,
            'p_values': stats,
            'failed': [name for name,flag in zip(stats.keys(), stat_flags) if not flag]
        }

        # Individualvariable significance
        pvals = res.pvalues.drop('const', errors='ignore')
        group_vars = {v for grp in (group_drivers or {}).values() for v in grp}
        indiv_vars = [v for v in pvals.index if v not in group_vars]
        failed_inds = [v for v in indiv_vars if pvals[v] >= pth]
        outcomes['individual'] = {
            'pass': not failed_inds,
            'p_values': pvals[indiv_vars].to_dict(),
            'failed': failed_inds
        }

        # Group-driver F-tests
        group_results = {}
        failed_groups = []
        if group_drivers:
            for name, vars_list in group_drivers.items():
                pv = float(res.results.f_test(' and '.join(f"{v}=0" for v in vars_list)).pvalue)
                ok = pv < pth
                group_results[name] = {'pass': ok, 'p_value': pv}
                if not ok: failed_groups.append(name)
        outcomes['groups'] = {
            'results': group_results,
            'failed': failed_groups
        }

        # Overall F-test & VIF
        f_p = getattr(res.results, 'f_pvalue', np.nan)
        max_vif = max(getattr(res, 'vif', {}).values(), default=np.nan)
        outcomes['model_f_test'] = {'pass': f_p < pth, 'p_value': f_p}
        outcomes['collinearity'] = {'pass': max_vif < vif_th, 'max_vif': max_vif}

        # Sign alignment
        if sign_map:
            signs = self.model_in.params.reindex(sign_map.keys()).apply(np.sign)
            alignment_fail = [v for v, exp in sign_map.items() if np.sign(self.model_in.params.get(v,0)) != exp]
            outcomes['sign_alignment'] = {
                'pass': not alignment_fail,
                'failed': alignment_fail
            }

        # MEV Sign Flip
        flip_out = {'details': {}, 'failed': []}
        if mev_sign_flip_vars and mev_flip_periods:
            records = self._roll_refit(mev_sign_flip_vars, mev_flip_periods, mev_model_cls)
            for var in mev_sign_flip_vars:
                exp_sign = sign_map.get(var, np.sign(self.model_in.params.get(var, np.nan)))
                flip_out['details'][var] = {}
                for m, params in records.items():
                    ok = np.sign(params.get(var, np.nan)) == exp_sign
                    flip_out['details'][var][m] = ok
                    if not ok: flip_out['failed'].append(var)
        outcomes['mev_sign_flip'] = {'pass': not flip_out['failed'], 'details': flip_out['details'], 'failed': flip_out['failed']}

        return outcomes

    def summary_tables(
        self,
        scenario_args: Optional[List[Any]] = None,
        scenario_kwargs: Optional[Dict[str, Any]] = None,
        poos_horizons: Optional[List[int]] = None,
        shock_se: float = 1.0,
        shock_levels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Return a comprehensive summary including:
          - In-sample and out-of-sample performance measures
          - Filter outcomes
          - Scenario test results
          - Pseudo-out-of-sample robustness
          - Parameter sensitivity projections
          - Input sensitivity projections

        Optional parameters allow passing args to scenario_test,
        specifying roll horizons for POoS, and shock levels.
        """
        # Performance
        in_perf = self.measure_in.in_perf_measures
        out_perf = self.measure_in.out_perf_measures

        # Filtering
        filter_res = self.filter()

        # Scenario testing
        sc_args = scenario_args or []
        sc_kwargs = scenario_kwargs or {}
        scenario_res = self.scenario_test(*sc_args, **sc_kwargs)

        # Pseudo-out-of-sample robustness
        horizons = poos_horizons or []
        poos_res = self.pseudo_oos_robustness(horizons)

        # Parameter sensitivity
        param_res = self.parameter_sensitivity(shock_se)

        # Input sensitivity
        levels = shock_levels or []
        input_res = self.input_sensitivity(levels)

        return {
            'in_sample_performance': in_perf,
            'out_of_sample_performance': out_perf,
            'filter_outcomes': filter_res,
            'scenario_test': scenario_res,
            'pseudo_oos_robustness': poos_res,
            'parameter_sensitivity': param_res,
            'input_sensitivity': input_res
        }


    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        Display sequentially:
          1) In-sample performance
          2) Optional out-of-sample performance
          3) Model parameters
          4) In-sample performance plot
          5) Optional testing metrics & plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # disable out-of-sample if none
        if not self.measure_in.out_perf_measures:
            show_out = False

        # 1) In-sample performance
        print(f"--- {self.model_id} — In-Sample Performance ---")
        print(self.report_in.show_perf_tbl().to_string(index=False))

        # 2) Out-of-sample performance
        if show_out:
            print(f"\n--- {self.model_id} — Out-of-Sample Performance ---")
            print(self.report_in.show_out_perf_tbl().to_string(index=False))

        # 3) Parameters
        def fmt_coef(x):
            try: val = float(x)
            except: return str(x)
            if abs(val)>=1e5 or (0<abs(val)<1e-3): return f"{val:.4e}"
            return f"{val:.4f}"
        def fmt_std(x):
            try: val = float(x)
            except: return str(x)
            if abs(val)>=1e5 or (0<abs(val)<1e-3): return f"{val:.4e}"
            return f"{val:.4f}"

        print(f"\n--- {self.model_id} — Model Parameters ---")
        params_df = self.report_in.show_params_tbl()
        print(params_df.to_string(index=False, formatters={
            'Coef': fmt_coef, 'Pvalue': '{:.3f}'.format,
            'VIF': '{:.2f}'.format, 'Std': fmt_std
        }))

        # 4) In-sample performance plot
        fig1 = self.report_in.plot_perf(**perf_kwargs)
        plt.show()

        # 5) Optional tests
        if show_tests:
            print(f"\n--- {self.model_id} — Test Metrics ---")
            print(self.report_in.show_test_tbl().to_string(index=False))
            fig2 = self.report_in.plot_tests(**test_kwargs)
            plt.show()
