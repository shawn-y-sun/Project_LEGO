# =============================================================================
# module: test.py
# Purpose: Model testing framework with base and concrete test implementations
# Dependencies: pandas, statsmodels, scipy, abc, typing
# =============================================================================
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
import pandas as pd
import numpy as np

from statsmodels.stats.stattools import jarque_bera, durbin_watson
from .helper import het_white
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, normal_ad
from scipy.stats import shapiro, kstest, cramervonmises
from statsmodels.tsa.stattools import adfuller, zivot_andrews, range_unit_root_test, kpss
from arch.unitroot import PhillipsPerron, DFGLS, engle_granger
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
# ignore out-of-range interpolation warnings
warnings.filterwarnings('ignore', category=InterpolationWarning)

# ----------------------------------------------------------------------------
# ModelTestBase class
# ----------------------------------------------------------------------------

class ModelTestBase(ABC):
    """
    Abstract base class for model testing frameworks.

    Parameters
    ----------
    alias : Optional[str]
        Custom and human-readable name for the test instance (defaults to class name).
    filter_mode : str, default 'moderate'
        How to evaluate passed results: 'strict' or 'moderate'.
    """
    category: str = 'base'
    _allowed_modes = {'strict', 'moderate'}  # Allowed evaluation modes

    def __init__(
        self,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True,
    ):
        if filter_mode not in self._allowed_modes:
            raise ValueError(f"filter_mode must be one of {self._allowed_modes}")
        self.alias = alias or ''
        self.filter_mode = filter_mode
        self.filter_on = filter_on

    @property
    def name(self) -> str:
        """
        Display name for the test: alias if provided, else class name.
        """
        return self.alias or type(self).__name__

    @property
    @abstractmethod
    def test_result(self) -> Dict[str, Any]:
        """
        Execute the test(s) and return a **print‐friendly** result object.
 
        Could be a DataFrame, namedtuple, or other lightweight struct that
        formats cleanly when printed or logged.  Implementations should
        ensure it's fast to construct.
        """
        ...

    @property
    @abstractmethod
    def test_filter(self) -> bool:
        """
        Return True/False based on the chosen filter_mode and the
        content of `test_result`.  Implementations must adapt if
        `test_result` no longer returns a dict.
        """
        ...

# ----------------------------------------------------------------------------
# TestSet class
# ----------------------------------------------------------------------------

class TestSet:
    """
    Aggregator for ModelTestBase instances, with filtering and reporting utilities.

    Parameters
    ----------
    tests : dict
        Mapping from test alias (str) to ModelTestBase instance.
    """
    def __init__(
        self,
        tests: Dict[str, ModelTestBase]
    ):
        # Override each test's alias and collect in defined order
        self.tests: List[ModelTestBase] = []
        for alias, test_obj in tests.items():
            test_obj.alias = alias
            self.tests.append(test_obj)

    @property
    def all_test_results(self) -> Dict[str, Any]:
        """
        Return the test_result dict for every test in this set,
        keyed by the test’s display name (alias or class name),
        including both active and inactive tests.
        """
        return {t.name: t.test_result for t in self.tests}
    

    def filter_pass(
        self,
        fast_filter: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Run active tests and return overall pass flag and failed test names.

        Parameters
        ----------
        fast_filter : bool, default True
            If True, stops on first failure.

        Returns
        -------
        passed : bool
            True if all active tests pass.
        failed_tests : list of str
            Names of tests that did not pass.
        """
        failed = []
        for t in self.tests:
            if not t.filter_on:
                continue
            if not t.test_filter:
                failed.append(t.name)
                if fast_filter:
                    return False, failed
        return len(failed) == 0, failed

    def print_test_info(self) -> None:
        """
        Print summary of test configurations:
          - Active tests: name, category, filter_mode, filter_mode_desc
          - Inactive tests: name only, with note.
        """
        print("Active Tests:")
        for t in self.tests:
            if t.filter_on:
                print(f"- {t.name} | category: {t.category} | filter_mode: {t.filter_mode} | desc: {t.filter_mode_desc}")
        print("\nInactive Tests:")
        inactive = [t for t in self.tests if not t.filter_on]
        for t in inactive:
            print(f"- {t.name}")
        if inactive:
            print(
                "\nNote: These tests are included but not turned on. "
                "Set `filter_on=True` on a test to include it in filter_pass results."
            )

# ----------------------------------------------------------------------------
# FitMeasure class
# ----------------------------------------------------------------------------
class FitMeasure(ModelTestBase):
    """
    Compute and expose fit metrics for a fitted model.

    Parameters
    ----------
    actual : pd.Series
        The observed target values.
    predicted : pd.Series
        The model’s fitted or predicted values (in-sample).
    n_features : int
        Number of predictors (not including the intercept) used in fitting.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        Not used—always passes. Exists to satisfy ModelTestBase interface.
    """
    category = 'measure'

    def __init__(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        n_features: int,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = False
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.actual = actual
        self.predicted = predicted
        self.n = len(actual)
        self.p = n_features
        # this is only for reporting: do not include in filter_pass

    @property
    def test_result(self) -> pd.Series:
        """
        Returns
        -------
        pd.Series
            {'R²': ..., 'Adj R²': ...}
        """
        # compute sum of squares
        ss_res = ((self.actual - self.predicted) ** 2).sum()
        ss_tot = ((self.actual - self.actual.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        # adjusted R² = 1 - (1−R²)*(n−1)/(n−p−1)
        adj_r2 = 1 - (1 - r2) * (self.n - 1) / (self.n - self.p - 1) if self.n > self.p + 1 else float('nan')

        return pd.Series({'R²': float(r2), 'Adj R²': float(adj_r2)}, name=self.name)

    @property
    def test_filter(self) -> bool:
        """
        Always pass—this test is for reporting measures, not for filtering.
        """
        return True


# ----------------------------------------------------------------------------
# ErrorMeasure class
# ----------------------------------------------------------------------------
class ErrorMeasure(ModelTestBase):
    """
    Compute and expose error diagnostics for a fitted model.

    Parameters
    ----------
    actual : pd.Series
        The observed target values.
    predicted : pd.Series
        The model’s fitted or predicted values (in- or out-of-sample).
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        Not used—always passes. Exists to satisfy ModelTestBase interface.
    """
    category = 'measure'

    def __init__(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = False
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.errors = actual - predicted
        # this is only for reporting: do not include in filter_pass

    @property
    def test_result(self) -> pd.Series:
        """
        Returns
        -------
        pd.Series
            {'ME': ..., 'MAE': ..., 'RMSE': ...}
        """
        abs_err = self.errors.abs()
        me = float(abs_err.max())
        mae = float(abs_err.mean())
        rmse = float(np.sqrt((self.errors ** 2).mean()))

        return pd.Series({'ME': me, 'MAE': mae, 'RMSE': rmse}, name=self.name)

    @property
    def test_filter(self) -> bool:
        """
        Always pass—this test is for reporting measures, not for filtering.
        """
        return True

# ----------------------------------------------------------------------------
# R2Test class
# ----------------------------------------------------------------------------

class R2Test(ModelTestBase):
    """
    Assess in-sample R² fit quality of regression models.

    Parameters
    ----------
    r2 : float
        Model’s coefficient of determination.
    thresholds : Dict[str, float], optional
        Minimum R² by filter_mode; defaults to {'strict': 0.8, 'moderate': 0.6}.
    alias : Optional[str]
        Display name for this test.
    filter_mode : str
        'strict' or 'moderate'.
    """
    category = 'fit'

    def __init__(
        self,
        r2: float,
        thresholds: Optional[Dict[str, float]] = {'strict': 0.6, 'moderate': 0.4},
        alias: Optional[str] = None,
        filter_mode: str = 'strict',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.r2 = r2
        self.thresholds = thresholds
        self.filter_mode_descs = {
            'strict':   f"Require R² ≥ {self.thresholds['strict']}.",
            'moderate': f"Require R² ≥ {self.thresholds['moderate']}."
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.Series:
        return pd.Series({'R²': self.r2}, name=self.name)

    @property
    def test_filter(self) -> bool:
        thr = self.thresholds[self.filter_mode]
        return self.r2 >= thr

# ----------------------------------------------------------------------------
# AutocorrTest class
# ----------------------------------------------------------------------------

# Default test functions for autocorrelation diagnostics
autocorr_test_dict: Dict[str, Callable] = {
    # 'Durbin–Watson': lambda res: float(durbin_watson(res)),
    # 'Breusch–Godfrey': lambda res: _bg_pvalue(res)
    'Durbin–Watson':             lambda m: float(durbin_watson(m.resid)),
    'Breusch–Godfrey': lambda m: acorr_breusch_godfrey(m, nlags=1)[1]
}

class AutocorrTest(ModelTestBase):
    """
    Test for autocorrelation in residuals using multiple diagnostics.

    Parameters
    ----------
    results : any
        Results from a fitted regression model (e.g., `model.resid`).
    alias : str, optional
        A label for this test when reporting. If None, uses `self.name`.
    filter_mode : {'strict', 'moderate'}, default 'moderate'
        - 'strict': all tests must pass.
        - 'moderate': at least half of the tests must pass.
    test_dict : dict, optional
        Mapping of test names to functions computing the statistic. Defaults to DEFAULT_AUTOCORR_TEST_FUNCS.

    Attributes
    ----------
    test_funcs : dict
        Mapping test names to statistic functions.
    thresholds : dict
        Threshold definitions per test name.
    filter_mode_descs : dict
        Descriptions of filter modes.
    """
    category = 'assumption'

    # Descriptions of filter_mode behaviors
    filter_mode_descs = {
        'strict': 'All tests must pass',
        'moderate': 'At least half of the tests must pass'
    }

    # Threshold definitions (Durbin–Watson: (lower, upper); BG: p-value cutoff)
    threshold_defs = {
        'Durbin–Watson': (1.5, 2.5),
        'Breusch–Godfrey': 0.1
    }

    def __init__(
        self,
        results: Any,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        test_dict: Optional[Dict[str, Callable]] = None,
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        # store residuals array
        self.results = results
        # assign test functions (default or user-provided)
        self.test_funcs = test_dict if test_dict is not None else autocorr_test_dict
        # assign thresholds
        self.thresholds = self.threshold_defs

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Compute each autocorrelation test and package with threshold & pass/fail.

        Returns
        -------
        pd.DataFrame
            Index: test names.
            Columns:
              - 'statistic': computed value (float or p-value)
              - 'threshold': tuple or float threshold
              - 'passed': bool if statistic meets threshold criteria
        """
        records = []
        for name, func in self.test_funcs.items():
            stat = func(self.results)
            thresh = self.thresholds[name]
            if name == 'Durbin–Watson':
                lower, upper = thresh
                passed = lower <= stat <= upper
            else:
                alpha = thresh
                passed = stat > alpha
            records.append({'Test': name, 'Statistic': stat, 'Threshold': thresh, 'Passed': passed})
        df = pd.DataFrame(records).set_index('Test')
        return df

    @property
    def test_filter(self) -> bool:
        """
        Aggregate pass/fail according to filter_mode:
        - strict: all tests must pass
        - moderate: at least half of tests must pass
        """
        results = self.test_result['Passed']
        passed_count = int(results.sum())
        total = len(results)
        if self.filter_mode == 'strict':
            return passed_count == total
        else:
            return passed_count >= (total / 2)

# ----------------------------------------------------------------------------
# HetTest class
# ----------------------------------------------------------------------------

# Default test functions for homoscedasticity diagnostics
het_test_dict: Dict[str, Callable] = {
    'Breusch–Pagan': lambda res, exog: het_breuschpagan(res, exog)[1],
    'White': lambda res, exog: het_white(res, exog)[1]
}

class HetTest(ModelTestBase):
    """
    Test for homoscedasticity using Breusch–Pagan and White's tests.

    Parameters
    ----------
    resids : array-like
        Residuals from a fitted regression model (e.g., `model.resid`).
    exog : array-like
        Exogenous regressors (design matrix) used in the original model.
    alias : str, optional
        A label for this test when reporting. If None, uses `self.name`.
    filter_mode : {'strict', 'moderate'}, default 'moderate'
        - 'strict': all tests must pass.
        - 'moderate': at least half of the tests must pass.
    test_dict : dict, optional
        Mapping of test names to functions computing the statistic. Defaults to DEFAULT_HETTEST_FUNCS.

    Attributes
    ----------
    test_funcs : dict
        Mapping test names to statistic functions.
    thresholds : dict
        Threshold definitions per test name.
    filter_mode_descs : dict
        Descriptions of filter modes.
    """
    category = 'assumption'

    filter_mode_descs = {
        'strict': 'All tests must pass',
        'moderate': 'At least half of the tests must pass'
    }

    threshold_defs = {
        'Breusch–Pagan': 0.05,
        'White': 0.05
    }

    def __init__(
        self,
        resids: Union[np.ndarray, List[float]],
        exog: Union[np.ndarray, pd.DataFrame, List[List[float]]],
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        test_dict: Optional[Dict[str, Callable]] = None,
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.resids = np.asarray(resids)
        self.exog = np.asarray(exog)
        self.test_funcs = test_dict if test_dict is not None else het_test_dict
        self.thresholds = self.threshold_defs

    @property
    def test_result(self) -> pd.DataFrame:
        records = []
        for name, func in self.test_funcs.items():
            pval = func(self.resids, self.exog)
            alpha = self.thresholds[name]
            passed = pval > alpha
            records.append({'Test': name, 'P-value': pval, 'Passed': passed})
        df = pd.DataFrame(records).set_index('Test')
        return df

    @property
    def test_filter(self) -> bool:
        passed_count = int(self.test_result['Passed'].sum())
        total = len(self.test_funcs)
        return passed_count == total if self.filter_mode == 'strict' else passed_count >= (total / 2)

# ----------------------------------------------------------------------------
# NormalityTest class
# ----------------------------------------------------------------------------

def _cvm_test_fn(series: pd.Series):
    """Cramér–von Mises test against fitted Normal (mean, std)."""
    res = cramervonmises(series, 'norm', args=(series.mean(), series.std(ddof=1)))
    return res.statistic, res.pvalue

# Dictionary of normality diagnostic tests
normality_test_dict: Dict[str, Callable] = {
   'JB': lambda s: jarque_bera(s)[0:2],
   'SW': lambda s: shapiro(s)[0:2],
   'KS': lambda s: kstest(s, 'norm', args=(s.mean(), s.std(ddof=1)))[0:2],
   'CM': _cvm_test_fn,
   'AD': lambda s: normal_ad(s)
}

class NormalityTest(ModelTestBase):
    """
    Concrete test for normality diagnostics on a pandas Series.

    Uses multiple tests (Jarque-Bera, Shapiro) and applies filter_mode logic.
    """
    category = 'assumption'

    def __init__(
        self,
        series: pd.Series,
        alpha: Union[float, Dict[str, float]] = 0.05,
        alias: Optional[str] = None,
        filter_mode: str = 'strict',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.series = series
        self.alpha = alpha
        self.test_dict = normality_test_dict
        self.filter_mode_descs = {
            'strict':   'All normality tests must pass.',
            'moderate': 'At least half of normality tests must pass.'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run each normality test and return a DataFrame:
        ┌──────┬──────────┬─────────┬────────┐
        │ Test │ Statistic│ P-value │ Passed │
        ├──────┼──────────┼─────────┼────────┤
        │ JB   │   …      │   …     │  True  │
        │ SW   │   …      │   …     │  True  │
        └──────┴──────────┴─────────┴────────┘
        """
        rows = []
        for name, fn in self.test_dict.items():
            stat, pvalue = fn(self.series)[0:2]
            level = self.alpha[name] if isinstance(self.alpha, dict) else self.alpha
            passed = pvalue > level
            rows.append({'Test': name, 'Statistic': stat, 'P-value': pvalue, 'Passed': passed})
        return pd.DataFrame(rows).set_index('Test')

    @property
    def test_filter(self) -> bool:
        passed = self.test_result['Passed']
        if self.filter_mode == 'strict':
            return passed.all()
        return passed.sum() >= len(passed) / 2
    

# ----------------------------------------------------------------------------
# StationarityTest class
# ----------------------------------------------------------------------------
# Wrapper for KPSS test
def _kpss_test_fn(series: pd.Series):
    """KPSS test for level stationarity (null: stationary), suppressing interpolation warnings."""
    stat, pvalue, _, _ = kpss(series, regression='c', nlags='auto')
    return stat, pvalue

# Wrapper for Zivot–Andrews test
def _za_test_fn(series: pd.Series):
    """Zivot–Andrews test for unit root with one structural break (null: unit root)."""
    try:
        stat, crit_vals, pvalue = zivot_andrews(series, regression='c', maxlag=3)
    except ValueError:
        # if auxiliary regression fails due to rank deficiency, return NaNs
        stat, pvalue = np.nan, np.nan
    return stat, pvalue

# Wrapper for DF-GLS test using arch.unitroot
def _dfgls_test_fn(series: pd.Series):
    """DF-GLS test for unit root after GLS detrending (null: unit root)."""
    test = DFGLS(series)
    return float(test.stat), float(test.pvalue)

# Wrapper for ADF test
def _adf_test_fn(series: pd.Series):
    """Augmented Dickey–Fuller test for unit root (null: unit root)."""
    stat, pvalue, *_ = adfuller(series, autolag='AIC')
    return stat, pvalue

# Wrapper for Phillips–Perron test using arch.unitroot
def _pp_test_fn(series: pd.Series):
    """Phillips–Perron test for unit root (null: unit root)."""
    test = PhillipsPerron(series)
    return float(test.stat), float(test.pvalue)

def _rur_test_fn(series: pd.Series):
    """Range Unit Root (RUR) test for stationarity (null: stationary)."""
    # range_unit_root_test may return an object or tuple
    result = range_unit_root_test(series)
    # If result has attributes stat and pvalue
    if hasattr(result, 'stat') and hasattr(result, 'pvalue'):
        return float(result.stat), float(result.pvalue)
    # If result is tuple-like
    try:
        return result[0], result[1]
    except Exception:
        raise ValueError('Unexpected RUR test output format')

# Dictionary of stationarity diagnostic tests
stationarity_test_dict: Dict[str, Callable] = {
    'ADF': _adf_test_fn,
    'PP': _pp_test_fn,
    'KPSS': _kpss_test_fn,
    'ZA': _za_test_fn,
    'DFGLS': _dfgls_test_fn,
    'RUR': _rur_test_fn
}

class StationarityTest(ModelTestBase):
    """
    Concrete ModelTestBase implementation for stationarity testing using ADF.

    Parameters
    ----------
    series : Optional[pd.Series]
        Time series to test for stationarity.
    test_dict : Dict[str, callable]
        Mapping of test names to functions; default is {'adf': adfuller}..
    """
    category = 'assumption'

    # Thresholds and directions: (alpha, direction)
    threshold_defs = {
        'ADF': (0.05, '<'),
        'PP': (0.05, '<'),
        'KPSS': (0.05, '>'),
        'ZA': (0.05, '<'),
        'DFGLS': (0.05, '<'),
        'RUR': (0.05, '>' )
    }

    def __init__(
        self,
        series: Union[np.ndarray, pd.Series, list],
        alias: Optional[str] = None,
        filter_mode: str = 'strict',
        test_dict: Optional[Dict[str, Callable]] = None,
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.series = pd.Series(series)
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = self.threshold_defs
        self.filter_mode_descs = {
            'strict':   'All stationarity tests must pass.',
            'moderate': 'At least half of stationarity tests must pass.'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run each stationarity test and return a DataFrame:
        ┌──────┬──────────┬─────────┬────────┐
        │ Test │ Statistic│ P-value │ Passed │
        ├──────┼──────────┼─────────┼────────┤
        │ ADF  │   …      │   …     │  True  │
        │ PP   │   …      │   …     │  True  │
        └──────┴──────────┴─────────┴────────┘
        """
        records = []
        for name, func in self.test_dict.items():
            stat, pvalue = func(self.series)
            alpha, direction = self.thresholds[name]
            passed = pvalue < alpha if direction == '<' else pvalue > alpha
            records.append({
                'Test': name,
                'Statistic': stat,
                'P-value': pvalue,
                'Passed': passed
            })
        df = pd.DataFrame(records).set_index('Test')
        return df

    @property
    def test_filter(self) -> bool:
        """
        Return True if stationarity tests meet the threshold based on filter_mode:
        - strict:  all tests must pass
        - moderate: at least half of tests must pass
        """
        results = self.test_result['Passed']
        passed_count = int(results.sum())
        total = len(results)
        if self.filter_mode == 'strict':
            return passed_count == total
        return passed_count >= (total / 2)
    

    @property
    def test_result_legacy(self) -> pd.DataFrame:
        """
        Returns a legacy-style stationarity test table similar to SAS ARIMA's
        Augmented Dickey-Fuller test output.
        """
        types = {'Zero Mean': 'nc', 'Single Mean': 'c', 'Trend': 'ct'}
        data = []
        series = self.series.dropna() if self.series is not None else None
        if series is None:
            return pd.DataFrame()
        for typ, reg in types.items():
            for lag in (0, 1, 2):
                # run ADF with fixed lag and regression type
                res = adfuller(series, maxlag=lag, regression=reg, autolag=None, store=True)
                adfstat = res[0]
                pval_tau = res[1]
                resstore = res[3]
                regres = resstore.resols
                # coefficient on lagged level term
                delta = regres.params[0]
                rho = float(delta + 1)
                # p-value for rho parameter (unit root test)
                try:
                    pval_rho = float(regres.pvalues[0])
                except Exception:
                    pval_rho = None
                # F-statistic and p-value for model (skip for Zero Mean)
                if typ != 'Zero Mean':
                    fval = getattr(regres, 'fvalue', None)
                    pr_f = getattr(regres, 'f_pvalue', None)
                else:
                    fval = None
                    pr_f = None
                data.append({
                    'Type': typ,
                    'Lags': lag,
                    'Rho': rho,
                    'Pr < Rho': pval_rho,
                    'Tau': float(adfstat),
                    'Pr < Tau': float(pval_tau),
                    'F': fval,
                    'Pr > F': pr_f
                })
        return pd.DataFrame(data).set_index(['Type', 'Lags'])


# ----------------------------------------------------------------------------
# PvalueTest class
# ----------------------------------------------------------------------------

class PvalueTest(ModelTestBase):
    """
    Concrete test for checking coefficient significance of model parameters.

    Parameters
    ----------
    alpha : float, optional
        Significance level for p-value (default=0.05 strict).
    alias : str, optional
        Display name for this test (defaults to 'SignificanceTest').
    filter_mode : {'strict','moderate'}, default 'moderate'
        'strict'   → p-value < alpha;
        'moderate' → p-value < 2*alpha.
    """
    category = 'performance'
    filter_mode_descs = {
        'strict':'p-value < 5%',   
        'moderate': 'p-value < 10%'
    }

    def __init__(
        self,
        pvalues: pd.Series,
        alpha: float = 0.05,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.pvalues = pvalues
        self.alpha = alpha
        self.filter_mode_descs = {
            'strict':   f'Require p < {alpha} for all.',
            'moderate': f'Require p < {alpha*2} for at least half.'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'P-value': self.pvalues,
            'Passed':  self.pvalues < self.alpha
        })
        return df

    @property
    def test_filter(self) -> bool:
        passed = self.test_result['Passed']
        if self.filter_mode == 'strict':
            return passed.all()
        return passed.sum() >= len(passed) / 2


# ----------------------------------------------------------------------------
# F-Test for Group Significance 
# ----------------------------------------------------------------------------

class FTest(ModelTestBase):
    """
    Joint F-test for significance of a group of regression coefficients.

    Parameters
    ----------
    model_result : any
        Fitted statsmodels regression result (must support .f_test).
    vars : list of str
        Names of coefficients to test jointly (e.g. ['x1','x2']).
    alpha : float, optional
        Significance level for p-value (default=0.05 strict).
    alias : str, optional
        Display name for this test (defaults to 'FTest').
    filter_mode : {'strict','moderate'}, default 'moderate'
        'strict'   → p-value < alpha;
        'moderate' → p-value < 2*alpha.
    """
    category = 'performance'

    def __init__(
        self,
        model_result: Any,
        vars: list,
        alpha: float = 0.05,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.model_result = model_result
        self.vars = vars
        self.alpha = alpha
        # set per-mode descriptions
        self.filter_mode_descs = {
            'strict':   f"F-test p < {self.alpha} for group {vars}.",
            'moderate': f"F-test p < {self.alpha*2} for group {vars}."
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Perform joint hypothesis test that all specified coefficients are zero.
        Returns DataFrame with columns ['F-statistic','P-value','Passed'] and index label alias.
        """
        # build restriction matrix string e.g. 'x1 = 0, x2 = 0'
        hypothesis = ' = 0, '.join(self.vars) + ' = 0'
        res = self.model_result.f_test(hypothesis)
        fstat = float(res.fvalue)
        pvalue = float(res.pvalue)
        passed = pvalue < (self.alpha if self.filter_mode=='strict' else self.alpha*2)
        df = pd.DataFrame([{
            'F-statistic': fstat,
            'P-value':     pvalue,
            'Passed':      passed
        }], index=['Joint F Test'])
        return df

    @property
    def test_filter(self) -> bool:
        """
        Return True if the F-test p-value meets threshold for filter_mode.
        """
        return bool(self.test_result['Passed'].iloc[0])


# ----------------------------------------------------------------------------
# VIF Test for Multicollinearity
# ----------------------------------------------------------------------------

class VIFTest(ModelTestBase):
    """
    Test for multicollinearity by computing Variance Inflation Factors (VIF) for each predictor.

    Parameters
    ----------
    exog : array-like or pandas.DataFrame
        Exogenous regressors (design matrix) including an intercept if appropriate.
    alias : str, optional
        Label for this test. If None, uses `self.name`.
    filter_mode : {'strict', 'moderate'}, default 'strict'
        - 'strict': threshold = 5
        - 'moderate': threshold = 10
    """
    def __init__(
        self,
        exog: Union[np.ndarray, pd.DataFrame, list],
        alias: Optional[str] = None,
        filter_mode: str = 'strict',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.exog = pd.DataFrame(exog)
        self.filter_mode_descs = {
        'strict': 'Threshold = 5',
        'moderate': 'Threshold = 10'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Compute VIF for each variable.

        Returns
        -------
        pandas.DataFrame
            Index: variable names
            Columns: 'VIF'
        """
        vif_values = []
        X = self.exog.values
        cols = self.exog.columns
        for i, col in enumerate(cols):
            vif = float(variance_inflation_factor(X, i))
            vif_values.append({'VIF': vif})
        df = pd.DataFrame(vif_values, index=cols)
        df.index.name = 'Variable'
        # drop the intercept (constant) if present
        df = df.drop(index='const', errors='ignore')
        return df

    @property
    def test_filter(self) -> bool:
        """
        Passes if all VIFs are below the threshold implied by filter_mode.
        """
        threshold = 5.0 if self.filter_mode == 'strict' else 10.0
        return (self.test_result['VIF'] <= threshold).all()
    
# ----------------------------------------------------------------------------
# Co-integration Test
# ----------------------------------------------------------------------------

class CointTest(ModelTestBase):
    """
    Test for cointegration of y with X via Engle–Granger using p-values.

    Parameters
    ----------
    y : array-like
        Dependent series (e.g., pd.Series).
    X : array-like
        Explanatory variables (2D array or DataFrame).
    alias : str, optional
        Label for this test. If None, uses self.name.
    filter_mode : {'strict', 'moderate'}, default 'strict'
        - 'strict':   require p-value < 0.05
        - 'moderate': require p-value < 0.10

    Attributes
    ----------
    alpha : float
        Significance level for pass criteria (0.05 or 0.10).
    filter_mode_descs : dict
        Descriptions of the pass criteria for each mode.
    """
    category = 'assumption'

    filter_mode_descs = {
        'strict':   'Require Engle–Granger p-value < 0.05',
        'moderate': 'Require Engle–Granger p-value < 0.10'
    }

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series, list],
        X: Union[np.ndarray, pd.DataFrame, list],
        alias: Optional[str] = None,
        filter_mode: str = 'strict'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
        # Align y and X, drop missing
        self.y = pd.Series(y).dropna()
        self.X = pd.DataFrame(X, index=self.y.index).dropna(axis=1, how='any')
        # Set alpha based on mode
        self.alpha = 0.05 if filter_mode == 'strict' else 0.10
        self.filter_mode_desc = self.filter_mode_descs[filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run Engle-Granger and report statistic, p-value, and pass/fail.
        """
        try:
            result = engle_granger(
                self.y.values,
                self.X.values,
                trend='c', lags=None, max_lags=None, method='bic'
            )
            stat = float(result.stat)
            pval = float(result.pvalue)
            passed = pval < self.alpha
        except Exception:
            stat, pval, passed = np.nan, np.nan, False

        df = pd.DataFrame([{
            'Statistic': stat,
            'P-value':   pval,
            'Passed':     passed
        }], index=[self.name])
        return df

    @property
    def test_filter(self) -> bool:
        """
        Indicates whether the cointegration test passed based on alpha.
        """
        return bool(self.test_result['Passed'].iloc[0])