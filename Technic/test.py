# =============================================================================
# module: test.py
# Purpose: Model testing framework with base and concrete test implementations
# Key Types/Classes: ModelTestBase, StationarityTest, FullStationarityTest,
#                    TargetStationarityTest, MultiFullStationarityTest, CoefTest
# Key Functions: _adf_test_fn, _pp_test_fn, stationarity_test_dict
# Dependencies: pandas, statsmodels, scipy, abc, typing
# =============================================================================
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union
import pandas as pd
import numpy as np

from statsmodels.stats.stattools import jarque_bera, durbin_watson
from .helper import het_white
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, normal_ad
from scipy.stats import shapiro, kstest, cramervonmises
from statsmodels.tsa.stattools import adfuller, zivot_andrews, range_unit_root_test, kpss
from arch.unitroot import PhillipsPerron, DFGLS, engle_granger
from arch.unitroot.unitroot import InfeasibleTestException
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .data import DataManager
from .transform import TSFM
from .regime import RgmVar
from .condition import CondVar
from .feature import Feature, DumVar

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
        Compute R² and adjusted R² and return as a small table.

        Returns
        -------
        pandas.DataFrame
            Index named 'Metric' with rows 'R²' and 'Adj R²' and a single
            column 'Value'.

        Example output structure
        ------------------------
        ┌──────────┬─────────┐
        │ Metric   │ Value   │
        ├──────────┼─────────┤
        │ R²       │ 0.87    │
        │ Adj R²   │ 0.85    │
        └──────────┴─────────┘
        """
        # compute sum of squares
        ss_res = ((self.actual - self.predicted) ** 2).sum()
        ss_tot = ((self.actual - self.actual.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        # adjusted R² = 1 - (1−R²)*(n−1)/(n−p−1)
        adj_r2 = 1 - (1 - r2) * (self.n - 1) / (self.n - self.p - 1) if self.n > self.p + 1 else float('nan')
        df = pd.DataFrame(
            [{'Metric': 'R²', 'Value': float(r2)},
             {'Metric': 'Adj R²', 'Value': float(adj_r2)}]
        ).set_index('Metric')
        df.index.name = 'Metric'
        return df

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
        Compute error diagnostics (ME, MAE, RMSE) and return as a table.

        Returns
        -------
        pandas.DataFrame
            Index named 'Metric' with rows 'ME', 'MAE', 'RMSE' and column 'Value'.

        Example output structure
        ------------------------
        ┌────────┬─────────┐
        │ Metric │ Value   │
        ├────────┼─────────┤
        │ ME     │ 1.23    │
        │ MAE    │ 0.54    │
        │ RMSE   │ 0.78    │
        └────────┴─────────┘
        """
        abs_err = self.errors.abs()
        me = float(abs_err.max())
        mae = float(abs_err.mean())
        rmse = float(np.sqrt((self.errors ** 2).mean()))
        df = pd.DataFrame(
            [{'Metric': 'ME', 'Value': me},
             {'Metric': 'MAE', 'Value': mae},
             {'Metric': 'RMSE', 'Value': rmse}]
        ).set_index('Metric')
        df.index.name = 'Metric'
        return df

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
        Minimum R² by filter_mode; defaults to {'strict': 0.6, 'moderate': 0.3}.
    alias : Optional[str]
        Display name for this test.
    filter_mode : str
        'strict' or 'moderate'.
    """
    category = 'performance'

    def __init__(
        self,
        r2: float,
        thresholds: Optional[Dict[str, float]] = {'strict': 0.6, 'moderate': 0.3},
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.r2 = r2
        self.thresholds = thresholds
        # self.filter_mode_descs = {
        #     'strict':   f"Require R² ≥ {self.thresholds['strict']}.",
        #     'moderate': f"Require R² ≥ {self.thresholds['moderate']}."
        # }
        # self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]
    
    @property
    def filter_mode_descs(self):
        return {
            'strict':   f"Require R² ≥ {self.thresholds['strict']}.",
            'moderate': f"Require R² ≥ {self.thresholds['moderate']}."
        }
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.Series:
        """
        Return the model R² as a one-line series with a named index.

        Returns
        -------
        pandas.Series
            Index name 'Metric' with a single entry 'R²'. The series name is
            the test instance name.

        Example output structure
        ------------------------
        ┌──────────┬───────┐
        │ Metric   │ value │
        ├──────────┼───────┤
        │ R²       │ 0.74  │
        └──────────┴───────┘
        """
        s = pd.Series({'R²': self.r2}, name=self.name)
        s.index.name = 'Metric'
        return s

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
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

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

        Example output structure
        ------------------------
        ┌───────────────────┬───────────┬────────────┬────────┐
        │ Test              │ Statistic │ Threshold  │ Passed │
        ├───────────────────┼───────────┼────────────┼────────┤
        │ Durbin–Watson     │ 2.01      │ (1.5, 2.5) │ True   │
        │ Breusch–Godfrey   │ 0.23      │ 0.1        │ True   │
        └───────────────────┴───────────┴────────────┴────────┘
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
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run homoscedasticity tests and return a table of p-values and pass/fail.

        Returns
        -------
        pandas.DataFrame
            Index 'Test' with columns 'P-value' and 'Passed'.

        Example output structure
        ------------------------
        ┌────────────────┬──────────┬────────┐
        │ Test           │ P-value  │ Passed │
        ├────────────────┼──────────┼────────┤
        │ Breusch–Pagan  │ 0.42     │ True   │
        │ White          │ 0.31     │ True   │
        └────────────────┴──────────┴────────┘
        """
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
   'CM': _cvm_test_fn,
#    'SW': lambda s: shapiro(s)[0:2],
#    'KS': lambda s: kstest(s, 'norm', args=(s.mean(), s.std(ddof=1)))[0:2],
#    'AD': lambda s: normal_ad(s)
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
        filter_mode: str = 'moderate',
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
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run each normality test and return a DataFrame.

        Example output structure
        ------------------------
        ┌──────┬──────────┬─────────┬────────┐
        │ Test │ Statistic│ P-value │ Passed │
        ├──────┼──────────┼─────────┼────────┤
        │ JB   │   …      │   …     │  True  │
        │ CM   │   …      │   …     │  True  │
        └──────┴──────────┴─────────┴────────┘
        """
        # Drop missing values to avoid test-function failures on NaNs.
        series = pd.to_numeric(self.series, errors='coerce').dropna()
        if series.empty:
            return pd.DataFrame(columns=['Statistic', 'P-value', 'Passed'])

        rows = []
        for name, fn in self.test_dict.items():
            stat, pvalue = fn(series)[0:2]
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
    # 'KPSS': _kpss_test_fn,
    # 'ZA': _za_test_fn,
    # 'DFGLS': _dfgls_test_fn,
    # 'RUR': _rur_test_fn
}

# Thresholds and directions for stationarity tests: (alpha, direction)
stationarity_test_threshold: Dict[str, Tuple[float, str]] = {
    'ADF': (0.05, '<'),
    'PP': (0.05, '<'),
    'KPSS': (0.05, '>'),
    'ZA': (0.05, '<'),
    'DFGLS': (0.05, '<'),
    'RUR': (0.05, '>' )
}

class StationarityTest(ModelTestBase):
    """
    Concrete ModelTestBase implementation for stationarity testing using ADF.

    Parameters
    ----------
    series : Optional[pd.Series]
        Time series to test for stationarity.
    test_dict : Dict[str, callable], optional
        Mapping of test names to functions; defaults to stationarity_test_dict.
    test_threshold : Dict[str, Tuple[float, str]], optional
        Test thresholds and directions; defaults to stationarity_test_threshold.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        - 'strict':   all stationarity tests must pass
        - 'moderate': at least half of stationarity tests must pass
    filter_on : bool, default True
        Whether this test is active in filtering.
    """
    category = 'assumption'

    def __init__(
        self,
        series: Union[np.ndarray, pd.Series, list],
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        test_dict: Optional[Dict[str, Callable]] = None,
        test_threshold: Optional[Dict[str, Tuple[float, str]]] = None,
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.series = pd.Series(series)
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = test_threshold if test_threshold is not None else stationarity_test_threshold
        self.filter_mode_descs = {
            'strict':   'All stationarity tests must pass.',
            'moderate': 'At least half of stationarity tests must pass.'
        }
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run each stationarity test and return a DataFrame.

        Example output structure
        ------------------------
        ┌──────┬──────────┬─────────┬────────┐
        │ Test │ Statistic│ P-value │ Passed │
        ├──────┼──────────┼─────────┼────────┤
        │ ADF  │   …      │   …     │  True  │
        │ PP   │   …      │   …     │  True  │
        └──────┴──────────┴─────────┴────────┘

        Notes
        -----
        If a diagnostic raises ``InfeasibleTestException`` (typically from the
        Phillips–Perron routine on very short samples), the statistic and
        p-value are recorded as ``None`` and the check is marked as failed so
        downstream filters can handle the edge case gracefully.
        """
        # Normalize dtype to float to avoid mixed-int/float dtype issues inside
        # tests such as Phillips–Perron which expect pure numeric input.
        # Drop missing values before running stationarity diagnostics to
        # prevent underlying tests from raising on NaN inputs.
        series = pd.to_numeric(self.series, errors='coerce').dropna().astype(float)
        if series.empty:
            return pd.DataFrame(columns=['Statistic', 'P-value', 'Passed'])

        records = []
        for name, func in self.test_dict.items():
            try:
                stat, pvalue = func(series)
                alpha, direction = self.thresholds[name]
                passed = pvalue < alpha if direction == '<' else pvalue > alpha
            except InfeasibleTestException:
                # Short samples can make long-run covariance estimators infeasible;
                # mark the test as failed while returning explicit null diagnostics.
                stat, pvalue, passed = None, None, False
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


class FullStationarityTest(ModelTestBase):
    """
    Run staged stationarity checks across in-sample and full-sample data.

    The test first evaluates stationarity on the in-sample portion of a
    constructed feature. If the check fails, it retries on the combined
    in- and out-of-sample data. For regime-shift or conditional variables,
    it optionally re-runs the sequence on the original (pre-regime or
    pre-condition) variable.

    Parameters
    ----------
    variable : Union[str, TSFM, Dict[str, pd.Series]]
        Variable identifier or transformation specification to build.
        Regime-aware (:class:`~Technic.regime.RgmVar`) and conditional
        (:class:`~Technic.condition.CondVar`) features are also supported.
        When a dictionary is provided, keys are treated as sample labels and
        values are pre-materialized series evaluated sequentially until a
        stationary series is found.
    dm : DataManager
        Data manager used to construct features and access sample indices.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict', 'moderate'}, default 'moderate'
        Passed through to the underlying stationarity test.
    test_dict : Dict[str, callable], optional
        Mapping of test names to functions; defaults to
        ``stationarity_test_dict``.
    test_threshold : Dict[str, Tuple[float, str]], optional
        Test thresholds and directions; defaults to
        ``stationarity_test_threshold``.
    filter_on : bool, default True
        Whether this test is active in filtering.
    test_class : Type[StationarityTest], optional
        Test class to instantiate for each sample evaluation. Defaults to
        :class:`StationarityTest`.

    Examples
    --------
    >>> fst = FullStationarityTest('GDP', dm)
    >>> fst.test_result
    >>> fst.test_filter
    """

    category = 'assumption'

    def __init__(
        self,
        variable: Union[str, TSFM, RgmVar, CondVar, Mapping[str, pd.Series]],
        dm: DataManager,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        test_dict: Optional[Dict[str, Callable]] = None,
        test_threshold: Optional[Dict[str, Tuple[float, str]]] = None,
        filter_on: bool = True,
        test_class: Type[StationarityTest] = StationarityTest,
    ) -> None:
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.variable = variable
        self.dm = dm
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = test_threshold if test_threshold is not None else stationarity_test_threshold
        self.test_class = test_class
        self._test_result_cache: Optional[pd.DataFrame] = None
        self._test_filter_cache: Optional[bool] = None

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Execute staged stationarity checks and return the most recent table.

        Returns
        -------
        pd.DataFrame
            Stationarity diagnostics matching :class:`StationarityTest`
            output with an added ``Sample`` column denoting the sample on
            which the reported results were obtained. Values are one of
            ``'In'``, ``'Full'``, ``'Original-In'``, or ``'Original-Full'``.

        Raises
        ------
        ValueError
            If feature construction yields an empty series for the provided
            variable specification.
        """

        if self._test_result_cache is not None:
            return self._test_result_cache

        # Evaluate the provided variable on in-sample then full-sample data.
        if isinstance(self.variable, Mapping):
            result_df, passed = self._evaluate_series_collection(self.variable)
        else:
            result_df, passed = self._evaluate_variable(self.variable, label_prefix='')

        # If still failing and the variable carries an original component,
        # retry using the unfiltered base variable for a broader assessment.
        if not passed and isinstance(self.variable, (RgmVar, CondVar)):
            original_spec = self._resolve_original_variable(self.variable)
            result_df, passed = self._evaluate_variable(original_spec, label_prefix='Original-')

        self._test_result_cache = result_df
        self._test_filter_cache = passed
        return result_df

    @property
    def test_filter(self) -> bool:
        """
        Return True if any staged stationarity check passes.
        """

        if self._test_filter_cache is None:
            _ = self.test_result
        return bool(self._test_filter_cache)

    def _evaluate_variable(
        self,
        variable_spec: Union[str, TSFM, RgmVar, CondVar],
        label_prefix: str,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Run stationarity tests on in-sample followed by full-sample data.

        Parameters
        ----------
        variable_spec : Union[str, TSFM, RgmVar, CondVar]
            Specification passed to :meth:`DataManager.build_features`.
        label_prefix : str
            Prefix for the ``Sample`` label (``''`` or ``'Original-'``).

        Returns
        -------
        Tuple[pandas.DataFrame, bool]
            The latest test result table and whether any stage passed.
        """

        series = self._build_feature_series(variable_spec)
        in_sample_df, in_passed = self._run_stationarity(series, self.dm.in_sample_idx, f"{label_prefix}In")
        if in_passed:
            return in_sample_df, True

        combined_idx = self.dm.in_sample_idx.append(self.dm.out_sample_idx)
        full_df, full_passed = self._run_stationarity(series, combined_idx, f"{label_prefix}Full")
        return full_df, full_passed

    def _evaluate_series_collection(
        self, series_map: Mapping[str, pd.Series]
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Evaluate a mapping of sample labels to series until one is stationary.

        Parameters
        ----------
        series_map : Mapping[str, pd.Series]
            Ordered collection of sample labels to candidate series. Series are
            tested sequentially until a stationary candidate is found.

        Returns
        -------
        Tuple[pandas.DataFrame, bool]
            The latest test result table with an inserted ``Sample`` column and
            whether any candidate series satisfied the configured filter mode.

        Raises
        ------
        ValueError
            If ``series_map`` is empty.
        """

        if not series_map:
            raise ValueError(
                "FullStationarityTest: variable dictionary is empty; unable to "
                "perform stationarity checks."
            )

        latest_result: Optional[pd.DataFrame] = None
        passed_any = False

        for sample_label, sample_series in series_map.items():
            if sample_series is None:
                continue

            # Normalize to Series to support array-like inputs while preserving
            # any explicit index supplied by the caller.
            normalized_series = sample_series if isinstance(sample_series, pd.Series) else pd.Series(sample_series)
            test_instance = self.test_class(
                series=normalized_series,
                alias=self.alias,
                filter_mode=self.filter_mode,
                test_dict=self.test_dict,
                test_threshold=self.thresholds,
                filter_on=self.filter_on,
            )
            result_df = test_instance.test_result.copy()
            insert_at = result_df.columns.get_loc('Passed') if 'Passed' in result_df.columns else len(result_df.columns)
            result_df.insert(insert_at, 'Sample', sample_label)
            latest_result = result_df
            if test_instance.test_filter:
                passed_any = True
                break

        return latest_result if latest_result is not None else pd.DataFrame(), passed_any

    def _build_feature_series(self, variable_spec: Union[str, TSFM, RgmVar, CondVar]) -> pd.Series:
        """
        Construct a single feature series from the provided specification.

        Parameters
        ----------
        variable_spec : Union[str, TSFM, RgmVar, CondVar]
            Feature identifier forwarded to :meth:`DataManager.build_features`.

        Returns
        -------
        pd.Series
            The first column from the constructed feature frame.

        Raises
        ------
        ValueError
            If the constructed DataFrame is empty.
        """

        feature_frame = self.dm.build_features([variable_spec])
        if feature_frame.empty:
            raise ValueError(
                "FullStationarityTest: constructed feature frame is empty; "
                "unable to perform stationarity checks."
            )
        series = feature_frame.iloc[:, 0]
        series.name = series.name or str(variable_spec)
        return series


class TargetStationarityTest(ModelTestBase):
    """
    Assemble target-focused stationarity diagnostics with optional outlier handling.

    Parameters
    ----------
    target : str
        Column name of the target variable within ``dm.internal_data``.
    dm : DataManager
        Data manager supplying target history and sample index boundaries.
    outlier_idx : Optional[List[Any]]
        Index labels to exclude when constructing outlier-adjusted target
        series. Missing labels are ignored.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict', 'moderate'}, default 'moderate'
        Passed through to the underlying :class:`StationarityTest` instances.
    test_dict : Dict[str, callable], optional
        Mapping of test names to functions; defaults to
        ``stationarity_test_dict``.
    test_threshold : Dict[str, Tuple[float, str]], optional
        Test thresholds and directions; defaults to
        ``stationarity_test_threshold``.
    filter_on : bool, default True
        Whether this test participates in filtering.
    test_class : Type[StationarityTest], optional
        Test class to instantiate for each sample evaluation. Defaults to
        :class:`StationarityTest`.

    Examples
    --------
    >>> tst = TargetStationarityTest(target='NII', dm=dm)
    >>> tst.test_result
    >>> tst.test_filter
    """

    category = 'assumption'

    def __init__(
        self,
        target: str,
        dm: DataManager,
        outlier_idx: Optional[List[Any]] = None,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        test_dict: Optional[Dict[str, Callable]] = None,
        test_threshold: Optional[Dict[str, Tuple[float, str]]] = None,
        filter_on: bool = True,
        test_class: Type[StationarityTest] = StationarityTest,
    ) -> None:
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.target = target
        self.dm = dm
        self.outlier_idx = list(outlier_idx) if outlier_idx else []
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = test_threshold if test_threshold is not None else stationarity_test_threshold
        self.test_class = test_class
        self._test_result_cache: Optional[pd.DataFrame] = None
        self._test_filter_cache: Optional[bool] = None

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Execute target stationarity checks across multiple sample definitions.

        Returns
        -------
        pd.DataFrame
            Stationarity diagnostics for the first stationary target sample.

        Raises
        ------
        KeyError
            If ``target`` is not present in :attr:`DataManager.internal_data`.
        """

        if self._test_result_cache is not None:
            return self._test_result_cache

        series_map = self._build_target_series_dictionary()
        full_test = FullStationarityTest(
            variable=series_map,
            dm=self.dm,
            alias=self.alias,
            filter_mode=self.filter_mode,
            test_dict=self.test_dict,
            test_threshold=self.thresholds,
            filter_on=self.filter_on,
            test_class=self.test_class,
        )
        self._test_result_cache = full_test.test_result
        self._test_filter_cache = full_test.test_filter
        return self._test_result_cache

    @property
    def test_filter(self) -> bool:
        """
        Return True if any staged target sample is stationary.
        """

        if self._test_filter_cache is None:
            _ = self.test_result
        return bool(self._test_filter_cache)

    @property
    def filter_mode_desc(self) -> str:
        """
        Describe the active filter mode using the underlying test class mapping.

        Returns
        -------
        str
            Human-readable description of the filter criteria, or an empty
            string when unavailable.
        """

        mode_descs = getattr(self.test_class, 'filter_mode_descs', None)
        if isinstance(mode_descs, dict):
            return mode_descs.get(self.filter_mode, '')
        return ''

    def _build_target_series_dictionary(self) -> Dict[str, pd.Series]:
        """
        Construct target sample variants for stationarity diagnostics.

        Returns
        -------
        Dict[str, pd.Series]
            Mapping of sample labels to target series variants: in-sample,
            full-sample, and their outlier-removed counterparts.

        Raises
        ------
        KeyError
            If the target is not available within ``dm.internal_data``.
        """

        internal_data = self.dm.internal_data
        if self.target not in internal_data.columns:
            raise KeyError(
                f"Target '{self.target}' not found in DataManager.internal_data columns."
            )

        target_series = internal_data[self.target]
        in_sample_series = target_series.loc[self.dm.in_sample_idx]
        out_sample_idx = getattr(self.dm, 'out_sample_idx', None)
        if out_sample_idx is None or len(out_sample_idx) == 0:
            full_sample_series = in_sample_series
        else:
            out_sample_series = target_series.loc[out_sample_idx]
            full_sample_series = pd.concat([in_sample_series, out_sample_series])

        if self.outlier_idx:
            outlier_labels = pd.Index(self.outlier_idx)
            in_no_outliers = in_sample_series.drop(index=outlier_labels, errors='ignore')
            full_no_outliers = full_sample_series.drop(index=outlier_labels, errors='ignore')
        else:
            in_no_outliers = in_sample_series
            full_no_outliers = full_sample_series

        return {
            'In': in_sample_series,
            'Full': full_sample_series,
            'In (No Outliers)': in_no_outliers,
            'Full (No Outliers)': full_no_outliers,
        }

    def _run_stationarity(
        self,
        series: pd.Series,
        sample_idx: pd.Index,
        sample_label: str,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Execute the configured stationarity test for a specific sample.

        Parameters
        ----------
        series : pd.Series
            Full feature series prior to sample slicing.
        sample_idx : pd.Index
            Index labels defining the sample to evaluate (e.g., in-sample).
        sample_label : str
            Label recorded in the ``Sample`` column of the result table.

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            The stationarity result table for the given sample and whether it
            satisfied the configured filter mode.
        """

        aligned_idx = sample_idx[sample_idx.isin(series.index)]
        test_instance = self.test_class(
            series=series.loc[aligned_idx],
            alias=self.alias,
            filter_mode=self.filter_mode,
            test_dict=self.test_dict,
            test_threshold=self.thresholds,
            filter_on=self.filter_on,
        )
        result_df = test_instance.test_result.copy()

        # Insert sample indicator before the Passed column for readability.
        insert_at = result_df.columns.get_loc('Passed')
        result_df.insert(insert_at, 'Sample', sample_label)
        return result_df, test_instance.test_filter

    @staticmethod
    def _resolve_original_variable(variable_spec: Union[RgmVar, CondVar]) -> Union[str, TSFM]:
        """
        Extract the base variable from a regime or conditional specification.

        Parameters
        ----------
        variable_spec : Union[RgmVar, CondVar]
            Regime or conditional feature wrapper.

        Returns
        -------
        Union[str, TSFM]
            The unwrapped variable specification suitable for feature
            construction.
        """
        # Prefer the original transform when present (e.g., a TSFM wrapped by
        # a regime indicator) so fallback testing mirrors the non-regime
        # specification rather than the raw base variable.
        if isinstance(variable_spec, RgmVar) and getattr(variable_spec, "var_feature", None) is not None:
            return variable_spec.var_feature
        return variable_spec.var


# ----------------------------------------------------------------------------
# PvalueTest class
# ----------------------------------------------------------------------------

class CoefTest(ModelTestBase):
    """
    Concrete test for checking coefficient significance of model parameters.

    Parameters
    ----------
    pvalues : pd.Series
        Series of p-values for each coefficient.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        - 'strict'   → require p-value < 0.05 for all.
        - 'moderate' → require p-value < 0.10 for all.
    """
    category = 'performance'

    # Updated descriptions
    filter_mode_descs = {
        'strict':   'Require p-value < 0.05 for all coefficients.',
        'moderate': 'Require p-value < 0.10 for all coefficients.'
    }

    def __init__(
        self,
        pvalues: pd.Series,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.pvalues = pvalues
        # Set α based on mode
        self.alpha = 0.05 if filter_mode == 'strict' else 0.10
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          - 'P-value': the original p-values
          - 'Passed' : True if p-value < α

        Example output structure
        ------------------------
        ┌──────────────┬──────────┬────────┐
        │ Coefficient  │ P-value  │ Passed │
        ├──────────────┼──────────┼────────┤
        │ x1_DF        │ 0.012    │ True   │
        │ x2_GR        │ 0.085    │ True   │
        └──────────────┴──────────┴────────┘
        """
        df = pd.DataFrame({
            'P-value': self.pvalues,
            'Passed':  self.pvalues < self.alpha
        })
        df.index.name = 'Coefficient'
        return df

    @property
    def test_filter(self) -> bool:
        """
        All coefficients must pass (p-value < α) to pass the test.
        """
        return self.test_result['Passed'].all()


# ----------------------------------------------------------------------------
# F-Test for Group Significance 
# ----------------------------------------------------------------------------

class GroupTest(ModelTestBase):
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
        Display name for this test (defaults to 'GroupTest').
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
    
    @property
    def filter_mode_descs(self):
        return {
            'strict':   f"F-test p < {self.alpha} for group {self.vars}.",
            'moderate': f"F-test p < {self.alpha*2} for group {self.vars}."
        }
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Perform joint hypothesis test that all specified coefficients are zero.
        Returns DataFrame with columns ['F-statistic','P-value','Passed'] and index label alias.

        Example output structure
        ------------------------
        ┌───────────────┬──────────────┬──────────┬────────┐
        │ Test          │ F-statistic  │ P-value  │ Passed │
        ├───────────────┼──────────────┼──────────┼────────┤
        │ Joint F Test  │ 5.23         │ 0.003    │ True   │
        └───────────────┴──────────────┴──────────┴────────┘
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
        df.index.name = 'Test'
        return df

    @property
    def test_filter(self) -> bool:
        """
        Return True if the F-test p-value meets threshold for filter_mode.
        """
        return bool(self.test_result['Passed'].iloc[0])


# ----------------------------------------------------------------------------
# SignCheck class
# ----------------------------------------------------------------------------

class SignCheck(ModelTestBase):
    """
    Test whether model coefficients have the expected signs based on Feature exp_sign values.

    Parameters
    ----------
    feature_list : List[Feature]
        List of Feature-like objects that expose ``exp_sign`` and ``name`` attributes.
    coefficients : pd.Series
        Series of model coefficients with variable names as index.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        Currently no difference between modes (may change in future).

    Example
    -------
    >>> from Technic.transform import TSFM
    >>> # Create some TSFM instances with expected signs
    >>> tsfms = [
    ...     TSFM('x1', 'DF', exp_sign=1),   # expect positive
    ...     TSFM('x2', 'GR', exp_sign=-1),  # expect negative
    ... ]
    >>> # Model coefficients (e.g., from fitted regression)
    >>> coeffs = pd.Series({'x1_DF': 0.5, 'x2_GR': -0.3})
    >>> 
    >>> # Create and run sign check
    >>> sign_test = SignCheck(tsfms, coeffs)
    >>> print(sign_test.test_result)
    """
    category = 'performance'

    def __init__(
        self,
        feature_list: List[Feature],
        coefficients: pd.Series,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.feature_list = feature_list
        self.coefficients = coefficients
        self.filter_mode_descs = {
            'strict':   'All coefficients must have expected signs.',
            'moderate': 'All coefficients must have expected signs.'
        }
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Check coefficient signs against expected signs from provided feature objects.

        Returns
        -------
        pd.DataFrame
            Index: feature names (where exp_sign != 0)
            Columns:
              - 'Expected': '+' for positive, '-' for negative expected sign
              - 'Coefficient': actual coefficient value
              - 'Passed': True if sign matches expectation, False otherwise

        Example output structure
        ------------------------
        ┌──────────┬───────────┬─────────────┬────────┐
        │ Variable │ Expected  │ Coefficient │ Passed │
        ├──────────┼───────────┼─────────────┼────────┤
        │ x1_DF    │ +         │ 0.52        │ True   │
        │ x2_GR    │ -         │ -0.31       │ True   │
        └──────────┴───────────┴─────────────┴────────┘
        """
        records = []

        for feature in self.feature_list:
            if not hasattr(feature, 'exp_sign'):
                raise AttributeError(
                    f"Feature '{feature}' does not define required 'exp_sign' attribute."
                )

            # Skip features where exp_sign is 0 (no expectation)
            if feature.exp_sign == 0:
                continue

            tsfm_name = feature.name

            # Check if coefficient exists for this feature
            if tsfm_name not in self.coefficients.index:
                # If coefficient not found, mark as failed
                expected_sign = '+' if feature.exp_sign > 0 else '-'
                records.append({
                    'Expected': expected_sign,
                    'Coefficient': np.nan,
                    'Passed': False
                })
                continue

            coeff_value = self.coefficients[tsfm_name]
            expected_sign = '+' if feature.exp_sign > 0 else '-'

            # Check if signs match
            if feature.exp_sign > 0:
                # Expect positive coefficient
                passed = coeff_value > 0
            else:
                # Expect negative coefficient
                passed = coeff_value < 0
            
            records.append({
                'Expected': expected_sign,
                'Coefficient': coeff_value,
                'Passed': passed
            })

        # Create DataFrame with feature names as index
        tsfm_names = [feature.name for feature in self.feature_list if feature.exp_sign != 0]
        df = pd.DataFrame(records, index=tsfm_names)
        df.index.name = 'Variable'

        return df

    @property
    def test_filter(self) -> bool:
        """
        Return True if all variables with expected signs have coefficients 
        with matching signs.
        """
        if self.test_result.empty:
            return True  # No expectations to check
        return self.test_result['Passed'].all()


# ----------------------------------------------------------------------------
# BaseGrowthTest class
# ----------------------------------------------------------------------------

class BaseGrowthTest(ModelTestBase):
    """
    Estimate a model's base growth rate implied by intercept and periodic dummies.

    The base growth rate measures the model's growth when all non-periodic drivers
    are held neutral (no change), allowing only periodical dummies to contribute.

    Parameters
    ----------
    coeffs : pd.Series
        Series of model coefficients indexed by variable names. Should include
        'const' if an intercept is present, and any monthly or quarterly dummy
        coefficients named like 'M:2', 'M:3', ... or 'Q:2', 'Q:3', ...
        (column naming convention produced by `DumVar`).
    freq : {'M','Q'}
        Frequency of the target variable. 'M' for monthly, 'Q' for quarterly.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        - 'strict': require base growth to be within ±0.10
        - 'moderate': require base growth to be within ±0.15
    filter_on : bool, default False
        Whether this test participates in filtering (default off).

    Example
    -------
    >>> coeffs = pd.Series({
    ...     'const': 0.01,
    ...     'M:2': 0.001,
    ...     'M:3': -0.0005,
    ...     'x1': 0.2
    ... })
    >>> test = BaseGrowthTest(coeffs=coeffs, freq='M')
    >>> test.test_result  # doctest: +SKIP
    
    Notes
    -----
    Base growth calculation:
    - If freq = 'M': base_growth = 12 * const + sum(M:"*")
    - If freq = 'Q': base_growth = 4  * const + sum(Q:"*")
    """
    category = 'performance'

    def __init__(
        self,
        coeffs: pd.Series,
        freq: str,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = False
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        if not isinstance(coeffs, pd.Series):
            raise TypeError("coeffs must be a pandas Series")
        self.coeffs = coeffs
        self.freq = (freq or '').upper()
        if self.freq not in {'M', 'Q'}:
            raise ValueError("freq must be 'M' or 'Q'")
        self._thresholds = {'strict': 0.10, 'moderate': 0.15}

    @property
    def filter_mode_descs(self):
        return {
            'strict':   f"Base growth must be within ±{self._thresholds['strict']:.2f}.",
            'moderate': f"Base growth must be within ±{self._thresholds['moderate']:.2f}."
        }

    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    def _compute_base_growth(self) -> float:
        const = float(self.coeffs.get('const', 0.0))
        if self.freq == 'M':
            scale = 12.0
            dummy_sum = float(self.coeffs[[c for c in self.coeffs.index if isinstance(c, str) and c.startswith('M:')]].sum())
        else:  # 'Q'
            scale = 4.0
            dummy_sum = float(self.coeffs[[c for c in self.coeffs.index if isinstance(c, str) and c.startswith('Q:')]].sum())
        return scale * const + dummy_sum

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Compute base growth and indicate pass/fail against mode-specific bounds.

        Returns
        -------
        pandas.DataFrame
            Single-row table with columns: 'Value', 'Lower', 'Upper', 'Passed'.

        Example output structure
        ------------------------
        ┌─────────────┬────────┬────────┬────────┬────────┐
        │ Metric      │ Value  │ Lower  │ Upper  │ Passed │
        ├─────────────┼────────┼────────┼────────┼────────┤
        │ Base Growth │  0.05  │ -0.15  │  0.15  │  True  │
        └─────────────┴────────┴────────┴────────┴────────┘
        """
        value = self._compute_base_growth()
        thr = self._thresholds[self.filter_mode]
        lower, upper = -thr, thr
        passed = (value >= lower) and (value <= upper)
        df = pd.DataFrame([
            {'Metric': 'Base Growth', 'Value': float(value), 'Lower': float(lower), 'Upper': float(upper), 'Passed': bool(passed)}
        ]).set_index('Metric')
        return df

    @property
    def test_filter(self) -> bool:
        value = self._compute_base_growth()
        thr = self._thresholds[self.filter_mode]
        return (-thr <= value <= thr)


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
    category = 'assumption'

    def __init__(
        self,
        exog: Union[np.ndarray, pd.DataFrame, list],
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.exog = pd.DataFrame(exog)
        self.filter_mode_descs = {
        'strict': 'Threshold = 5',
        'moderate': 'Threshold = 10'
        }
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Compute VIF for each variable.

        Returns
        -------
        pandas.DataFrame
            Index: variable names
            Columns: 'VIF'

        Example output structure
        ------------------------
        ┌──────────┬──────┐
        │ Variable │ VIF  │
        ├──────────┼──────┤
        │ x1       │ 3.4  │
        │ x2       │ 6.1  │
        └──────────┴──────┘
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
    Test for cointegration by checking if X variables are non-stationary and residuals are stationary.

    Parameters
    ----------
    X_vars : pd.DataFrame
        DataFrame containing all X variables that are applicable to stationarity testing.
    resids : pd.Series
        Residual series from the fitted model.
    test_dict : Dict[str, Callable], optional
        Mapping of test names to functions; defaults to stationarity_test_dict.
    test_threshold : Dict[str, Tuple[float, str]], optional
        Test thresholds and directions; defaults to stationarity_test_threshold.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        - 'strict': all tests must pass for residuals and NOT pass for X variables
        - 'moderate': at least half of tests must pass for residuals and NOT pass for each X variable
    filter_on : bool, default True
        Whether this test is active in filtering.

    Example
    -------
    >>> import pandas as pd
    >>> # X variables (should be non-stationary)
    >>> X_data = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]})
    >>> # Model residuals (should be stationary)
    >>> resids = pd.Series([0.1, -0.2, 0.1, -0.1, 0.0])
    >>> 
    >>> # Create cointegration test
    >>> coint_test = CointTest(X_data, resids)
    >>> print(coint_test.test_result)
    """
    category = 'assumption'

    def __init__(
        self,
        X_vars: pd.DataFrame,
        resids: pd.Series,
        test_dict: Optional[Dict[str, Callable]] = None,
        test_threshold: Optional[Dict[str, Tuple[float, str]]] = None,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.X_vars = X_vars
        self.resids = resids
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = test_threshold if test_threshold is not None else stationarity_test_threshold
        self.filter_mode_descs = {
            'strict':   'All X variables must be non-stationary and residuals must be stationary.',
            'moderate': 'At least half of tests must show X variables are non-stationary and residuals are stationary.'
        }
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Test stationarity of X variables and residuals.

        Returns
        -------
        pd.DataFrame
            Index: Variable names (X variables + 'Residuals')
            Columns:
              - 'Type': 'X Variable' or 'Residuals'
              - 'Expected': 'Non-stationary' for X, 'Stationary' for residuals
              - Individual test columns (e.g., 'ADF', 'PP'): True if test passed expectation
              - 'Result': 'Non-stationary'/'Stationary' based on filter_mode aggregation
              - 'Passed': True if meets expectation, False otherwise

        Example output structure
        ------------------------
        ┌────────────┬──────────────┬───────────────────┬──────┬─────┬──────────────┬────────┐
        │ Variable   │ Type         │ Expected          │ ADF  │ PP  │ Result       │ Passed │
        ├────────────┼──────────────┼───────────────────┼──────┼─────┼──────────────┼────────┤
        │ x1         │ X Variable   │ Non-stationary    │ True │ …   │ Non-stationary│ True  │
        │ Residuals  │ Residuals    │ Stationary        │ True │ …   │ Stationary    │ True  │
        └────────────┴──────────────┴───────────────────┴──────┴─────┴──────────────┴────────┘
        """
        records = []
        test_names = list(self.test_dict.keys())
        
        # Test each X variable (expect non-stationary)
        for col in self.X_vars.columns:
            # Ensure numeric dtype to prevent unit-root tests from failing when
            # a column mixes ints and floats.
            series = pd.to_numeric(self.X_vars[col], errors='coerce').dropna().astype(float)
            if len(series) < 10:  # Skip series that are too short
                continue
                
            record = {
                'Type': 'X Variable',
                'Expected': 'Non-stationary'
            }
            
            # Individual test results
            test_results = {}
            for test_name, test_func in self.test_dict.items():
                if test_name not in self.thresholds:
                    test_results[test_name] = False
                    continue
                    
                try:
                    stat, pvalue = test_func(series)
                    alpha, direction = self.thresholds[test_name]
                    
                    # For stationarity tests: 
                    # - Tests like ADF/PP have null hypothesis of unit root (non-stationary)
                    #   So p > alpha means non-stationary (null not rejected)
                    # - Tests like KPSS have null hypothesis of stationarity
                    #   So p > alpha means stationary (null not rejected)
                    
                    if direction == '<':
                        # Null: non-stationary, reject null if p < alpha (stationary)
                        test_indicates_stationary = pvalue < alpha
                    else:
                        # Null: stationary, reject null if p < alpha (non-stationary)
                        test_indicates_stationary = pvalue > alpha
                    
                    # For X variables, we expect non-stationary, so pass if test indicates non-stationary
                    test_results[test_name] = not test_indicates_stationary
                    
                except Exception:
                    test_results[test_name] = False
                    continue
            
            # Add individual test results to record
            for test_name in test_names:
                record[test_name] = test_results.get(test_name, False)
            
            # Determine overall result based on filter_mode
            passed_count = sum(test_results.values())
            total_count = len([v for v in test_results.values() if v is not None])
            
            if self.filter_mode == 'strict':
                is_nonstationary = passed_count == total_count and total_count > 0
            else:  # moderate
                is_nonstationary = passed_count > (total_count / 2) if total_count > 0 else False
                
            result_str = 'Non-stationary' if is_nonstationary else 'Stationary'
            expected_result = True if is_nonstationary else False  # Expect non-stationary for X
            
            record['Result'] = result_str
            record['Passed'] = expected_result
            records.append(record)
        
        # Test residuals (expect stationary)
        # Standardize residual dtype for unit-root diagnostics.
        resid_series = pd.to_numeric(self.resids, errors='coerce').dropna().astype(float)
        if len(resid_series) >= 10:
            record = {
                'Type': 'Residuals',
                'Expected': 'Stationary'
            }
            
            # Individual test results
            test_results = {}
            for test_name, test_func in self.test_dict.items():
                if test_name not in self.thresholds:
                    test_results[test_name] = False
                    continue
                    
                try:
                    stat, pvalue = test_func(resid_series)
                    alpha, direction = self.thresholds[test_name]
                    
                    if direction == '<':
                        # Null: non-stationary, reject null if p < alpha (stationary)
                        test_indicates_stationary = pvalue < alpha
                    else:
                        # Null: stationary, reject null if p < alpha (non-stationary)
                        test_indicates_stationary = pvalue > alpha
                    
                    # For residuals, we expect stationary, so pass if test indicates stationary
                    test_results[test_name] = test_indicates_stationary
                    
                except Exception:
                    test_results[test_name] = False
                    continue
            
            # Add individual test results to record
            for test_name in test_names:
                record[test_name] = test_results.get(test_name, False)
            
            # Determine overall result based on filter_mode
            passed_count = sum(test_results.values())
            total_count = len([v for v in test_results.values() if v is not None])
            
            if self.filter_mode == 'strict':
                is_stationary = passed_count == total_count and total_count > 0
            else:  # moderate
                is_stationary = passed_count > (total_count / 2) if total_count > 0 else False
                
            result_str = 'Stationary' if is_stationary else 'Non-stationary'
            expected_result = True if is_stationary else False  # Expect stationary for residuals
            
            record['Result'] = result_str
            record['Passed'] = expected_result
            records.append(record)
        
        # Create index
        var_names = list(self.X_vars.columns) + ['Residuals']
        df = pd.DataFrame(records, index=var_names[:len(records)])
        df.index.name = 'Variable'
        
        return df

    @property
    def test_filter(self) -> bool:
        """
        Return True if all X variables are non-stationary AND residuals are stationary.
        
        The filter_mode logic is already incorporated in the test_result calculation,
        so we just need to check if all variables passed their expectations.
        """
        results = self.test_result
        if results.empty:
            return False
            
        # All variables must pass their expectations (logic already handled in test_result)
        return results['Passed'].all()

class MultiFullStationarityTest(ModelTestBase):
    """
    Run staged stationarity tests across all feature specifications in a model option.

    The class builds a :class:`FullStationarityTest` for each string or Feature
    spec provided, excluding dummy specifications. It consolidates the
    sample-aware diagnostics into a single DataFrame to highlight which sample
    (in-sample, full-sample, or original-variable reruns) determined the
    outcome.

    Parameters
    ----------
    specs : List[Union[str, Feature]]
        Feature specifications, as accepted by :class:`ModelBase`, flattened
        automatically to individual entries. Dummy specs are ignored.
    dm : DataManager
        Data manager used to build features and provide sample indices.
    test_dict : Dict[str, callable], optional
        Mapping of test names to functions; defaults to ``stationarity_test_dict``.
    test_threshold : Dict[str, Tuple[float, str]], optional
        Thresholds and inequality directions for each test; defaults to
        ``stationarity_test_threshold``.
    alias : str, optional
        Display name for this test suite (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        - 'strict': all individual stationarity diagnostics must pass per feature
        - 'moderate': at least half of the diagnostics must pass per feature
    filter_on : bool, default True
        Whether this test participates in filtering decisions.
    full_test_class : Type[FullStationarityTest], optional
        Class used to instantiate staged stationarity checks; defaults to
        :class:`FullStationarityTest`.
    stationarity_test_class : Type[StationarityTest], optional
        Underlying stationarity implementation passed through to each
        :class:`FullStationarityTest` instance.

    Raises
    ------
    ValueError
        If ``specs`` is empty.
    TypeError
        If a spec is neither a string nor a Feature instance (excluding
        :class:`DumVar`).

    Examples
    --------
    >>> multi_full = MultiFullStationarityTest(specs=['GDP', TSFM('UNRATE', diff)], dm=dm)
    >>> multi_full.test_result
    >>> multi_full.test_filter
    """

    category = 'assumption'

    def __init__(
        self,
        specs: List[Union[str, Feature]],
        dm: DataManager,
        test_dict: Optional[Dict[str, Callable]] = None,
        test_threshold: Optional[Dict[str, Tuple[float, str]]] = None,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True,
        full_test_class: Type[FullStationarityTest] = FullStationarityTest,
        stationarity_test_class: Type[StationarityTest] = StationarityTest,
    ) -> None:
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        if not specs:
            raise ValueError("specs must contain at least one feature specification.")
        self.specs = specs
        self.dm = dm
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = test_threshold if test_threshold is not None else stationarity_test_threshold
        self.filter_mode_descs = {
            'strict':   'All individual tests must pass for each variable.',
            'moderate': 'At least half of individual tests must pass for each variable.'
        }
        self.full_test_class = full_test_class
        self.stationarity_test_class = stationarity_test_class

        self._individual_tests: Dict[str, FullStationarityTest] = {}
        for spec in self._flatten_specs(self.specs):
            if isinstance(spec, DumVar):
                # Dummy variables are intentionally excluded from stationarity checks.
                continue
            if isinstance(spec, (str, Feature)):
                label = self._spec_label(spec)
                self._individual_tests[label] = self.full_test_class(
                    variable=spec,
                    dm=self.dm,
                    alias=self.alias,
                    filter_mode=self.filter_mode,
                    test_dict=self.test_dict,
                    test_threshold=self.thresholds,
                    filter_on=self.filter_on,
                    test_class=self.stationarity_test_class,
                )
            else:
                raise TypeError(
                    f"Unsupported spec type {type(spec)} for MultiFullStationarityTest."
                )

    @property
    def filter_mode_desc(self) -> str:
        """Human-readable description of the configured filter mode."""

        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run staged stationarity tests on all valid specs and consolidate results.

        Returns
        -------
        pd.DataFrame
            Index: Feature names
            Columns: For each test in ``test_dict``
                - ``{test_name}_Statistic``
                - ``{test_name}_P-value``
                - ``{test_name}_Passed``
            Plus:
                - ``Sample`` indicating which sample produced the recorded result
                - ``Passed`` indicating the overall outcome per feature
        """

        if not self._individual_tests:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        test_names = list(self.test_dict.keys())

        for var_name, stat_test in self._individual_tests.items():
            record: Dict[str, Any] = {'Variable': var_name}

            individual_results = stat_test.test_result
            sample_value = individual_results['Sample'].iloc[0] if 'Sample' in individual_results.columns else np.nan
            record['Sample'] = sample_value

            for test_name in test_names:
                if test_name in individual_results.index:
                    record[f'{test_name}_Statistic'] = individual_results.loc[test_name, 'Statistic']
                    record[f'{test_name}_P-value'] = individual_results.loc[test_name, 'P-value']
                    record[f'{test_name}_Passed'] = individual_results.loc[test_name, 'Passed']
                else:
                    record[f'{test_name}_Statistic'] = np.nan
                    record[f'{test_name}_P-value'] = np.nan
                    record[f'{test_name}_Passed'] = False

            record['Passed'] = stat_test.test_filter
            records.append(record)

        return pd.DataFrame(records).set_index('Variable')

    @property
    def test_filter(self) -> bool:
        """Return True if all staged stationarity tests pass for every spec."""

        if not self._individual_tests:
            return True

        return all(test.test_filter for test in self._individual_tests.values())

    @staticmethod
    def _flatten_specs(items: Any) -> List[Any]:
        """Flatten nested spec containers to a single list."""

        flattened: List[Any] = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flattened.extend(MultiFullStationarityTest._flatten_specs(item))
            else:
                flattened.append(item)
        return flattened

    @staticmethod
    def _spec_label(spec: Union[str, Feature]) -> str:
        """Derive a user-friendly label for a feature specification."""

        if isinstance(spec, str):
            return spec
        if getattr(spec, 'alias', None):
            return str(spec.alias)
        if getattr(spec, 'name', None):
            return str(spec.name)
        return str(spec)


class MultiStationarityTest(ModelTestBase):
    """
    Conduct stationarity tests on multiple variables (DataFrame columns) simultaneously.
    
    This class creates individual StationarityTest instances for each column in the input
    DataFrame and consolidates the results into a comprehensive test result.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing all series to test for stationarity.
    test_dict : Dict[str, callable], optional
        Mapping of test names to functions; defaults to stationarity_test_dict.
    test_threshold : Dict[str, Tuple[float, str]], optional
        Test thresholds and directions; defaults to stationarity_test_threshold.
    alias : str, optional
        Display name for this test (defaults to class name).
    filter_mode : {'strict','moderate'}, default 'moderate'
        - 'strict': all individual tests must pass for each variable
        - 'moderate': at least half of individual tests must pass for each variable
    filter_on : bool, default True
        Whether this test is active in filtering.
        
    Examples
    --------
    >>> import pandas as pd
    >>> # Create test data
    >>> df = pd.DataFrame({
    ...     'var1': [1, 2, 3, 4, 5],
    ...     'var2': [2, 4, 6, 8, 10],
    ...     'var3': [0.1, -0.2, 0.1, -0.1, 0.0]
    ... })
    >>> 
    >>> # Create multi-variable stationarity test
    >>> multi_test = MultiStationarityTest(df, filter_mode='moderate')
    >>> print(multi_test.test_result)
    >>> print(f"Overall passed: {multi_test.test_filter}")
    """
    category = 'assumption'

    def __init__(
        self,
        dataframe: pd.DataFrame,
        test_dict: Optional[Dict[str, Callable]] = None,
        test_threshold: Optional[Dict[str, Tuple[float, str]]] = None,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate',
        filter_on: bool = True
    ):
        super().__init__(alias=alias, filter_mode=filter_mode, filter_on=filter_on)
        self.dataframe = dataframe
        self.test_dict = test_dict if test_dict is not None else stationarity_test_dict
        self.thresholds = test_threshold if test_threshold is not None else stationarity_test_threshold
        self.filter_mode_descs = {
            'strict':   'All individual tests must pass for each variable.',
            'moderate': 'At least half of individual tests must pass for each variable.'
        }
        
        # Create individual StationarityTest instances for each column
        self._individual_tests = {}
        for col in self.dataframe.columns:
            if len(self.dataframe[col].dropna()) >= 10:  # Skip columns with insufficient data
                self._individual_tests[col] = StationarityTest(
                    series=self.dataframe[col],
                    test_dict=self.test_dict,
                    test_threshold=self.thresholds,
                    filter_mode=self.filter_mode,
                    filter_on=True
                )
    
    @property
    def filter_mode_desc(self):
        return self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> pd.DataFrame:
        """
        Run stationarity tests on all variables and return consolidated DataFrame.

        Returns
        -------
        pd.DataFrame
            Index: Variable names
            Columns: For each test in test_dict:
                - '{test_name}_Statistic': Test statistic value
                - '{test_name}_P-value': P-value from test
                - '{test_name}_Passed': Boolean indicating if test passed
            Plus final 'Passed' column indicating overall result for each variable
            
        Example output structure:
        ┌──────┬─────────────────┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬──────────────────┬────────┐
        │ Var  │ ADF_Statistic   │ ADF_P-value     │ ADF_Passed       │ PP_Statistic    │ PP_P-value      │ PP_Passed        │ Passed │
        ├──────┼─────────────────┼─────────────────┼──────────────────┼─────────────────┼─────────────────┼──────────────────┼────────┤
        │ var1 │ -2.1            │ 0.03            │ True             │ -1.8            │ 0.07            │ False            │ True   │
        │ var2 │ -1.5            │ 0.12            │ False            │ -1.2            │ 0.15            │ False            │ False  │
        └──────┴─────────────────┴─────────────────┴──────────────────┴─────────────────┴─────────────────┴──────────────────┴────────┘
        """
        if not self._individual_tests:
            return pd.DataFrame()
        
        records = []
        test_names = list(self.test_dict.keys())
        
        for var_name, stat_test in self._individual_tests.items():
            record = {'Variable': var_name}
            
            # Get individual test results
            individual_results = stat_test.test_result
            
            # Add columns for each test
            for test_name in test_names:
                if test_name in individual_results.index:
                    record[f'{test_name}_Statistic'] = individual_results.loc[test_name, 'Statistic']
                    record[f'{test_name}_P-value'] = individual_results.loc[test_name, 'P-value']
                    record[f'{test_name}_Passed'] = individual_results.loc[test_name, 'Passed']
                else:
                    record[f'{test_name}_Statistic'] = np.nan
                    record[f'{test_name}_P-value'] = np.nan
                    record[f'{test_name}_Passed'] = False
            
            # Determine overall pass/fail for this variable
            record['Passed'] = stat_test.test_filter
            
            records.append(record)
        
        df = pd.DataFrame(records).set_index('Variable')
        return df

    @property
    def test_filter(self) -> bool:
        """
        Return True if all variables pass their stationarity tests.
        
        The individual filter_mode logic is already handled by each StationarityTest instance,
        so we just need to check if all variables passed.
        """
        if not self._individual_tests:
            return True  # No tests to run
            
        # All variables must pass their individual stationarity tests
        return all(test.test_filter for test in self._individual_tests.values())