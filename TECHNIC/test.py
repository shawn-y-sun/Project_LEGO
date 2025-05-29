# =============================================================================
# module: test.py
# Purpose: Model testing framework with base and concrete test implementations
# Dependencies: pandas, statsmodels, scipy, abc, typing
# =============================================================================
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, List, Tuple
import pandas as pd
import numpy as np

from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

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
        filter_mode: str = 'moderate'
    ):
        if filter_mode not in self._allowed_modes:
            raise ValueError(f"filter_mode must be one of {self._allowed_modes}")
        self.alias = alias or ''
        self.filter_mode = filter_mode
        self.filter_on = True

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
        filter_mode: str = 'moderate'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
        self.actual = actual
        self.predicted = predicted
        self.n = len(actual)
        self.p = n_features
        # this is only for reporting: do not include in filter_pass
        self.filter_on = False

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
        filter_mode: str = 'moderate'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
        self.errors = actual - predicted
        # this is only for reporting: do not include in filter_pass
        self.filter_on = False

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
        filter_mode: str = 'strict'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
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
# NormalityTest class
# ----------------------------------------------------------------------------
    
# Dictionary of normality diagnostic tests
normality_test_dict: Dict[str, callable] = {
    'Jarque_Bera': jarque_bera,
    'Shapiro': shapiro
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
        filter_mode: str = 'strict'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
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
        │ Sh   │   …      │   …     │  True  │
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

# Dictionary of stationarity diagnostic tests
stationarity_test_dict: Dict[str, Any] = {
    'ADF': lambda s: adfuller(s.dropna(), autolag='AIC'),
    'PP': lambda s: (  # Phillips–Perron unit-root test
        (lambda pp: (pp.stat, pp.pvalue))(PhillipsPerron(s.dropna()))
    )
}

class StationarityTest(ModelTestBase):
    """
    Concrete ModelTestBase implementation for stationarity testing using ADF.

    Parameters
    ----------
    series : Optional[pd.Series]
        Time series to test for stationarity.
    alpha  : float or Dict[str, float]
        Significance level(s) for test(s); default is 0.05.
    test_dict : Dict[str, callable]
        Mapping of test names to functions; default is {'adf': adfuller}.
    model : Optional[ModelBase]
        Optional, a ModelBase instance whose residuals or target can be tested.
    """
    category = 'assumption'

    def __init__(
        self,
        series: pd.Series,
        alpha: Union[float, Dict[str, float]] = 0.05,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
        self.series = series
        self.alpha = alpha
        self.test_dict = stationarity_test_dict
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
        rows = []
        for name, fn in self.test_dict.items():
            stat, pvalue = fn(self.series)[0:2]
            level = self.alpha[name] if isinstance(self.alpha, dict) else self.alpha
            passed = pvalue < level
            rows.append({
                'Test': name,
                'Statistic': stat,
                'P-value': pvalue,
                'Passed': passed
            })
        return pd.DataFrame(rows).set_index('Test')

    @property
    def test_filter(self) -> bool:
        """
        Return True if stationarity tests meet the threshold based on filter_mode:
        - strict:  all tests must pass
        - moderate: at least half of tests must pass
        """
        df = self.test_result
        passed = df['Passed']
        if self.filter_mode == 'strict':
            return passed.all()
        # moderate
        return passed.sum() >= len(passed) / 2
    

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
# SignificanceTest class
# ----------------------------------------------------------------------------

class SignificanceTest(ModelTestBase):
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

    def __init__(
        self,
        pvalues: pd.Series,
        alpha: float = 0.05,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
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
    category = 'group'

    def __init__(
        self,
        model_result: Any,
        vars: list,
        alpha: float = 0.05,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate'
    ):
        super().__init__(alias=alias, filter_mode=filter_mode)
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
