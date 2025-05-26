# TECHNIC/test.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, List, Tuple
import pandas as pd

from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller


class ModelTestBase(ABC):
    """
    Abstract base class for model testing frameworks.

    Parameters
    ----------
    model : Optional[Any]
        Model instance for context.
    X : Optional[pd.DataFrame]
        In-sample features.
    y : Optional[pd.Series]
        In-sample target.
    test_dict : Optional[Dict[str, Any]]
        Dictionary mapping test names to functions or parameters.
    alias : Optional[str]
        Custom display name for this test instance.
    filter_mode : str, default 'moderate'
        How to evaluate passed results: 'strict' or 'moderate'.
    """
    category: str = 'base'
    _allowed_modes = {'strict', 'moderate'}  # Allowed evaluation modes

    def __init__(
        self,
        model: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        test_dict: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None,
        filter_mode: str = 'moderate'
    ):
        # Basic context and test configuration
        self.model = model
        self.X = X
        self.y = y
        self.test_dict = test_dict or {}
        self.alias = alias or ''
        # Validate filter_mode
        if filter_mode not in self._allowed_modes:
            raise ValueError(f"filter_mode must be one of {self._allowed_modes}")
        self.filter_mode = filter_mode
        # Status flag to enable/disable this test
        self.status = 'on'

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
        Execute the test(s) and return a mapping of results for each metric.

        Returns
        -------
        Dict[str, Any]
            Keys are test labels; values include metric results and pass/fail.
        """
        pass

    @property
    @abstractmethod
    def test_filter(self) -> bool:
        """
        Evaluate the overall test pass/fail based on filter_mode.

        Implementations should consider filter_mode ('strict' or 'moderate')
        and use results from test_result.
        """
        pass

class TestSet:
    """
    Aggregator for ModelTestBase instances, with easy filtering and pass/fail logic.
    """

    def __init__(
        self,
        test_kwargs: Dict[Type[ModelTestBase], dict]
    ):
        """
        :param test_kwargs: Mapping of ModelTestBase subclasses to their init kwargs.
        """
        self.tests: List[ModelTestBase] = [test_cls(**kwargs) for test_cls, kwargs in test_kwargs.items()]

    @property
    def assumption_tests(self) -> List[ModelTestBase]:
        """
        Return only the tests categorized as 'assumption'.
        """
        return [t for t in self.tests if getattr(t, 'category', None) == 'assumption']
    
    @property
    def assumption_results(self) -> Dict[str, Any]:
        """
        Return the test_result dict for each assumption test.
        """
        return {t.name: t.test_result for t in self.assumption_tests}

    @property
    def performance_tests(self) -> List[ModelTestBase]:
        """
        Return only the tests categorized as 'performance'.
        """
        return [t for t in self.tests if getattr(t, 'category', None) == 'performance']

    def search_pass(
        self,
        test_list: Optional[List[str]] = None,
        fast_test: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate selected tests and return overall pass flag and list of failed test names.

        :param test_list: List of test.name to include (in that order). If None, use all tests in original order.
        :param fast_test: If True, stops at first failure (fail-fast). If False, evaluates all tests (collecting all failures).
        :return: (passed: bool, failed_tests: List[str])
        """
        # Determine execution sequence
        if test_list:
            name_map = {t.name: t for t in self.tests}
            seq = [name_map[n] for n in test_list if n in name_map]
        else:
            seq = list(self.tests)

        failed: List[str] = []
        for t in seq:
            if not t.test_filter:
                failed.append(t.name)
                if fast_test:
                    # fail-fast: stop at first failure
                    return False, failed
        # if not fast_test, or no failures occurred during fast test
        return (len(failed) == 0), failed


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
        filter_mode: str = 'moderate'
    ):
        super().__init__(filter_mode=filter_mode)
        self.series = series
        self.alpha = alpha
        self.test_dict = normality_test_dict
        # Custom descriptions for strict vs moderate
        self.filter_mode_descs = {
            'strict': 'All normality tests must pass.',
            'moderate': 'At least half of normality tests must pass.'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> Dict[str, Dict[str, Any]]:
        # Run each normality test and record results
        results = {}
        for name, fn in self.test_dict.items():
            stat, pvalue = fn(self.series)[0:2]
            level = self.alpha[name] if isinstance(self.alpha, dict) else self.alpha
            passed = pvalue > level
            results[name] = {'statistic': stat, 'pvalue': pvalue, 'passed': passed}
        return results

    @property
    def test_filter(self) -> bool:
        # Base logic: strict requires all pass, moderate requires half
        passed_flags = [info['passed'] for info in self.test_result.values()]
        if self.filter_mode == 'strict':
            return all(passed_flags)
        return sum(passed_flags) >= (len(passed_flags) / 2)
    

# Dictionary of stationarity diagnostic tests
stationarity_test_dict: Dict[str, callable] = {
    'ADF': lambda series: adfuller(series.dropna(), autolag='AIC')
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
        regression: str = 'c',
        autolag: str = 'AIC',
        filter_mode: str = 'moderate'
    ):
        super().__init__(filter_mode=filter_mode)
        self.series = series
        self.regression = regression
        self.autolag = autolag
        self.test_dict = stationarity_test_dict
        # Single-test descriptions
        self.filter_mode_descs = {
            'strict': 'Stationarity must pass.',
            'moderate': 'Stationarity must pass.'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]

    @property
    def test_result(self) -> Dict[str, Dict[str, Any]]:
        stat, pvalue = adfuller(self.series, regression=self.regression, autolag=self.autolag)[0:2]
        passed = pvalue < 0.05
        return {'ADF': {'statistic': stat, 'pvalue': pvalue, 'passed': passed}}

    @property
    def test_filter(self) -> bool:
        # Single test; same threshold for both modes
        return list(self.test_result.values())[0]['passed']
    

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


class SignificanceTest(ModelTestBase):
    """
    Concrete test for checking coefficient significance of model parameters.

    Evaluates p-values against alpha thresholds.
    """
    category = 'performance'

    def __init__(
        self,
        pvalues: pd.Series,
        filter_mode: str = 'strict'
    ):
        super().__init__(filter_mode=filter_mode)
        self.pvalues = pvalues
        # Custom descriptions and alpha per mode
        self.filter_mode_descs = {
            'strict': 'Require p-value < 0.05 for all coefficients.',
            'moderate': 'Require p-value < 0.1 for all coefficients.'
        }
        self.filter_mode_desc = self.filter_mode_descs[self.filter_mode]
        self.alpha = 0.05 if self.filter_mode == 'strict' else 0.1

    @property
    def test_result(self) -> Dict[str, Dict[str, Any]]:
        # Map each variable's p-value to pass/fail
        results = {}
        for var, p in self.pvalues.items():
            results[var] = {'pvalue': float(p), 'passed': p < self.alpha}
        return results

    @property
    def test_filter(self) -> bool:
        # All coefficients must meet their threshold
        return all(info['passed'] for info in self.test_result.values())