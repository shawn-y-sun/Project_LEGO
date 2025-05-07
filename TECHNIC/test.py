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
    model : Optional[ModelBase]
        Fitted model instance to test.
    X : Optional[pd.DataFrame]
        Feature matrix for testing.
    y : Optional[pd.Series]
        Target vector for testing.
    test_dict : Optional[Dict[str, Any]]
        Dictionary of test configuration parameters.
    alias : Optional[str]
        Custom display name for this test instance.
    """
    # Category of the test: 'assumption', 'performance', or 'base'
    category: str = 'base'

    def __init__(
        self,
        model: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        test_dict: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None
    ):
        self.model = model
        self.X = X
        self.y = y
        self.test_dict = test_dict or {}
        self.alias = alias or ''

    @property
    def name(self) -> str:
        """
        Display name for the test: alias if provided, else class name.
        """
        return self.alias if self.alias else type(self).__name__

    @property
    @abstractmethod
    def test_result(self) -> Any:
        """
        Returns the computed test result (outcome) of the testing.
        """
        ...

    @property
    @abstractmethod
    def test_filter(self) -> bool:
        """
        Returns True if test conditions are met, False otherwise.
        """
        ...

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
    Concrete ModelTestBase implementation for normality diagnostics on a series of data.

    Performs specified normality tests on the provided series.
    """
    category = 'assumption'

    def __init__(
        self,
        series: pd.Series,
        alpha: Union[float, Dict[str, float]] = 0.05,
        alias: Optional[str] = None
    ):
        super().__init__(
            model=None,
            X=None,
            y=None,
            test_dict=normality_test_dict,
            alias=alias
        )
        self.series = series
        self.alpha = alpha
    
    @property
    def test_result(self) -> Dict[str, Dict[str, Any]]:
        """Run each normality test, returning stat, pvalue, and pass flag."""
        results: Dict[str, Dict[str, Any]] = {}
        for test_name, fn in self.test_dict.items():
            output = fn(self.series)
            stat = output[0]
            pvalue = output[1]
            # allow alpha per test if dict
            level = self.alpha[test_name] if isinstance(self.alpha, dict) else self.alpha
            passed = pvalue > level
            results[test_name] = {
                'statistic': stat,
                'pvalue': pvalue,
                'passed': passed
            }
        return results


    @property
    def test_filter(self) -> bool:
        """
        Returns True if all tests in test_result are passed, False otherwise.
        """
        results = self.test_result
        if not results:
            return False
        return all(
            test_info.get('passed', False) for test_info in results.values()
        )
    

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
        series: Optional[pd.Series] = None,
        alpha : Union[float, Dict[str, float]] = 0.05,
        test_dict: Dict[str, Any] = stationarity_test_dict,
        model: Optional[Any] = None
    ):
        super().__init__(model=model, X=None, y=None, test_dict=test_dict)
        self.alpha  = alpha
        # Assign series: explicit or from model residuals/target
        if series is not None:
            self.series = series
        elif self.model is not None and hasattr(self.model, 'resid'):
            self.series = getattr(self.model, 'resid')
        else:
            self.series = None

    @property
    def test_result(self) -> Dict[str, Dict[str, Any]]:
        """Run each stationarity test, returning stat, pvalue, and pass flag."""
        results: Dict[str, Dict[str, Any]] = {}
        for test_name, fn in self.test_dict.items():
            output = fn(self.series)
            stat = output[0]
            pvalue = output[1]
            level = self.alpha[test_name] if isinstance(self.alpha, dict) else self.alpha
            passed = pvalue < level
            results[test_name] = {
                'statistic': stat,
                'pvalue': pvalue,
                'passed': passed
            }
        return results

    @property
    def test_filter(self) -> bool:
        """
        Returns True if all stationarity tests passed, False otherwise.
        """
        results = self.test_result
        if not results:
            return False
        return all(info.get('passed', False) for info in results.values())

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
    Concrete ModelTestBase implementation for checking that each p-value
    for a set of variables meets a significance threshold.
    """
    category = 'performance'

    def __init__(self, pvalues: pd.Series, alpha: float = 0.05):
        super().__init__(model=None, X=None, y=None, test_dict=None)
        self.pvalues = pvalues
        self.alpha = alpha

    @property
    def test_result(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for var, p in self.pvalues.items():
            passed = (p < self.alpha)
            results[var] = {
                'pvalue': float(p),
                'passed': bool(passed)
            }
        return results

    @property
    def test_filter(self) -> bool:
        return all(info.get('passed', False) for info in self.test_result.values())