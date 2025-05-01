# TECHNIC/test.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, List
import pandas as pd

from .model import ModelBase
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
    """
    def __init__(
        self,
        model: Optional[ModelBase] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        test_dict: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.X = X
        self.y = y
        self.test_dict = test_dict or {}

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

class TestSetBase:
    """
    Base class for collecting and assessing multiple ModelTestBase instances.
    """

    def __init__(self, test_kwargs: Dict[Type[ModelTestBase], dict]):
        """
        Initialize TestSetBase with a mapping of ModelTestBase subclasses to init kwargs.
        :param test_kwargs: dict mapping each ModelTestBase subclass to its init arguments
        """
        # Instantiate each test with its corresponding kwargs
        self.tests: List[ModelTestBase] = [
            test_cls(**kwargs) for test_cls, kwargs in test_kwargs.items()
        ]

    @property
    def test_results(self) -> Dict[str, Any]:
        """
        Gather results from each test in a dict keyed by test class name.
        """
        return {type(test).__name__: test.test_result for test in self.tests}

    def search_pass(self) -> bool:
        """
        Quickly determine if all tests pass by returning False on first failure.
        """
        for test in self.tests:
            if not test.test_filter:
                return False
        return True


# Dictionary of normality diagnostic tests
normality_test_dict: Dict[str, callable] = {
    'Jarque_Bera': jarque_bera,
    'Shapiro': shapiro
}

class NormalityTest(ModelTestBase):
    """
    Concrete ModelTestBase implementation for normality diagnostics on residuals.

    Provides access to model residuals and performs specified normality tests.
    """
    def __init__(
        self,
        model: Optional[ModelBase] = None,
        test_dict: Dict[str, Any] = normality_test_dict,
        resid: Optional[pd.Series] = None,
        alpha : Union[float, Dict[str, float]] = 0.05
    ):
        super().__init__(model=model, X=None, y=None, test_dict=test_dict)
        self.alpha  = alpha
        # Use provided residuals if given, otherwise extract from model
        if resid is not None:
            self.resid = resid
        elif self.model is not None and hasattr(self.model, 'resid'):
            self.resid = getattr(self.model, 'resid')
        else:
            self.resid = None

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
    'adf': adfuller
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
    def __init__(
        self,
        series: Optional[pd.Series] = None,
        alpha : Union[float, Dict[str, float]] = 0.05,
        test_dict: Dict[str, Any] = stationarity_test_dict,
        model: Optional[ModelBase] = None
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
            for lag in (0, 1):
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
