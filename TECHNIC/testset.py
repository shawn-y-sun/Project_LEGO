from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .test import *
from .cm import CM


class TestSetBase(ABC):
    """
    Abstract base class for collecting and assessing multiple ModelTestBase instances.
    """
    def __init__(self, tests: List[ModelTestBase]):
        """
        :param tests: List of ModelTestBase subclasses to run.
        """
        self.tests = tests

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


# Default list of test instances for PPNR OLS
PPNR_OLS_tests: List[ModelTestBase] = [
    NormalityTest(),
    StationarityTest()
]

# Default thresholds for PPNR OLS tests
PPNR_OLS_thresholds: Dict[str, float] = {
    'NormalityTest': 0.05,
    'StationarityTest': 0.05
}



class PPNR_OLS_TestSet(TestSetBase):
    """
    Test set for PPNR OLS models using supplied model tests and thresholds.

    Parameters
    ----------
    tests : List[ModelTestBase]
        List of pre-configured test instances to evaluate.
    thresholds : Dict[str, float]
        Mapping of test class names to significance thresholds.
    """
    def __init__(
        self,
        tests: List[ModelTestBase] = DEFAULT_TESTS,
        thresholds: Dict[str, float] = DEFAULT_THRESHOLDS
    ):
        self,
        tests: List[ModelTestBase],
        thresholds: Dict[str, float] = DEFAULT_THRESHOLDS
    ):
        """
        :param tests: List of ModelTestBase instances to run.
        :param thresholds: Dict mapping test class names to threshold values.
        """
        # Apply thresholds to each test if attribute exists
        for test in tests:
            test_name = type(test).__name__
            if hasattr(test, 'threshold') and test_name in thresholds:
                setattr(test, 'threshold', thresholds[test_name])
        super().__init__(tests)
        self.thresholds = thresholds