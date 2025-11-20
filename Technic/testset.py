# =============================================================================
# module: testset.py
# Purpose: Test set builder functions for different model types.
# Key Types/Classes: TestSet
# Key Functions: ppnr_ols_testset_func, fixed_ols_testset_func
# Dependencies: pandas, numpy, statsmodels, typing, .test module classes
#
# TESTSET FUNCTION REQUIREMENTS:
# ==============================
# All testset functions should define these measure tests FIRST:
# 1. 'Fit Measures' - FitMeasure for R² and Adj R²
# 2. 'IS Error Measures' - ErrorMeasure for in-sample ME, MAE, RMSE
# 3. 'OOS Error Measures' - ErrorMeasure for out-of-sample ME, MAE, RMSE (if data available)
#
# These are used by ModelBase.in_perf_measures and ModelBase.out_perf_measures
# for model reporting and evaluation. Define these before other tests.
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, TYPE_CHECKING, List, Tuple
from .test import *
from .modeltype import Growth

if TYPE_CHECKING:
    from .model import ModelBase


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
        keyed by the test's display name (alias or class name),
        including both active and inactive tests.
        """
        return {t.name: t.test_result for t in self.tests}
    
    @property
    def test_info(self) -> Dict[str, Dict[str, str]]:
        """
        Return key information of each test in dictionary format.
        
        Returns
        -------
        dict
            Keys: test names
            Values: dict containing 'filter_mode' and 'desc' for each test
        """
        info = {}
        for test in self.tests:
            info[test.name] = {
                'filter_mode': test.filter_mode,
                'desc': test.filter_mode_desc if hasattr(test, 'filter_mode_desc') else ''
            }
        return info
    
    @property
    def filter_test_info(self) -> Dict[str, Dict[str, str]]:
        """
        Return key information of only active tests (filter_on=True) in dictionary format.
        
        Returns
        -------
        dict
            Keys: test names for tests with filter_on=True
            Values: dict containing 'filter_mode' and 'desc' for each active test
        """
        info = {}
        for test in self.tests:
            if test.filter_on:
                info[test.name] = {
                    'filter_mode': test.filter_mode,
                    'desc': test.filter_mode_desc if hasattr(test, 'filter_mode_desc') else ''
                }
        return info

    def filter_pass(
        self,
        fast_filter: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Run active tests and return overall pass flag and failed test names.

        Parameters
        ----------
        fast_filter : bool, default False
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
        Print summary of test configurations using test_info property:
          - Filtering Tests: name, filter_mode, desc
          - No-Filtering Tests: name only (excluding measures), with note
          - Measures: list of tests in 'measure' category
        """
        info = self.test_info

        # Filtering tests (filter_on=True)
        print("Filtering Tests:")
        for test in self.tests:
            if test.filter_on:
                test_info = info[test.name]
                print(f"- {test.name} | filter_mode: {test_info['filter_mode']} | desc: {test_info['desc']}")

        # No-filtering tests (filter_on=False), excluding measures
        print("\nNo-Filtering Tests:")
        inactive = [t for t in self.tests if (not t.filter_on) and getattr(t, 'category', None) != 'measure']
        for test in inactive:
            print(f"- {test.name}")
        
        if inactive:
            print(
                "\nNote: These tests are included but not turned on. "
                "Set `filter_on=True` on a test to include it in filter_pass results."
            )

        # Measures (category == 'measure'), shown separately
        measures = [t for t in self.tests if getattr(t, 'category', None) == 'measure']
        if measures:
            print("\nMeasures:")
            for test in measures:
                print(f"- {test.name}")


def ppnr_ols_testset_func(mdl: 'ModelBase') -> Dict[str, ModelTestBase]:
    """
    Pre-defined TestSet for PPNR OLS models with improved group labels:
    - In-sample R-sq
    - Individual significance (CoefTest drivers)
    - Joint F-tests (GroupTest drivers)
    - Residual stationarity & normality
    - Target stationarity & cointegration
    - Sign checking for features with exp_sign
    
    GUIDANCE FOR TESTSET FUNCTIONS:
    ===============================
    All future testset functions should define the following measure tests FIRST,
    before any other assumption and performance tests:
    
    1. 'Fit Measures' - FitMeasure test for R-sq and Adj R-sq metrics
    2. 'IS Error Measures' - ErrorMeasure test for in-sample ME, MAE, RMSE
    3. 'OOS Error Measures' - ErrorMeasure test for out-of-sample ME, MAE, RMSE (if applicable)
    
    These measures will be used by ModelBase.in_perf_measures and ModelBase.out_perf_measures
    properties for model reporting and evaluation. The order matters as these are the 
    foundation metrics that other tests may reference.
    """
    tests: Dict[str, ModelTestBase] = {}

    #---Fit & Error Measures (inactive for filtering)---
    # Goodness of fit (in-sample)
    tests['Fit Measures'] = FitMeasure(
        actual=mdl.y,
        predicted=mdl.y_fitted_in,
        n_features=len(mdl.params) - 1  # subtract intercept
    )

    # Add error measures (in-sample)
    tests['IS Error Measures'] = ErrorMeasure(
        actual=mdl.y,
        predicted=mdl.y_fitted_in
    )

    # Optionally, out-of-sample:
    if not mdl.X_out.empty:
        tests['OOS Error Measures'] = ErrorMeasure(
            actual=mdl.y_out,
            predicted=mdl.y_pred_out
        )

    #---Filtering Test---
    # In-sample R-sq
    tests['In-Sample R-sq'] = R2Test(
        r2=mdl.rsquared,
        filter_mode='moderate'
    )

    # Individual coefficient significance using CoefTest
    coef_test_vars = mdl.spec_map.get('CoefTest', [])
    if coef_test_vars:
        # Filter to only include variables that exist in the model
        available_vars = [var for var in coef_test_vars if var in mdl.pvalues.index]
        if available_vars:
            tests['Coefficient Significance'] = CoefTest(
                pvalues=mdl.pvalues.loc[available_vars],
                filter_mode='moderate'
            )

    # Group-driver significance using GroupTest
    for grp in mdl.spec_map.get('GroupTest', []):
        # list of names
        if isinstance(grp, (list, tuple)):
            names = list(grp)
            # Filter to only include variables that exist in the model
            available_names = [name for name in names if name in mdl.pvalues.index]
            if not available_names:
                continue
                
            parts = [name.split(':', 1) if ':' in name else [None, name] for name in available_names]
            prefixes = [p[0] for p in parts]
            suffixes = [p[1] for p in parts]
            # detect common prefix
            if None not in prefixes and len(set(prefixes)) == 1:
                prefix = prefixes[0] + ':'
                label_body = "'".join(suffixes)
                group_label = f"{prefix}{label_body}"
            else:
                group_label = "'".join(available_names)
            vars_for = available_names
        else:
            group_label = str(grp)
            vars_for = [grp] if grp in mdl.pvalues.index else []

        if vars_for:  # Only create test if variables exist
            alias = f"Group Driver F-Test {group_label}"
            tests[alias] = GroupTest(
                model_result=mdl.fitted,
                vars=vars_for,
                filter_mode='moderate'
            )
    
    # Coefficient Multicollinearity
    tests['Multicollinearity'] = VIFTest(
        exog=sm.add_constant(mdl.X),
        filter_mode='moderate'
     )
    
    # Residual diagnostics
    tests['Residual Stationarity'] = StationarityTest(
        series=mdl.resid,
        filter_mode='moderate'
    )
    tests['Residual Normality'] = NormalityTest(
        series=mdl.resid,
        filter_mode='moderate',
        filter_on=False
    )
    tests['Residual Autocorrelation'] = AutocorrTest(
        results=mdl.fitted,
        filter_mode='moderate',
        filter_on=False
    )
    tests['Residual Heteroscedasticity'] = HetTest(
        resids=mdl.resid,
        exog=sm.add_constant(mdl.X),
        filter_mode='moderate',
        filter_on=False
    )

    # --- Target Stationarity & Cointegration ---
    # 1) Check if Y itself is stationary
    y_stat = StationarityTest(
        series=mdl.y.copy(),
        filter_mode='moderate',
        filter_on=False
    )
    tests['Y Stationarity'] = y_stat

    # 2) Get variables applicable for stationarity testing
    stationarity_vars = mdl.spec_map.get('StationarityTest', [])
    
    if y_stat.test_filter:
        # Y is stationary - check that all X variables are also stationary
        if stationarity_vars:
            # Filter to only include variables that exist in X
            available_vars = [var for var in stationarity_vars if var in mdl.X.columns]
            if available_vars:
                # Stationarity screening leverages staged, sample-aware checks across
                # all model specifications; search_cms() already performs pretests,
                # so filtering is disabled here to avoid double-counting.
                tests['X Stationarity'] = MultiFullStationarityTest(
                    specs=mdl.specs,
                    dm=mdl.dm,
                    filter_mode='moderate',
                    filter_on=False
                )
    else:
        # Y is non-stationary - test cointegration with applicable X variables
        if stationarity_vars:
            # Filter to only include variables that exist in X
            available_vars = [var for var in stationarity_vars if var in mdl.X.columns]
            if available_vars:
                X_vars_df = mdl.X[available_vars].copy()
                tests['Y–X Cointegration'] = CointTest(
                    X_vars=X_vars_df,
                    resids=mdl.resid.copy(),
                    filter_mode='moderate'
                )

    # --- Sign Check Test ---
    sign_check_features = mdl.spec_map.get('SignCheck', [])
    if sign_check_features:
        tests['Sign Check'] = SignCheck(
            feature_list=sign_check_features,
            coefficients=mdl.params,
            filter_mode='moderate'
        )

    # --- Base Growth Test (for Growth model types) ---
    if getattr(mdl, 'model_type', None) is Growth:
        try:
            freq = mdl.dm.freq if hasattr(mdl, 'dm') and hasattr(mdl.dm, 'freq') else 'M'
            tests['Base Growth'] = BaseGrowthTest(
                coeffs=mdl.params,
                freq=freq,
                filter_on=False
            )
        except Exception:
            # If anything goes wrong (e.g., params not ready), skip adding this test
            pass

    return tests 

def fixed_ols_testset_func(mdl: 'ModelBase') -> Dict[str, ModelTestBase]:
    """
    Minimal TestSet for fixed-coefficient OLS-style models.

    Includes only fundamental measures that do not require a statsmodels fit:
    - 'Fit Measures' (R², Adj R²) from in-sample actual vs fitted
    - 'IS Error Measures' (ME, MAE, RMSE)
    - 'OOS Error Measures' (ME, MAE, RMSE) if OOS data available
    """
    tests: Dict[str, ModelTestBase] = {}

    tests['Fit Measures'] = FitMeasure(
        actual=mdl.y,
        predicted=mdl.y_fitted_in,
        n_features=max(0, len(getattr(mdl, 'params', [])) - 1)
    )
    tests['IS Error Measures'] = ErrorMeasure(
        actual=mdl.y,
        predicted=mdl.y_fitted_in
    )
    if not mdl.X_out.empty:
        tests['OOS Error Measures'] = ErrorMeasure(
            actual=mdl.y_out,
            predicted=mdl.y_pred_out
        )
    return tests