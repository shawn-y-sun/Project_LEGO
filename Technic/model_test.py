# =============================================================================
# module: model_test.py
# Purpose: Test set builder functions for different model types
# Dependencies: pandas, numpy, statsmodels, typing, .test module classes
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, TYPE_CHECKING
from .test import *

if TYPE_CHECKING:
    from .model import ModelBase


def ppnr_ols_testset_func(mdl: 'ModelBase') -> Dict[str, ModelTestBase]:
    """
    Pre-defined TestSet for PPNR OLS models with improved group labels:
    - In-sample R²
    - Individual significance (CoefTest drivers)
    - Joint F-tests (GroupTest drivers)
    - Residual stationarity & normality
    - Target stationarity & cointegration
    - Sign checking for features with exp_sign
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
    # In-sample R²
    tests['In-Sample R²'] = R2Test(
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
        for i, var in enumerate(stationarity_vars):
            if var in mdl.X.columns:
                tests[f'X Stationarity {var}'] = StationarityTest(
                    series=mdl.X[var].copy(),
                    filter_mode='moderate',
                    alias=f'X Stationarity {var}'
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
            tsfm_list=sign_check_features,
            coefficients=mdl.params,
            filter_mode='moderate'
        )

    return tests 