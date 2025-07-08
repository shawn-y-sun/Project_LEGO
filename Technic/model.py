# =============================================================================
# module: model.py
# Purpose: Define base and OLS regression models with testing and reporting hooks
# Dependencies: pandas, numpy, statsmodels, typing, .testset.TestSet, .report.OLS_ModelReport
# =============================================================================

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Any, Callable, Type, Dict, List, Union, Tuple
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .test import *
from .report import ModelReportBase, OLS_ModelReport
from .testset import ppnr_ols_testset_func, TestSet

class ModelBase(ABC):
    """
    Abstract base class for statistical models with testing and reporting.

    This class now handles data preparation, validation, and scenario management
    internally. It can work with a DataManager to build features or accept
    pre-built data matrices.

    Parameters
    ----------
    dm : DataManager
        DataManager instance for building features and accessing data.
    specs : List[Union[str, Dict[str, Any]]]
        Feature specifications to pass to DataManager.build_features().
    sample : {'in', 'full'}
        Which sample to use for fitting:
        - 'in': fit using in-sample data only
        - 'full': fit using full-sample data
    outlier_idx : List[Any], optional
        List of index labels corresponding to outlier records to remove 
        from the data. Each label must exist within the sample period.
    target : str
        Name of the target column in the DataManager's internal_data.
    testset_func : callable, optional
        Builds initial mapping of tests.
    test_update_func : callable, optional
        Updates or adds tests post initial mapping.
    testset_cls : type, default TestSet
        Class for aggregating ModelTestBase instances into a TestSet.
    scen_cls : type, optional
        Class to use for scenario management. If None, defaults to ScenManager.
    report_cls : type, optional
        Class for generating model reports.
    X : DataFrame, optional
        Pre-built in-sample features. If provided, overrides feature building.
    y : Series, optional
        Pre-built in-sample target. If provided, overrides target extraction.
    X_out : DataFrame, optional
        Pre-built out-of-sample features.
    y_out : Series, optional
        Pre-built out-of-sample target.
        
    Attributes
    ----------
    scen_manager : Any, optional
        Scenario manager instance created using scen_cls.
    """
    def __init__(
        self,
        dm: Any = None,
        specs: List[Union[str, Dict[str, Any]]] = None,
        sample: str = 'in',
        outlier_idx: Optional[List[Any]] = None,
        target: str = None,
        testset_func: Optional[Callable[['ModelBase'], Dict[str, ModelTestBase]]] = None,
        test_update_func: Optional[Callable[['ModelBase'], Dict[str, Any]]] = None,
        testset_cls: Type = TestSet,
        scen_cls: Optional[Type] = None,
        report_cls: Optional[Type] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        X_out: Optional[pd.DataFrame] = None,
        y_out: Optional[pd.Series] = None
    ):
        # Core data preparation parameters
        self.dm = dm
        self.specs = specs
        self.sample = sample
        self.outlier_idx = outlier_idx or []
        self.target = target
        
        # Validation
        if sample not in {'in', 'full'}:
            raise ValueError("sample must be 'in' or 'full'.")
        
        # If X and y are not provided, dm and specs are required
        if X is None or y is None:
            if self.dm is None:
                raise ValueError("DataManager (dm) is required when X and y are not provided.")
            if self.specs is None:
                raise ValueError("specs is required when X and y are not provided.")
            if self.target is None:
                raise ValueError("target is required when X and y are not provided.")
        
        # Cached data attributes (only set if provided by user)
        self._X_cache = X
        self._y_cache = y
        self._X_out_cache = X_out
        self._y_out_cache = y_out
        
        # Test configuration
        self.testset_func = testset_func
        self.test_update_func = test_update_func
        self.testset_cls = testset_cls
        self.testset: Optional[TestSet] = None
        
        # Scenario management
        if scen_cls is None:
            # Import ScenManager here to avoid circular imports
            # (scenario.py imports from model.py)
            from .scenario import ScenManager
            self.scen_cls = ScenManager
        else:
            self.scen_cls = scen_cls
        self.scen_manager: Optional[Any] = None
        
        # Reporting configuration
        self.report_cls = report_cls
        
        # Model metadata
        self.coefs_ = None
        self.is_fitted = False
        
        # Cache for out-of-sample predictions
        self._y_pred_out: Optional[pd.Series] = None

    @staticmethod
    def _validate_data(X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate X (features) and y (target) for NaNs or infinite values.

        Raises
        ------
        ValueError
            If any column in X or the series y contains NaNs or infinite values.
        """
        nan_cols: List[str] = []
        inf_cols: List[str] = []

        for col in X.columns:
            s = X[col]
            if s.isna().any():
                nan_cols.append(col)
            elif is_numeric_dtype(s) and not np.isfinite(s.dropna()).all():
                inf_cols.append(col)

        has_nan_y = y.isna().any()
        has_inf_y = is_numeric_dtype(y) and not np.isfinite(y.dropna()).all()

        if nan_cols or inf_cols or has_nan_y or has_inf_y:
            msgs: List[str] = []
            if nan_cols:
                msgs.append(f"X contains NaNs: {nan_cols}")
            if inf_cols:
                msgs.append(f"X contains infinite values: {inf_cols}")
            if has_nan_y:
                msgs.append("y contains NaNs")
            if has_inf_y:
                msgs.append("y contains infinite values")
            raise ValueError("Data validation error: " + "; ".join(msgs))

    @property
    def X_full(self) -> pd.DataFrame:
        """
        Get full-sample features, building from DataManager if not cached.
        
        Returns
        -------
        pd.DataFrame
            Full-sample feature matrix.
        """
        if self._X_cache is not None:
            return self._X_cache
        
        # Get the union of in-sample and out-of-sample indices
        idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        
        # Build features
        X_full = self.dm.build_features(self.specs)
        
        # Align to index
        X_full = X_full.reindex(idx).astype(float)
        
        return X_full

    @property
    def y_full(self) -> pd.Series:
        """
        Get full-sample target, extracting from DataManager if not cached.
        
        Returns
        -------
        pd.Series
            Full-sample target series.
        """
        if self._y_cache is not None:
            return self._y_cache
        
        # Get the union of in-sample and out-of-sample indices
        idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        
        # Extract target
        y_full = self.dm.internal_data[self.target].copy()
        
        # Align to index
        y_full = y_full.reindex(idx).astype(float)
        
        return y_full

    @property
    def X_in(self) -> pd.DataFrame:
        """
        Get in-sample features with outliers removed.
        
        Returns
        -------
        pd.DataFrame
            In-sample feature matrix with outliers removed.
        """
        X_in = self.X_full.loc[self.dm.in_sample_idx].copy()
        
        # Remove outliers if specified
        if self.outlier_idx:
            # Convert outlier_idx to match X_in.index dtype
            if is_datetime64_any_dtype(X_in.index):
                converted_idx = pd.to_datetime(self.outlier_idx)
            else:
                converted_idx = self.outlier_idx

            # Check whether every converted index exists in X_in.index
            missing = [i for i in converted_idx if i not in X_in.index]
            if missing:
                raise ValueError(f"Outlier indices {missing} not in in-sample period.")

            # Drop those rows from X_in
            X_in = X_in.drop(index=converted_idx)
        
        return X_in

    @property
    def y_in(self) -> pd.Series:
        """
        Get in-sample target with outliers removed.
        
        Returns
        -------
        pd.Series
            In-sample target series with outliers removed.
        """
        y_in = self.y_full.loc[self.dm.in_sample_idx].copy()
        
        # Remove outliers if specified
        if self.outlier_idx:
            # Convert outlier_idx to match y_in.index dtype
            if is_datetime64_any_dtype(y_in.index):
                converted_idx = pd.to_datetime(self.outlier_idx)
            else:
                converted_idx = self.outlier_idx

            # Check whether every converted index exists in y_in.index
            missing = [i for i in converted_idx if i not in y_in.index]
            if missing:
                raise ValueError(f"Outlier indices {missing} not in in-sample period.")

            # Drop those rows from y_in
            y_in = y_in.drop(index=converted_idx)
        
        return y_in

    @property
    def X_out(self) -> pd.DataFrame:
        """
        Get out-of-sample features.
        
        Returns
        -------
        pd.DataFrame
            Out-of-sample feature matrix.
        """
        if self._X_out_cache is not None:
            return self._X_out_cache
        
        return self.X_full.loc[self.dm.out_sample_idx].copy()

    @property
    def y_out(self) -> pd.Series:
        """
        Get out-of-sample target.
        
        Returns
        -------
        pd.Series
            Out-of-sample target series.
        """
        if self._y_out_cache is not None:
            return self._y_out_cache
        
        return self.y_full.loc[self.dm.out_sample_idx].copy()

    @property
    def X(self) -> pd.DataFrame:
        """
        Get features based on sample setting.
        
        Returns
        -------
        pd.DataFrame
            Feature matrix for the specified sample.
        """
        if self.sample == 'in':
            return self.X_in
        else:  # sample == 'full'
            return self.X_full

    @property
    def y(self) -> pd.Series:
        """
        Get target based on sample setting.
        
        Returns
        -------
        pd.Series
            Target series for the specified sample.
        """
        if self.sample == 'in':
            return self.y_in
        else:  # sample == 'full'
            return self.y_full

    @abstractmethod
    def fit(self) -> 'ModelBase':
        """
        Fit the model to sample data.

        Returns
        -------
        self : ModelBase
        """
        ...

    @abstractmethod
    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for new data.
        """
        ...

    @property
    def y_pred_out(self) -> pd.Series:
        """
        Out-of-sample predictions generated by calling predict on X_out.

        Returns empty Series if X_out is empty.
        """
        if self.X_out.empty:
            return pd.Series(dtype=float)
        self._y_pred_out = self.predict(self.X_out)
        return self._y_pred_out

    @property
    def report(self) -> ModelReportBase:
        """
        Build and return the report instance using report_cls and this model.
        """
        if not self.report_cls:
            raise ValueError("No report_cls provided for building report.")
        # Now report_cls must accept only model=self
        return self.report_cls(model=self)

    @property
    def in_perf_measures(self) -> pd.Series:
        """
        In-sample performance measures from testset results.
        
        Combines 'Fit Measures' and 'IS Error Measures' test results into a single Series.
        
        Returns
        -------
        pd.Series
            Combined performance measures with metric names as index.
        """
        if self.testset is None:
            return pd.Series(dtype=float)
        
        # Get test results from testset
        all_results = self.testset.all_test_results
        
        # Combine Fit Measures and IS Error Measures
        combined_series = pd.Series(dtype=float)
        
        # Add Fit Measures if available
        if 'Fit Measures' in all_results:
            fit_result = all_results['Fit Measures']
            if isinstance(fit_result, pd.Series):
                combined_series = pd.concat([combined_series, fit_result])
        
        # Add IS Error Measures if available
        if 'IS Error Measures' in all_results:
            error_result = all_results['IS Error Measures']
            if isinstance(error_result, pd.Series):
                combined_series = pd.concat([combined_series, error_result])
        
        return combined_series

    @property
    def out_perf_measures(self) -> pd.Series:
        """
        Out-of-sample performance measures from testset results.
        
        Returns 'OOS Error Measures' test result as a Series.
        
        Returns
        -------
        pd.Series
            Out-of-sample performance measures, empty if no OOS data available.
        """
        if self.testset is None:
            return pd.Series(dtype=float)
        
        # Get test results from testset
        all_results = self.testset.all_test_results
        
        # Return OOS Error Measures if available
        if 'OOS Error Measures' in all_results:
            oos_result = all_results['OOS Error Measures']
            if isinstance(oos_result, pd.Series):
                return oos_result
        
        return pd.Series(dtype=float)

    def _create_scenario_manager(self) -> None:
        """
        Create ScenManager instance using scen_cls.
        """
        if self.scen_cls is not None:
            self.scen_manager = self.scen_cls(
                dm=self.dm,
                model=self,
                specs=self.specs,
                target=self.target
            )

    @property
    def spec_map(self) -> Dict[str, List[Union[str, Tuple[str, ...]]]]:
        """
        Categorize self.specs into driver lists for different test purposes.

        Returns
        -------
        Dict[str, List[Union[str, Tuple[str, ...]]]]
            - 'CoefTest': variable names for individual coefficient tests
            - 'GroupTest': tuples of variable names for group F-tests (only full monthly/quarterly dummies)
            - 'StationarityTest': variable names applicable for stationarity tests
            - 'SignCheck': Feature instances with exp_sign attribute
        """
        if self.specs is None:
            return {
                'CoefTest': [],
                'GroupTest': [],
                'StationarityTest': [],
                'SignCheck': []
            }
        
        # Import here to avoid circular imports
        from .feature import DumVar, Feature
        
        coef_test_vars: List[str] = []
        group_test_vars: List[Tuple[str, ...]] = []
        stationarity_test_vars: List[str] = []
        sign_check_features: List[Feature] = []

        def _flatten(items):
            """Recursively flatten nested lists but leave tuples intact."""
            for item in items:
                if isinstance(item, list):
                    yield from _flatten(item)
                else:
                    yield item

        def _is_full_monthly_quarterly_dummy(dumvar: DumVar) -> bool:
            """Check if DumVar represents full monthly (M:2-12) or quarterly (Q:2-4) dummies."""
            if not isinstance(dumvar, DumVar):
                return False
            
            # Extract numbers from output_names that have the pattern var:number
            dummy_names = [name for name in dumvar.output_names if ':' in name]
            if not dummy_names:
                return False
                
            # Check for monthly dummies: var should be 'M' and output should cover M:2 to M:12
            if dumvar.var == 'M':
                months = set()
                for name in dummy_names:
                    if name.startswith('M:'):
                        try:
                            month_num = int(name.split(':')[1])
                            months.add(month_num)
                        except (ValueError, IndexError):
                            continue
                # Check if covers months 2-12 (reference month 1 dropped)
                if months == set(range(2, 13)):
                    return True
            
            # Check for quarterly dummies: var should be 'Q' and output should cover Q:2 to Q:4
            elif dumvar.var == 'Q':
                quarters = set()
                for name in dummy_names:
                    if name.startswith('Q:'):
                        try:
                            quarter_num = int(name.split(':')[1])
                            quarters.add(quarter_num)
                        except (ValueError, IndexError):
                            continue
                # Check if covers quarters 2-4 (reference quarter 1 dropped)
                if quarters == set(range(2, 5)):
                    return True
            
            return False
        
        for spec in _flatten(self.specs):
            # 1) Tuple of specs → one tuple of their output names
            if isinstance(spec, tuple):
                names = tuple(
                    n
                    for s in spec
                    for n in (
                        s.output_names if isinstance(s, Feature) else [str(s)]
                    )
                )
                # Tuples go to CoefTest only (no overlap with GroupTest)
                coef_test_vars.extend(names)
                # Individual variables go to stationarity test if not dummies
                for name in names:
                    if not ':' in name:  # Not a dummy variable
                        stationarity_test_vars.append(name)

            # 2) DumVar instance → one tuple of its dummy-column names
            elif isinstance(spec, DumVar):
                names = tuple(spec.output_names)
                
                # Check if this is a full monthly/quarterly dummy group
                if _is_full_monthly_quarterly_dummy(spec):
                    # Full monthly/quarterly dummies go to GroupTest only
                    group_test_vars.append(names)
                else:
                    # Other dummy groups go to CoefTest only
                    coef_test_vars.extend(names)

            # 3) Everything else → individual features
            else:
                if isinstance(spec, Feature):
                    # Add to coef_test_vars
                    coef_test_vars.extend(spec.output_names)
                    
                    # Check for stationarity test eligibility (non-dummy variables)
                    for name in spec.output_names:
                        if not ':' in name:  # Not a dummy variable
                            stationarity_test_vars.append(name)
                    
                    # Check for SignCheck eligibility (Features with exp_sign attribute)
                    if hasattr(spec, 'exp_sign'):
                        sign_check_features.append(spec)
                        
                else:
                    # String specifications
                    coef_test_vars.append(str(spec))
                    stationarity_test_vars.append(str(spec))

        return {
            'CoefTest': coef_test_vars,
            'GroupTest': group_test_vars,
            'StationarityTest': stationarity_test_vars,
            'SignCheck': sign_check_features
        }

    def load_testset(
        self,
        testset_func: Optional[Callable[['ModelBase'], Dict[str, ModelTestBase]]] = None,
        test_update_func: Optional[Callable[['ModelBase'], Dict[str, Any]]] = None
    ) -> TestSet:
        """
        Rebuild TestSet from provided functions and cache it.
        """
        func_init = testset_func or self.testset_func
        if func_init is None:
            raise ValueError("No testset_func provided.")
        tests = func_init(self)
        func_update = test_update_func or self.test_update_func
        if func_update:
            updates = func_update(self)
            for alias, val in updates.items():
                if isinstance(val, ModelTestBase):
                    tests[alias] = val
                elif isinstance(val, dict):
                    if alias in tests:
                        for k, v in val.items(): setattr(tests[alias], k, v)
                    else:
                        raise KeyError(f"Unknown test '{alias}' in update_map")
                else:
                    raise TypeError("test_update_map values must be ModelTestBase or kwargs dict")
        # Apply aliases and assemble TestSet
        for alias, obj in tests.items(): obj.alias = alias
        self.testset = self.testset_cls(tests)
        return self.testset


class OLS(ModelBase):
    """
    Ordinary Least Squares regression model with built-in testing and reporting.
    """
    def __init__(
        self,
        dm: Any,
        specs: List[Union[str, Dict[str, Any]]],
        sample: str,
        outlier_idx: Optional[List[Any]] = None,
        target: str = None,
        testset_func: Callable[['ModelBase'], Dict[str, ModelTestBase]] = ppnr_ols_testset_func,
        test_update_func: Optional[Callable[['ModelBase'], Dict[str, Any]]] = None,
        testset_cls: Type = TestSet,
        scen_cls: Optional[Type] = None,
        report_cls: Type = OLS_ModelReport,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        X_out: Optional[pd.DataFrame] = None,
        y_out: Optional[pd.Series] = None
    ):
        super().__init__(
            dm=dm,
            specs=specs,
            sample=sample,
            outlier_idx=outlier_idx,
            target=target,
            testset_func=testset_func,
            test_update_func=test_update_func,
            testset_cls=testset_cls,
            scen_cls=scen_cls,
            report_cls=report_cls,
            X=X,
            y=y,
            X_out=X_out,
            y_out=y_out
        )
        # Fit result placeholders
        self.fitted = None
        self.params = None
        self.pvalues = None
        self.rsquared = None
        self.rsquared_adj = None
        self.y_fitted_in = None
        self.resid = None
        self.bse = None
        self.tvalues = None
        self.vif = None
        # track covariance type
        self.cov_type: str = 'OLS'
        self.is_fitted = False

    def fit(self) -> 'OLS':
        """
        Fit OLS, detect residual issues, and if needed refit with robust covariances.
        """
        # Validate data before fitting
        self._validate_data(self.X, self.y)
        
        # initial OLS fit
        Xc = sm.add_constant(self.X)
        res = sm.OLS(self.y, Xc).fit()
        # store core attributes
        self.fitted = res
        self.params = res.params
        self.pvalues = res.pvalues
        self.rsquared = res.rsquared
        self.rsquared_adj = res.rsquared_adj
        self.y_fitted_in = res.fittedvalues
        self.resid = res.resid
        self.bse = res.bse
        self.tvalues = res.tvalues
        # VIF
        self.vif = pd.Series({
            col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)
        })
        self.is_fitted = True

        # Residual diagnostics
        ac_test = AutocorrTest(
            results=self.fitted,
            alias='Residual Autocorrelation',
            filter_mode='moderate'
        )
        het_test = HetTest(
            resids=self.resid,
            exog=Xc,
            alias='Residual Heteroscedasticity',
            filter_mode='moderate'
        )
        ac_fail = not ac_test.test_filter
        het_fail = not het_test.test_filter

        # Determine cov_type based on diagnostics
        if ac_fail or het_fail:
            if het_fail and not ac_fail:
                # heteroskedasticity only
                robust = self.fitted.get_robustcov_results(cov_type='HC1')
                self.cov_type = 'HC1'
            else:
                # autocorrelation (with or without heteroskedasticity)
                n = len(self.y)
                lag = int(np.floor(4 * (n / 100) ** (2/9)))
                robust = self.fitted.get_robustcov_results(
                    cov_type='HAC', maxlags=lag
                )
                self.cov_type = f'HAC({lag})'
            # update fitted results and inferential attributes
            self.fitted = robust
            # ensure pandas Series for consistency
            idx = self.params.index
            self.bse = pd.Series(robust.bse, index=idx)
            self.tvalues = pd.Series(robust.tvalues, index=idx)
            self.pvalues = pd.Series(robust.pvalues, index=idx)
        else:
            self.cov_type = 'NR'
        
        # load tests
        self.load_testset()
        
        # Create scenario manager
        self._create_scenario_manager()
        
        return self

    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """
        Predict using the fitted statsmodels results.
        """
        if not self.is_fitted or self.fitted is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xc_new = sm.add_constant(X_new, has_constant='add')
        return self.fitted.predict(Xc_new)
    
    @property
    def param_measures(self) -> Dict[str, Dict[str, Any]]:
        """
        Parameter measures: coefficient, pvalue, VIF, and standard error for each term.
        """
        return {
            var: {
                'coef': float(self.params[var]),
                'pvalue': float(self.pvalues[var]),
                'vif': float(self.vif.get(var, np.nan)),
                'std': float(self.bse.get(var, np.nan))
            }
            for var in self.params.index
        }

    def __repr__(self) -> str:
        return f'OLS-{self.cov_type}'