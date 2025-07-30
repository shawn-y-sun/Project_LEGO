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
from .modeltype import RateLevel, BalanceLevel

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
    model_type : type, optional
        ModelType subclass for converting predictions to base variables.
    target_base : str, optional
        Name of the base variable of interest (highly recommended if available).
    target_exposure : str, optional
        Name of the exposure variable (required for Ratio model types).
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
    stability_test_cls : type, optional
        Class for model stability testing. If None, defaults to WalkForwardTest.
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
    base_predictor : Any, optional
        Model type instance for converting predictions to base variables.
    """
    def __init__(
        self,
        dm: Any = None,
        specs: List[Union[str, Dict[str, Any]]] = None,
        sample: str = 'in',
        outlier_idx: Optional[List[Any]] = None,
        target: str = None,
        model_type: Optional[Type] = None,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None,
        testset_func: Optional[Callable[['ModelBase'], Dict[str, ModelTestBase]]] = None,
        test_update_func: Optional[Callable[['ModelBase'], Dict[str, Any]]] = None,
        testset_cls: Type = TestSet,
        scen_cls: Optional[Type] = None,
        report_cls: Optional[Type] = None,
        stability_test_cls: Optional[Type] = None,
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
        self.target_base = target_base
        self.target_exposure = target_exposure
        
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
        
        # Model type and base prediction
        self.model_type = model_type
        self.base_predictor: Optional[Any] = None
        if model_type is not None:
            # Validate that model_type is a subclass of ModelType
            from .modeltype import ModelType
            if not issubclass(model_type, ModelType):
                raise ValueError("model_type must be a subclass of ModelType")
            
            # Create base predictor instance
            if self.dm is not None:
                # Only pass target_exposure if it's not None
                if self.target_exposure is not None:
                    self.base_predictor = model_type(
                        dm=self.dm,
                        target=self.target,
                        target_base=self.target_base,
                        target_exposure=self.target_exposure
                    )
                else:
                    self.base_predictor = model_type(
                        dm=self.dm,
                        target=self.target,
                        target_base=self.target_base
                    )
        
        # Reporting configuration
        self.report_cls = report_cls
        
        # Stability testing configuration
        self.stability_test_cls = stability_test_cls
        
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
        
        When has_lookback_var is True, uses rolling_predict to generate features
        that account for conditional variable effects.
        
        Returns
        -------
        pd.DataFrame
            Out-of-sample feature matrix.
        """
        if self._X_out_cache is not None:
            return self._X_out_cache
        
        # If model has lookback variables, use rolling_predict
        if self.has_lookback_var:
            # Get out-of-sample time frame
            oos_time_frame = (str(self.dm.out_sample_idx.min()), str(self.dm.out_sample_idx.max()))
            
            # Use rolling_predict to get both predictions and features
            _, X_new = self.rolling_predict(
                df_internal=self.dm.internal_data,
                df_mev=self.dm.model_mev,
                y=self.y_full,
                time_frame=oos_time_frame
            )
            return X_new
        
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

    def rolling_predict(
        self,
        df_internal: pd.DataFrame,
        df_mev: pd.DataFrame,
        y: Optional[pd.Series] = None,
        time_frame: Tuple[str, str] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate iterative predictions with lookback handling for conditional variables.
        
        This method is designed for scenario prediction where conditional variables (like BO)
        need to be updated iteratively based on predicted target values. It's primarily used
        when predicting on scenario data (dm.scen_internal_data and dm.scen_mevs) rather than
        modeling data, since scenario variables haven't been pre-adjusted for conditional effects.
        
        The method performs iterative prediction:
        1. Builds features for the entire time frame
        2. Predicts one period at a time
        3. Updates the target series with each prediction
        4. Rebuilds features to capture conditional variable effects
        5. Continues until all periods are predicted
        
        Parameters
        ----------
        df_internal : pd.DataFrame
            Internal data DataFrame, ideally from dm.scen_internal_data.
            Should contain the target variable for historical periods.
        df_mev : pd.DataFrame
            MEV data DataFrame, ideally from dm.scen_mevs.
            Should have the same scenario set and scenario name as df_internal.
        y : pd.Series, optional
            Historical target values before prediction period. If None, uses self.y_full.
        time_frame : tuple of str
            Start and end dates for prediction horizon, e.g., ('2025-01-31', '2025-06-30').
            Dates should be in a format pandas can parse.
            
        Returns
        -------
        tuple
            (predicted_series, X_new) where:
            - predicted_series: pd.Series of predicted target values for the specified time frame
            - X_new: pd.DataFrame of the latest feature matrix after all predictions
            
        Raises
        ------
        ValueError
            If no lookback variables found, insufficient historical data, or invalid time_frame.
        RuntimeError
            If model is not fitted.
            
        Example
        -------
        >>> # Get scenario data
        >>> scen_internal = dm.scen_internal_data['EWST2024']['Base']
        >>> scen_mev = dm.scen_mevs['EWST2024']['Base']
        >>> 
        >>> # Predict with lookback handling for scenario
        >>> predictions, X_final = model.rolling_predict(
        ...     df_internal=scen_internal,
        ...     df_mev=scen_mev,
        ...     time_frame=('2025-01-31', '2025-06-30')
        ... )
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        if time_frame is None:
            raise ValueError("time_frame must be specified as (start_date, end_date)")
        
        # Parse time frame dates
        start_date = pd.to_datetime(time_frame[0])
        end_date = pd.to_datetime(time_frame[1])
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Get lookback variables from spec_map
        lookback_vars = self.spec_map.get('lookback_var', [])
        
        if not lookback_vars:
            # No lookback variables, use standard feature building and prediction
            X_full = self.dm.build_features(self.specs, df_internal, df_mev)
            X_new = X_full[(X_full.index >= start_date) & (X_full.index <= end_date)]
            return self.predict(X_new)
        
        # Get the maximum lookback_n required
        max_lookback_n = max(var.lookback_n for var in lookback_vars)
        
        # Get historical target values
        if y is None:
            y = self.y_full
        
        # Ensure we have enough historical data
        if len(y) < max_lookback_n:
            raise ValueError(
                f"Insufficient historical data. Need at least {max_lookback_n} periods, "
                f"but only have {len(y)} periods."
            )
        
        # Get the latest lookback_n periods of target values before start date
        y_lookback = y[y.index < start_date].tail(max_lookback_n)
        
        if len(y_lookback) < max_lookback_n:
            raise ValueError(
                f"Insufficient historical data before {start_date}. "
                f"Need at least {max_lookback_n} periods, but only have {len(y_lookback)}."
            )
        
        # Create a copy of df_internal and replace target series with lookback data
        df_internal_lb = df_internal.copy()
        
        # Replace entire target series with lookback data
        df_internal_lb[self.target] = y_lookback
        
        # Build initial features
        X_full = self.dm.build_features(self.specs, df_internal_lb, df_mev)
        X_new = X_full[(X_full.index >= start_date) & (X_full.index <= end_date)]
        
        if X_new.empty:
            raise ValueError(f"No data found in time frame {time_frame}")
        
        # Get all indices for iteration
        prediction_indices = X_new.index.tolist()
        
        # Iterative prediction
        for idx in prediction_indices:
            # Predict for current index only
            X_current = X_new.loc[[idx]]
            y_hat = self.predict(X_current).iloc[0]
            
            # Update target series in df_internal_lb
            df_internal_lb.loc[idx, self.target] = y_hat
            
            # Rebuild features with updated target series
            X_full = self.dm.build_features(self.specs, df_internal_lb, df_mev)
            X_new = X_full[(X_full.index >= start_date) & (X_full.index <= end_date)]
        
        # Return the predicted target series from the time frame and the latest X_new
        predicted_series = df_internal_lb.loc[prediction_indices, self.target].copy()
        predicted_series.name = self.target
        
        return predicted_series, X_new

    @property
    def y_pred_out(self) -> pd.Series:
        """
        Out-of-sample predictions generated by calling predict on X_out.
        
        When has_lookback_var is True, uses rolling_predict to generate predictions
        that account for conditional variable effects.

        Returns empty Series if X_out is empty.
        """
        if self.X_out.empty:
            return pd.Series(dtype=float)
        
        # If model has lookback variables, use rolling_predict
        if self.has_lookback_var:
            # Get out-of-sample time frame
            oos_time_frame = (str(self.dm.out_sample_idx.min()), str(self.dm.out_sample_idx.max()))
            
            # Use rolling_predict to get predictions
            y_pred, _ = self.rolling_predict(
                df_internal=self.dm.internal_data,
                df_mev=self.dm.model_mev,
                y=self.y_full,
                time_frame=oos_time_frame
            )
            return y_pred
        
        return self.predict(self.X_out)

    @property
    def y_base_full(self) -> pd.Series:
        """
        Get full-sample target base series.
        
        Returns
        -------
        pd.Series
            Full-sample target base series including p0, in-sample, and out-sample indices.
        """
        if self.target_base is None:
            return pd.Series(dtype=float)
        
        # Get the union of p0, in-sample and out-of-sample indices
        idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        if self.dm.p0 is not None:
            idx = idx.union([self.dm.p0])
        
        # Extract target base
        y_base_full = self.dm.internal_data[self.target_base].copy()
        
        # Align to index
        y_base_full = y_base_full.reindex(idx).astype(float)
        
        return y_base_full

    @property
    def y_base_in(self) -> pd.Series:
        """
        Get in-sample target base series including p0.
        
        Returns
        -------
        pd.Series
            In-sample target base series including p0 if available.
        """
        if self.target_base is None:
            return pd.Series(dtype=float)
        
        # Get in-sample indices and add p0 if available
        idx = self.dm.in_sample_idx
        if self.dm.p0 is not None:
            idx = idx.union([self.dm.p0])
        
        return self.y_base_full.reindex(idx)

    @property
    def y_base_out(self) -> pd.Series:
        """
        Get out-of-sample target base series including out_p0.
        
        Returns
        -------
        pd.Series
            Out-of-sample target base series including out_p0 if available.
        """
        if self.target_base is None:
            return pd.Series(dtype=float)
        
        # Get out-of-sample indices and add p0 if available
        idx = self.dm.out_sample_idx
        if self.dm.out_p0 is not None:
            idx = idx.union([self.dm.out_p0])
        
        return self.y_base_full.reindex(idx)

    @property
    def y_exposure_full(self) -> pd.Series:
        """
        Get full-sample target exposure series.
        
        Returns
        -------
        pd.Series
            Full-sample target exposure series for in-sample and out-sample indices.
        """
        if self.target_exposure is None:
            return pd.Series(dtype=float)
        
        # Get the union of in-sample and out-of-sample indices
        idx = self.dm.in_sample_idx.union(self.dm.out_sample_idx)
        
        # Extract target exposure
        y_exposure_full = self.dm.internal_data[self.target_exposure].copy()
        
        # Align to index
        y_exposure_full = y_exposure_full.reindex(idx).astype(float)
        
        return y_exposure_full

    @property
    def y_base_fitted_in(self) -> Optional[pd.Series]:
        """
        Get in-sample fitted base predictions.
        
        For level models (RateLevel, BalanceLevel), directly returns y_fitted_in.
        For other models, uses base predictor to convert predictions to base variable.
        
        Returns
        -------
        Optional[pd.Series]
            In-sample fitted base predictions excluding p0, None if there are outliers 
            or no base predictor available.
        """
        # Return None if there are outliers (outlier_idx is not None and not empty)
        if self.outlier_idx:
            return None
            
        # For level models (RateLevel, BalanceLevel), return fitted values directly
        if self.model_type is not None:
            if self.model_type in {RateLevel, BalanceLevel}:
                if not hasattr(self, 'y_fitted_in') or self.y_fitted_in is None:
                    return pd.Series(dtype=float)
                return self.y_fitted_in
        
        # For other models, use base predictor
        if self.base_predictor is None or not hasattr(self, 'y_fitted_in') or self.y_fitted_in is None:
            return pd.Series(dtype=float)
        
        if self.dm.p0 is None:
            return pd.Series(dtype=float)
        
        try:
            base_predictions = self.base_predictor.predict_base(self.y_fitted_in, self.dm.p0)
            # Exclude p0 from the result
            if self.dm.p0 in base_predictions.index:
                base_predictions = base_predictions.drop(self.dm.p0)
            return base_predictions
        except Exception:
            # Return empty series if conversion fails
            return pd.Series(dtype=float)

    @property 
    def y_base_pred_out(self) -> Optional[pd.Series]:
        """
        Get out-of-sample base predictions.
        
        For level models (RateLevel, BalanceLevel), directly returns y_pred_out.
        For other models, uses base predictor to convert predictions to base variable.
        
        Returns
        -------
        Optional[pd.Series]
            Out-of-sample base predictions excluding out_p0, None if there are outliers
            or no base predictor available.
        """
        # Return None if there are outliers (outlier_idx is not None and not empty)
        if self.outlier_idx:
            return None
            
        # For level models (RateLevel, BalanceLevel), return predicted values directly
        if self.model_type is not None:
            if self.model_type in {RateLevel, BalanceLevel}:
                y_pred_out = self.y_pred_out
                if y_pred_out.empty:
                    return pd.Series(dtype=float)
                return y_pred_out
        
        # For other models, use base predictor
        if self.base_predictor is None or self.dm.out_p0 is None:
            return pd.Series(dtype=float)
        
        # Get out-of-sample predictions
        y_pred_out = self.y_pred_out
        if y_pred_out.empty:
            return pd.Series(dtype=float)
        
        try:
            base_predictions = self.base_predictor.predict_base(y_pred_out, self.dm.out_p0)
            # Exclude out_p0 from the result
            if self.dm.out_p0 in base_predictions.index:
                base_predictions = base_predictions.drop(self.dm.out_p0)
            return base_predictions
        except Exception:
            # Return empty series if conversion fails
            return pd.Series(dtype=float)

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
    def stability_test(self) -> Any:
        """
        Build and return the stability test instance using stability_test_cls.
        
        Creates a stability test instance (e.g., WalkForwardTest) using
        the model's configuration parameters. This enables model stability analysis
        through various testing methodologies.
        
        Returns
        -------
        ModelStabilityTest instance
            Stability test instance configured with this model's parameters.
            
        Raises
        ------
        ValueError
            If no stability_test_cls is provided or required parameters are not available.
            
        Example
        -------
        >>> # Create model and access stability test
        >>> model = OLS(dm=dm, specs=['GDP', 'UNRATE'], target='balance')
        >>> wft = model.stability_test
        >>> 
        >>> # Get stability metrics
        >>> stability_metrics = wft.get_stability_metrics()
        >>> print(f"R² stability: {stability_metrics['r_squared_stability']}")
        """
        if not self.stability_test_cls:
            raise ValueError("No stability_test_cls provided for building stability test.")
        
        # Check that required parameters are available
        if self.dm is None:
            raise ValueError("DataManager (dm) is required for stability testing.")
        if self.specs is None:
            raise ValueError("Feature specs are required for stability testing.")
        if self.target is None:
            raise ValueError("Target variable is required for stability testing.")
        
        # Extract model kwargs from current instance
        model_kwargs = {
            'sample': getattr(self, 'sample', 'in'),
            'outlier_idx': getattr(self, 'outlier_idx', []),
        }
        
        # Add optional parameters if they exist
        if hasattr(self, 'model_type') and self.model_type is not None:
            model_kwargs['model_type'] = self.model_type
        if hasattr(self, 'target_base') and self.target_base is not None:
            model_kwargs['target_base'] = self.target_base
        if hasattr(self, 'target_exposure') and self.target_exposure is not None:
            model_kwargs['target_exposure'] = self.target_exposure
        if hasattr(self, 'testset_func') and self.testset_func is not None:
            model_kwargs['testset_func'] = self.testset_func
        if hasattr(self, 'test_update_func') and self.test_update_func is not None:
            model_kwargs['test_update_func'] = self.test_update_func
        if hasattr(self, 'testset_cls') and self.testset_cls is not None:
            model_kwargs['testset_cls'] = self.testset_cls
        if hasattr(self, 'scen_cls') and self.scen_cls is not None:
            model_kwargs['scen_cls'] = self.scen_cls
        if hasattr(self, 'report_cls') and self.report_cls is not None:
            model_kwargs['report_cls'] = self.report_cls
        
        # Create stability test instance
        return self.stability_test_cls(
            model_cls=type(self),
            dm=self.dm,
            specs=self.specs,
            target=self.target,
            model_kwargs=model_kwargs
        )

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
                model=self
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
            - 'lookback_var': CondVar objects with lookback_n > 0
        """
        if self.specs is None:
            return {
                'CoefTest': [],
                'GroupTest': [],
                'StationarityTest': [],
                'SignCheck': [],
                'lookback_var': []
            }
        
        # Import here to avoid circular imports
        from .feature import DumVar, Feature
        
        coef_test_vars: List[str] = []
        group_test_vars: List[Tuple[str, ...]] = []
        stationarity_test_vars: List[str] = []
        sign_check_features: List[Feature] = []
        lookback_vars: List[Any] = []

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
                    
                    # Check for lookback_var eligibility (CondVar objects with lookback_n > 0)
                    if hasattr(spec, 'lookback_n') and spec.lookback_n > 0:
                        lookback_vars.append(spec)
                        
                else:
                    # String specifications
                    coef_test_vars.append(str(spec))
                    stationarity_test_vars.append(str(spec))

        return {
            'CoefTest': coef_test_vars,
            'GroupTest': group_test_vars,
            'StationarityTest': stationarity_test_vars,
            'SignCheck': sign_check_features,
            'lookback_var': lookback_vars
        }

    @property
    def has_lookback_var(self) -> bool:
        """
        Check if the model has lookback variables in spec_map.
        
        Returns
        -------
        bool
            True if spec_map contains lookback_var and the list is not empty.
        """
        lookback_vars = self.spec_map.get('lookback_var', [])
        return len(lookback_vars) > 0

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
        model_type: Optional[Type] = None,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None,
        report_cls: Type = OLS_ModelReport,
        stability_test_cls: Optional[Type] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        X_out: Optional[pd.DataFrame] = None,
        y_out: Optional[pd.Series] = None
    ):
        # Set default stability test class if not provided
        if stability_test_cls is None:
            from .stability import WalkForwardTest
            stability_test_cls = WalkForwardTest
            
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
            model_type=model_type,
            target_base=target_base,
            target_exposure=target_exposure,
            report_cls=report_cls,
            stability_test_cls=stability_test_cls,
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
        self.fvalue = None
        self.f_pvalue = None
        self.llf = None  # Log-likelihood
        # track covariance type
        self.cov_type: str = 'OLS'
        self.is_fitted = False
        # Additional attributes for conf_int, AIC, BIC
        self.conf_int_alpha: float = 0.05
        self.conf_int_df: Optional[pd.DataFrame] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None

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
        self.se = self.bse
        self.tvalues = res.tvalues
        self.fvalue = res.fvalue
        self.f_pvalue = res.f_pvalue
        self.llf = res.llf  # Store log-likelihood
        # VIF
        self.vif = pd.Series({
            col: variance_inflation_factor(Xc.values, i)
            for i, col in enumerate(Xc.columns)
        })
        self.is_fitted = True

        # Set initial conf_int, AIC, BIC
        self.conf_int_df = pd.DataFrame(
            self.fitted.conf_int(alpha=self.conf_int_alpha),
            index=self.params.index,
            columns=[0, 1]
        )
        self.aic = self.fitted.aic
        self.bic = self.fitted.bic

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
            self.se = self.bse
            self.tvalues = pd.Series(robust.tvalues, index=idx)
            self.pvalues = pd.Series(robust.pvalues, index=idx)
            self.fvalue = robust.fvalue
            self.f_pvalue = robust.f_pvalue
            self.llf = robust.llf  # Update log-likelihood after robust covariance estimation
            # Update conf_int, AIC, BIC after robust covariance estimation
            self.conf_int_df = pd.DataFrame(
                self.fitted.conf_int(alpha=self.conf_int_alpha),
                index=self.params.index,
                columns=[0, 1]
            )
            self.aic = self.fitted.aic
            self.bic = self.fitted.bic
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
    
    def predict_param_shock(self, X_new: pd.DataFrame, param: str, shock: int) -> pd.Series:
        """
        Predict with parameter shock testing by applying standard error shocks to coefficients.
        
        This method adjusts a specific parameter's coefficient by adding a multiple of its 
        standard error, then uses the adjusted coefficients to make predictions.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            Independent variables on which to make predictions.
        param : str
            Name of the parameter whose coefficient will be shocked.
        shock : int
            Number of standard errors to apply (e.g., 1, 2, -1, -2).
            Positive values increase the coefficient, negative values decrease it.
        
        Returns
        -------
        pd.Series
            Predictions using the shocked parameter coefficients.
        
        Example
        -------
        >>> # Predict with +1 standard error shock to 'GDP' parameter
        >>> predictions = model.predict_param_shock(X_new, 'GDP', 1)
        >>> 
        >>> # Predict with -2 standard error shock to 'UNRATE' parameter  
        >>> predictions = model.predict_param_shock(X_new, 'UNRATE', -2)
        """
        if not self.is_fitted or self.fitted is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        if param not in self.params.index:
            raise ValueError(f"Parameter '{param}' not found in model parameters.")
        
        # Get current coefficients and create a copy
        coeffs = self.params.copy()
        
        # Adjust the specified parameter's coefficient
        coeffs[param] = coeffs[param] + shock * self.se[param]
        
        # Add constant to X_new
        Xc_new = sm.add_constant(X_new, has_constant='add')
        
        # Make predictions using the adjusted coefficients
        predictions = Xc_new.dot(coeffs)
        
        return predictions
    
    def predict_input_shock(self, X_new: pd.DataFrame, param: str, shock: int, std: float) -> pd.Series:
        """
        Predict with input shock testing by applying standard deviation shocks to independent variable values.
        
        This method adjusts a specific independent variable's values by adding a multiple of its 
        standard deviation, then uses the adjusted input data to make predictions.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            Independent variables on which to make predictions.
        param : str
            Name of the independent variable whose values will be shocked.
        shock : int
            Number of standard deviations to apply (e.g., 1, 2, -1, -2).
            Positive values increase the input values, negative values decrease them.
        std : float
            Standard deviation of the parameter's input history.
        
        Returns
        -------
        pd.Series
            Predictions using the shocked input values.
        
        Example
        -------
        >>> # Predict with +1 standard deviation shock to 'GDP' input values
        >>> predictions = model.predict_input_shock(X_new, 'GDP', 1, 0.5)
        >>> 
        >>> # Predict with -2 standard deviation shock to 'UNRATE' input values  
        >>> predictions = model.predict_input_shock(X_new, 'UNRATE', -2, 0.3)
        """
        if not self.is_fitted or self.fitted is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        if param not in X_new.columns:
            raise ValueError(f"Parameter '{param}' not found in X_new columns.")
        
        # Create a copy of X_new to avoid modifying the original
        X_new_adjusted = X_new.copy()
        
        # Adjust the specified parameter's input values
        X_new_adjusted[param] = X_new_adjusted[param] + shock * std
        
        # Use the standard predict method with adjusted input
        return self.predict(X_new_adjusted)
    
    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compute confidence intervals for the fitted parameters.
        
        Args:
            alpha: The significance level for the confidence interval.
                  Default is 0.05 for 95% confidence intervals.
        
        Returns:
            DataFrame with confidence intervals for each parameter.
        """
        if not self.is_fitted or self.fitted is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Update stored alpha and confidence intervals if different
        if alpha != self.conf_int_alpha:
            self.conf_int_alpha = alpha
            self.conf_int_df = pd.DataFrame(
                self.fitted.conf_int(alpha=alpha),
                index=self.params.index,
                columns=[0, 1]
            )
        
        return self.conf_int_df
    
    @property
    def param_measures(self) -> pd.DataFrame:
        """
        Parameter measures: coefficient, pvalue, VIF, standard error, and confidence intervals.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter measures including coef, pvalue, vif, se, CI_2_5, 
            and CI_97_5.
        """
        if not self.is_fitted or self.fitted is None:
            return pd.DataFrame()
        
        # Create base parameter measures
        param_data = []
        for var in self.params.index:
            param_dict = {
                'variable': var,
                'coef': float(self.params[var]),
                'pvalue': float(self.pvalues[var]),
                'vif': float(self.vif.get(var, np.nan)) if var != 'const' else np.nan,
                'se': float(self.bse.get(var, np.nan))
            }
            
            # Add confidence intervals if available
            if self.conf_int_df is not None and var in self.conf_int_df.index:
                param_dict['CI_2_5'] = float(self.conf_int_df.loc[var, 0])
                param_dict['CI_97_5'] = float(self.conf_int_df.loc[var, 1])
            else:
                param_dict['CI_2_5'] = np.nan
                param_dict['CI_97_5'] = np.nan
            
            param_data.append(param_dict)
        
        # Create DataFrame
        df = pd.DataFrame(param_data)
        
        return df

    def __repr__(self) -> str:
        return f'OLS-{self.cov_type}'