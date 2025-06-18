# =============================================================================
# module: cm.py
# Purpose: Candidate Model wrapper managing in-sample, out-of-sample,
#          and full-sample fitting, handling outliers, and exposing reports
# Dependencies: pandas, numpy, typing, .model.ModelBase, .feature.Feature,
#               .feature.DumVar, .scenario.ScenManager
# =============================================================================

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import Type, List, Dict, Any, Optional, Union, Tuple

from .model import ModelBase
from .feature import DumVar, Feature
from .scenario import ScenManager


class CM:
    """
    Candidate Model wrapper.

    Manages in-sample, out-of-sample, and full-sample fitting. Does not persist
    feature DataFrames internally; instead, passes data directly into model instances.
    Provides access to report and testset properties of the underlying models.
    Handles removal of outlier records from the in-sample dataset.
    Stores column names to reconstruct formula and representation.

    Attributes
    ----------
    model_id : str
        Unique identifier for this candidate model.
    target : str
        Name of the target column.
    model_cls : Type[ModelBase]
        ModelBase subclass used for fitting.
    dm : Any
        DataManager instance providing build_features() and internal_data.
    specs : List[Any]
        Cached feature specifications passed to DataManager.
    cols_in : List[str]
        Column names of in-sample features after build().
    cols_full : List[str]
        Column names of full-sample features after build().
    model_in : Optional[ModelBase]
        In-sample fitted model instance.
    model_full : Optional[ModelBase]
        Full-sample fitted model instance.
    scen_manager_in : Optional[ScenManager]
        Scenario manager for in-sample model (created during build()).
    scen_manager_full : Optional[ScenManager]
        Scenario manager for full-sample model (created during build()).
    """

    def __init__(
        self,
        model_id: str,
        target: str,
        model_cls: Type[ModelBase],
        data_manager: Any = None,
    ):
        """
        Initialize CM.

        Parameters
        ----------
        model_id : str
            Unique identifier for this candidate model.
        target : str
            Name of the target column in the DataManager's internal_data.
        model_cls : Type[ModelBase]
            A class that extends ModelBase, to be used for fitting.
        data_manager : Any, optional
            DataManager instance; if None, must be provided to build().
        """
        self.model_id = model_id
        self.target = target
        self.model_cls = model_cls
        self.dm = data_manager

        # Placeholders; we do NOT store large DataFrames to save memory
        self.specs: List[Any] = []
        self.cols_in: List[str] = []
        self.cols_full: List[str] = []
        self.model_in: Optional[ModelBase] = None
        self.model_full: Optional[ModelBase] = None
        self.outlier_idx: List[str] = []
        self.scen_manager_in: Optional[ScenManager] = None
        self.scen_manager_full: Optional[ScenManager] = None

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

    def build(
        self,
        specs: List[Union[str, Dict[str, Any]]],
        sample: str = 'in',
        data_manager: Any = None,
        outlier_idx: Optional[List[Any]] = None
    ) -> None:
        """
        Build features and target, validate data, split into in-/out-of-sample,
        remove specified outliers from in-sample data, store column names, and
        fit the model(s). After calling, model_in and/or model_full are defined.

        Parameters
        ----------
        specs : List[Union[str, Dict[str, Any]]]
            Feature specifications to pass to DataManager.build_features().
        sample : {'in', 'full', 'both'}, default 'in'
            Which sample(s) to build and fit:
            - 'in': fit only the in-sample model;
            - 'full': fit only the full-sample model;
            - 'both': fit both in-sample and full-sample models.
        data_manager : Any, optional
            If provided, overrides self.dm.
        outlier_idx : List[Any], optional
            List of index labels (e.g. timestamps or keys) corresponding to outlier
            records to remove from the in-sample data. If provided and `build_in`
            is True, each label must exist within the in-sample period; otherwise,
            a ValueError is raised.

        Raises
        ------
        ValueError
            If no DataManager is available or if `sample` is invalid,
            or if data validation fails, or if any provided outlier index is
            outside the in-sample period.
        """
        # Cache input specs
        self.specs = specs

        dm = data_manager or self.dm
        if dm is None:
            raise ValueError("No data_manager provided to CM.build().")
        if sample not in {'in', 'full', 'both'}:
            raise ValueError("sample must be 'in', 'full', or 'both'.")

        build_in = sample in {'in', 'both'}
        build_full = sample in {'full', 'both'}

        # Get the union of in-sample and out-of-sample indices
        idx = dm.in_sample_idx.union(dm.out_sample_idx)

        # Prepare full DataFrame X_full and Series y_full
        X_full = dm.build_features(specs)
        y_full = dm.internal_data[self.target].copy()

        # Align to index
        X_full = X_full.reindex(idx).astype(float)
        y_full = y_full.reindex(idx).astype(float)

        # Validate full-sample data
        self._validate_data(X_full, y_full)

        # Split into in-sample and out-of-sample using DataLoader indices
        X_in = X_full.loc[dm.in_sample_idx].copy()
        y_in = y_full.loc[dm.in_sample_idx].copy()
        X_out = X_full.loc[dm.out_sample_idx].copy()
        y_out = y_full.loc[dm.out_sample_idx].copy()

        # If in-sample fit is requested, remove specified outliers
        if build_in and outlier_idx:
            # 1) Convert outlier_idx to match X_in.index dtype
            if is_datetime64_any_dtype(X_in.index):
                converted_idx = pd.to_datetime(outlier_idx)
            else:
                converted_idx = outlier_idx

            # 2) Check whether every converted index exists in X_in.index
            missing = [i for i in converted_idx if i not in X_in.index]
            if missing:
                raise ValueError(f"Outlier indices {missing} not in in-sample period.")

            # 3) Drop those rows from X_in and y_in
            X_in = X_in.drop(index=converted_idx)
            y_in = y_in.drop(index=converted_idx)

            #4) Record outlier indices
            self.outlier_idx = outlier_idx

        # Store column names for representation/formula
        self.cols_in = list(X_in.columns)
        self.cols_full = list(X_full.columns)

        # Fit in-sample model (no model_id argument)
        if build_in:
            # Re-validate after dropping outliers
            self._validate_data(X_in, y_in)
            self.model_in = self.model_cls(
                X_in,
                y_in,
                X_out=X_out,
                y_out=y_out,
                spec_map=self.spec_map
            ).fit()
            
            # Create ScenManager for in-sample model
            self.scen_manager_in = ScenManager(
                dm=dm,
                model=self.model_in,
                specs=specs,
                target=self.target
            )
            # Attach ScenManager to model_in
            self.model_in.scen_manager = self.scen_manager_in

        # Fit full-sample model (no model_id argument)
        if build_full:
            self.model_full = self.model_cls(
                X_full,
                y_full,
                X_out=X_out,
                y_out=y_out,
                spec_map=self.spec_map
            ).fit()
            
            # Create ScenManager for full-sample model
            self.scen_manager_full = ScenManager(
                dm=dm,
                model=self.model_full,
                specs=specs,
                target=self.target
            )
            # Attach ScenManager to model_full
            self.model_full.scen_manager = self.scen_manager_full

    @property
    def spec_map(self) -> Dict[str, List[Union[str, Tuple[str, ...]]]]:
        """
        Categorize self.specs into 'common' and 'group' driver lists.

        Returns
        -------
        Dict[str, List[Union[str, Tuple[str, ...]]]]
            - 'common': flat list of column-name strings
            - 'group': list of tuples, each containing column-name strings
                       for a grouped driver set
        """
        common_drivers: List[str] = []
        group_drivers: List[Tuple[str, ...]] = []

        def _flatten(items):
            """Recursively flatten nested lists but leave tuples intact."""
            for item in items:
                if isinstance(item, list):
                    yield from _flatten(item)
                else:
                    yield item

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
                group_drivers.append(names)

            # 2) DumVar instance → one tuple of its dummy-column names
            elif isinstance(spec, DumVar):
                group_drivers.append(tuple(spec.output_names))

            # 3) Everything else → common driver(s)
            else:
                if isinstance(spec, Feature):
                    common_drivers.extend(spec.output_names)
                else:
                    common_drivers.append(str(spec))

        return {
            'common': common_drivers,
            'group': group_drivers
        }

    def __repr__(self) -> str:
        """
        Return a concise string representation of this CM, showing the formula.

        Uses the __repr__ of the in-sample model if available; otherwise,
        uses the __repr__ of the full-sample model. The formula portion
        (":target~C+...") remains unchanged.

        Returns
        -------
        str
            e.g., "UnderlyingModelRepr:target~C+var1+var2"
        """
        # Determine which underlying model repr to use
        if self.model_in is not None:
            prefix = self.model_in.__repr__()
        elif self.model_full is not None:
            prefix = self.model_full.__repr__()
        else:
            return f"<CM {self.model_id}: no model data>"

        # Build the formula string exactly as before
        if self.model_in is not None and self.cols_in:
            cols = self.cols_in
        elif self.model_full is not None and self.cols_full:
            cols = self.cols_full
        else:
            cols = []

        formula = f"{self.target}~C"
        if cols:
            formula += "+" + "+".join(cols)

        return f"{prefix}:{formula}"

    @property
    def formula(self) -> str:
        """
        Expose the formula string (same as __repr__).

        Returns
        -------
        str
            Formula representation, e.g., "Model1:target~C+var1+var2".
        """
        return self.__repr__()

    @property
    def report_in(self) -> Any:
        """
        Expose the in-sample report of the fitted in-sample model.

        Returns
        -------
        Any
            The in-sample report object from model_in.

        Raises
        ------
        RuntimeError
            If model_in is not yet built.
        """
        if self.model_in is None:
            raise RuntimeError("In-sample model not built; call build(sample='in' or 'both').")
        return self.model_in.report

    @property
    def report_full(self) -> Any:
        """
        Expose the full-sample report of the fitted full-sample model.

        Returns
        -------
        Any
            The full-sample report object from model_full.

        Raises
        ------
        RuntimeError
            If model_full is not yet built.
        """
        if self.model_full is None:
            raise RuntimeError("Full-sample model not built; call build(sample='full' or 'both').")
        return self.model_full.report

    @property
    def testset_in(self) -> Any:
        """
        Expose the in-sample testset of the fitted in-sample model.

        Returns
        -------
        Any
            The in-sample testset object from model_in.

        Raises
        ------
        RuntimeError
            If model_in is not yet built.
        """
        if self.model_in is None:
            raise RuntimeError("In-sample model not built; call build(sample='in' or 'both').")
        return self.model_in.testset

    @property
    def testset_full(self) -> Any:
        """
        Expose the full-sample testset of the fitted full-sample model.

        Returns
        -------
        Any
            The full-sample testset object from model_full.

        Raises
        ------
        RuntimeError
            If model_full is not yet built.
        """
        if self.model_full is None:
            raise RuntimeError("Full-sample model not built; call build(sample='full' or 'both').")
        return self.model_full.testset

    def show_report(
        self,
        show_full: bool = False,
        show_out: bool = True,
        show_tests: bool = False,
        show_scens: bool = False,
        perf_kwargs: Optional[Dict[str, Any]] = None,
        test_kwargs: Optional[Dict[str, Any]] = None,
        scen_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Display the in-sample report (and optionally full-sample report) with scenario plots.

        Parameters
        ----------
        show_full : bool, default False
            If True, also display the full-sample report.
        show_out : bool, default True
            If True, include out-of-sample performance in in-sample report.
        show_tests : bool, default False
            If True, include diagnostic test results.
        show_scens : bool, default False
            If True, display scenario forecast and variable plots.
        perf_kwargs : dict, optional
            Keyword arguments passed to performance plotting methods.
        test_kwargs : dict, optional
            Keyword arguments passed to test-result display methods.
        scen_kwargs : dict, optional
            Keyword arguments passed to scenario plotting methods.

        Raises
        ------
        RuntimeError
            If the requested report (in or full) is not yet available.
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}
        scen_kwargs = scen_kwargs or {}

        # In-sample report
        if self.report_in is None:
            raise RuntimeError("report_in is not defined. Call build(sample='in' or 'both').")
        self.report_in.show_report(
            show_out=show_out,
            show_tests=show_tests,
            perf_kwargs=perf_kwargs,
            test_kwargs=test_kwargs
        )

        # Full-sample report
        if show_full:
            if self.report_full is None:
                raise RuntimeError("report_full is not defined. Call build(sample='full' or 'both').")
            self.report_full.show_report(
                show_out=False,
                show_tests=show_tests,
                perf_kwargs=perf_kwargs,
                test_kwargs=test_kwargs
            )

        # Scenario plots
        if show_scens:
            # Plot scenarios for in-sample model
            if hasattr(self, 'scen_manager_in') and self.scen_manager_in is not None:
                print(f"\n=== Model: {self.model_id} — Scenario Analysis ===")
                try:
                    figures = self.scen_manager_in.plot_all(**scen_kwargs)
                    # Display each figure immediately after creation
                    for scen_set, plot_dict in figures.items():
                        print(f"Scenario plots for {scen_set} generated successfully.")
                        for plot_type, fig in plot_dict.items():
                            plt.show()
                except Exception as e:
                    print(f"Error generating in-sample scenario plots: {e}")
            
            # Plot scenarios for full-sample model if requested
            if show_full and hasattr(self, 'scen_manager_full') and self.scen_manager_full is not None:
                print(f"\n=== Model: {self.model_id} — Full-Sample Scenario Analysis ===")
                try:
                    figures = self.scen_manager_full.plot_all(**scen_kwargs)
                    # Display each figure immediately after creation
                    for scen_set, plot_dict in figures.items():
                        print(f"Full-sample scenario plots for {scen_set} generated successfully.")
                        for plot_type, fig in plot_dict.items():
                            plt.show()
                except Exception as e:
                    print(f"Error generating full-sample scenario plots: {e}")
            
            # If no scenario managers are available, inform the user
            if not (hasattr(self, 'scen_manager_in') and self.scen_manager_in is not None):
                if not show_full or not (hasattr(self, 'scen_manager_full') and self.scen_manager_full is not None):
                    print("\nNo scenario managers available. Scenario data may not be loaded in DataManager.")
