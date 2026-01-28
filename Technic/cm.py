# =============================================================================
# module: cm.py
# Purpose: Candidate Model wrapper managing in-sample, out-of-sample, and full-sample fitting.
# Key Types/Classes: CM
# Key Functions: build, bind_data_manager
# Dependencies: pandas, numpy, matplotlib.pyplot, typing, .model.ModelBase, .feature.Feature,
#               .feature.DumVar, .scenario.ScenManager
# =============================================================================

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import copy
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import Type, List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt

from .model import ModelBase
from .feature import DumVar, Feature
from .scenario import ScenManager

if TYPE_CHECKING:
    from .data import DataManager


class CM:
    """
    Candidate Model wrapper.

    Manages creation and coordination of in-sample, out-of-sample, and full-sample
    models. This class serves as a factory for creating ModelBase instances and
    provides unified access to their reports and test results.

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
    model_in : Optional[ModelBase]
        In-sample fitted model instance.
    model_full : Optional[ModelBase]
        Full-sample fitted model instance.
    scen_cls : Type
        Class to use for scenario management.
    scen_manager_in : Optional[ScenManager]
        Scenario manager for in-sample model (from model_in.scen_manager).
    scen_manager_full : Optional[ScenManager]
        Scenario manager for full-sample model (from model_full.scen_manager).
    outlier_idx : List[str]
        List of outlier indices that were removed from in-sample data.
    """

    def __init__(
        self,
        model_id: str,
        target: str,
        model_type: Optional[Any] = None,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None,
        model_cls: Type[ModelBase] = None,
        data_manager: Any = None,
        scen_cls: Type = None,
        qtr_method: str = 'mean',
    ):
        """
        Initialize CM.

        Parameters
        ----------
        model_id : str
            Unique identifier for this candidate model.
        target : str
            Name of the target column in the DataManager's internal_data.
        target_base : str, optional
            Name of the base variable of interest (highly recommended if available).
        target_exposure : str, optional
            Name of the exposure variable (required for Ratio model types).
        model_cls : Type[ModelBase]
            A class that extends ModelBase, to be used for fitting.
        data_manager : Any, optional
            DataManager instance; if None, must be provided to build().
        scen_cls : Type, optional
            Class to use for scenario management. If None, defaults to ScenManager.
        """
        self.model_id = model_id
        self.target = target
        self.model_type = model_type
        self.target_base = target_base
        self.target_exposure = target_exposure
        self.model_cls = model_cls
        self.dm = data_manager
        self.qtr_method = qtr_method
        
        # Import and set default ScenManager if not provided
        if scen_cls is None:
            self.scen_cls = ScenManager
        else:
            self.scen_cls = scen_cls

        # Model instances
        self.model_in: Optional[ModelBase] = None
        self.model_full: Optional[ModelBase] = None
        
        # Cached specs and outlier indices
        self.specs: List[Any] = []
        self.outlier_idx: List[str] = []

    def bind_data_manager(self, dm: "DataManager") -> None:
        """
        Attach a DataManager instance to the candidate model and its fitted models.

        Parameters
        ----------
        dm : DataManager
            The DataManager instance to bind to this candidate model.

        Raises
        ------
        ValueError
            If ``dm`` is ``None``.
        """
        if dm is None:
            raise ValueError("Cannot bind a null DataManager to CM.")

        # Store the data manager on the CM and propagate it to fitted models.
        self.dm = dm
        for model in (self.model_in, self.model_full):
            if model is not None and hasattr(model, "dm"):
                model.dm = dm

    def build(
        self,
        specs: List[Union[str, Dict[str, Any]]],
        sample: str = 'in',
        outlier_idx: Optional[List[Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Build model instance(s) using the specified sample type.

        Parameters
        ----------
        specs : List[Union[str, Dict[str, Any]]]
            Feature specifications to pass to DataManager.build_features().
        sample : {'in', 'full', 'both'}, default 'in'
            Which sample(s) to build and fit:
            - 'in': create only the in-sample model;
            - 'full': create only the full-sample model;
            - 'both': create both in-sample and full-sample models.
        outlier_idx : List[Any], optional
            List of index labels (e.g. timestamps or keys) corresponding to outlier
            records to remove from the in-sample data.

        Raises
        ------
        ValueError
            If no DataManager is available or if `sample` is invalid.
        """
        # Cache input specs and outlier indices
        self.specs = specs
        self.outlier_idx = outlier_idx or []
        model_kwargs = model_kwargs or {}

        if self.dm is None:
            raise ValueError("No DataManager available for CM.build().")
        if sample not in {'in', 'full', 'both'}:
            raise ValueError("sample must be 'in', 'full', or 'both'.")

        build_in = sample in {'in', 'both'}
        build_full = sample in {'full', 'both'}

        # Create in-sample model if requested
        if build_in:
            self.model_in = self.model_cls(
                dm=self.dm,
                specs=specs,
                sample='in',
                outlier_idx=outlier_idx,
                target=self.target,
                model_type=self.model_type,
                target_base=self.target_base,
                target_exposure=self.target_exposure,
                scen_cls=self.scen_cls,
                qtr_method=self.qtr_method,
                **model_kwargs
            ).fit()

        # Create full-sample model if requested
        if build_full:
            self.model_full = self.model_cls(
                dm=self.dm,
                specs=specs,
                sample='full',
                outlier_idx=outlier_idx,
                target=self.target,
                model_type=self.model_type,
                target_base=self.target_base,
                target_exposure=self.target_exposure,
                scen_cls=self.scen_cls,
                qtr_method=self.qtr_method,
                **model_kwargs
            ).fit()

    @property
    def scen_manager_in(self) -> Optional[ScenManager]:
        """
        Get the scenario manager from the in-sample model.

        Returns
        -------
        Optional[ScenManager]
            The scenario manager from model_in, or None if not available.
        """
        if self.model_in is not None:
            return self.model_in.scen_manager
        return None

    @property
    def scen_manager_full(self) -> Optional[ScenManager]:
        """
        Get the scenario manager from the full-sample model.

        Returns
        -------
        Optional[ScenManager]
            The scenario manager from model_full, or None if not available.
        """
        if self.model_full is not None:
            return self.model_full.scen_manager
        return None

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare a picklable CM state without DataManager references.

        Returns
        -------
        Dict[str, Any]
            Dictionary representing the CM state with DataManager references
            removed from the CM and any fitted models.
        """
        state = self.__dict__.copy()
        state["dm"] = None

        # Strip DataManager references from fitted models without mutating
        # the live instances in memory.
        for attr in ("model_in", "model_full"):
            model = state.get(attr)
            if model is not None and hasattr(model, "dm"):
                model_copy = copy.copy(model)
                model_copy.dm = None
                state[attr] = model_copy

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore CM state after unpickling.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary produced by :meth:`__getstate__`.
        """
        self.__dict__.update(state)
        self.dm = None

    def __repr__(self) -> str:
        """
        Return a concise string representation of this CM, showing the formula.

        Uses the __repr__ of the in-sample model if available; otherwise,
        uses the __repr__ of the full-sample model. The formula portion
        (":target~C+...") is built from the model's feature columns.

        Returns
        -------
        str
            e.g., "ModelRepr:target~C+var1+var2"
        """
        # Determine which underlying model repr to use
        if self.model_in is not None:
            prefix = self.model_in.__repr__()
            cols = list(self.model_in.X.columns)
        elif self.model_full is not None:
            prefix = self.model_full.__repr__()
            cols = list(self.model_full.X.columns)
        else:
            return f"<CM {self.model_id}: no model data>"

        # Build the formula string
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
        show_sens: bool = False,
        show_stab: bool = False,
        show_backtest: bool = False,
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
        show_sens : bool, default False
            If True, display sensitivity testing plots for all scenarios.
        show_stab : bool, default False
            If True, display stability test results using the in-sample model's stability test.
        show_backtest : bool, default False
            If True, display rolling in-sample backtesting summaries.
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
            if self.scen_manager_in is not None:
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
            if show_full and self.scen_manager_full is not None:
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
            if not self.scen_manager_in and (not show_full or not self.scen_manager_full):
                print("\nNo scenario managers available. Scenario data may not be loaded in DataManager.")

        # Sensitivity testing plots
        if show_sens:
            # Run sensitivity testing for in-sample model
            if self.scen_manager_in is not None:
                print(f"\n=== Model: {self.model_id} — Sensitivity Analysis ===")
                try:
                    self.scen_manager_in.sens_test.plot_all()
                except Exception as e:
                    print(f"Error generating in-sample sensitivity plots: {e}")
            
            # Run sensitivity testing for full-sample model if requested
            if show_full and self.scen_manager_full is not None:
                print(f"\n=== Model: {self.model_id} — Full-Sample Sensitivity Analysis ===")
                try:
                    self.scen_manager_full.sens_test.plot_all()
                except Exception as e:
                    print(f"Error generating full-sample sensitivity plots: {e}")
            
            # If no scenario managers are available, inform the user
            if not self.scen_manager_in and (not show_full or not self.scen_manager_full):
                print("\nNo scenario managers available for sensitivity testing. Scenario data may not be loaded in DataManager.")

        # Stability testing
        if show_stab:
            # In-sample stability testing
            if self.model_in is not None:
                print(f"\n=== Model: {self.model_id} — Model Stability Analysis ===")
                try:
                    self.model_in.stability_test.show_all()
                except Exception as e:
                    print(f"Error generating in-sample stability test results: {e}")
            else:
                print("\nNo in-sample model available for stability testing. Call build(sample='in' or 'both').")
            
            # Full-sample stability testing (if show_full is True)
            if show_full and self.model_full is not None:
                print(f"\n=== Model: {self.model_id} — Model Stability Analysis ===")
                try:
                    self.model_full.stability_test.show_all()
                except Exception as e:
                    print(f"Error generating full-sample stability test results: {e}")
            elif show_full and self.model_full is None:
                print("\nNo full-sample model available for stability testing. Call build(sample='full' or 'both').")

        # Backtesting results
        if show_backtest:
            def _print_backtest_summary(model, label: str) -> None:
                try:
                    backtest = model.backtesting_test
                    results_df = getattr(backtest, 'results_df', None)
                    if results_df is None or results_df.empty:
                        print(f"\n=== Model: {self.model_id} — {label} Backtesting ===")
                        print("No backtesting results available.")
                        return
                    routes = results_df['route'].dropna().unique().tolist()
                    print(f"\n=== Model: {self.model_id} — {label} Backtesting ===")
                    print(f"Routes: {', '.join(routes)}")
                    print(f"Rows: {len(results_df):,}")
                except Exception as e:
                    print(f"Error generating {label.lower()} backtesting results: {e}")

            if self.model_in is not None:
                _print_backtest_summary(self.model_in, "In-Sample")
            else:
                print("\nNo in-sample model available for backtesting. Call build(sample='in' or 'both').")

            if show_full:
                if self.model_full is not None:
                    _print_backtest_summary(self.model_full, "Full-Sample")
                else:
                    print("\nNo full-sample model available for backtesting. Call build(sample='full' or 'both').")
