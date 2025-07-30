# TECHNIC/report.py
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Dict
from .plot import *

class ModelReportBase(ABC):
    """
    Abstract base for model-specific reports, now initialized with a ModelBase.

    Parameters
    ----------
    model : ModelBase
        Fitted ModelBase instance containing data, predictions, and measures.
    perf_set_plot_fn : callable, optional
        Function to plot performance across multiple reports.
    """
    def __init__(
        self,
        model: Any,
        perf_set_plot_fn: Optional[Callable[[Dict[str, 'ModelReportBase']], Any]] = None
    ):
        # Store model for attribute access
        self.model = model
        # Optional performance-set plotting function
        self.perf_set_plot_fn = perf_set_plot_fn

    def show_test_tbl(self) -> None:
        """
        Print all test results from the model's TestSet in a reader-friendly format.
        """
        # Gather all test results (both active and inactive)
        results = self.model.testset.all_test_results
        for test_name, result in results.items():
            print(f"--- {test_name} ---")
            # Print DataFrame or Series, fallback to default print
            if hasattr(result, 'to_string'):
                print(result.to_string())
            else:
                print(result)
            print()

    def show_in_perf_tbl(self) -> DataFrame:
        """
        Return in-sample performance measures as a DataFrame.
        
        Converts the Series returned by model.in_perf_measures to a single-row DataFrame.
        
        Returns
        -------
        DataFrame
            Single-row DataFrame with performance measures as columns.
        """
        perf_series = self.model.in_perf_measures
        if perf_series.empty:
            return pd.DataFrame()
        return pd.DataFrame([perf_series])

    def show_out_perf_tbl(self) -> DataFrame:
        """
        Return out-of-sample performance measures as a DataFrame.
        
        Converts the Series returned by model.out_perf_measures to a single-row DataFrame.
        
        Returns
        -------
        DataFrame
            Single-row DataFrame with out-of-sample performance measures as columns.
            Empty DataFrame if no out-of-sample data available.
        """
        perf_series = self.model.out_perf_measures
        if perf_series.empty:
            return pd.DataFrame()
        return pd.DataFrame([perf_series])

    @abstractmethod
    def show_params_tbl(self) -> DataFrame:
        """Return parameter measures as a DataFrame."""
        ...

    def plot_perf(self, **kwargs) -> Any:
        """Plot performance metrics for this model."""
        return ols_model_perf_plot(model=self.model, **kwargs)

class ReportSet:
    """
    Aggregates multiple ModelReportBase instances into consolidated performance tables and plots.

    :param reports: dict mapping model_id to ModelReportBase.
    """
    def __init__(
        self,
        reports: Dict[str, ModelReportBase]
    ):
        if not isinstance(reports, dict) or not all(isinstance(r, ModelReportBase) for r in reports.values()):
            raise TypeError("`reports` must be a dict mapping model_id to ModelReportBase instances.")
        self._reports = reports

    def show_in_perf_set(self, **kwargs) -> pd.DataFrame:
        """
        Return a DataFrame with each model's in-sample performance as separate rows,
        keeping the same columns from individual reports and using model_id as index.
        """
        dfs = []
        for mid, rpt in self._reports.items():
            df = rpt.show_in_perf_tbl(**kwargs).copy()
            df.index = [mid] * len(df)
            dfs.append(df)
        if dfs:
            result = pd.concat(dfs, axis=0)
            result.index.name = 'Model'
            return result
        return pd.DataFrame()

    def show_out_perf_set(self, **kwargs) -> pd.DataFrame:
        """
        Return a DataFrame with each model's out-of-sample performance as separate rows,
        keeping the same columns from individual reports and using model_id as index.
        """
        dfs = []
        for mid, rpt in self._reports.items():
            df = rpt.show_out_perf_tbl(**kwargs).copy()
            df.index = [mid] * len(df)
            dfs.append(df)
        if dfs:
            result = pd.concat(dfs, axis=0)
            result.index.name = 'Model'
            return result
        return pd.DataFrame()

    def plot_perf_set(
        self,
        plot_fn: Callable[[Dict[str, ModelReportBase]], Any] = None,
        **kwargs
    ) -> Any:
        """
        Delegate to a plot function (defaulting to each report's own perf_set_plot_fn)
        that accepts the full mapping of model_id -> ModelReportBase and any additional kwargs.
        """
        # Use provided plot_fn, or default to the first report's perf_set_plot_fn
        fn = plot_fn
        if fn is None:
            # assume all reports share the same default, take from first
            fn = next(iter(self._reports.values())).perf_set_plot_fn
        return fn(self._reports, **kwargs)
    
    def show_report(
        self,
        show_out: bool = True,
        show_params: bool = False,
        show_tests: bool = False,
        show_scens: bool = False,
        perf_kwargs: dict = None,
        params_kwargs: dict = None,
        test_kwargs: dict = None,
        scen_kwargs: dict = None
    ) -> None:
        """
        Sequentially display:
          1) In-sample performance (always)
          2) Out-of-sample performance (if show_out)
          3) Performance plot
          4) Parameter tables per model (if show_params)
          5) Testing tables per model (if show_tests)
          6) Scenario plots per model (if show_scens)
        """
        perf_kwargs   = perf_kwargs or {}
        params_kwargs = params_kwargs or {}
        test_kwargs   = test_kwargs or {}
        scen_kwargs   = scen_kwargs or {}

        # 1) In-sample performance
        print("=== In-Sample Performance ===")
        df_in = self.show_in_perf_set(**perf_kwargs)
        print(df_in.to_string(float_format='{:.3f}'.format))

        # 2) Optional out-of-sample performance
        if show_out:
            print("\n=== Out-of-Sample Performance ===")
            df_out = self.show_out_perf_set(**perf_kwargs)
            print(df_out.to_string(float_format='{:.3f}'.format))

        # 3) Performance plot
        print("\n=== Performance Plot ===")
        fig = self.plot_perf_set(**perf_kwargs)
        plt.show()

        # 4) Optional parameter tables per model
        if show_params:
            for model_id, report in self._reports.items():
                print(f"\n=== Model: {model_id} — Parameters ===")
                df_params = report.show_params_tbl(**params_kwargs)
                print(df_params.to_string())

        # 5) Optional per-model testing metrics
        if show_tests:
            for model_id, report in self._reports.items():
                print(f"\n=== Model: {model_id} — Testing Metrics ===\n")
                report.show_test_tbl(**test_kwargs)

        # 6) Optional per-model scenario plots
        if show_scens:
            for model_id, report in self._reports.items():
                if hasattr(report.model, 'scen_manager') and report.model.scen_manager is not None:
                    print(f"\n=== Model: {model_id} — Scenario Analysis ===")
                    try:
                        figures = report.model.scen_manager.plot_all(**scen_kwargs)
                        # Display each figure immediately after creation
                        for scen_set, plot_dict in figures.items():
                            print(f"Scenario plots for {scen_set} generated successfully.")
                            for plot_type, fig in plot_dict.items():
                                plt.show()
                    except Exception as e:
                        print(f"Error generating scenario plots for {model_id}: {e}")
                else:
                    print(f"\n=== Model: {model_id} — No Scenario Manager Available ===")
                    print("Scenario data may not be loaded or model not built through CM.")


class OLS_ModelReport(ModelReportBase):
    """
    Report for OLS models: displays performance, tests, and parameter tables.
    """
    def __init__(
        self,
        model: Any,
        perf_plot_fn: Callable[['ModelReportBase'], Any] = ols_model_perf_plot,
        test_plot_fn: Callable[['ModelReportBase'], Any] = ols_model_test_plot,
        perf_set_plot_fn: Callable[[Dict[str, 'ModelReportBase']], Any] = ols_plot_perf_set
    ):
        super().__init__(model=model, perf_set_plot_fn=perf_set_plot_fn)
        # Store specific plot routines
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn



    def show_params_tbl(self) -> pd.DataFrame:
        """
        Parameter table with columns: Variable, Coef, Pvalue, VIF, SE, CI_2_5, CI_97_5.
        
        Returns
        -------
        pd.DataFrame
            Parameter table with all available measures.
        """
        # Get the param_measures DataFrame
        df = self.model.param_measures
        
        if df.empty:
            return pd.DataFrame()
        
        # Rename columns for display
        column_mapping = {
            'variable': 'Variable',
            'coef': 'Coef',
            'pvalue': 'Pvalue',
            'vif': 'VIF',
            'se': 'SE',
            'CI_2_5': 'CI_2_5',
            'CI_97_5': 'CI_97_5'
        }
        
        # Apply column mapping for existing columns
        df = df.rename(columns=column_mapping)
        
        # Define the order of columns to display
        display_cols = ['Variable', 'Coef', 'Pvalue', 'VIF', 'SE', 'CI_2_5', 'CI_97_5']
        
        # Select only existing columns
        existing_cols = [col for col in display_cols if col in df.columns]
        
        return df[existing_cols]



    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: dict = None,
        test_kwargs: dict = None
    ) -> None:
        """
        Display report sections sequentially:
          1) In-sample performance
          2) Optional out-of-sample performance
          3) Parameter table
          4) In-sample performance plot
          5) Optional testing metrics & plot
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        if self.model.out_perf_measures.empty:
            show_out = False

        print('=== In-Sample Performance ===')
        in_perf_df = self.show_in_perf_tbl()
        print(in_perf_df.to_string(index=False, float_format='{:.3f}'.format))

        if show_out:
            print('\n=== Out-of-Sample Performance ===')
            out_perf_df = self.show_out_perf_tbl()
            print(out_perf_df.to_string(index=False, float_format='{:.3f}'.format))

        # Parameters
        def fmt_coef(x):
            try:
                val = float(x)
            except:
                return str(x)
            if abs(val) >= 1e5 or (abs(val) > 0 and abs(val) < 1e-3):
                return f"{val:.4e}"
            return f"{val:.4f}"

        def fmt_std(x):
            try:
                val = float(x)
            except:
                return str(x)
            if abs(val) >= 1e5 or (abs(val) > 0 and abs(val) < 1e-3):
                return f"{val:.4e}"
            return f"{val:.4f}"

        def fmt_conf_int(x):
            try:
                val = float(x)
            except:
                return str(x)
            if pd.isna(val):
                return "nan"
            if abs(val) >= 1e5 or (abs(val) > 0 and abs(val) < 1e-3):
                return f"{val:.4e}"
            return f"{val:.4f}"

        print("\n=== Model Parameters ===")
        params_df = self.show_params_tbl()
        
        # Create formatters dictionary with all possible columns
        formatters = {
            'Coef': fmt_coef,
            'Pvalue': '{:.3f}'.format,
            'VIF': '{:.2f}'.format,
            'SE': fmt_std,
            'CI_2_5': '{:.4f}'.format,
            'CI_97_5': '{:.4f}'.format
        }
        
        # Filter formatters to only include columns that exist in the DataFrame
        existing_formatters = {col: formatters[col] for col in formatters if col in params_df.columns}
        
        print(
            params_df.to_string(
                index=False,
                formatters=existing_formatters
            )
        )

        # Performance plot
        fig1 = self.plot_perf(**(perf_kwargs or {}))
        plt.show()

        if show_tests:
            print('\n=== Model Testing ===')
            self.show_test_tbl()