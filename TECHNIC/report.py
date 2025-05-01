import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, Any
from .measure import *
from .plot import ols_model_perf_plot, ols_model_test_plot, ols_seg_perf_plot
from .cm import CM

class ModelReportBase(ABC):
    """
    Abstract base for model‐specific reports.
    Subclasses must implement methods to display in‑sample performance,
    out‑of‑sample performance, testing measures, parameter tables, and plotting.
    """
    def __init__(
        self,
        in_perf_measures: Dict[str, Any],
        out_perf_measures: Dict[str, Any],
        test_measures: Dict[str, Any],
        param_measures: Dict[str, Dict[str, Any]],
        perf_plot_fn: Callable[..., Any],
        test_plot_fn: Callable[..., Any]
    ):
        self.in_perf_measures = in_perf_measures
        self.out_perf_measures = out_perf_measures
        self.test_measures = test_measures
        self.param_measures = param_measures
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn

    @abstractmethod
    def show_in_perf_tbl(self) -> pd.DataFrame:
        """Return in‑sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Return out‑of‑sample performance measures as a DataFrame."""
        ...

    @abstractmethod
    def show_test_tbl(self) -> pd.DataFrame:
        """Return testing measures as a DataFrame."""
        ...

    @abstractmethod
    def show_params_tbl(self) -> pd.DataFrame:
        """Return parameter measures as a DataFrame."""
        ...

    @abstractmethod
    def plot_perf(self, **kwargs) -> Any:
        """Plot performance metrics; implementation must be provided by subclass."""
        ...

    def plot_tests(self, **kwargs) -> Any:
        """Render diagnostic plot via injected test_plot_fn."""
        return self.test_plot_fn(**kwargs)
    

class SegmentReportBase(ABC):
    """
    Abstract base for segment reports. Takes a dict of CM objects (keyed by model_id) and provides
    methods to display combined performance, out-of-sample results,
    testing measures, parameter summaries, and plotting across segments.
    """
    def __init__(
        self,
        cms: Dict[str, CM],
        perf_plot_fn: Callable[..., Any],
        test_plot_fn: Callable[..., Any]
    ):
        self.cms = cms
        self.perf_plot_fn = perf_plot_fn
        self.test_plot_fn = test_plot_fn

    @abstractmethod
    def show_perf_tbl(self) -> pd.DataFrame:
        """Return combined in‑sample performance across segments."""
        ...

    @abstractmethod
    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Return combined out‑of‑sample performance across segments."""
        ...

    @abstractmethod
    def show_test_tbl(self) -> pd.DataFrame:
        """Return combined test measures across segments."""
        ...

    @abstractmethod
    def show_params_tbl(self) -> pd.DataFrame:
        """Return combined parameter summaries across segments."""
        ...

    @abstractmethod
    def plot_perf(self, **kwargs) -> Any:
        """Render performance plots for each segment."""
        ...

    @abstractmethod
    def plot_tests(self, **kwargs) -> Any:
        """Render test diagnostic plots for each segment."""
        ...


class OLS_ModelReport(ModelReportBase):
    """
    OLS-specific report: implements display methods for in-sample performance,
    out-of-sample performance, testing measures, parameter tables, and plotting.
    """
    def __init__(self, measures):
        super().__init__(
            measure=measures,
            perf_plot_fn=ols_model_perf_plot,
            test_plot_fn=ols_model_test_plot
        )

    def show_in_perf_tbl(self) -> pd.DataFrame:
        """In-sample performance metrics as a single-row DataFrame."""
        return pd.DataFrame([self.measure.in_perf_measures])

    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Out-of-sample performance metrics as a single-row DataFrame (empty if none)."""
        out = self.measure.out_perf_measures
        return pd.DataFrame([out]) if out else pd.DataFrame()

    def show_test_tbl(self) -> pd.DataFrame:
        """In-sample testing measures as a single-row DataFrame."""
        return pd.json_normalize(self.measure.test_measures)

    def show_params_tbl(self) -> pd.DataFrame:
        """Parameter table with columns: Variable, Coef, Pvalue, Sig, VIF, Std."""
        pm = self.measure.param_measures
        df = pd.DataFrame.from_dict(pm, orient='index')
        df.index.name = 'Variable'
        df = df.reset_index()
        df = df.rename(columns={
            'coef': 'Coef',
            'pvalue': 'Pvalue',
            'sig': 'Sig',
            'vif': 'VIF',
            'std': 'Std'
        })
        cols = ['Variable', 'Coef', 'Pvalue', 'Sig', 'VIF', 'Std']
        cols_existing = [c for c in cols if c in df.columns]
        return df[cols_existing]

    def plot_perf(self, **kwargs) -> Any:
        """
        Plot actual vs. fitted (in-sample) and predicted (out-of-sample) values,
        with absolute errors as a bar chart (alpha=0.7).
        """
        # Build full target series
        y_list = [self.measure.y]
        if getattr(self.measure, 'y_out', None) is not None:
            y_list.append(self.measure.y_out)
        y_full = pd.concat(y_list).sort_index()

        return self.perf_plot_fn(
            self.measure.model,
            self.measure.X,
            y_full,
            X_out=getattr(self.measure, 'X_out', None),
            y_pred_out=getattr(self.measure, 'y_pred_out', None),
            **kwargs
        )

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

        if not getattr(self.measure, 'out_perf_measures', None):
            show_out = False

        print('=== In-Sample Performance ===')
        print(self.show_in_perf_tbl().to_string(index=False))

        if show_out:
            print('\n=== Out-of-Sample Performance ===')
            print(self.show_out_perf_tbl().to_string(index=False))

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

        print("\n=== Model Parameters ===")
        params_df = self.show_params_tbl()
        print(
            params_df.to_string(
                index=False,
                formatters={
                    'Coef':   fmt_coef,
                    'Pvalue': '{:.3f}'.format,
                    'VIF':    '{:.2f}'.format,
                    'Std':    fmt_std
                }
            )
        )

        # Performance plot
        fig1 = self.plot_perf(**(perf_kwargs or {}))
        plt.show()

        if show_tests:
            print('\n=== Test Metrics ===')
            print(self.show_test_tbl().to_string(index=False))
            fig2 = self.plot_tests(**(test_kwargs or {}))
            plt.show()



class OLS_SegmentReport:
    """
    Candidate-model-level reporting for OLS models. Stores per-CM measures
    (measure_in and measure_full) from CM instances. Excludes CMs with missing measures.
    Uses injected plotting functions for segment-level performance and diagnostics.
    """
    def __init__(self,
        cms: Dict[str, CM],
        seg_perf_plot_fn=ols_seg_perf_plot,
        seg_test_plot_fn=ols_model_test_plot
    ):
        """
        cms: dict mapping cm_id to CM objects.
        seg_perf_plot_fn: function to plot per-CM performance.
        seg_test_plot_fn: function to plot per-CM diagnostic tests.

        Only CMs with 'measure_in' or 'measure_full' attributes set are stored.
        Raises warnings for any cm_ids missing one of the measures.
        """
        missing_in, missing_full = [], []
        self.measures_in: Dict[str, Any] = {}
        self.measures_full: Dict[str, Any] = {}
        self.seg_perf_plot_fn = seg_perf_plot_fn
        self.seg_test_plot_fn = seg_test_plot_fn

        for cm_id, cm in cms.items():
            if hasattr(cm, 'measure_in') and cm.measure_in is not None:
                self.measures_in[cm_id] = cm.measure_in
            else:
                missing_in.append(cm_id)

            if hasattr(cm, 'measure_full') and cm.measure_full is not None:
                self.measures_full[cm_id] = cm.measure_full
            else:
                missing_full.append(cm_id)

        if missing_in:
            warnings.warn(
                f"CMs missing 'measure_in' and excluded: {missing_in}",
                UserWarning
            )
        if missing_full:
            warnings.warn(
                f"CMs missing 'measure_full' and excluded: {missing_full}",
                UserWarning
            )

    def show_in_perf_tbl(self) -> pd.DataFrame:
        """Combined in-sample performance across CMs."""
        rows = []
        for cm_id, m in self.measures_in.items():
            r = m.in_perf_measures.copy()
            r['cm_id'] = cm_id
            rows.append(r)
        df = pd.DataFrame(rows)
        if 'cm_id' in df.columns:
            cols = ['cm_id'] + [c for c in df.columns if c != 'cm_id']
            df = df[cols]
        return df

    def show_out_perf_tbl(self) -> pd.DataFrame:
        """Combined out-of-sample performance across CMs."""
        rows = []
        for cm_id, m in self.measures_full.items():
            r = m.out_perf_measures.copy()
            r['cm_id'] = cm_id
            rows.append(r)
        df = pd.DataFrame(rows)
        if 'cm_id' in df.columns:
            cols = ['cm_id'] + [c for c in df.columns if c != 'cm_id']
            df = df[cols]
        return df

    def show_test_tbl(self) -> pd.DataFrame:
        """Combine in-sample test diagnostics across candidate models."""
        rows = []
        for cm_id, m in self.measures_in.items():
            row = m.test_measures.copy()
            row['cm_id'] = cm_id
            rows.append(row)
        return pd.DataFrame(rows)

    def show_params_tbl(self) -> pd.DataFrame:
        """Combined in-sample parameter summaries across CMs."""
        rows = []
        for cm_id, m in self.measures_in.items():
            for var, stats in m.param_measures.items():
                sr = stats.copy()
                sr['Variable'] = var
                sr['cm_id'] = cm_id
                rows.append(sr)
        df = pd.DataFrame(rows)
        # Reorder columns: cm_id first, Variable second
        if 'cm_id' in df.columns and 'Variable' in df.columns:
            cols = ['cm_id', 'Variable'] + [c for c in df.columns if c not in ['cm_id', 'Variable']]
            df = df[cols]
        return df

    def plot_perf(self, **kwargs) -> Any:
        """Plot in-sample performance comparison across candidate models."""
        return self.seg_perf_plot_fn(self.measures_in, full=False, **kwargs)

    def plot_full_perf(self, **kwargs) -> Any:
        """Plot full-sample performance comparison across candidate models."""
        return self.seg_perf_plot_fn(self.measures_full, full=True, **kwargs)
    
    def plot_tests(self, **kwargs) -> Any:
        """Plot in-sample diagnostic tests comparison across candidate models."""
        return self.seg_test_plot_fn(self.measures_in, **kwargs)
    
    def show_report(
        self,
        show_out: bool = True,
        show_tests: bool = False,
        perf_kwargs: Dict[str, Any] = None,
        test_kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Display segment-level tables and plots, mirroring OLSReport.show_report structure.
        """
        perf_kwargs = perf_kwargs or {}
        test_kwargs = test_kwargs or {}

        # In-sample performance
        print('=== In-Sample Performance Across CMs ===')
        print(self.show_in_perf_tbl().to_string(index=False))

        # Out-of-sample performance
        if show_out and not self.show_out_perf_tbl().empty:
            print('\n=== Out-of-Sample Performance Across CMs ===')
            print(self.show_out_perf_tbl().to_string(index=False))

        # Parameters
        print('\n=== Parameter Summaries Across CMs ===')
        print(self.show_params_tbl().to_string(index=False))

        # Performance plot
        fig1 = self.plot_perf(**perf_kwargs)
        plt.show()

        # Full-sample plot
        if show_out:
            fig2 = self.plot_full_perf(**perf_kwargs)
            plt.show()

        # Diagnostic tests
        if show_tests:
            print('\n=== Diagnostic Tests Across CMs ===')
            print(self.show_test_tbl().to_string(index=False))
            fig3 = self.plot_tests(**test_kwargs)
            plt.show()