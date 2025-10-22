# =============================================================================
# module: plot.py
# Purpose: Visualization helpers for OLS diagnostics and performance reporting.
# Key Types/Classes: None
# Key Functions: ols_model_perf_plot, ols_model_test_plot, ols_plot_perf_set
# Dependencies: numpy, pandas, matplotlib.pyplot, statsmodels.api, statsmodels.stats
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _infer_typical_gap(sorted_index: pd.Index) -> Union[pd.Timedelta, float]:
    """
    Estimate the typical spacing between consecutive index values.

    Parameters
    ----------
    sorted_index : pd.Index
        Index values sorted in ascending order.

    Returns
    -------
    Union[pd.Timedelta, float]
        Median difference between consecutive index entries. Time-based indexes
        yield ``pd.Timedelta`` values, whereas numeric indexes return floats.
    """
    if len(sorted_index) <= 1:
        if isinstance(sorted_index, pd.DatetimeIndex):
            return pd.Timedelta(days=30)
        return 1.0

    diffs = sorted_index.to_series().diff().dropna()
    if diffs.empty:
        if isinstance(sorted_index, pd.DatetimeIndex):
            return pd.Timedelta(days=30)
        return 1.0

    median_diff = diffs.median()
    if pd.isna(median_diff):
        if isinstance(sorted_index, pd.DatetimeIndex):
            return pd.Timedelta(days=30)
        return 1.0

    if isinstance(median_diff, np.timedelta64):
        return pd.to_timedelta(median_diff)

    if isinstance(median_diff, pd.Timedelta):
        return median_diff

    return float(median_diff)


def ols_model_perf_plot(
    model: Optional['OLS'] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    X_out: Optional[pd.DataFrame] = None,
    y_out: Optional[pd.Series] = None,
    y_fitted_in: Optional[pd.Series] = None,
    y_pred_out: Optional[pd.Series] = None,
    figsize: tuple = (8, 4),
    **kwargs
) -> plt.Figure:
    """
    Plot actual vs. fitted (in-sample) and predicted (out-of-sample) values,
    with a secondary bar chart of absolute errors. When a model is supplied,
    periods flagged as outliers are rendered as gaps in the actual series.

    Parameters
    ----------
    model : OLS, optional
        Fitted OLS model instance. If provided, data will be extracted from this model.
    X : pd.DataFrame, optional
        In-sample feature DataFrame used for fitting. Ignored if model is provided.
    y : pd.Series, optional
        In-sample target values. Ignored if model is provided.
    X_out : pd.DataFrame, optional
        Out-of-sample feature DataFrame for predictions. Ignored if model is provided.
    y_out : pd.Series, optional
        Actual target values for out-of-sample. Ignored if model is provided.
    y_fitted_in : pd.Series, optional
        Fitted in-sample values. Ignored if model is provided.
    y_pred_out : pd.Series, optional
        Predicted out-of-sample values. Ignored if model is provided.
    figsize : tuple, default (8,4)
        Figure size.
    **kwargs
        Additional kwargs passed to plt.subplots().

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with target and base variable performance plots.

    Raises
    ------
    ValueError
        If the supplied model is not fitted or required data arrays are missing.
    """
    # Extract data from model if provided
    if model is not None:
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Extract data from model
        X = model.X
        y = model.y
        X_out = model.X_out if hasattr(model, 'X_out') else None
        y_out = model.y_out if hasattr(model, 'y_out') else None
        y_fitted_in = model.y_fitted_in if hasattr(model, 'y_fitted_in') else None
        y_pred_out = model.y_pred_out if hasattr(model, 'y_pred_out') else None
    
    # Validate required parameters
    if X is None or y is None:
        raise ValueError("Either model or X and y must be provided")
    
    # Determine full actual series; use model-provided view when available so that
    # outlier periods are represented as ``NaN`` (rendered as gaps on the plot).
    if model is not None and hasattr(model, 'y_full'):
        y_full = model.y_full.sort_index()
    else:
        y_sorted = y.sort_index()
        if y_out is not None:
            y_full = pd.concat([y_sorted, y_out.sort_index()]).sort_index()
        else:
            y_full = y_sorted

    # In-sample fitted series aligned to in-sample index (including outliers).
    if y_fitted_in is None:
        raise ValueError("y_fitted_in must be provided for in-sample fitted values")
    y_fitted = y_fitted_in.sort_index()

    if (
        model is not None
        and hasattr(model, 'dm')
        and getattr(model.dm, 'in_sample_idx', None) is not None
    ):
        in_sample_index = pd.Index(model.dm.in_sample_idx).sort_values()
    else:
        in_sample_index = pd.Index(y_fitted.index).sort_values()

    y_fitted_aligned = y_fitted.reindex(in_sample_index)

    # Out-of-sample predictions
    if X_out is not None:
        if y_pred_out is None:
            raise ValueError("y_pred_out must be provided when X_out is not None")
        y_pred = y_pred_out.sort_index()
    else:
        y_pred = pd.Series(dtype=float)

    # Combine fitted and predicted series and align to full actual index
    y_pred_components = []
    if not y_fitted_aligned.empty:
        y_pred_components.append(y_fitted_aligned)
    if not y_pred.empty:
        y_pred_components.append(y_pred)

    if y_pred_components:
        y_pred_full = pd.concat(y_pred_components).sort_index()
        y_pred_full = y_pred_full.reindex(y_full.index)
    else:
        y_pred_full = pd.Series(index=y_full.index, dtype=float)

    # Compute absolute errors
    abs_err = (y_full - y_pred_full).abs() if not y_full.empty else pd.Series(dtype=float)

    # Create subplots for side-by-side comparison
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]), **kwargs)
    
    # Left plot: Target variable (original functionality)
    # Plot actual values
    ax1.plot(y_full.index, y_full, label="Actual", color="black", linewidth=2)

    # Plot fitted values while preserving gaps created by NaN placeholders.
    fitted_valid = y_fitted_aligned.dropna()
    if not fitted_valid.empty:
        if len(fitted_valid) == 1:
            ax1.plot(
                fitted_valid.index,
                fitted_valid.values,
                marker='o',
                markersize=2,
                color="tab:blue",
                label="Fitted (IS)",
                linestyle='None'
            )
        else:
            ax1.plot(
                y_fitted_aligned.index,
                y_fitted_aligned,
                label="Fitted (IS)",
                color="tab:blue",
                linewidth=2
            )
    
    # Plot out-of-sample predictions
    if not y_pred.empty:
        ax1.plot(
            y_pred.index,
            y_pred,
            linestyle="--",
            label="Predicted (OOS)",
            color="tab:blue",
            linewidth=2,
        )
    
    ax1.set_ylabel("Value")
    ax1.set_title("Target Variable: Actual vs. Fitted/OOS")
    ax1.legend(loc="upper left")

    # Plot absolute errors on secondary axis with improved width calculation
    ax2 = ax1.twinx()
    
    # Calculate bar width more robustly
    if len(abs_err) > 1:
        # Calculate median time difference for more robust width estimation
        time_diffs = []
        abs_err_sorted = abs_err.sort_index()
        for i in range(1, len(abs_err_sorted)):
            delta = abs_err_sorted.index[i] - abs_err_sorted.index[i-1]
            if isinstance(delta, pd.Timedelta):
                time_diffs.append(float(delta / pd.Timedelta(days=1)))
            else:
                time_diffs.append(delta)
        
        # Use median difference for more robust width calculation
        if time_diffs:
            median_diff = np.median(time_diffs)
            width = median_diff * 0.6  # Use 60% of median difference
        else:
            width = 0.8
    else:
        width = 0.8
    
    ax2.bar(abs_err.index, abs_err, width=width, alpha=0.2, color="grey", label="|Error|")
    ax2.set_ylabel("Absolute Error")
    ax2.legend(loc="upper right")

    # Right plot: Base variable (new functionality)
    if model is not None and hasattr(model, 'y_base_full') and not model.y_base_full.empty:
        # Get base variable data
        y_base_full = model.y_base_full
        y_base_fitted_in = model.y_base_fitted_in if hasattr(model, 'y_base_fitted_in') else None
        y_base_pred_out = model.y_base_pred_out if hasattr(model, 'y_base_pred_out') else None
        
        # Plot actual base values
        ax3.plot(y_base_full.index, y_base_full, label="Actual", color="black", linewidth=2)
        
        # Combine fitted and predicted base values
        y_base_pred_full = pd.Series(dtype=float)
        if y_base_fitted_in is not None and not y_base_fitted_in.empty:
            y_base_pred_full = pd.concat([y_base_pred_full, y_base_fitted_in])
        if y_base_pred_out is not None and not y_base_pred_out.empty:
            y_base_pred_full = pd.concat([y_base_pred_full, y_base_pred_out])
        
        # Plot base predictions with gap handling (similar to target variable)
        if not y_base_pred_full.empty:
            y_base_pred_full = y_base_pred_full.sort_index()
            
            # Check for gaps in base fitted data
            if y_base_fitted_in is not None and not y_base_fitted_in.empty:
                fitted_base_idx_sorted = y_base_fitted_in.index.sort_values()

                # Infer the typical spacing to spot discontinuities created by outliers.
                typical_diff_td = _infer_typical_gap(fitted_base_idx_sorted)

                # Find gaps in base fitted data (similar logic to target)
                base_segments = []
                if len(fitted_base_idx_sorted) > 0:
                    current_base_segment = [fitted_base_idx_sorted[0]]

                    for i in range(1, len(fitted_base_idx_sorted)):
                        prev_idx = fitted_base_idx_sorted[i-1]
                        curr_idx = fitted_base_idx_sorted[i]

                        # Check if there's a gap larger than expected
                        time_gap = curr_idx - prev_idx

                        if time_gap > typical_diff_td * 1.5:  # Allow some tolerance
                            # Gap detected, finish current segment and start new one
                            base_segments.append(current_base_segment)
                            current_base_segment = [curr_idx]
                        else:
                            # No gap, continue current segment
                            current_base_segment.append(curr_idx)

                    # Add the last segment
                    if current_base_segment:
                        base_segments.append(current_base_segment)

                    # Plot each segment separately
                    for i, segment in enumerate(base_segments):
                        if len(segment) > 0:
                            segment_data = y_base_fitted_in.loc[segment]
                            label = "Fitted (IS)" if i == 0 else None  # Only label first segment

                            if len(segment) == 1:
                                # Single point - plot as marker
                                ax3.plot(segment_data.index, segment_data.values,
                                        marker='o', markersize=2, color="tab:orange",
                                        label=label, linestyle='None')
                            else:
                                # Multiple points - plot as line
                                ax3.plot(segment_data.index, segment_data.values,
                                        label=label, color="tab:orange", linewidth=2)
            
            # Plot out-of-sample base predictions
            if y_base_pred_out is not None and not y_base_pred_out.empty:
                ax3.plot(
                    y_base_pred_out.index,
                    y_base_pred_out,
                    linestyle="--",
                    label="Predicted (OOS)",
                    color="tab:orange",
                    linewidth=2,
                )
        
        # Calculate and plot absolute errors for base variable
        if not y_base_pred_full.empty:
            abs_err_base = (y_base_full - y_base_pred_full).abs()
            
            # Plot absolute errors on secondary axis
            ax4 = ax3.twinx()
            
            # Calculate bar width for base variable
            if len(abs_err_base) > 1:
                time_diffs_base = []
                abs_err_base_sorted = abs_err_base.sort_index()
                for i in range(1, len(abs_err_base_sorted)):
                    delta = abs_err_base_sorted.index[i] - abs_err_base_sorted.index[i-1]
                    if isinstance(delta, pd.Timedelta):
                        time_diffs_base.append(float(delta / pd.Timedelta(days=1)))
                    else:
                        time_diffs_base.append(delta)
                
                if time_diffs_base:
                    median_diff_base = np.median(time_diffs_base)
                    width_base = median_diff_base * 0.6
                else:
                    width_base = 0.8
            else:
                width_base = 0.8
            
            ax4.bar(abs_err_base.index, abs_err_base, width=width_base, alpha=0.2, color="grey", label="|Error|")
            ax4.set_ylabel("Absolute Error")
            ax4.legend(loc="upper right")
        
        ax3.set_ylabel("Value")
        ax3.set_title("Base Variable: Actual vs. Fitted/OOS")
        ax3.legend(loc="upper left")
    else:
        # No base variable data available
        ax3.text(0.5, 0.5, "No base variable data available", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Base Variable: Actual vs. Fitted/OOS")

    fig.tight_layout()
    return fig


def ols_model_test_plot(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    figsize: tuple = (6, 4),
    **kwargs
) -> plt.Figure:
    """
    Plot residuals against fitted values for a fitted regression model.

    Parameters
    ----------
    model : Any
        Statsmodels OLS result instance containing ``fittedvalues`` and ``resid``.
    X : pd.DataFrame
        Feature matrix (unused; kept for compatibility with existing calls).
    y : pd.Series
        Target series (unused; kept for compatibility with existing calls).
    figsize : tuple, optional
        Size of the matplotlib figure in inches.
    **kwargs
        Additional keyword arguments forwarded to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot figure showing residuals versus fitted values.
    """
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    ax.scatter(model.fittedvalues, model.resid)
    ax.axhline(0, color='grey', linewidth=1)
    ax.set_title("Residuals vs Fitted")
    return fig


def ols_plot_perf_set(
    reports: Dict[str, Any],
    full: bool = False,
    figsize: tuple = (12, 6),
    **kwargs
) -> plt.Figure:
    """
    Plot actual vs. fitted/in-sample and predicted/out-of-sample values for multiple candidate models.
    In-sample fits are solid; out-of-sample predictions are dashed. When reports
    originate from models with configured outliers, those periods appear as gaps in
    the actual series.

    Parameters
    ----------
    reports : dict
        Mapping of model_id to ModelReportBase instances (must have .model.y, .model.y_out,
        .model.y_fitted_in, and .model.y_pred_out attributes).
    full : bool
        If True, plot only in-sample fits; otherwise include out-of-sample predictions.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Passed to plt.subplots().

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing target and base variable performance comparisons.
    """
    # Determine the actual series for the target variable (top row)
    first_report = next(iter(reports.values()))
    model = first_report.model
    if hasattr(model, 'y_full') and model.y_full is not None and not model.y_full.empty:
        y_full_series = model.y_full.sort_index()
        if full:
            if (
                hasattr(model, 'dm')
                and getattr(model.dm, 'in_sample_idx', None) is not None
            ):
                in_idx = pd.Index(model.dm.in_sample_idx).sort_values()
                actual = y_full_series.reindex(in_idx).sort_index()
            else:
                actual = model.y.sort_index()
        else:
            actual = y_full_series
    else:
        if (
            not full
            and hasattr(model, 'y_out')
            and model.y_out is not None
            and not model.y_out.empty
        ):
            actual = pd.concat([model.y, model.y_out]).sort_index()
        else:
            actual = model.y.sort_index()

    # Prepare for two rows: top for target, bottom for base variable
    fig, (ax, ax_base) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*2), sharex=False, **kwargs)

    # --- Top row: Target variable performance (as before) ---
    ax.plot(actual.index, actual, label='Actual', color='black', linewidth=2)
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    for idx, (mid, rpt) in enumerate(reports.items()):
        color = colors[idx % len(colors)] if colors else None
        y_in = rpt.model.y_fitted_in.sort_index()
        if (
            hasattr(rpt.model, 'dm')
            and getattr(rpt.model.dm, 'in_sample_idx', None) is not None
        ):
            in_idx = pd.Index(rpt.model.dm.in_sample_idx).sort_values()
        else:
            in_idx = pd.Index(y_in.index).sort_values()

        y_in_aligned = y_in.reindex(in_idx)
        y_in_valid = y_in_aligned.dropna()
        if not y_in_valid.empty:
            label = f"{mid} (IS)"
            if len(y_in_valid) == 1:
                ax.plot(
                    y_in_valid.index,
                    y_in_valid.values,
                    marker='o',
                    markersize=2,
                    color=color,
                    label=label,
                    linestyle='None'
                )
            else:
                ax.plot(
                    y_in_aligned.index,
                    y_in_aligned,
                    linestyle='-',
                    label=label,
                    color=color,
                    linewidth=2
                )
        # Out-of-sample predicted
        if (
            not full
            and hasattr(rpt.model, 'y_pred_out')
            and rpt.model.y_pred_out is not None
            and not rpt.model.y_pred_out.empty
        ):
            y_out = rpt.model.y_pred_out.sort_index()
            ax.plot(
                y_out.index,
                y_out,
                linestyle='--',
                label=None,
                color=color,
                linewidth=2
            )
    ax.set_title('Model Performance Comparison (Target Variable)')
    ax.set_ylabel('Value')
    ax.legend(loc='best')

    # --- Bottom row: Base variable performance ---
    # Plot actual base variable (if available) and predictions for each model
    base_plotted = False
    for idx, (mid, rpt) in enumerate(reports.items()):
        color = colors[idx % len(colors)] if colors else None
        model = rpt.model
        if hasattr(model, 'y_base_full') and model.y_base_full is not None and not model.y_base_full.empty:
            y_base_full = model.y_base_full.sort_index()
            if not base_plotted:
                ax_base.plot(
                    y_base_full.index,
                    y_base_full,
                    label='Actual',
                    color='black',
                    linewidth=2
                )
                base_plotted = True
            # Fitted and predicted base
            y_base_fitted_in = getattr(model, 'y_base_fitted_in', None)
            y_base_pred_out = getattr(model, 'y_base_pred_out', None)
            y_base_pred_full = pd.Series(dtype=float)
            if y_base_fitted_in is not None and not y_base_fitted_in.empty:
                y_base_pred_full = pd.concat([y_base_pred_full, y_base_fitted_in])
            if y_base_pred_out is not None and not y_base_pred_out.empty:
                y_base_pred_full = pd.concat([y_base_pred_full, y_base_pred_out])
            y_base_pred_full = y_base_pred_full.sort_index()
            # Gap handling for in-sample fitted base
            if y_base_fitted_in is not None and not y_base_fitted_in.empty:
                fitted_base_idx_sorted = y_base_fitted_in.index.sort_values()
                typical_diff = _infer_typical_gap(fitted_base_idx_sorted)
                base_segments = []
                current_base_segment = [fitted_base_idx_sorted[0]]
                for i in range(1, len(fitted_base_idx_sorted)):
                    prev_idx = fitted_base_idx_sorted[i-1]
                    curr_idx = fitted_base_idx_sorted[i]
                    time_gap = curr_idx - prev_idx
                    if time_gap > typical_diff * 1.5:
                        base_segments.append(current_base_segment)
                        current_base_segment = [curr_idx]
                    else:
                        current_base_segment.append(curr_idx)
                if current_base_segment:
                    base_segments.append(current_base_segment)
                for i, segment in enumerate(base_segments):
                    if len(segment) > 0:
                        segment_data = y_base_fitted_in.loc[segment]
                        label = f"{mid} (IS)" if i == 0 else None
                        if len(segment) == 1:
                            ax_base.plot(
                                segment_data.index,
                                segment_data.values,
                                marker='o', markersize=2, color=color,
                                label=label, linestyle='None'
                            )
                        else:
                            ax_base.plot(
                                segment_data.index,
                                segment_data.values,
                                linestyle='-',
                                label=label,
                                color=color,
                                linewidth=2
                            )
            # Out-of-sample predicted base
            if y_base_pred_out is not None and not y_base_pred_out.empty:
                ax_base.plot(
                    y_base_pred_out.index,
                    y_base_pred_out,
                    linestyle='--',
                    label=None,
                    color=color,
                    linewidth=2
                )
        else:
            # No base variable data for this model
            ax_base.text(0.5, 0.5, f"No base variable for {mid}",
                         ha='center', va='center', transform=ax_base.transAxes)
    ax_base.set_title('Model Performance Comparison (Base Variable)')
    ax_base.set_ylabel('Value')
    ax_base.legend(loc='best')
    fig.tight_layout()
    return fig
