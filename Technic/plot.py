# TECHNIC/plot.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

def ols_model_perf_plot(
    X: pd.DataFrame,
    y: pd.Series,
    X_out: Optional[pd.DataFrame] = None,
    y_out: Optional[pd.Series] = None,
    y_fitted_in: Optional[pd.Series] = None,
    y_pred_out: Optional[pd.Series] = None,
    figsize: tuple = (8, 4),
    **kwargs
) -> plt.Figure:
    """
    Plot actual vs. fitted (in-sample) and predicted (out-of-sample) values,
    with a secondary bar chart of absolute errors.

    Parameters
    ----------
    X : pd.DataFrame
        In-sample feature DataFrame used for fitting.
    y : pd.Series
        In-sample target values.
    X_out : pd.DataFrame, optional
        Out-of-sample feature DataFrame for predictions.
    y_out : pd.Series, optional
        Actual target values for out-of-sample.
    y_fitted_in : pd.Series
        Fitted in-sample values (must be provided).
    y_pred_out : pd.Series, optional
        Predicted out-of-sample values (must be provided if X_out is given).
    figsize : tuple, default (8,4)
        Figure size.
    **kwargs
        Additional kwargs passed to plt.subplots().
    """
    # Determine full index for actual values
    if X_out is not None:
        X_full = pd.concat([X, X_out]).sort_index()
    else:
        X_full = X.sort_index()

    if y_out is not None:
        y_full = pd.concat([y, y_out]).sort_index().reindex(X_full.index)
    else:
        y_full = y.sort_index().reindex(X_full.index)

    # In-sample fitted series
    if y_fitted_in is None:
        raise ValueError("y_fitted_in must be provided for in-sample fitted values")
    y_fitted = y_fitted_in.sort_index()

    # Out-of-sample predictions
    if X_out is not None:
        if y_pred_out is None:
            raise ValueError("y_pred_out must be provided when X_out is not None")
        y_pred = y_pred_out.sort_index()
    else:
        y_pred = pd.Series(dtype=float)

    # Combine fitted and predicted series
    y_pred_full = pd.concat([y_fitted, y_pred]).sort_index()

    # Compute absolute errors
    abs_err = (y_full - y_pred_full).abs()

    # Create plot
    fig, ax1 = plt.subplots(figsize=figsize, **kwargs)
    
    # Plot actual values
    ax1.plot(y_full.index, y_full, label="Actual", color="black", linewidth=2)
    
    # Plot fitted values with potential gaps for outliers
    # Check if there are gaps in the fitted data (indicating outliers)
    if len(y_fitted) > 0:
        # Simple approach: detect gaps by looking at consecutive fitted indices
        fitted_idx_sorted = y_fitted.index.sort_values()
        
        # Find expected frequency from the original data
        if len(y.index) > 1:
            # Infer frequency from original data
            freq = pd.infer_freq(y.index)
            if freq is None:
                # Fallback: calculate median difference
                diffs = y.index.to_series().diff().dropna()
                if len(diffs) > 0:
                    typical_diff = diffs.median()
                else:
                    typical_diff = pd.Timedelta(days=30)  # Default fallback
            else:
                # Convert frequency to offset object
                typical_diff = pd.tseries.frequencies.to_offset(freq)
        else:
            typical_diff = pd.Timedelta(days=30)  # Default fallback
        
        # Find gaps in fitted data
        segments = []
        current_segment = [fitted_idx_sorted[0]]
        
        for i in range(1, len(fitted_idx_sorted)):
            prev_idx = fitted_idx_sorted[i-1]
            curr_idx = fitted_idx_sorted[i]
            
            # Check if there's a gap larger than expected
            time_gap = curr_idx - prev_idx
            
            # Convert typical_diff to Timedelta if it's an offset object for comparison
            if hasattr(typical_diff, 'delta'):
                # It's an offset object, get the underlying timedelta
                typical_diff_td = typical_diff.delta
            elif isinstance(typical_diff, pd.Timedelta):
                typical_diff_td = typical_diff
            else:
                # Try to convert to timedelta
                try:
                    typical_diff_td = pd.Timedelta(typical_diff)
                except:
                    typical_diff_td = pd.Timedelta(days=30)  # Fallback
            
            if time_gap > typical_diff_td * 1.5:  # Allow some tolerance
                # Gap detected, finish current segment and start new one
                segments.append(current_segment)
                current_segment = [curr_idx]
            else:
                # No gap, continue current segment
                current_segment.append(curr_idx)
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        # Plot each segment separately
        for i, segment in enumerate(segments):
            if len(segment) > 0:
                segment_data = y_fitted.loc[segment]
                label = "Fitted (IS)" if i == 0 else None  # Only label first segment
                
                if len(segment) == 1:
                    # Single point - plot as marker
                    ax1.plot(segment_data.index, segment_data.values, 
                            marker='o', markersize=2, color="tab:blue", 
                            label=label, linestyle='None')
                else:
                    # Multiple points - plot as line
                    ax1.plot(segment_data.index, segment_data.values, 
                            label=label, color="tab:blue", linewidth=2)
    else:
        # No fitted data to plot
        pass
    
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
    ax1.set_title("Actual vs. Fitted/OOS")
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

    fig.tight_layout()
    return fig


def ols_model_test_plot(model, X, y, figsize=(6,4), **kwargs):
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
    In-sample fits are solid; out-of-sample predictions are dashed.

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
    """
    # Determine the actual series
    first_report = next(iter(reports.values()))
    model = first_report.model
    if (
        not full
        and hasattr(model, 'y_out')
        and model.y_out is not None
        and not model.y_out.empty
    ):
        actual = pd.concat([model.y, model.y_out]).sort_index()
    else:
        actual = model.y.sort_index()

    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    ax.plot(actual.index, actual, label='Actual', color='black', linewidth=2)

    # Color cycle for model lines
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    for idx, (mid, rpt) in enumerate(reports.items()):
        color = colors[idx % len(colors)] if colors else None
        
        # In-sample fitted with gap handling
        y_in = rpt.model.y_fitted_in.sort_index()
        
        # Check for gaps in fitted data (indicating outliers)
        if len(y_in) > 0:
            # Simple approach: detect gaps by looking at consecutive fitted indices
            fitted_idx_sorted = y_in.index.sort_values()
            
            # Find expected frequency from the original data
            if len(rpt.model.y.index) > 1:
                # Infer frequency from original data
                freq = pd.infer_freq(rpt.model.y.index)
                if freq is None:
                    # Fallback: calculate median difference
                    diffs = rpt.model.y.index.to_series().diff().dropna()
                    if len(diffs) > 0:
                        typical_diff = diffs.median()
                    else:
                        typical_diff = pd.Timedelta(days=30)  # Default fallback
                else:
                    typical_diff = pd.tseries.frequencies.to_offset(freq)
            else:
                typical_diff = pd.Timedelta(days=30)  # Default fallback
            
            # Find gaps in fitted data
            segments = []
            current_segment = [fitted_idx_sorted[0]]
            
            for i in range(1, len(fitted_idx_sorted)):
                prev_idx = fitted_idx_sorted[i-1]
                curr_idx = fitted_idx_sorted[i]
                
                # Check if there's a gap larger than expected
                time_gap = curr_idx - prev_idx
                
                # Convert typical_diff to Timedelta if it's an offset object for comparison
                if hasattr(typical_diff, 'delta'):
                    # It's an offset object, get the underlying timedelta
                    typical_diff_td = typical_diff.delta
                elif isinstance(typical_diff, pd.Timedelta):
                    typical_diff_td = typical_diff
                else:
                    # Try to convert to timedelta
                    try:
                        typical_diff_td = pd.Timedelta(typical_diff)
                    except:
                        typical_diff_td = pd.Timedelta(days=30)  # Fallback
                
                if time_gap > typical_diff_td * 1.5:  # Allow some tolerance
                    # Gap detected, finish current segment and start new one
                    segments.append(current_segment)
                    current_segment = [curr_idx]
                else:
                    # No gap, continue current segment
                    current_segment.append(curr_idx)
            
            # Add the last segment
            if current_segment:
                segments.append(current_segment)
            
            # Plot each segment separately
            for i, segment in enumerate(segments):
                if len(segment) > 0:
                    segment_data = y_in.loc[segment]
                    label = f"{mid} (IS)" if i == 0 else None  # Only label first segment
                    
                    if len(segment) == 1:
                        # Single point - plot as marker
                        ax.plot(
                            segment_data.index,
                            segment_data.values,
                            marker='o', markersize=2, color=color,
                            label=label, linestyle='None'
                        )
                    else:
                        # Multiple points - plot as line
                        ax.plot(
                            segment_data.index,
                            segment_data.values,
                            linestyle='-',  # solid for in-sample
                            label=label,
                            color=color,
                            linewidth=2
                        )
        else:
            # No fitted data to plot
            pass
        
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
                linestyle='--',  # dashed for out-of-sample
                label=None,
                color=color,
                linewidth=2
            )

    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig
