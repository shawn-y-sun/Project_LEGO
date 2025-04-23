# TECHNIC/plot.py
import numpy as np
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

def ols_model_perf_plot(model, X, y, X_out=None, y_pred_out=None, figsize=(8,4), **kwargs):
    """
    Plot actual vs. fitted/in-sample and predicted/out-of-sample values,
    with a secondary bar chart of absolute errors (alpha=0.7).

    Parameters
    ----------
    model : statsmodels RegressionResults
        Fitted in-sample OLS model with .fittedvalues attribute.
    X : pd.DataFrame
        In-sample feature DataFrame used for fitting.
    y : pd.Series
        Target values for full sample (index covering X and optional X_out).
    X_out : pd.DataFrame, optional
        Out-of-sample feature DataFrame for predictions.
    y_pred_out : pd.Series, optional
        Predicted values for out-of-sample X_out. If None and X_out provided,
        uses model.predict(X_out).
    figsize : tuple, default (8,4)
        Figure size.
    **kwargs
        Additional kwargs passed to plt.subplots().
    """
    # Combine full index
    if X_out is not None:
        X_full = pd.concat([X, X_out]).sort_index()
    else:
        X_full = X.sort_index()
    y_full = y.sort_index().reindex(X_full.index)

    # In-sample fitted values
    y_fitted_in = pd.Series(model.fittedvalues, index=X.index).sort_index()

    # Out-of-sample predictions
    if X_out is not None:
        y_pred = (
            y_pred_out.sort_index()
            if (y_pred_out is not None)
            else pd.Series(model.predict(X_out), index=X_out.index).sort_index()
        )
    else:
        y_pred = pd.Series(dtype=float)

    # Combine predictions
    y_pred_full = pd.concat([y_fitted_in, y_pred]).sort_index()

    # Absolute errors
    abs_err = (y_full - y_pred_full).abs()

    # Plotting
    fig, ax1 = plt.subplots(figsize=figsize, **kwargs)
    ax1.plot(y_full.index, y_full, label="Actual", color="black", linewidth=2)
    ax1.plot(y_fitted_in.index, y_fitted_in, label="Fitted (In-sample)", color="tab:blue", linewidth=2)
    if not y_pred.empty:
        ax1.plot(
            y_pred.index,
            y_pred,
            linestyle="--",
            label="Predicted (Out-of-sample)",
            color="tab:blue",
            linewidth=2,
        )
    ax1.set_ylabel("Value")
    ax1.set_title("Actual vs. Fitted/Predicted")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    if len(abs_err) > 1:
        # calculate bar width based on first interval
        width = (abs_err.index[1] - abs_err.index[0]) * 0.8
    else:
        width = 0.8
    ax2.bar(abs_err.index, abs_err, width=width, alpha=0.7, color="grey", label="|Error|")
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


def ols_seg_perf_plot(
    measures: Dict[str, Any],
    full: bool = False,
    figsize: tuple = (12, 6),
    **kwargs
) -> plt.Figure:
    """
    Plot actual vs. fitted/predicted values for multiple candidate models on one axis.
    In-sample fits are solid; out-of-sample predictions are dashed (same color per CM).
    """
    # Determine common actual series
    first_m = next(iter(measures.values()))
    if full and getattr(first_m, 'y_out', None) is not None:
        actual = pd.concat([first_m.y, first_m.y_out]).sort_index()
    else:
        actual = first_m.y.sort_index()

    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    ax.plot(actual.index, actual, label='Actual', color='black', linewidth=2)

    # use default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    for idx, (cm_id, m) in enumerate(measures.items()):
        color = colors[idx % len(colors)] if colors else None
        y_in = pd.Series(m.model.predict(m.X), index=m.X.index).sort_index()
        ax.plot(
            y_in.index, y_in,
            linestyle='-', label=f'{cm_id} (in)',
            linewidth=2, color=color
        )
        if full and getattr(m, 'X_out', None) is not None:
            if getattr(m, 'y_pred_out', None) is not None:
                y_out = m.y_pred_out.sort_index()
            else:
                y_out = pd.Series(m.model.predict(m.X_out), index=m.X_out.index).sort_index()
            ax.plot(
                y_out.index, y_out,
                linestyle='--', label=f'{cm_id} (out)',
                linewidth=2, color=color
            )

    ax.set_ylabel('Value')
    ax.set_title('Segment Performance Comparison')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig