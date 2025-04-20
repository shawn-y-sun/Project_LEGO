# TECHNIC/plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

def ols_perf_plot(model, X, y, figsize=(8,4), **kwargs):
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    ax.plot(y.index, y, label="Actual")
    ax.plot(y.index, model.fittedvalues, label="Fitted")
    ax.legend()
    ax.set_title("Actual vs Fitted")
    return fig

def ols_test_plot(model, X, y, figsize=(6,4), **kwargs):
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    ax.scatter(model.fittedvalues, model.resid)
    ax.axhline(0, color='grey', linewidth=1)
    ax.set_title("Residuals vs Fitted")
    return fig