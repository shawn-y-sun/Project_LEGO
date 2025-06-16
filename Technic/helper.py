import numpy as np
from scipy import stats
from statsmodels.tools.validation import (array_like, bool_like)
from statsmodels.regression.linear_model import OLS



def _check_het_test(x: np.ndarray, test_name: str) -> None:
    """
    Check validity of the exogenous regressors in a heteroskedasticity test

    Parameters
    ----------
    x : ndarray
        The exogenous regressor array
    test_name : str
        The test name for the exception
    """
    x_max = x.max(axis=0)
    if (
        not np.any(((x_max - x.min(axis=0)) == 0) & (x_max != 0))
        or x.shape[1] < 2
    ):
        raise ValueError(
            f"{test_name} test requires exog to have at least "
            "two columns where one is a constant."
        )


def het_white(resid, exog):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity.

    Parameters
    ----------
    resid : array_like
        The residuals. The squared residuals are used as the endogenous
        variable.
    exog : array_like
        The explanatory variables for the variance. Squares and interaction
        terms are automatically included in the auxiliary regression.

    Returns
    -------
    lm : float
        The lagrange multiplier statistic.
    lm_pvalue :float
        The p-value of lagrange multiplier test.
    fvalue : float
        The f-statistic of the hypothesis that the error variance does not
        depend on x. This is an alternative test variant not the original
        LM test.
    f_pvalue : float
        The p-value for the f-statistic.

    Notes
    -----
    Assumes x contains constant (for counting dof).

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222. Test statistic reproduces
    Greene 5th, example 11.3.
    """
    x = array_like(exog, "exog", ndim=2)
    y = array_like(resid, "resid", ndim=2, shape=(x.shape[0], 1))
    _check_het_test(x, "White's heteroskedasticity")
    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0
    resols = OLS(y ** 2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    # degrees of freedom take possible reduced rank in exog into account
    # df_model checks the rank to determine df
    # extra calculation that can be removed:
    # assert resols.df_model == np.linalg.matrix_rank(exog) - 1
    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval