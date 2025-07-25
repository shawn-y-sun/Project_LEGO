# =============================================================================
# module: condition.py
# Purpose: Conditional feature specification as a subclass of Feature
# Dependencies: pandas, typing, .feature.Feature
# =============================================================================

import pandas as pd
from typing import Callable, Union, List, Any, Dict, Optional
from .feature import Feature

class CondVar(Feature):
    """
    Conditional variable feature.

    Applies a function to a main variable and one or more condition variables
    to produce a new feature series.

    Parameters
    ----------
    main_var : str or pandas.Series
        Name or series of the main variable.
    cond_var : str, pandas.Series, or list thereof
        One or more condition variable names or Series.
    cond_fn : Callable[..., pandas.Series]
        Function taking (main_series, *cond_series) and returning a Series.
    cond_fn_kwargs : dict, optional
        Keyword arguments for cond_fn.
    alias : str, optional
        Custom name for the output feature.
    """
    def __init__(
        self,
        main_var: Union[str, pd.Series],
        cond_var: Union[str, pd.Series, List[Union[str, pd.Series]]],
        cond_fn: Callable[..., pd.Series],
        cond_fn_kwargs: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None
    ):
        super().__init__(var=main_var, alias=alias)
        # Initialize list of conditional variables (names or Series)
        if isinstance(cond_var, (str, pd.Series)):
            self.cond_var = [cond_var]
        elif (isinstance(cond_var, list) and
              all(isinstance(cv, (str, pd.Series)) for cv in cond_var)):
            self.cond_var = cond_var
        else:
            raise TypeError("`cond_var` must be a column name, Series, or list thereof")
        self.cond_fn = cond_fn
        self.cond_fn_kwargs = cond_fn_kwargs or {}

    @property
    def name(self) -> str:
        """
        Name of the output feature: alias if provided else mainVar_condFnName.
        """
        fn_name = getattr(self.cond_fn, "__name__", "cond")
        return self.alias if self.alias else f"{self.var}_{fn_name}"

    def lookup_map(self) -> Dict[str, str]:
        """
        Map attribute 'var_series' to the main variable name for lookup.
        """
        return {"var_series": self.var}

    def apply(
        self,
        *dfs: pd.DataFrame
    ) -> pd.Series:
        """
        Execute conditional function to build the feature Series.

        Parameters
        ----------
        *dfs : pandas.DataFrame
            DataFrame sources to search for variables.

        Returns
        -------
        pandas.Series
            Resulting feature series with name set to self.name.

        Raises
        ------
        KeyError
            If any variable name is not found in provided DataFrames.
        TypeError
            If cond_fn does not return a pandas Series.
        """
        # Populate main_series via Feature.lookup
        self.lookup(*dfs)
        main_series = self.var_series  # type: pd.Series

        # Resolve each condition series
        cond_series_list: List[pd.Series] = []
        for cv in self.cond_var:
            if isinstance(cv, pd.Series):
                cond_series_list.append(cv)
            elif isinstance(cv, str):
                # search provided DataFrames for the variable
                for df in dfs:
                    if df is not None and cv in df.columns:
                        cond_series_list.append(df[cv])
                        break
                else:
                    raise KeyError(f"CondVar: var '{cv}' not found in any DataFrame.")
            else:
                raise TypeError("`cond_var` elements must be column names or Series")

        # Apply the conditional function
        result = self.cond_fn(main_series, *cond_series_list, **self.cond_fn_kwargs)
        if not isinstance(result, pd.Series):
            raise TypeError("`cond_fn` must return a pandas Series")
        result.name = self.name
        self.output_names = [self.name]
        return result
    

def zero_if_exceeds(
    main_series: pd.Series,
    condition_series: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Conditional function: sets values of main_series to 0 where
    condition_series exceeds the threshold; otherwise retains original values.

    :param main_series: Series with original values.
    :param condition_series: Series to compare against threshold.
    :param threshold: Numeric cutoff above which main_series is zeroed.
    :return: A new Series with same index as main_series.
    """
    # Zero out values where condition exceeds the threshold
    result = main_series.where(condition_series <= threshold, other=0)
    return result


def BO(
    main_series: pd.Series,
    condition_series: pd.Series,
    threshold: float,
    lag: int = 1,
    burn_periods: int = 1
) -> pd.Series:
    """
    Burn-Out function
    Sets values of main_series to 0 for `burn_periods` after
    condition_series at time t-lag exceeds threshold, but only if
    the original main_series values are positive.

    :param main_series: Series with original values.
    :param condition_series: Series to compare against threshold.
    :param threshold: Numeric cutoff above which burn is triggered.
    :param lag: Number of periods to look back in condition_series (default=1).
    :param burn_periods: Number of consecutive periods to zero out in main_series (default=1).
    :return: A new Series with burn applied.
    """
    # Align and prepare
    cond_shifted = condition_series.shift(lag)
    events = cond_shifted > threshold

    original = main_series.copy()
    result = main_series.copy()

    # For each event, zero out next burn_periods only if original>0
    for idx in events[events].index:
        try:
            start_loc = result.index.get_loc(idx)
        except KeyError:
            continue
        end_loc = start_loc + burn_periods
        positions = result.index[start_loc:end_loc]
        # Mask positions where original > 0
        mask = original.loc[positions] > 0
        zero_positions = positions[mask.values]
        result.loc[zero_positions] = 0

    return result


def create_conditional_var(
    dm,
    main_var: str,
    condition: Callable[[pd.Series], pd.Series],
    cond_var: Optional[str] = None,
    alias: Optional[str] = None,
    add_to_data: bool = True
) -> Optional[pd.Series]:
    """
    Create a conditional variable and optionally add it to all datasets.
    
    This is a user-friendly wrapper to create conditional variables and add them
    to both internal and MEV data. The condition function should take a Series
    and return a Series with the transformed values.
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance containing the data to modify
    main_var : str
        Name of the main variable to transform
    condition : Callable[[pd.Series], pd.Series]
        Function that takes a Series and returns a transformed Series.
        This function defines how to transform the main_var based on conditions.
    cond_var : str, optional
        Name of the conditional variable if different from main_var.
        If None, uses main_var as the condition variable.
    alias : str, optional
        Name for the new variable. If None, uses main_var + '_COND'
    add_to_data : bool, default True
        If True, adds the variable to all datasets in the DataManager.
        If False, only returns the Series without modifying data.
    
    Returns
    -------
    Optional[pd.Series]
        If add_to_data is False, returns the conditional Series.
        If add_to_data is True, returns None (modifies data in place).
    
    Examples
    --------
    >>> # Create high unemployment regime (UNRATE > 10%)
    >>> def high_unemp(series):
    ...     return series.where(series > 0.10, other=0)
    >>> 
    >>> create_conditional_var(
    ...     dm,
    ...     main_var='UNRATE',
    ...     condition=high_unemp,
    ...     alias='UNEMP_REGIME'
    ... )
    >>> 
    >>> # Create GDP growth regime based on unemployment
    >>> def gdp_in_high_unemp(series, unrate):
    ...     return series.where(unrate > 0.10, other=0)
    >>> 
    >>> create_conditional_var(
    ...     dm,
    ...     main_var='GDP',
    ...     condition=lambda x: gdp_in_high_unemp(x, dm.model_mev['UNRATE']),
    ...     alias='GDP_HIGH_UNEMP'
    ... )
    >>> 
    >>> # Create conditional without adding to data
    >>> def recession_gdp(series):
    ...     return series.where(series < 0, other=0)
    >>> 
    >>> gdp_recession = create_conditional_var(
    ...     dm,
    ...     main_var='GDP',
    ...     condition=recession_gdp,
    ...     alias='GDP_RECESSION',
    ...     add_to_data=False
    ... )
    
    Notes
    -----
    - The condition function should handle any necessary data validation
    - If add_to_data=True, the variable is added to:
        - Internal data (dm.internal_data)
        - Model MEV data (dm.model_mev)
        - All scenario MEV data (dm.scen_mevs)
    - The function preserves the original data structure and index
    """
    # Validate inputs
    if not callable(condition):
        raise TypeError("condition must be a callable function")
    
    # Set default names
    cond_var = cond_var or main_var
    alias = alias or f"{main_var}_COND"
    
    def add_conditional_var(df, internal_df=None):
        """Inner function to add conditional variable to a DataFrame."""
        if main_var not in df.columns:
            return df
            
        # Apply the condition
        df[alias] = condition(df[main_var])
        return df
    
    if add_to_data:
        # Add to all datasets
        dm.apply_to_internal(add_conditional_var)
        dm.apply_to_mevs(add_conditional_var)
        return None
    else:
        # Just return the conditional series for model MEV
        if main_var not in dm.model_mev.columns:
            raise KeyError(f"Variable '{main_var}' not found in model MEV data")
        return condition(dm.model_mev[main_var])