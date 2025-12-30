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
    var : str or pandas.Series
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
        var: Union[str, pd.Series],
        cond_var: Union[str, pd.Series, List[Union[str, pd.Series]]],
        cond_fn: Callable[..., pd.Series],
        cond_fn_kwargs: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None
    ):
        super().__init__(var=var, alias=alias)
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
        Also maps 'cond_var_series' to condition variable names for lookup.
        """
        lookup_dict = {"var_series": self.var}
        
        # Add condition variables to lookup map
        if len(self.cond_var) == 1:
            # Single condition variable - use simple name
            cv = self.cond_var[0]
            if isinstance(cv, str):
                lookup_dict["cond_var_series"] = cv
        else:
            # Multiple condition variables - use indexed names
            for i, cv in enumerate(self.cond_var):
                if isinstance(cv, str):
                    lookup_dict[f"cond_var_series_{i}"] = cv
        
        return lookup_dict

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
        result = self.apply_fn(main_series, *cond_series_list)
        return result

    def apply_fn(self, main_series: pd.Series, *cond_series: pd.Series) -> pd.Series:
        """
        Apply the conditional function to the main series and condition series.
        
        Parameters
        ----------
        main_series : pandas.Series
            The main series to apply the conditional function to.
        *cond_series : pandas.Series
            Variable number of condition series to pass to the conditional function.
            
        Returns
        -------
        pandas.Series
            Result of applying the conditional function.
            
        Raises
        ------
        TypeError
            If cond_fn does not return a pandas Series.
        """
        result = self.cond_fn(main_series, *cond_series, **self.cond_fn_kwargs)
        if not isinstance(result, pd.Series):
            raise TypeError("`cond_fn` must return a pandas Series")
        result.name = self.name
        self.output_names = [self.name]
        return result


class BO(CondVar):
    """
    Burn-Out conditional variable feature.
    
    A specialized subclass of CondVar that applies burn-out logic to a main variable
    based on a condition variable. The burn-out effect zeros out the main variable
    for a specified number of periods after the condition variable exceeds a threshold.
    
    Parameters
    ----------
    var : str or pandas.Series
        Name or series of the main variable to apply burn-out to.
    cond_var : str or pandas.Series
        Name or series of the condition variable that triggers burn-out.
    cond_thresh : float
        Threshold on condition series that triggers burn-out effect.
    main_thresh : float, optional
        Threshold on main series - changes only made when main_series exceeds this threshold.
        Default is None (no threshold on main series).
    effect_periods : int, default 1
        Number of consecutive periods to zero out in main_series after trigger.
    lag : int, default 1
        Number of periods to look back in condition_series.
    cond_thresh_sign : str, default '>'
        Comparison operator for condition threshold: '>', '>=', '<', '<=', '=='.
    main_thresh_sign : str, default '>'
        Comparison operator for main threshold: '>', '>=', '<', '<=', '=='.
    alias : str, optional
        Custom name for the output feature.
        
    Example
    -------
    >>> # Create burn-out variable that zeros GDP for 2 periods after UNRATE > 5%
    >>> bo_gdp = BO(
    ...     var='GDP',
    ...     cond_var='UNRATE',
    ...     cond_thresh=0.05,
    ...     effect_periods=2,
    ...     alias='GDP_BO'
    ... )
    """
    def __init__(
        self,
        var: Union[str, pd.Series],
        cond_var: Union[str, pd.Series],
        cond_thresh: float,
        main_thresh: Optional[float] = None,
        effect_periods: int = 1,
        lag: int = 1,
        cond_thresh_sign: str = '>',
        main_thresh_sign: str = '>',
        alias: Optional[str] = None
    ):
        # Validate threshold signs
        valid_signs = ['>', '>=', '<', '<=', '==']
        if cond_thresh_sign not in valid_signs:
            raise ValueError(f"cond_thresh_sign must be one of {valid_signs}")
        if main_thresh_sign not in valid_signs:
            raise ValueError(f"main_thresh_sign must be one of {valid_signs}")
        
        # Store parameters for padding calculation
        self.effect_periods = effect_periods
        self.lag = lag
        
        # Create kwargs for BO function
        cond_fn_kwargs = {
            'cond_thresh': cond_thresh,
            'main_thresh': main_thresh,
            'effect_periods': effect_periods,
            'lag': lag,
            'cond_thresh_sign': cond_thresh_sign,
            'main_thresh_sign': main_thresh_sign
        }
        
        super().__init__(
            var=var,
            cond_var=cond_var,
            cond_fn=BO_func,
            cond_fn_kwargs=cond_fn_kwargs,
            alias=alias
        )
    
    @property
    def lookback_n(self) -> int:
        """
        Number of periods needed for conditional forecasting/iterative prediction.
        
        This property calculates the minimum number of lookback periods required
        to determine if a target variable is under burn-out effect during prediction.
        
        Formula: effect_periods + lag - 1
        
        Example
        -------
        If predicting monthly target from Jan 2025 onwards:
        - lag = 1: Need Dec 2024 condition value to check Jan 2025 effect
        - effect_periods = 1: Effect lasts 1 period
        - lookback_n = 1 + 1 - 1 = 1 period needed
        
        So input X must have at least Dec 2024's target to determine
        if Jan 2025 should be zeroed due to burn-out effect.
        
        Returns
        -------
        int
            Number of periods needed for conditional forecasting.
        """
        return self.effect_periods + self.lag - 1


def zero_if_exceeds(
    main_series: pd.Series,
    condition_series: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Conditional function: sets values of main_series to 0 where
    condition_series exceeds the threshold; otherwise retains original values.

    Parameters
    ----------
    main_series : pandas.Series
        Series with original values.
    condition_series : pandas.Series
        Series to compare against threshold.
    threshold : float
        Numeric cutoff above which main_series is zeroed.
        
    Returns
    -------
    pandas.Series
        A new Series with same index as main_series.
    """
    # Zero out values where condition exceeds the threshold
    result = main_series.where(condition_series <= threshold, other=0)
    return result


def BO_func(
    main_series: pd.Series,
    condition_series: pd.Series,
    cond_thresh: float,
    main_thresh: Optional[float] = None,
    effect_periods: int = 1,
    lag: int = 1,
    cond_thresh_sign: str = '>',
    main_thresh_sign: str = '>'
) -> pd.Series:
    """
    Burn-Out function with enhanced threshold controls.
    
    Sets values of main_series to 0 for `effect_periods` after
    condition_series at time t-lag meets the condition threshold, but only if
    the original main_series values meet the main threshold condition.
    
    Parameters
    ----------
    main_series : pandas.Series
        Series with original values.
    condition_series : pandas.Series
        Series to compare against threshold.
    cond_thresh : float
        Threshold on condition series that triggers burn-out effect.
    main_thresh : float, optional
        Threshold on main series - changes only made when main_series meets this threshold.
        If None, no threshold is applied to main series.
    effect_periods : int, default 1
        Number of consecutive periods to zero out in main_series after trigger.
    lag : int, default 1
        Number of periods to look back in condition_series.
    cond_thresh_sign : str, default '>'
        Comparison operator for condition threshold: '>', '>=', '<', '<=', '=='.
    main_thresh_sign : str, default '>'
        Comparison operator for main threshold: '>', '>=', '<', '<=', '=='.
        
    Returns
    -------
    pandas.Series
        A new Series with burn-out effect applied.
        
    Example
    -------
    >>> # Zero GDP for 2 periods after UNRATE > 5%, but only if GDP > 0
    >>> result = BO_func(
    ...     main_series=gdp_series,
    ...     condition_series=unrate_series,
    ...     cond_thresh=0.05,
    ...     main_thresh=0,
    ...     effect_periods=2,
    ...     cond_thresh_sign='>',
    ...     main_thresh_sign='>'
    ... )
    """
    # Align and prepare condition series
    cond_shifted = condition_series.shift(lag)
    
    # Apply condition threshold comparison
    if cond_thresh_sign == '>':
        events = cond_shifted > cond_thresh
    elif cond_thresh_sign == '>=':
        events = cond_shifted >= cond_thresh
    elif cond_thresh_sign == '<':
        events = cond_shifted < cond_thresh
    elif cond_thresh_sign == '<=':
        events = cond_shifted <= cond_thresh
    elif cond_thresh_sign == '==':
        events = cond_shifted == cond_thresh
    else:
        raise ValueError(f"Invalid cond_thresh_sign: {cond_thresh_sign}")

    original = main_series.copy()
    result = main_series.copy()

    # For each event, zero out next effect_periods only if main_series meets threshold
    for idx in events[events].index:
        try:
            start_loc = result.index.get_loc(idx)
        except KeyError:
            continue
        end_loc = start_loc + effect_periods
        positions = result.index[start_loc:end_loc]
        
        # Apply main threshold condition if specified
        if main_thresh is not None:
            if main_thresh_sign == '>':
                mask = original.loc[positions] > main_thresh
            elif main_thresh_sign == '>=':
                mask = original.loc[positions] >= main_thresh
            elif main_thresh_sign == '<':
                mask = original.loc[positions] < main_thresh
            elif main_thresh_sign == '<=':
                mask = original.loc[positions] <= main_thresh
            elif main_thresh_sign == '==':
                mask = original.loc[positions] == main_thresh
            else:
                raise ValueError(f"Invalid main_thresh_sign: {main_thresh_sign}")
        else:
            # No main threshold - apply effect to all positions
            mask = pd.Series([True] * len(positions), index=positions)
        
        zero_positions = positions[mask.values]
        result.loc[zero_positions] = 0

    return result



def create_cond_var(
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
    >>> create_cond_var(
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
    >>> create_cond_var(
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
    >>> gdp_recession = create_cond_var(
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


# =============================================================================
# DOCUMENTATION FOR FUTURE CONDITIONAL FUNCTIONS
# =============================================================================
"""
Guidelines for creating new conditional functions that target variables as conditional series:

1. Required Parameters:
   - effect_periods: int - Number of periods the effect lasts
   - lag: int - Number of periods to look back in condition series
   
2. Optional Parameters:
   - cond_thresh: float - Threshold on condition series
   - main_thresh: float - Threshold on main series
   - cond_thresh_sign: str - Comparison operator for condition ('>', '>=', '<', '<=', '==')
   - main_thresh_sign: str - Comparison operator for main series ('>', '>=', '<', '<=', '==')

3. Function Signature Pattern:
   def new_cond_func(
       main_series: pd.Series,
       condition_series: pd.Series,
       cond_thresh: float,
       main_thresh: Optional[float] = None,
       effect_periods: int = 1,
       lag: int = 1,
       cond_thresh_sign: str = '>',
       main_thresh_sign: str = '>'
   ) -> pd.Series:
       # Implementation
       pass

4. Corresponding CondVar Subclass:
   class NewCond(CondVar):
       def __init__(
           self,
           var: Union[str, pd.Series],
           cond_var: Union[str, pd.Series],
           cond_thresh: float,
           main_thresh: Optional[float] = None,
           effect_periods: int = 1,
           lag: int = 1,
           cond_thresh_sign: str = '>',
           main_thresh_sign: str = '>',
           alias: Optional[str] = None
       ):
           cond_fn_kwargs = {
               'cond_thresh': cond_thresh,
               'main_thresh': main_thresh,
               'effect_periods': effect_periods,
               'lag': lag,
               'cond_thresh_sign': cond_thresh_sign,
               'main_thresh_sign': main_thresh_sign
           }
           super().__init__(
               var=var,
               cond_var=cond_var,
               cond_fn=new_cond_func,
               cond_fn_kwargs=cond_fn_kwargs,
               alias=alias
           )
"""