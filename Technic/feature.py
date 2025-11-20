# =============================================================================
# module: feature.py
# Purpose: Abstract base for feature specifications that turn raw variables
#          into model-ready pandas Series.
# Dependencies: abc, pandas
# =============================================================================

import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional


class Feature(ABC):
    """
    Base class for declarative feature engineering.

    Subclasses define how one or more raw variables are transformed into
    model features. The base class provides var lookup across
    DataFrame sources and enforces a consistent interface.
    """

    def __init__(
        self,
        var: Union[str, pd.Series],
        alias: Optional[str] = None
    ):
        """
        Initialize with a var name or Series and optional alias.

        Parameters
        ----------
        var : str or pandas.Series
            Input column name to lookup or Series to use directly.
        alias : str, optional
            Custom name for the output feature; defaults to the input name.
        var_series : pandas.Series
            Cached series obtained via lookup().

        Raises
        ------
        TypeError
            If `var` is neither a string nor a pandas Series.
        """
        if isinstance(var, pd.Series):
            self.var_series = var
            self.var = var.name
        elif isinstance(var, str):
            self.var_series = None
            self.var = var
        else:
            raise TypeError("`var` must be a column name or pandas Series")

        self.alias = alias or ""
        # List of output column names after apply()
        self.output_names: List[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Identifier for the output feature: subclasses return alias or custom name.
        """
        ...

    @abstractmethod
    def lookup_map(self) -> Dict[str, str]:
        """
        Mapping of attribute names to var names for input resolution.

        E.g., {'var_series': 'GDP'} or
              {'main_series': 'price', 'cond_series': 'volume'}
        """
        ...

    def lookup(self, *dfs: pd.DataFrame) -> None:
        """
        Resolve all attributes listed in `lookup_map` against provided DataFrames.

        Always re-resolves attributes to ensure fresh data is used, which is essential
        for scenario analysis where the same Feature object may be used with different
        datasets (e.g., model MEV vs scenario-specific MEV data).

        Parameters
        ----------
        *dfs : pandas.DataFrame
            DataFrame sources to search in order.

        Raises
        ------
        KeyError
            If any var name is not found in any DataFrame.
        """
        for attr, var_name in self.lookup_map().items():
            # Always re-resolve to ensure fresh data (removed caching check)
            # This is essential for scenario analysis where Feature objects
            # are reused with different datasets
            for df in dfs:
                if df is not None and var_name in df.columns:
                    setattr(self, attr, df[var_name])
                    break
            else:
                raise KeyError(f"Var '{var_name}' for '{attr}' not found.")

    @abstractmethod
    def apply(self, *dfs: pd.DataFrame) -> pd.Series:
        """
        Execute transformation logic and return the feature series.

        Subclasses must first call `self.lookup(...)` to populate inputs.
        The returned Series must have its `.name` set to `self.name`.
        """
        ...


class DumVar(Feature):
    """
    One-hot encoding for categorical or continuous variables with multiple modes.

    Modes
    -----
    'categories' : simple dummy for each unique value or specified list (e.g. period dummies)
    'group'      : group specified categories and optionally Others
    'quantile'   : cut into equal-frequency bins (default n=5)
    'custom'     : cut into bins defined by user-provided edges

    Parameters
    ----------
    var : str or pandas.Series
        Source variable name or series.
    ode : str, default 'categories'
        One of 'categories', 'group', 'quantile', or 'custom'.
    categories : list of Any, optional
        Values or list of values to dummy; for 'categories' or 'group'.
    bins : int, default 5
        Number of bins for 'quantile' mode.
    bin_edges : list of float, optional
        Explicit bin edges for 'custom' mode.
    drop_first : bool, default True
        Drop the first dummy to avoid multicollinearity if full set.
    alias : str, optional
        Base name for resulting dummy columns.
    """
    def __init__(
        self,
        var: Union[str, pd.Series],
        mode: str = 'categories',
        categories: Optional[List[Any]] = None,
        bins: int = 5,
        bin_edges: Optional[List[float]] = None,
        drop_first: bool = True,
        alias: Optional[str] = None
    ):
        super().__init__(var=var, alias=alias)
        self.mode = mode
        self.categories = categories
        self.bins = bins
        self.bin_edges = bin_edges
        self.drop_first = drop_first

    @property
    def name(self) -> str:
        """
        Identifier for the dummy‐variable group, including selected categories or bins.

        - In 'categories' or 'group' mode, shows the exact levels joined by apostrophes:
          e.g. M:2'3'4
        - In 'quantile' mode, appends the number of bins: e.g. M:q5
        - In 'custom' mode, shows the explicit cut edges: e.g. X:bins(0-10-20)
        """
        base = self.alias or str(self.var)

        # Categorical/grouped levels
        if hasattr(self, 'categories') and self.categories:
            levels = []
            for lvl in self.categories:
                if isinstance(lvl, (list, tuple)):
                    levels.append('/'.join(map(str, lvl)))
                else:
                    levels.append(str(lvl))
            # simpler quoting: build the sep once
            sep = "'"
            return f"{base}:{sep.join(levels)}"

        # Even‐spaced quantile bins
        if getattr(self, 'mode', None) == 'quantile' and getattr(self, 'bins', None):
            return f"{base}:q{self.bins}"

        # User‐defined custom cut edges
        if getattr(self, 'mode', None) == 'custom' and getattr(self, 'bin_edges', None):
            edge_str = '-'.join(map(str, self.bin_edges))
            return f"{base}:bins({edge_str})"

        # Fallback to just the variable name or alias
        return base

    def lookup_map(self) -> Dict[str, str]:
        """
        Map var_series to source var for lookup().
        """
        return {"var_series": self.var}

    def apply(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        """
        Generate dummy variables based on the specified mode.

        Parameters
        ----------
        *dfs : pandas.DataFrame
            DataFrame sources for variable lookup.

        Returns
        -------
        pandas.DataFrame
            DataFrame of dummy columns named 'var:value'.
        """
        # Resolve series
        self.lookup(*dfs)
        series = self.var_series
        var_name = self.var

        # Prepare raw dummy mapping
        if self.mode == 'categories':
            levels = self.categories or sorted(series.dropna().unique())
            raw = pd.get_dummies(series.astype(object))
            raw = raw.reindex(columns=levels, fill_value=0)
        elif self.mode == 'group':
            groups = self.categories or []
            def mapper(x):
                for grp in groups:
                    if isinstance(grp, (list, tuple)) and x in grp:
                        return '/'.join(map(str, grp))
                    if x == grp:
                        return str(grp)
                return 'Others'
            mapped = series.map(mapper)
            raw = pd.get_dummies(mapped)
            # If user specified all categories exactly, drop 'Others'
            if self.categories:
                # flatten specified categories
                specified = set()
                for grp in self.categories:
                    if isinstance(grp, (list, tuple)):
                        specified.update(grp)
                    else:
                        specified.add(grp)
                if specified >= set(series.dropna().unique()):
                    raw = raw.drop(columns=['Others'], errors='ignore')
        elif self.mode == 'quantile':
            labels = list(range(1, self.bins + 1))
            binned = pd.qcut(series, q=self.bins, labels=labels, duplicates='drop')
            raw = pd.get_dummies(binned)
        elif self.mode == 'custom':
            if not self.bin_edges:
                raise ValueError("custom mode requires bin_edges")
            binned = pd.cut(series, bins=self.bin_edges, include_lowest=True)
            raw = pd.get_dummies(binned)
        else:
            raise ValueError(f"Unknown mode '{self.mode}' for DummyVar.")

        # Rename to 'var:value'
        raw.columns = [f"{var_name}:{col}" for col in raw.columns]

        # Determine effective drop_first
        if self.mode in ['categories', 'group']:
            unique_vals = set(series.dropna().unique())
            if self.mode == 'categories':
                specified = set(self.categories) if self.categories else unique_vals
            else:
                specified = set()
                if self.categories:
                    for grp in self.categories:
                        if isinstance(grp, (list, tuple)):
                            specified.update(grp)
                        else:
                            specified.add(grp)
                else:
                    specified = unique_vals
            full_coverage = specified >= unique_vals
            drop_first_effective = self.drop_first and full_coverage
        else:
            drop_first_effective = self.drop_first

        # Drop first column if effective and more than one
        if drop_first_effective and raw.shape[1] > 1:
            raw = raw.iloc[:, 1:]
        
        # capture all generated column names
        self.output_names = list(raw.columns)
        return raw
    
    def __repr__(self) -> str:
        """Use the `name` property as the representation, prefixed with 'DumVar:'."""
        return f"DumVar:{self.name}"

class Interaction(Feature):
    """
    Interaction term feature for combining multiple variables.
    
    Supports various interaction types:
    - 'multiply': var1 * var2
    - 'divide': var1 / var2  
    - 'add': var1 + var2
    - 'subtract': var1 - var2
    - 'ratio': var1 / var2 (with safety checks)
    - 'polynomial': var1^n * var2^m
    
    Parameters
    ----------
    var : list of str or pandas.Series
        List of variable names or Series to interact.
    interaction_type : str, default 'multiply'
        Type of interaction: 'multiply', 'divide', 'add', 'subtract', 'ratio', 'polynomial'
    powers : list of int, optional
        For polynomial interactions, powers for each variable [power1, power2, ...]
    lag : int, default 0
        Lag to apply to the second variable (useful for lead-lag interactions)
    exp_sign : int, default 0
        Expected coefficient sign for economic validation
    alias : str, optional
        Custom name for the output feature.
        
    Examples
    --------
    >>> # Simple multiplication
    >>> gdp_unemp = Interaction(['GDP', 'UNRATE'])
    >>> 
    >>> # Ratio with lag
    >>> gdp_lag_unemp = Interaction(
    ...     ['GDP', 'UNRATE'],
    ...     interaction_type='ratio',
    ...     lag=1
    ... )
    >>> 
    >>> # Polynomial interaction
    >>> gdp_sq_unemp = Interaction(
    ...     ['GDP', 'UNRATE'],
    ...     interaction_type='polynomial',
    ...     powers=[2, 1]  # GDP squared * UNRATE
    ... )
    """
    def __init__(
        self,
        var: List[Union[str, pd.Series]],
        interaction_type: str = 'multiply',
        powers: Optional[List[int]] = None,
        lag: int = 0,
        exp_sign: int = 0,
        alias: Optional[str] = None
    ):
        # Initialize with first variable as the main var
        super().__init__(var=var[0], alias=alias)

        # Store additional variables
        self.all_vars = var
        self.interaction_type = interaction_type.lower()
        self.powers = powers if powers else [1] * len(var)
        self.lag = lag
        self.exp_sign = exp_sign

        # Validate inputs
        if len(var) < 2:
            raise ValueError("At least two variables required for interaction")
        if interaction_type not in ['multiply', 'divide', 'add', 'subtract', 'ratio', 'polynomial']:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
        if powers and len(powers) != len(var):
            raise ValueError("Number of powers must match number of variables")
        
        # Initialize series cache for additional variables
        self.var_series_list: List[Optional[pd.Series]] = [None] * len(var)

    @property
    def name(self) -> str:
        """
        Generate descriptive name for the interaction feature.
        
        Format: var1_var2_TYPE[_LAGn] or var1_var2_POW[n,m]
        Examples:
        - GDP_UNRATE_MUL
        - GDP_UNRATE_DIV_LAG1
        - GDP_UNRATE_POW2,1
        """
        if self.alias:
            return self.alias
            
        # Get base variable names
        var_names = []
        for var in self.all_vars:
            if isinstance(var, pd.Series):
                var_names.append(var.name if var.name is not None else 'unnamed')
            else:
                var_names.append(str(var))
        
        # Build type suffix
        if self.interaction_type == 'polynomial':
            type_suffix = f"POW{','.join(map(str, self.powers))}"
        else:
            type_map = {
                'multiply': 'MUL',
                'divide': 'DIV',
                'add': 'ADD',
                'subtract': 'SUB',
                'ratio': 'RATIO'
            }
            type_suffix = type_map[self.interaction_type]
        
        # Add lag suffix if needed
        if self.lag > 0:
            type_suffix = f"{type_suffix}_LAG{self.lag}"
            
        return "_".join(var_names + [type_suffix])

    def lookup_map(self) -> Dict[str, str]:
        """
        Map all variables to their lookup names.
        """
        lookup_dict: Dict[str, str] = {}
        for i, var in enumerate(self.all_vars):
            if isinstance(var, str):
                lookup_dict[f"var_{i}"] = var
        return lookup_dict

    def apply(self, *dfs: pd.DataFrame) -> pd.Series:
        """
        Apply the interaction transformation to the input variables.
        
        Parameters
        ----------
        *dfs : pandas.DataFrame
            DataFrame sources for variable lookup.
            
        Returns
        -------
        pandas.Series
            Transformed interaction series.
        """
        # Resolve all variables
        series_list: List[pd.Series] = []
        for i, var in enumerate(self.all_vars):
            if isinstance(var, pd.Series):
                series = var.copy()  # Make a copy to avoid modifying original
            else:
                # Use lookup to find the variable
                for df in dfs:
                    if df is not None and var in df.columns:
                        series = df[var].copy()  # Make a copy
                        break
                else:
                    raise KeyError(f"Variable '{var}' not found in any DataFrame")
            
            # Apply lag to second variable if specified
            if i == 1 and self.lag > 0:
                series = series.shift(self.lag)
                
            series_list.append(series)
        
        # Apply powers if specified
        if self.powers:
            series_list = [s.pow(p) for s, p in zip(series_list, self.powers)]
        
        # Perform the interaction
        if self.interaction_type == 'multiply' or self.interaction_type == 'polynomial':
            result = series_list[0].copy()
            for s in series_list[1:]:
                result = result * s
        elif self.interaction_type == 'divide':
            result = series_list[0] / series_list[1]
        elif self.interaction_type == 'ratio':
            # Add small constant to denominator to avoid division by zero
            result = series_list[0] / (series_list[1] + 1e-10)
        elif self.interaction_type == 'add':
            result = pd.Series(0, index=series_list[0].index)
            for s in series_list:
                result = result + s
        elif self.interaction_type == 'subtract':
            result = series_list[0] - series_list[1]
        else:
            raise ValueError(f"Unsupported interaction type: {self.interaction_type}")
        
        # Set name and return
        result.name = self.name
        self.output_names = [self.name]
        return result

    def __repr__(self) -> str:
        """Use the name property as the representation."""
        return f"Interaction:{self.name}"