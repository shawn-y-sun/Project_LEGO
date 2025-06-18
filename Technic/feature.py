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
            if getattr(self, attr, None) is not None:
                continue
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