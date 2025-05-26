# =============================================================================
# module: feature.py
# Purpose: Abstract base for feature specifications that turn raw variables
#          into model-ready pandas Series.
# Dependencies: abc, pandas
# =============================================================================

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, Dict

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
    def apply(self) -> pd.Series:
        """
        Execute transformation logic and return the feature series.

        Subclasses must first call `self.lookup(...)` to populate inputs.
        The returned Series must have its `.name` set to `self.name`.
        """
        ...
