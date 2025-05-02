# TECHNIC/condition.py

import pandas as pd
from typing import Callable, Union, List, Any


class CondVar:
    """
    Conditional variable manager.

    Creates a new Series by applying a function to a main variable
    and one or more condition variables.

    Parameters:
    - main_var: pd.Series
        The main data series to be transformed.
    - cond_vars: Union[pd.Series, List[pd.Series]]
        One or more condition data series.
    - cond_fn: Callable[..., pd.Series]
        Function taking (main_series, *cond_series, **kwargs) -> Series.
    - kwargs: Any
        Additional keyword arguments for cond_fn.
    """
    def __init__(
        self,
        main_var: pd.Series,
        cond_vars: Union[pd.Series, List[pd.Series]],
        cond_fn: Callable[..., pd.Series],
        **kwargs: Any
    ):
        if not isinstance(main_var, pd.Series):
            raise TypeError("`main_var` must be a pandas Series")
        self.main_var = main_var
        self.main_name = main_var.name or "main"

        if isinstance(cond_vars, pd.Series):
            self.cond_vars = [cond_vars]
        elif isinstance(cond_vars, list) and all(isinstance(cv, pd.Series) for cv in cond_vars):
            self.cond_vars = cond_vars
        else:
            raise TypeError(
                "`cond_vars` must be a pandas Series or list of pandas Series"
            )

        self.cond_fn = cond_fn
        self.fn_kwargs = kwargs

    def apply(self) -> pd.Series:
        """
        Apply the conditional function to main_var and cond_vars.
        Returns a Series named by the `name` property.
        """
        result = self.cond_fn(self.main_var, *self.cond_vars, **self.fn_kwargs)
        if not isinstance(result, pd.Series):
            raise ValueError("`cond_fn` must return a pandas Series")
        result.name = self.name
        return result

    @property
    def name(self) -> str:
        """
        Generate a name for the conditional variable:
        mainName_condFnName.
        """
        fn_name = getattr(self.cond_fn, "__name__", "cond")
        return f"{self.main_name}_{fn_name}"