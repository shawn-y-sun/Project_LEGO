# TECHNIC/condition.py

import pandas as pd
from typing import Callable, Union, List, Any, Dict


class CondVar:
    """
    Conditional variable manager.

    Creates a new Series by applying a function to a main variable
    and one or more condition variables.

    Parameters:
    - main_var: Union[str, pd.Series]
        Name or series of the main variable to be transformed.
    - cond_var: Union[str, pd.Series, List[Union[str, pd.Series]]]
        One or more condition variable names or Series.
    - cond_fn: Callable[..., pd.Series]
        Function taking (main, *cond_vars, **cond_fn_kwargs) -> Series.
    - cond_fn_kwargs: Dict[str, Any]
        Keyword arguments to pass to cond_fn.
    """
    def __init__(
        self,
        main_var: Union[str, pd.Series],
        cond_var: Union[str, pd.Series, List[Union[str, pd.Series]]],
        cond_fn: Callable[..., pd.Series],
        cond_fn_kwargs: Dict[str, Any]
    ):
        # Main variable can be name or series
        if isinstance(main_var, pd.Series):
            self.main_series = main_var
            self.main_name = main_var.name or "main"
        elif isinstance(main_var, str):
            self.main_series = None
            self.main_name = main_var
        else:
            raise TypeError("`main_var` must be a column name string or a pandas Series")

        # Condition variables list
        if isinstance(cond_var, (str, pd.Series)):
            self.cond_var = [cond_var]
        elif isinstance(cond_var, list) and all(
            isinstance(cv, (str, pd.Series)) for cv in cond_var
        ):
            self.cond_var = cond_var
        else:
            raise TypeError(
                "`cond_var` must be a column name, Series, or list thereof"
            )

        self.cond_fn = cond_fn
        self.cond_fn_kwargs = cond_fn_kwargs

    def apply(self) -> pd.Series:
        """
        Apply the conditional function to main_var and cond_var(s).
        Returns a Series named by the `name` property.
        """
        # Prepare args: use series or names
        args = [
            self.main_series if self.main_series is not None else self.main_name
        ]
        for cv in self.cond_var:
            args.append(cv if isinstance(cv, pd.Series) else cv)

        result = self.cond_fn(*args, **self.cond_fn_kwargs)
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