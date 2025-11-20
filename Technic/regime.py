# =============================================================================
# module: regime.py
# Purpose: Regime-aware feature that activates a variable based on a regime flag.
# Key Types/Classes: RgmVar
# Key Functions: None
# Dependencies: pandas, typing, .feature.Feature, .transform.TSFM
# =============================================================================

from typing import Dict, Optional, Union

import pandas as pd

from .feature import Feature
from .transform import TSFM


class RgmVar(Feature):
    """
    Regime-aware feature that activates or deactivates a variable.

    Parameters
    ----------
    var : str or TSFM
        Name of the base variable or a :class:`TSFM` transformation to evaluate.
    regime : str
        Column name for the regime indicator (expected 0/1 or boolean series).
    on : bool or int, default True
        When True/1 the variable is active during regime==1; when False/0 the
        variable is active during regime==0.
    exp_sign : int, default 0
        Expected coefficient sign for economic validation:
        - 1: expect positive coefficient (positive relationship with target)
        - -1: expect negative coefficient (negative relationship with target)
        - 0: no expectation (no sign constraint)
    alias : str, optional
        Custom output name. Defaults to ``"{var_name}@{regime}_{on}"``.
    freq : str, optional
        Data frequency (e.g., ``"M"`` or ``"Q"``) carried for downstream
        compatibility. Not used directly in transformation logic.

    Raises
    ------
    TypeError
        If ``regime`` is not a string or ``var`` is neither string nor TSFM.
    ValueError
        If ``on`` is not interpretable as 0 or 1.

    Examples
    --------
    >>> rgm = RgmVar("GDP", regime="recession", on=1)
    >>> rgm.apply(df)  # activates GDP only when `recession` == 1
    >>> tsfm_var = TSFM("RATE", "GR")
    >>> RgmVar(tsfm_var, regime="tightening", on=0).apply(df)
    """

    def __init__(
        self,
        var: Union[str, TSFM],
        regime: str,
        on: Union[bool, int] = True,
        exp_sign: int = 0,
        alias: Optional[str] = None,
        freq: Optional[str] = None,
    ) -> None:
        if not isinstance(regime, str):
            raise TypeError("`regime` must be provided as a column name string.")

        if isinstance(var, TSFM):
            self.var_feature = var
            var_for_super: Union[str, pd.Series] = var.var
            var_label = var.name
        elif isinstance(var, str):
            self.var_feature = None
            var_for_super = var
            var_label = var
        else:
            raise TypeError("`var` must be a column name string or a TSFM instance.")

        normalized_on = int(on)
        if normalized_on not in (0, 1):
            raise ValueError("`on` must be interpretable as 0/1 or boolean.")

        super().__init__(var=var_for_super, alias=alias)
        self.regime = regime
        self.on = normalized_on
        self.exp_sign = exp_sign
        self.freq = freq
        self._var_label = var_label

    @property
    def name(self) -> str:
        """
        Output series name combining base variable and regime indicator.

        Returns
        -------
        str
            Alias if provided, otherwise ``"{var_name}@{regime}_{on}"``.
        """

        return self.alias or f"{self._var_label}@{self.regime}_{self.on}"

    def __repr__(self) -> str:
        """
        Represent the feature using its resolved name for readability.

        Returns
        -------
        str
            ``self.name`` to mirror the output feature label.
        """

        return self.name

    def lookup_map(self) -> Dict[str, str]:
        """
        Map attributes to column names for lookup.

        Returns
        -------
        dict
            Includes ``regime_series`` and, when ``var`` is a string, ``var_series``.
        """

        mapping: Dict[str, str] = {"regime_series": self.regime}
        if self.var_feature is None:
            mapping["var_series"] = self.var
        return mapping

    def apply(self, *dfs: pd.DataFrame) -> pd.Series:
        """
        Apply the regime filter to the underlying variable.

        Parameters
        ----------
        *dfs : pandas.DataFrame
            DataFrames to search for the base variable and regime indicator.

        Returns
        -------
        pandas.Series
            Regime-adjusted series with name set to ``self.name``.

        Raises
        ------
        KeyError
            If either the variable or regime column cannot be resolved.
        """

        # Resolve regime (and base variable when provided as a string).
        self.lookup(*dfs)

        if self.var_feature is None:
            var_series = self.var_series
        else:
            # Delegate transformation to provided TSFM feature using the same data sources.
            var_series = self.var_feature.apply(*dfs)

        regime_series = self.regime_series.astype(int)

        if self.on == 1:
            final_series = var_series * regime_series
        else:
            final_series = var_series * (1 - regime_series)

        final_series = final_series.copy()
        final_series.name = self.name
        self.output_names = [final_series.name]
        return final_series
