# =============================================================================
# module: periods.py
# Purpose: Normalize transform period arguments across modeling utilities.
# Key Types/Classes: None
# Key Functions: resolve_periods_argument, default_periods_for_freq
# Dependencies: warnings, typing
# =============================================================================
"""Utility helpers for reconciling period arguments across the codebase."""

from typing import List, Optional, Sequence, Union
import warnings

PeriodInput = Optional[Sequence[int]]


def default_periods_for_freq(freq: Optional[str]) -> List[int]:
    """Return the recommended default periods for a given frequency.

    Parameters
    ----------
    freq : str, optional
        Frequency code such as ``'M'`` or ``'Q'``. Monthly data receives the
        canonical ``[1, 3, 6, 12]`` window set, quarterly data receives
        ``[1, 2, 3, 4]``. Any other frequency falls back to ``[1]``.

    Returns
    -------
    list of int
        Recommended period values for the supplied frequency.

    Examples
    --------
    >>> default_periods_for_freq('M')
    [1, 3, 6, 12]
    >>> default_periods_for_freq('Q')
    [1, 2, 3, 4]
    """

    if freq == 'M':
        return [1, 3, 6, 12]
    if freq == 'Q':
        return [1, 2, 3, 4]
    return [1]


def resolve_periods_argument(
    freq: Optional[str],
    periods: PeriodInput,
    *,
    legacy_max_periods: Optional[Union[int, Sequence[int]]] = None,
    ensure_quarterly_floor: bool = False
) -> Optional[List[int]]:
    """Return a normalized list of positive periods or ``None``.

    Parameters
    ----------
    freq : str, optional
        Frequency code (for example ``'M'`` or ``'Q'``) used to match legacy
        behavior when expanding integer inputs.
    periods : Sequence[int], optional
        Candidate period configuration. ``None`` delegates to downstream
        defaults (monthly ``[1, 3, 6, 12]``; quarterly ``[1, 2, 3, 4]``). The
        sequence must contain strictly positive integers. Passing an integer is
        not supported; use the legacy ``max_periods`` argument for that
        behaviour.
    legacy_max_periods : int or Sequence[int], optional
        Deprecated argument mirroring the previous ``max_periods`` keyword.
        When supplied, a :class:`DeprecationWarning` is emitted and
        ``periods`` must be ``None``. Monthly inputs expanded from the legacy
        integer form exclude periods ``2`` and ``9`` to align with updated
        defaults.
    ensure_quarterly_floor : bool, default False
        When ``True`` and quarterly frequency is detected, guarantee that the
        normalized list covers ``[1, 2, 3, 4]`` even if the caller requested
        a shorter range.

    Returns
    -------
    list of int or None
        Normalized periods (positive integers without duplicates) or ``None``
        when the caller defers to downstream defaults.

    Raises
    ------
    TypeError
        If provided sequences contain non-integers or unsupported types.
    ValueError
        If any provided integer is less than one.

    Examples
    --------
    >>> resolve_periods_argument('M', [1, 3, 6, 12])
    [1, 3, 6, 12]
    >>> resolve_periods_argument('Q', None)
    >>> resolve_periods_argument('Q', None, legacy_max_periods=2, ensure_quarterly_floor=True)
    [1, 2, 3, 4]
    """

    use_legacy = legacy_max_periods is not None
    legacy_from_int = isinstance(legacy_max_periods, int)

    if use_legacy:
        warnings.warn(
            "'max_periods' is deprecated; pass 'periods' with explicit values instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if periods is not None:
            raise ValueError("Cannot supply both 'periods' and deprecated 'max_periods'.")
        candidate = legacy_max_periods
    else:
        candidate = periods

    if candidate is None:
        return None

    def _expand_from_int(limit: int) -> List[int]:
        if limit < 1:
            raise ValueError("Period values must be positive integers.")
        if freq == 'M' and limit > 3:
            values = [1, 2, 3]
            for value in range(6, limit + 1, 3):
                values.append(value)
            return values
        return list(range(1, limit + 1))

    normalized: List[int]

    if isinstance(candidate, int):
        if not use_legacy:
            raise TypeError("'periods' must be provided as a sequence of positive integers.")
        normalized = _expand_from_int(candidate)
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        normalized = []
        for value in candidate:
            if not isinstance(value, int):
                raise TypeError("All period entries must be integers.")
            if value < 1:
                raise ValueError("Period values must be positive integers.")
            if value not in normalized:
                normalized.append(value)
    else:
        raise TypeError("'periods' must be None or a sequence of integers.")

    if use_legacy and legacy_from_int and freq == 'M':
        # Legacy monthly expansion historically included 2 and 9, but both are
        # now excluded to better align with the recommended defaults.
        normalized = [value for value in normalized if value not in {2, 9}]
        if not normalized:
            normalized = [1]

    if ensure_quarterly_floor and freq == 'Q':
        floor = {1, 2, 3, 4}
        normalized = sorted(set(normalized).union(floor))

    return normalized
