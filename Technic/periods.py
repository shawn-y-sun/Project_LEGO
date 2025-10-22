# =============================================================================
# module: periods.py
# Purpose: Normalize transform period arguments across modeling utilities.
# Key Types/Classes: None
# Key Functions: resolve_periods_argument
# Dependencies: warnings, typing
# =============================================================================
"""Utility helpers for reconciling period arguments across the codebase."""

from typing import List, Optional, Sequence, Union
import warnings

PeriodInput = Optional[Union[int, Sequence[int]]]


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
    periods : int or Sequence[int], optional
        Candidate period configuration. ``None`` delegates to downstream
        defaults (monthly ``[1, 3, 6, 12]``; quarterly ``[1, 2, 3, 4]``).
        Integer inputs mimic the legacy ``max_periods`` semantics by expanding
        to consecutive values (and, for monthly data, multiples of three).
    legacy_max_periods : int or Sequence[int], optional
        Deprecated argument mirroring the previous ``max_periods`` keyword.
        When supplied, a :class:`DeprecationWarning` is emitted and
        ``periods`` must be ``None``.
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
    >>> resolve_periods_argument('M', 12)
    [1, 2, 3, 6, 9, 12]
    >>> resolve_periods_argument('Q', None)
    >>> resolve_periods_argument('Q', 2, ensure_quarterly_floor=True)
    [1, 2, 3, 4]
    """

    if legacy_max_periods is not None:
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
        raise TypeError("'periods' must be None, an int, or a sequence of integers.")

    if ensure_quarterly_floor and freq == 'Q':
        floor = {1, 2, 3, 4}
        normalized = sorted(set(normalized).union(floor))

    return normalized
