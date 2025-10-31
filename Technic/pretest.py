# =============================================================================
# module: pretest.py
# Purpose: Define pre-fitting test abstractions for model search workflows.
# Key Types/Classes: PreTestSet, TargetTest, FeatureTest, SpecTest
# Key Functions: ppnr_ols_target_test_func, ppnr_ols_feature_test_func
# Dependencies: typing, pandas, .data, .test, .transform
# =============================================================================
"""Public API
=================
PreTestSet orchestrates optional pre-fitting tests for model search routines.

The module exposes lightweight wrappers that capture the required arguments for
performing target, feature, and specification level tests before expensive
model fitting occurs.

Examples
--------
>>> pretests = PreTestSet(
...     target_test=TargetTest("sales", dm, lambda data_manager, tgt: True)
... )
"""

from typing import Any, Callable, Optional, Sequence, Union

import pandas as pd

from .data import DataManager
from .feature import Feature
from .transform import TSFM
from .test import StationarityTest


class PreTestSet:
    """Container for optional pre-fitting validation tests.

    Parameters
    ----------
    target_test : Optional["TargetTest"], optional
        Test applied to the modeling target. Defaults to ``None`` when no
        target-level validation is needed.
    feature_test : Optional["FeatureTest"], optional
        Test applied to each candidate feature. Defaults to ``None`` when no
        feature-level validation is needed.
    spec_test : Optional["SpecTest"], optional
        Test applied to model specification combinations. Defaults to ``None``
        when no specification-level validation is needed.

    Examples
    --------
    >>> pretests = PreTestSet()
    >>> pretests.target_test is None
    True
    """

    def __init__(
        self,
        target_test: Optional["TargetTest"] = None,
        feature_test: Optional["FeatureTest"] = None,
        spec_test: Optional["SpecTest"] = None,
    ) -> None:
        self.target_test = target_test
        self.feature_test = feature_test
        self.spec_test = spec_test


class TargetTest:
    """Test definition for validating a modeling target prior to fitting.

    Parameters
    ----------
    target : Optional[str], optional
        Name of the target variable present in the :class:`DataManager` data.
    dm : Optional[DataManager], optional
        Data manager that provides access to the underlying dataset and helper
        utilities required for the test.
    test_func : Optional[Callable[[DataManager, str], Any]], optional
        Callable that executes the validation logic. The callable should
        typically return ``True`` when the target passes validation and
        ``False`` otherwise. Some implementations may return richer diagnostic
        objects for downstream inspection.

    Examples
    --------
    >>> def has_target(data_manager, target_name):
    ...     return target_name in data_manager.data.columns
    >>> target_test = TargetTest("sales", dm, has_target)
    """

    def __init__(
        self,
        target: Optional[str] = None,
        dm: Optional[DataManager] = None,
        test_func: Optional[Callable[[DataManager, str], Any]] = None,
    ) -> None:
        self.target = target
        self.dm = dm
        self.test_func = test_func

    @property
    def test_filter(self) -> Any:
        """Return the result of the configured target test.

        Returns
        -------
        Any
            Result produced by ``test_func``. Implementations usually return a
            boolean flag, though richer diagnostic objects may be surfaced by
            specialized tests.

        Raises
        ------
        ValueError
            If any of ``target``, ``dm``, or ``test_func`` have not been
            configured.

        Examples
        --------
        >>> target_test = TargetTest("sales", dm, lambda manager, name: True)
        >>> target_test.test_filter
        True
        """

        if self.test_func is None:
            raise ValueError(
                "TargetTest.test_func is not configured; unable to execute test."
            )
        if self.dm is None:
            raise ValueError(
                "TargetTest.dm is not configured; unable to execute test."
            )
        if self.target is None:
            raise ValueError(
                "TargetTest.target is not configured; unable to execute test."
            )

        # The property wraps the callable so callers do not need to remember
        # the invocation signature when only the stored parameters are needed.
        return self.test_func(self.dm, self.target)


class FeatureTest:
    """Test definition for validating candidate features before modeling.

    Parameters
    ----------
    feature : Optional[Union[str, Feature, TSFM]], optional
        Identifier used by :meth:`DataManager.build_feature` to create the
        feature under evaluation, or a pre-instantiated :class:`Feature`
        object ready for evaluation.
    dm : Optional[DataManager], optional
        Data manager that provides access to feature construction utilities.
    test_func : Optional[
        Callable[[Union[str, Feature, TSFM], DataManager, Optional[Any]], bool]
    ], optional
        Callable invoked with the feature identifier (or object), the data
        manager, and the result of the target test (if available). The
        callable should return ``True`` when the feature passes validation and
        ``False`` otherwise.

    Examples
    --------
    >>> def feature_exists(feature_candidate, data_manager, target_result):
    ...     return True
    >>> feature_test = FeatureTest("price", dm, feature_exists)
    >>> feature_object_test = FeatureTest(
    ...     SomeFeatureSubclass("price"), dm, feature_exists
    ... )
    >>> feature_tsfm_test = FeatureTest(SomeTSFM("price"), dm, feature_exists)
    """

    def __init__(
        self,
        feature: Optional[Union[str, Feature, TSFM]] = None,
        dm: Optional[DataManager] = None,
        test_func: Optional[
            Callable[[Union[str, Feature, TSFM], DataManager, Optional[Any]], bool]
        ] = None,
    ) -> None:
        self.feature = feature
        self.dm = dm
        self.test_func = test_func
        self.target_test_result: Optional[Any] = None

    @property
    def test_filter(self) -> bool:
        """Return the outcome of the feature validation callable.

        Returns
        -------
        bool
            ``True`` when the feature is considered valid given the stored
            target test result and ``False`` otherwise.

        Raises
        ------
        ValueError
            If any of ``feature``, ``dm``, or ``test_func`` have not been
            configured.

        Examples
        --------
        >>> feature_test = FeatureTest("price", dm, lambda feat, manager, _: True)
        >>> feature_test.test_filter
        True
        """

        if self.test_func is None:
            raise ValueError(
                "FeatureTest.test_func is not configured; unable to execute test."
            )
        if self.dm is None:
            raise ValueError(
                "FeatureTest.dm is not configured; unable to execute test."
            )
        if self.feature is None:
            raise ValueError(
                "FeatureTest.feature is not configured; unable to execute test."
            )

        # Include the cached target test result (possibly ``None``) to respect
        # the callable's full signature without burdening callers.
        return self.test_func(self.feature, self.dm, self.target_test_result)


class SpecTest:
    """Test definition for validating model specification combinations.

    Parameters
    ----------
    specs : Optional[Sequence[object]], optional
        Model specification objects compatible with :meth:`CM.build`.
    dm : Optional[DataManager], optional
        Data manager that provides context for assessing the specifications.
    test_func : Optional[Callable[[Sequence[object], DataManager], bool]], optional
        Callable that evaluates the specifications. The callable should return
        ``True`` when the specification passes validation and ``False``
        otherwise.

    Examples
    --------
    >>> def specs_non_empty(spec_list, data_manager):
    ...     return bool(spec_list)
    >>> spec_test = SpecTest(["price"], dm, specs_non_empty)
    """

    def __init__(
        self,
        specs: Optional[Sequence[object]] = None,
        dm: Optional[DataManager] = None,
        test_func: Optional[Callable[[Sequence[object], DataManager], bool]] = None,
    ) -> None:
        self.specs = specs
        self.dm = dm
        self.test_func = test_func

    @property
    def test_filter(self) -> bool:
        """Return whether the specification validation passes.

        Returns
        -------
        bool
            ``True`` when the specification passes validation and ``False``
            otherwise.

        Raises
        ------
        ValueError
            If any of ``specs``, ``dm``, or ``test_func`` have not been
            configured.

        Examples
        --------
        >>> spec_test = SpecTest(["price"], dm, lambda spec_list, manager: True)
        >>> spec_test.test_filter
        True
        """

        if self.test_func is None:
            raise ValueError(
                "SpecTest.test_func is not configured; unable to execute test."
            )
        if self.dm is None:
            raise ValueError(
                "SpecTest.dm is not configured; unable to execute test."
            )
        if self.specs is None:
            raise ValueError(
                "SpecTest.specs is not configured; unable to execute test."
            )

        # Encapsulate the callable invocation to keep the property usage simple.
        return self.test_func(self.specs, self.dm)


def ppnr_ols_target_test_func(
    dm: DataManager,
    target: str,
    sample: str = "in",
) -> pd.DataFrame:
    """Run stationarity diagnostics for a PPNR OLS target series.

    Parameters
    ----------
    dm : DataManager
        Data manager containing the internal data and sample indices.
    target : str
        Column name of the target variable within ``dm.internal_data``.
    sample : {"in", "full"}, default "in"
        Which portion of the internal data to test. ``"in"`` restricts the
        analysis to in-sample observations. ``"full"`` concatenates the
        in-sample and out-of-sample segments to reflect the complete available
        history.

    Returns
    -------
    pd.DataFrame
        Stationarity diagnostics returned by :class:`StationarityTest.test_result`.
        The ``filter_mode_desc`` entry is mirrored into ``DataFrame.attrs`` so
        callers can present the descriptive text before printing the table.

    Raises
    ------
    KeyError
        If ``target`` is not present in the internal data.
    ValueError
        If ``sample`` is not one of the supported options.

    Examples
    --------
    >>> diagnostics = ppnr_ols_target_test_func(dm, "NII", sample="full")
    >>> diagnostics.loc["ADF", "Passed"]
    True
    """

    normalized_sample = str(sample).lower()
    if normalized_sample not in {"in", "full"}:
        raise ValueError(
            "sample must be either 'in' or 'full'; "
            f"received {sample!r}."
        )

    internal_data = dm.internal_data

    if target not in internal_data.columns:
        raise KeyError(
            f"Target '{target}' not found in DataManager.internal_data columns."
        )

    target_series = internal_data[target]

    # Align the target series to the requested sample scope while preserving
    # chronological ordering across contiguous segments.
    in_sample_series = target_series.loc[dm.in_sample_idx]
    if normalized_sample == "in":
        scoped_series = in_sample_series
    else:
        out_sample_idx = dm.out_sample_idx
        if out_sample_idx is None or len(out_sample_idx) == 0:
            scoped_series = in_sample_series
        else:
            out_sample_series = target_series.loc[out_sample_idx]
            scoped_series = pd.concat([in_sample_series, out_sample_series])

    stationarity_test = StationarityTest(series=scoped_series)
    result = stationarity_test.test_result
    # Preserve the descriptive text so upstream callers can surface it prior to
    # displaying the full diagnostic table.
    result.attrs["filter_mode_desc"] = getattr(
        stationarity_test,
        "filter_mode_desc",
        ""
    )
    return result


def _summarize_stationarity_result(result: pd.DataFrame) -> Optional[bool]:
    """Return a boolean summary of StationarityTest results."""

    if result.empty or "Passed" not in result.columns:
        return None

    passed_series = result["Passed"].fillna(False).astype(bool)
    total_tests = len(passed_series)
    if total_tests == 0:
        return None

    return bool(passed_series.sum() >= (total_tests / 2))


def _coerce_target_result(target_result: Optional[Any]) -> Optional[bool]:
    """Convert a stored target test result into a boolean when possible."""

    if target_result is None:
        return None
    if isinstance(target_result, bool):
        return target_result
    if isinstance(target_result, pd.DataFrame):
        return _summarize_stationarity_result(target_result)
    if hasattr(target_result, "test_filter"):
        try:
            return bool(target_result.test_filter)
        except Exception:
            return None
    if isinstance(target_result, pd.Series):
        try:
            return bool(target_result.all())
        except ValueError:
            return None
    try:
        return bool(target_result)
    except Exception:
        return None


def ppnr_ols_feature_test_func(
    feature: Union[str, Feature, TSFM],
    dm: DataManager,
    target_test_result: Optional[Any],
    sample: str = "in",
) -> bool:
    """Evaluate PPNR OLS feature stationarity relative to the target outcome.

    Parameters
    ----------
    feature : Union[str, Feature, TSFM]
        Identifier, feature object, or TSFM transform to be materialized via
        :meth:`DataManager.build_features`.
    dm : DataManager
        Data manager that provides feature construction utilities and sample
        indices.
    target_test_result : Any, optional
        Cached outcome from the target-level test. ``True`` indicates a
        stationary target, ``False`` denotes a non-stationary target, and
        ``None`` defers the decision to the feature diagnostics.
    sample : {"in", "full"}, default "in"
        Portion of the feature history to evaluate. ``"in"`` restricts to the
        in-sample span, while ``"full"`` includes both in-sample and
        out-of-sample observations when available.

    Returns
    -------
    bool
        ``True`` when the feature satisfies the stationarity preference implied
        by the target test outcome.

    Raises
    ------
    ValueError
        If ``sample`` is not ``"in"`` or ``"full"``.

    Examples
    --------
    >>> ppnr_ols_feature_test_func("GDP", dm, True)
    True
    """

    normalized_sample = str(sample).lower()
    if normalized_sample not in {"in", "full"}:
        raise ValueError(
            "sample must be either 'in' or 'full'; "
            f"received {sample!r}."
        )

    feature_frame = dm.build_features([feature])
    if isinstance(feature_frame, pd.Series):
        feature_frame = feature_frame.to_frame()

    if feature_frame.empty:
        # No data implies nothing to invalidate; treat as passing.
        return True

    in_sample_idx = dm.in_sample_idx
    scoped_segments = []

    # Build the evaluation window according to the requested sample scope.

    if in_sample_idx is not None:
        in_sample_idx = feature_frame.index.intersection(in_sample_idx)
        if len(in_sample_idx) > 0:
            scoped_segments.append(feature_frame.loc[in_sample_idx])

    if normalized_sample == "full":
        out_sample_idx = dm.out_sample_idx
        if out_sample_idx is not None:
            out_sample_idx = feature_frame.index.intersection(out_sample_idx)
            if len(out_sample_idx) > 0:
                scoped_segments.append(feature_frame.loc[out_sample_idx])

    if scoped_segments:
        scoped_frame = pd.concat(scoped_segments, axis=0).sort_index()
    else:
        scoped_frame = feature_frame.sort_index()

    eligible_columns = [
        col for col in scoped_frame.columns if ":" not in str(col)
    ]

    if isinstance(feature, Feature):
        desired_columns = [
            name for name in feature.output_names if ":" not in str(name)
        ]
        if desired_columns:
            eligible_columns = [
                col for col in eligible_columns if col in desired_columns
            ]

    if not eligible_columns:
        # Nothing qualifies for stationarity testing, so accept by default.
        return True

    # Each feature invocation yields a single logical output set, so iterate
    # directly over the filtered columns in their materialized order.
    column_outcomes = []
    for column in eligible_columns:
        series = scoped_frame[column].dropna()
        if series.empty:
            column_outcomes.append(False)
            continue

        stationarity_test = StationarityTest(series=series)
        result_df = stationarity_test.test_result
        column_pass = _summarize_stationarity_result(result_df)
        column_outcomes.append(False if column_pass is None else column_pass)

    # Require all eligible outputs for the feature to align with the target.
    feature_pass = all(column_outcomes) if column_outcomes else True

    target_pass = _coerce_target_result(target_test_result)
    if target_pass is False:
        return not feature_pass
    return feature_pass


# NOTE: Provide a reusable pre-test bundle for PPNR OLS model searches.
ppnr_ols_pretestset: PreTestSet = PreTestSet(
    target_test=TargetTest(test_func=ppnr_ols_target_test_func),
    feature_test=FeatureTest(test_func=ppnr_ols_feature_test_func),
)
