# =============================================================================
# module: pretest.py
# Purpose: Define pre-fitting test abstractions for model search workflows.
# Key Types/Classes: PreTestSet, BasePreTest, TargetTest, FeatureTest, SpecTest
# Key Functions: ppnr_ols_target_pretestset_func, ppnr_ols_feature_pretestset_func
# Dependencies: typing, .data, .test, .transform, .testset
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
...     target_test=TargetTest(subject="sales", dm=dm, testset_func=lambda *args: {})
... )
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from .data import DataManager
from .feature import Feature
from .test import FullStationarityTest, ModelTestBase, TargetStationarityTest
from .testset import TestSet
from .transform import TSFM


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
    context_map : Mapping[str, Sequence[str]], optional
        Routing rules controlling which context keys are forwarded to which
        pre-tests during :meth:`propagate_context`. Keys correspond to
        ``"target_test"``, ``"feature_test"``, ``"spec_test"``, or ``"*"`` for
        a default rule. ``None`` (default) forwards all keys to all configured
        pre-tests.

    Examples
    --------
    >>> pretests = PreTestSet()
    >>> pretests.target_test is None
    True

    Notes
    -----
    Use :meth:`propagate_target_result` to share a target pre-test outcome with
    feature-level tests so downstream expectations stay aligned with target
    stationarity.
    """

    def __init__(
        self,
        target_test: Optional["TargetTest"] = None,
        feature_test: Optional["FeatureTest"] = None,
        spec_test: Optional["SpecTest"] = None,
        context_map: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.target_test = target_test
        self.feature_test = feature_test
        self.spec_test = spec_test
        self.context_map = self._normalize_context_map(context_map)
        self._context: Dict[str, Any] = {}

    @staticmethod
    def _normalize_context_map(
        context_map: Optional[Mapping[str, Sequence[str]]]
    ) -> Dict[str, Sequence[str]]:
        """Validate and normalize context routing rules.

        Parameters
        ----------
        context_map : mapping, optional
            Mapping of pre-test attribute names (``"target_test"``,
            ``"feature_test"``, ``"spec_test"``, or ``"*"`` for a default) to
            sequences of context keys that should be forwarded when
            :meth:`propagate_context` is called. ``None`` permits all keys to be
            forwarded to every configured pre-test.

        Returns
        -------
        dict
            Normalized mapping where keys are strings and values are tuples of
            context keys. An empty tuple indicates that no context keys are
            forwarded for that entry.

        Raises
        ------
        TypeError
            If a mapping key is not a string or any mapping value is not a
            sequence of strings.
        """

        if context_map is None:
            return {}

        normalized: Dict[str, Sequence[str]] = {}
        for pretest_name, keys in context_map.items():
            if not isinstance(pretest_name, str):
                raise TypeError("context_map keys must be strings")

            if keys is None:
                normalized[pretest_name] = ()
                continue

            if isinstance(keys, (str, bytes)):
                raise TypeError("context_map values must be sequences of strings")

            if not isinstance(keys, Sequence):
                raise TypeError("context_map values must be sequences of strings")

            normalized[pretest_name] = tuple(str(key) for key in keys)

        return normalized

    def propagate_context(self, context: Mapping[str, Any]) -> None:
        """Share contextual test outcomes across managed pre-tests.

        Parameters
        ----------
        context : Mapping[str, Any]
            Arbitrary context payload (e.g., target outcomes, domain flags)
            that downstream pre-tests can interpret via their
            :meth:`BasePreTest.apply_context` implementations. Context routing
            follows :pyattr:`context_map` so only the configured keys are
            forwarded to each pre-test.

        Examples
        --------
        >>> bundle = PreTestSet(
        ...     feature_test=FeatureTest(testset_func=lambda *args: {}),
        ...     context_map={"feature_test": ("target_result",)},
        ... )
        >>> bundle.propagate_context({"target_result": False})
        """

        # Store the combined context so repeated calls merge new values without
        # discarding previous entries that may be needed by multiple tests.
        self._context.update(context)

        pretests = {
            "target_test": self.target_test,
            "feature_test": self.feature_test,
            "spec_test": self.spec_test,
        }

        # Honor per-pretest routing rules; fall back to the default mapping or
        # broadcast the full context when no rules are provided. An empty tuple
        # explicitly blocks all keys from being forwarded to a given pre-test.
        for name, pretest in pretests.items():
            if pretest is None:
                continue

            allowed_keys = self.context_map.get(name, self.context_map.get("*"))

            if allowed_keys is None:
                payload = self._context
            elif len(allowed_keys) == 0:
                payload = {}
            else:
                payload = {key: value for key, value in self._context.items() if key in allowed_keys}

            pretest.apply_context(payload)

    def propagate_target_result(self, target_result: Optional[bool]) -> None:
        """Propagate the target pre-test outcome to the feature pre-test.

        Parameters
        ----------
        target_result : bool, optional
            Pass/fail flag produced by :class:`TargetTest.test_filter`. ``True``
            indicates a stationary target and keeps feature expectations
            aligned with the default stationary requirement. ``False`` flips the
            expectation so non-stationary features are treated as passing.

        Examples
        --------
        >>> bundle = PreTestSet(
        ...     target_test=TargetTest(testset_func=lambda *args: {}),
        ...     feature_test=FeatureTest(testset_func=lambda *args: {}),
        ... )
        >>> bundle.propagate_target_result(False)
        """

        if target_result is None:
            return

        # Preserve backward compatibility with existing flows while delegating
        # to the broader context propagation mechanism so additional
        # dependencies can opt-in without stationarity-specific wiring.
        self.propagate_context({"target_result": target_result})


class BasePreTest(ABC):
    """Abstract base for pre-model tests.

    Parameters
    ----------
    subject : object, optional
        Domain object evaluated by the test (e.g., target name, feature spec,
        or spec list).
    dm : DataManager, optional
        Data manager providing context and sample indices.
    sample : {"in", "full"}, default "in"
        Sample slice to use when constructing the test set.
    outlier_idx : Sequence[Any], optional
        Indices to remove when the downstream tests support outlier exclusion.
    testset_func : callable, optional
        Function that returns a mapping of test alias to :class:`ModelTestBase`
        given ``subject``, ``dm``, ``sample``, and ``outlier_idx``.
    test_update_func : callable, optional
        Function that returns updates to apply to the base test mapping. The
        callable may accept no arguments or any subset of the constructor
        context (``subject``, ``dm``, ``sample``, ``outlier_idx``) and should
        return a dictionary whose values are :class:`ModelTestBase` instances
        (added or replaced) or dictionaries of attribute overrides targeting
        existing aliases.
    force_filter_pass : Optional[bool], optional
        Override for the computed :pyattr:`test_filter` result. When provided,
        the boolean value is returned directly, bypassing assembled test set
        evaluation.

    Raises
    ------
    ValueError
        If ``sample`` is not ``"in"`` or ``"full"``.

    Notes
    -----
    Subclasses expose :pyattr:`testset` and :pyattr:`test_filter` properties
    built from ``testset_func`` and ``test_update_func`` so callers only need to
    populate the constructor fields before evaluation.
    """

    def __init__(
        self,
        *,
        subject: Optional[object] = None,
        dm: Optional[DataManager] = None,
        sample: str = "in",
        outlier_idx: Optional[Sequence[Any]] = None,
        testset_func: Optional[
            Callable[[object, DataManager, str, Optional[Sequence[Any]]], Dict[str, ModelTestBase]]
        ] = None,
        test_update_func: Optional[Callable[..., Dict[str, Any]]] = None,
        force_filter_pass: Optional[bool] = None,
    ) -> None:
        normalized_sample = str(sample).lower()
        if normalized_sample not in {"in", "full"}:
            raise ValueError("sample must be either 'in' or 'full'")

        if force_filter_pass is not None and not isinstance(force_filter_pass, bool):
            raise TypeError("force_filter_pass must be a boolean or None")

        self.subject = subject
        self.dm = dm
        self.sample = normalized_sample
        self.outlier_idx = list(outlier_idx) if outlier_idx else []
        self.testset_func = testset_func
        self.test_update_func = test_update_func
        self.force_filter_pass = force_filter_pass
        self.external_context: Dict[str, Any] = {}

    @property
    def testset(self) -> TestSet:
        """Instantiate the test set for the configured subject.

        Returns
        -------
        TestSet
            TestSet instance reflecting ``testset_func`` with optional updates
            from ``test_update_func``.

        Raises
        ------
        ValueError
            If ``dm`` or ``testset_func`` is not provided.
        TypeError
            If an update entry is neither a :class:`ModelTestBase` nor an
            override dictionary.
        """

        if self.dm is None:
            raise ValueError("dm must be provided before building testset")
        if self.testset_func is None:
            raise ValueError("testset_func must be provided before building testset")

        # Wrap the pretest-specific builder to match the
        # ``TestSet.from_functions`` signature while preserving the subject,
        # sample, and outlier context captured by this instance and updates.
        def _base_builder(_: object) -> Dict[str, ModelTestBase]:
            return self.testset_func(self.subject, self.dm, self.sample, self.outlier_idx)

        return TestSet.from_functions(
            self,
            _base_builder,
            self.test_update_func if self.test_update_func else None,
            subject=self.subject,
            dm=self.dm,
            sample=self.sample,
            outlier_idx=self.outlier_idx,
        )

    @property
    def test_filter(self) -> bool:
        """Return the filter outcome from the assembled test set."""

        passed, _ = self.testset.filter_pass()
        return self._apply_force_filter_pass(passed)

    def _apply_force_filter_pass(self, result: bool) -> bool:
        """Apply the forced filter pass override when supplied."""

        if self.force_filter_pass is None:
            return bool(result)
        return bool(self.force_filter_pass)

    def apply_context(self, context: Mapping[str, Any]) -> None:
        """Store externally provided context for dependency-aware tests.

        Parameters
        ----------
        context : Mapping[str, Any]
            Arbitrary context payload supplied by orchestrators (e.g.,
            :class:`PreTestSet`) to coordinate expectations across related
            pre-tests.
        """

        # Shallow copy preserves immutability expectations for callers while
        # giving subclasses a consistent context attribute to inspect.
        self.external_context = dict(context)


class TargetTest(BasePreTest):
    """Test definition for validating a modeling target prior to fitting.

    Parameters
    ----------
    subject : str, optional
        Target variable name within :class:`DataManager.internal_data`.
    dm : DataManager, optional
        Data manager that provides access to the underlying dataset.
    sample : {"in", "full"}, default "in"
        Sample slice used by the downstream test set.
    outlier_idx : Sequence[Any], optional
        Outlier indices to exclude when the underlying tests support it.
    testset_func : callable, optional
        Function returning a dictionary of target tests; see
        :class:`BasePreTest` for details.
    test_update_func : callable, optional
        Optional update function; see :class:`BasePreTest` for details.

    Examples
    --------
    >>> target_test = TargetTest(subject="sales", dm=dm, testset_func=lambda *args: {})
    >>> target_test.test_filter
    True
    """

    def __init__(
        self,
        subject: Optional[str] = None,
        dm: Optional[DataManager] = None,
        sample: str = "in",
        outlier_idx: Optional[Sequence[Any]] = None,
        testset_func: Optional[
            Callable[[object, DataManager, str, Optional[Sequence[Any]]], Dict[str, ModelTestBase]]
        ] = None,
        test_update_func: Optional[Callable[..., Dict[str, Any]]] = None,
        force_filter_pass: Optional[bool] = None,
    ) -> None:
        super().__init__(
            subject=subject,
            dm=dm,
            sample=sample,
            outlier_idx=outlier_idx,
            testset_func=testset_func,
            test_update_func=test_update_func,
            force_filter_pass=force_filter_pass,
        )

    @property
    def target(self) -> Optional[str]:
        """Backward-compatible alias for ``subject``."""

        return self.subject if isinstance(self.subject, str) else None

    @target.setter
    def target(self, value: Optional[str]) -> None:
        self.subject = value


class FeatureTest(BasePreTest):
    """Test definition for validating candidate features before modeling.

    Parameters
    ----------
    subject : Optional[Union[str, Feature, TSFM]], optional
        Identifier used by :meth:`DataManager.build_feature` to create the
        feature under evaluation, or a pre-instantiated :class:`Feature` object.
    dm : DataManager, optional
        Data manager that provides access to feature construction utilities.
    sample : {"in", "full"}, default "in"
        Sample slice used by the downstream test set.
    outlier_idx : Sequence[Any], optional
        Outlier indices to exclude when supported by the underlying tests.
    testset_func : callable, optional
        Function returning a dictionary of feature tests; see
        :class:`BasePreTest` for details.
    test_update_func : callable, optional
        Optional update function; see :class:`BasePreTest` for details.
    target_test_result : bool, optional
        Outcome from the target-level pretest. When provided, ``False`` flips
        the expectation so non-stationary features satisfy the filter to mirror
        a non-stationary target.

    Attributes
    ----------
    target_test_result : Optional[bool]
        Cached outcome from the target-level test for downstream consumers.
    """

    def __init__(
        self,
        subject: Optional[Union[str, Feature, TSFM]] = None,
        dm: Optional[DataManager] = None,
        sample: str = "in",
        outlier_idx: Optional[Sequence[Any]] = None,
        testset_func: Optional[
            Callable[[object, DataManager, str, Optional[Sequence[Any]]], Dict[str, ModelTestBase]]
        ] = None,
        test_update_func: Optional[Callable[..., Dict[str, Any]]] = None,
        target_test_result: Optional[bool] = None,
        force_filter_pass: Optional[bool] = None,
    ) -> None:
        super().__init__(
            subject=subject,
            dm=dm,
            sample=sample,
            outlier_idx=outlier_idx,
            testset_func=testset_func,
            test_update_func=test_update_func,
            force_filter_pass=force_filter_pass,
        )
        self.target_test_result: Optional[bool] = (
            None if target_test_result is None else bool(target_test_result)
        )

    @property
    def feature(self) -> Optional[Union[str, Feature, TSFM]]:
        """Backward-compatible alias for ``subject``."""

        return self.subject if isinstance(self.subject, (str, Feature, TSFM)) else None

    @feature.setter
    def feature(self, value: Optional[Union[str, Feature, TSFM]]) -> None:
        self.subject = value

    def apply_target_test_result(self, target_result: Optional[bool]) -> None:
        """Store the target pre-test outcome for expectation alignment.

        Parameters
        ----------
        target_result : bool, optional
            Pass/fail flag returned by :class:`TargetTest.test_filter`. ``True``
            preserves the standard feature requirement of stationarity, while
            ``False`` flips expectations so non-stationary features satisfy the
            filter.

        Examples
        --------
        >>> feat_test = FeatureTest(testset_func=lambda *args: {})
        >>> feat_test.apply_target_test_result(False)
        """

        if target_result is None:
            self.target_test_result = None
            return

        # Maintain backwards compatibility for direct calls while funneling
        # through the shared context mechanism.
        self.apply_context({"target_result": target_result})

    def apply_context(self, context: Mapping[str, Any]) -> None:
        """Ingest shared context and update target-driven expectations.

        Parameters
        ----------
        context : Mapping[str, Any]
            Context payload produced by :class:`PreTestSet` or external
            orchestrators. The feature test consumes ``"target_result"`` (or
            ``"target_test_result"``) when available to align expectations with
            target stationarity outcomes. Additional keys are retained on
            :pyattr:`external_context` for downstream customization in other
            model types.
        """

        super().apply_context(context)
        context_value = context.get("target_result", context.get("target_test_result"))
        self.target_test_result = None if context_value is None else bool(context_value)

    @property
    def expected_stationary(self) -> Optional[bool]:
        """Return the target-informed expectation for feature stationarity."""

        return None if self.target_test_result is None else bool(self.target_test_result)

    @property
    def test_filter(self) -> bool:
        """
        Evaluate feature tests respecting target-driven stationarity expectations.

        Returns
        -------
        bool
            ``True`` when feature outcomes align with the desired stationarity
            state inferred from the target pre-test; defaults to the raw test
            results when no target guidance is present.
        """

        passed, _ = self.testset.filter_pass()
        expected_stationary = self.expected_stationary
        if expected_stationary is None:
            return self._apply_force_filter_pass(passed)

        # When the target is non-stationary, invert the outcome so non-stationary
        # features satisfy the dependency requirement.
        return self._apply_force_filter_pass(passed if expected_stationary else not passed)


class SpecTest(BasePreTest):
    """Test definition for validating model specification combinations.

    Parameters
    ----------
    subject : Optional[Sequence[object]], optional
        Model specification objects compatible with :meth:`CM.build`.
    dm : DataManager, optional
        Data manager that provides context for assessing the specifications.
    sample : {"in", "full"}, default "in"
        Sample slice used by the downstream test set.
    outlier_idx : Sequence[Any], optional
        Outlier indices to exclude when supported by the underlying tests.
    testset_func : callable, optional
        Function returning a dictionary of specification tests; see
        :class:`BasePreTest` for details.
    test_update_func : callable, optional
        Optional update function; see :class:`BasePreTest` for details.
    """

    def __init__(
        self,
        subject: Optional[Sequence[object]] = None,
        dm: Optional[DataManager] = None,
        sample: str = "in",
        outlier_idx: Optional[Sequence[Any]] = None,
        testset_func: Optional[
            Callable[[object, DataManager, str, Optional[Sequence[Any]]], Dict[str, ModelTestBase]]
        ] = None,
        test_update_func: Optional[Callable[..., Dict[str, Any]]] = None,
        force_filter_pass: Optional[bool] = None,
    ) -> None:
        super().__init__(
            subject=subject,
            dm=dm,
            sample=sample,
            outlier_idx=outlier_idx,
            testset_func=testset_func,
            test_update_func=test_update_func,
            force_filter_pass=force_filter_pass,
        )

    @property
    def specs(self) -> Optional[Sequence[object]]:
        """Backward-compatible alias for ``subject``."""

        return self.subject if isinstance(self.subject, Sequence) else None

    @specs.setter
    def specs(self, value: Optional[Sequence[object]]) -> None:
        self.subject = value


def ppnr_ols_target_pretestset_func(
    subject: object,
    dm: DataManager,
    sample: str,
    outlier_idx: Optional[Sequence[Any]],
) -> Dict[str, ModelTestBase]:
    """Build target-level PPNR OLS tests.

    Parameters
    ----------
    subject : object
        Target identifier expected to be a column name in
        :attr:`DataManager.internal_data`.
    dm : DataManager
        Data manager providing access to target history and sample indices.
    sample : {"in", "full"}
        Sample scope used by :class:`TargetStationarityTest`.
    outlier_idx : Sequence[Any], optional
        Index labels removed for outlier-adjusted diagnostics.

    Returns
    -------
    dict
        Mapping with a single ``"Target Stationarity"`` entry.
    """

    if subject is None:
        raise ValueError("subject must be provided for target pretests")

    target_name = str(subject)
    test = TargetStationarityTest(
        target=target_name,
        dm=dm,
        sample=sample,
        outlier_idx=list(outlier_idx) if outlier_idx else None,
        filter_mode="moderate",
        filter_on=True,
    )
    return {"Target Stationarity": test}


def ppnr_ols_feature_pretestset_func(
    subject: object,
    dm: DataManager,
    sample: str,
    outlier_idx: Optional[Sequence[Any]],
) -> Dict[str, ModelTestBase]:
    """Build feature-level PPNR OLS tests.

    Parameters
    ----------
    subject : object
        Feature identifier passed through to :class:`FullStationarityTest`.
    dm : DataManager
        Data manager supplying feature construction utilities and sample spans.
    sample : {"in", "full"}
        Sample scope used by :class:`FullStationarityTest`.
    outlier_idx : Sequence[Any], optional
        Index labels removed for outlier-adjusted diagnostics.

    Returns
    -------
    dict
        Mapping with a single ``"Feature Stationarity"`` entry.
    """

    if subject is None:
        raise ValueError("subject must be provided for feature pretests")

    test = FullStationarityTest(
        variable=subject,
        dm=dm,
        sample=sample,
        outlier_idx=list(outlier_idx) if outlier_idx else None,
        filter_mode="moderate",
        filter_on=True,
    )
    return {"Feature Stationarity": test}


# NOTE: Provide a reusable pre-test bundle for PPNR OLS model searches.
ppnr_ols_pretestset: PreTestSet = PreTestSet(
    target_test=TargetTest(testset_func=ppnr_ols_target_pretestset_func),
    feature_test=FeatureTest(testset_func=ppnr_ols_feature_pretestset_func),
    context_map={"feature_test": ("target_result", "target_test_result")},
)
