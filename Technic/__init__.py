# =============================================================================
# Package: TECHNIC
# Purpose: Expose core classes and functions for data loading, feature engineering,
#          modeling, reporting, and scenario analysis in a unified API.
# =============================================================================

"""
Project LEGO API

This package provides:
  - InternalDataLoader and MEVLoader for loading and interpolating time series data.
  - DataManager for combining internal and MEV data and engineering features.
  - FeatureSpec subclasses (TSFM, CondVar, etc.) for declarative feature transformations.
  - Candidate Models (CM) and ModelSearch for model fitting and exhaustive search.
  - Diagnostic tests (NormalityTest, StationarityTest, SignificanceTest) under test.py.
  - Segment for aggregating and reporting on multiple candidate models.
  - Scenario for scenario-based forecasting analysis.
  - Writer and template utilities for exporting results.

Importing * from this package will provide all top-level modules and classes.
"""

from .internal import *
from .mev import *
from .data import *
from .model import *
from .modeltype import *
from .feature import *
from .transform import *
from .cm import *
from .plot import*
from .report import *
from .segment import *
# from .writer import *
# from .template import *
from .test import *
from .testset import *
from .condition import *
from .scenario import *
from .search import *