# =============================================================================
# module: internal.py
# Purpose: InternalDataLoader for loading, standardizing, and enriching internal time-series data
# Dependencies: pandas
# =============================================================================

import pandas as pd
from typing import Optional, Tuple


class InternalDataLoader:
    """
    Load and standardize internal time-series data for modeling.

    Supports:
      - CSV or Excel file input (.csv, .xls, .xlsx)
      - Raw pandas DataFrame input
      - Pre-indexed pandas DataFrame input

    After loading, the data is reindexed to period-end timestamps (monthly or quarterly),
    normalized (date-only), and enriched with period dummy variables (Q1–Q4, and M1–M12 if monthly).

    Attributes
    ----------
    source : Optional[str]
        Path to the CSV/Excel file, if provided.
    raw_df : Optional[pd.DataFrame]
        Raw DataFrame provided directly, if no file source.
    date_col : Optional[str]
        Column name to parse as dates when reading from file.
    start : Optional[str]
        Start date (YYYY-MM-DD) when no date_col is given.
    end : Optional[str]
        End date (YYYY-MM-DD) when no date_col is given.
    freq : str
        Frequency code ('M' for monthly, 'Q' for quarterly) to use for period indexing.
    _internal_data : Optional[pd.DataFrame]
        Cached, processed DataFrame after `load()`.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        date_col: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        freq: str = 'M'
    ):
        # Input parameters
        self.source = source            # file path if provided
        self.raw_df = df                # DataFrame directly passed
        self.date_col = date_col       # column to parse as dates
        self.start = start             # fallback start date string
        self.end = end                 # fallback end date string
        self.freq = freq               # period frequency ('M' or 'Q')
        self._internal_data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load the internal data from file or raw_df, standardize its index,
        and add period dummy variables.

        Returns
        -------
        pd.DataFrame
            Processed internal data with period-end index and dummy columns.

        Raises
        ------
        ValueError
            If neither `source` nor `raw_df` is provided.
        """
        # Load from file or use provided DataFrame
        if self.source:
            df = self._load_from_file(self.source)
        elif self.raw_df is not None:
            df = self.raw_df.copy()
        else:
            raise ValueError("No source file or DataFrame provided.")

        # Standardize index and add dummy variables
        self._internal_data = self._standardize_index(df)
        return self._internal_data

    def _load_from_file(self, path: str) -> pd.DataFrame:
        """
        Read a CSV or Excel file into a DataFrame.

        Parameters
        ----------
        path : str
            File path ending with .csv, .xls, or .xlsx.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame from file.

        Raises
        ------
        ValueError
            If file extension is not supported.
        """
        ext = path.lower()
        if ext.endswith('.csv'):
            return pd.read_csv(path)
        elif ext.endswith(('.xls', '.xlsx')):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def _standardize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a date column or provided start/end into a period-end DatetimeIndex,
        sort the DataFrame, and append period dummy variables.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with either a date column or to be indexed by start/end.

        Returns
        -------
        pd.DataFrame
            DataFrame reindexed and enriched with dummy variables.
        """
        df = df.copy()
        if self.date_col:
            # Parse the date column and set as period-end index
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            periods = pd.PeriodIndex(df[self.date_col], freq=self.freq)
            idx = periods.to_timestamp(how='end').normalize()
            df.index = idx
            df.drop(columns=[self.date_col], inplace=True)
        else:
            # Use explicit start/end to generate a full period index
            start_p = pd.to_datetime(self.start).to_period(self.freq)
            end_p   = pd.to_datetime(self.end).to_period(self.freq)
            periods = pd.period_range(start=start_p, end=end_p, freq=self.freq)
            idx = periods.to_timestamp(how='end').normalize()
            df.index = idx

        df.sort_index(inplace=True)
        return self._add_period_dummies(df)

    def _add_period_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append quarter and month dummy variables based on DataFrame index.

        Quarter dummies: Q1, Q2, Q3, Q4
        Monthly dummies (if freq='M'): M1, M2, ..., M12

        Existing dummy columns are dropped first to avoid duplication.
        """
        df = df.copy()

        # Define dummy column names
        q_cols = [f"Q{i}" for i in range(1, 5)]
        m_cols = [f"M{i}" for i in range(1, 13)] if self.freq.upper() == 'M' else []

        # Drop any existing dummy columns
        drop_cols = [col for col in q_cols + m_cols if col in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        # Generate quarter dummies
        quarters = df.index.quarter
        qd = pd.get_dummies(quarters, prefix='', prefix_sep='').rename(
            columns={i: f"Q{i}" for i in range(1,5)}
        )
        for col in q_cols:
            df[col] = qd[col]

        # Generate month dummies if monthly frequency
        if m_cols:
            months = df.index.month
            md = pd.get_dummies(months, prefix='', prefix_sep='').rename(
                columns={i: f"M{i}" for i in range(1,13)}
            )
            for col in m_cols:
                df[col] = md[col]

        return df

    @property
    def internal_data(self) -> pd.DataFrame:
        """
        Accessor for the processed internal data.

        Raises
        ------
        ValueError
            If `load()` has not been called yet.
        """
        if self._internal_data is None:
            raise ValueError("Internal data not loaded. Call `load()` first.")
        return self._internal_data
