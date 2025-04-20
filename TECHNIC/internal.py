# TECHNIC/internal.py
import pandas as pd
from typing import Optional

class InternalDataLoader:
    """
    Load and standardize internal time-series data for modeling.

    Supports:
      1. CSV or Excel file input (.csv, .xlsx)
      2. Raw pandas DataFrame input
      3. Pre-indexed pandas DataFrame input

    Standardizes index to month‑ or quarter‑end dates (date‑only, no timestamps),
    then adds period dummy variables (Q1–Q4, and M1–M12 if monthly).
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
        self.source = source
        self.raw_df = df
        self.date_col = date_col
        self.start = start
        self.end = end
        self.freq = freq
        self._internal_data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        if self.source:
            df = self._load_from_file(self.source)
        elif self.raw_df is not None:
            df = self.raw_df.copy()
        else:
            raise ValueError("No source file or DataFrame provided.")
        self._internal_data = self._standardize_index(df)
        return self._internal_data

    def _load_from_file(self, path: str) -> pd.DataFrame:
        if path.lower().endswith('.csv'):
            return pd.read_csv(path)
        elif path.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def _standardize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.date_col:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            periods = pd.PeriodIndex(df[self.date_col], freq=self.freq)
            idx = periods.to_timestamp(how='end').normalize()
            df.index = idx
            df.drop(columns=[self.date_col], inplace=True)
        else:
            start_p = pd.to_datetime(self.start).to_period(self.freq)
            end_p   = pd.to_datetime(self.end).to_period(self.freq)
            periods = pd.period_range(start=start_p, end=end_p, freq=self.freq)
            idx     = periods.to_timestamp(how='end').normalize()
            df.index = idx
        df.sort_index(inplace=True)
        df = self._add_period_dummies(df)
        return df

    def _add_period_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append quarter dummies Q1–Q4, and if monthly, month dummies M1–M12.
        """
        # Quarter dummies
        quarters = df.index.quarter
        qd = pd.get_dummies(quarters, prefix='Q')
        df = pd.concat([df, qd], axis=1)

        # Monthly dummies if freq='M'
        if self.freq.upper() == 'M':
            months = df.index.month
            md = pd.get_dummies(months, prefix='M')
            df = pd.concat([df, md], axis=1)

        return df

    @property
    def internal_data(self) -> pd.DataFrame:
        if self._internal_data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._internal_data