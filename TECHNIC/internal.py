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

    Standardizes index to month- or quarter-end dates (date-only, no timestamps).
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
            self._internal_data = self._load_from_file(self.source)
        elif self.raw_df is not None:
            self._internal_data = self.raw_df.copy()
        else:
            raise ValueError("No source file or DataFrame provided.")
        self._standardize_index()
        return self._internal_data

    def _load_from_file(self, path: str) -> pd.DataFrame:
        if path.lower().endswith('.csv'):
            return pd.read_csv(path)
        elif path.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def _standardize_index(self):
        df = self._internal_data.copy()
        if self.date_col:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            periods = pd.PeriodIndex(df[self.date_col], freq=self.freq)
            idx = periods.to_timestamp(how='end').normalize()
            df.index = idx
            df.drop(columns=[self.date_col], inplace=True)
        else:
            start_period = pd.to_datetime(self.start).to_period(self.freq)
            end_period = pd.to_datetime(self.end).to_period(self.freq)
            periods = pd.period_range(start=start_period, end=end_period, freq=self.freq)
            idx = periods.to_timestamp(how='end').normalize()
            df.index = idx
        df.sort_index(inplace=True)
        self._internal_data = df

    @property
    def internal_data(self) -> pd.DataFrame:
        if self._internal_data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._internal_data
