# TECHNIC/base.py
from abc import ABC, abstractmethod
import pandas as pd

class MoedlBase(ABC):
    """
    Abstract base class for statistical models.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.coefs_ = None
        self.fitted = False

    @abstractmethod
    def fit(self):
        """Fit the model to X and y."""
        pass

    @abstractmethod
    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data."""
        pass
