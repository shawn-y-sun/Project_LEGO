# =============================================================================
# module: modeltype.py
# Purpose: Model type classes for handling different modeling approaches and conversions
# Dependencies: pandas, numpy, abc
# =============================================================================

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union
import warnings

from .data import DataManager


class ModelType(ABC):
    """
    Abstract base class for model types.
    
    Model types define how to convert between the modeled target variable
    and the original variable of interest. This is essential for model
    performance assessment and scenario analysis.
    
    Example
    -------
    >>> # This is an abstract class - use concrete subclasses
    >>> # growth_model = Growth(dm, target="gdp_growth", original_var="gdp_level")
    """
    pass


class TimeModelType(ModelType):
    """
    Base class for time series model types.
    
    Handles conversion between modeled target variables and base target variables
    of interest for time series data. Each subclass implements a specific
    modeling approach (growth, change, levels, ratios).
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance containing the data
    target : str
        Name of the target variable being modeled
    target_base : str, optional
        Name of the base variable of interest.
        Not required for RateLevel and BalanceLevel classes.
    target_exposure : str, optional
        Name of the exposure variable (denominator).
        Only required for Ratio class.
        
    Attributes
    ----------
    dm : DataManager
        Reference to the DataManager instance
    target : str
        Name of the target variable
    target_base : str or None
        Name of the base variable of interest
    target_exposure : str or None
        Name of the exposure variable
        
    Example
    -------
    >>> # This is a base class - use concrete subclasses like Growth, Change, etc.
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> growth_model = Growth(dm, target="balance_growth", target_base="balance")
    """
    
    def __init__(
        self,
        dm: DataManager,
        target: str,
        target_base: Optional[str] = None,
        target_exposure: Optional[str] = None
    ):
        """
        Initialize TimeModelType.
        
        Parameters
        ----------
        dm : DataManager
            DataManager instance
        target : str
            Name of target variable
        target_base : str, optional
            Name of base variable of interest
        target_exposure : str, optional
            Name of exposure variable (for Ratio class)
        """
        self.dm = dm
        self.target = target
        self.target_base = target_base
        self.target_exposure = target_exposure
        
        # Validate that target exists in internal data
        if target not in self.dm.internal_data.columns:
            raise ValueError(f"Target variable '{target}' not found in internal data")
        
        # Validate target_base if provided
        if target_base and target_base not in self.dm.internal_data.columns:
            raise ValueError(f"Base variable '{target_base}' not found in internal data")
            
        # Validate target_exposure if provided
        if target_exposure and target_exposure not in self.dm.internal_data.columns:
            raise ValueError(f"Exposure variable '{target_exposure}' not found in internal data")

    @abstractmethod
    def predict_base(
        self,
        y_pred: pd.Series,
        p0: pd.Timestamp
    ) -> pd.Series:
        """
        Convert predicted target values back to base variable of interest.
        
        Parameters
        ----------
        y_pred : pd.Series
            Series of predicted target values from the model
        p0 : pd.Timestamp
            Date index to use as the starting point for conversion
            
        Returns
        -------
        pd.Series
            Series of converted values for the base variable of interest
        """
        pass


class PanelModelType(ModelType):
    """
    Base class for panel data model types.
    
    Placeholder for future panel data model type implementations.
    Currently not implemented as the focus is on time series data.
    
    Example
    -------
    >>> # Future implementation for panel data
    >>> # panel_model = PanelGrowth(dm, target="account_growth", original_var="account_balance")
    """
    pass


class Growth(TimeModelType):
    """
    Model type for percentage change/growth modeling.
    
    Models the percentage change or growth rate of data over time.
    Converts growth rate predictions back to level values using compound growth.
    
    Formula: final_result[t] = final_result[t-1] * (1 + predicted_growth[t])
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance
    target : str
        Name of the growth rate target variable
    target_base : str
        Name of the base level variable of interest
        
    Example
    -------
    >>> # Model GDP growth rate, convert back to GDP level
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> growth_model = Growth(
    ...     dm=dm,
    ...     target="gdp_growth_rate",
    ...     target_base="gdp_level"
    ... )
    >>> 
    >>> # Convert growth predictions to level predictions
    >>> growth_preds = pd.Series([0.02, 0.025, 0.03], name="gdp_growth_rate")
    >>> p0 = pd.Timestamp("2023-12-31")
    >>> level_preds = growth_model.predict_base(growth_preds, p0)
    """
    
    def __init__(self, dm: DataManager, target: str, target_base: str):
        """
        Initialize Growth model type.
        
        Parameters
        ----------
        dm : DataManager
            DataManager instance
        target : str
            Name of growth rate target variable
        target_base : str
            Name of base level variable (required)
        """
        if not target_base:
            raise ValueError("Growth model type requires target_base parameter")
        super().__init__(dm, target, target_base)

    def predict_base(
        self,
        y_pred: pd.Series,
        p0: pd.Timestamp
    ) -> pd.Series:
        """
        Convert growth rate predictions to level predictions using compound growth.
        
        Parameters
        ----------
        y_pred : pd.Series
            Series of predicted growth rates
        p0 : pd.Timestamp
            Starting date for conversion (uses level at this date)
            
        Returns
        -------
        pd.Series
            Series of predicted levels for the base variable
            
        Example
        -------
        >>> # Growth rates: 2%, 2.5%, 3%
        >>> growth_preds = pd.Series([0.02, 0.025, 0.03])
        >>> p0 = pd.Timestamp("2023-12-31")
        >>> # If base level at p0 is 1000:
        >>> # Result: [1020, 1045.5, 1076.865]
        >>> levels = growth_model.predict_base(growth_preds, p0)
        """
        if y_pred.empty:
            return pd.Series([], name=self.target_base)
            
        # Get the jumpoff level from base variable
        base_data = self.dm.internal_data[self.target_base]
        
        if p0 not in base_data.index:
            raise ValueError(f"P0 date {p0} not found in base variable data")
            
        jumpoff_level = base_data.loc[p0]
        
        # Apply compound growth formula
        result = pd.Series(index=y_pred.index, name=self.target_base, dtype=float)
        current_level = jumpoff_level
        
        for date in y_pred.index:
            current_level = current_level * (1 + y_pred.loc[date])
            result.loc[date] = current_level
            
        return result


class Change(TimeModelType):
    """
    Model type for difference/change modeling.
    
    Models the period-to-period difference of data over time.
    Converts change predictions back to level values using cumulative addition.
    
    Formula: final_result[t] = final_result[t-1] + predicted_change[t]
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance
    target : str
        Name of the change target variable
    target_base : str
        Name of the base level variable of interest
        
    Example
    -------
    >>> # Model balance changes, convert back to balance levels
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> change_model = Change(
    ...     dm=dm,
    ...     target="balance_change",
    ...     target_base="balance_level"
    ... )
    >>> 
    >>> # Convert change predictions to level predictions
    >>> change_preds = pd.Series([100, 150, -50], name="balance_change")
    >>> p0 = pd.Timestamp("2023-12-31")
    >>> level_preds = change_model.predict_base(change_preds, p0)
    """
    
    def __init__(self, dm: DataManager, target: str, target_base: str):
        """
        Initialize Change model type.
        
        Parameters
        ----------
        dm : DataManager
            DataManager instance
        target : str
            Name of change target variable
        target_base : str
            Name of base level variable (required)
        """
        if not target_base:
            raise ValueError("Change model type requires target_base parameter")
        super().__init__(dm, target, target_base)

    def predict_base(
        self,
        y_pred: pd.Series,
        p0: pd.Timestamp
    ) -> pd.Series:
        """
        Convert change predictions to level predictions using cumulative addition.
        
        Parameters
        ----------
        y_pred : pd.Series
            Series of predicted changes
        p0 : pd.Timestamp
            Starting date for conversion (uses level at this date)
            
        Returns
        -------
        pd.Series
            Series of predicted levels for the base variable
            
        Example
        -------
        >>> # Changes: +100, +150, -50
        >>> change_preds = pd.Series([100, 150, -50])
        >>> p0 = pd.Timestamp("2023-12-31")
        >>> # If base level at p0 is 1000:
        >>> # Result: [1100, 1250, 1200]
        >>> levels = change_model.predict_base(change_preds, p0)
        """
        if y_pred.empty:
            return pd.Series([], name=self.target_base)
            
        # Get the jumpoff level from base variable
        base_data = self.dm.internal_data[self.target_base]
        
        if p0 not in base_data.index:
            raise ValueError(f"P0 date {p0} not found in base variable data")
            
        jumpoff_level = base_data.loc[p0]
        
        # Apply cumulative change formula
        result = pd.Series(index=y_pred.index, name=self.target_base, dtype=float)
        current_level = jumpoff_level
        
        for date in y_pred.index:
            current_level = current_level + y_pred.loc[date]
            result.loc[date] = current_level
            
        return result


class RateLevel(TimeModelType):
    """
    Model type for rate level modeling.
    
    Models the rate itself without transformation from base variable.
    For more comparable results, converts the modeled target to differences
    first, then applies them back to the actual jumpoff rate level.
    
    Formula: final_result[t] = final_result[t-1] + diff(predicted_rate)[t]
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance
    target : str
        Name of the rate target variable (also serves as base variable)
        
    Example
    -------
    >>> # Model interest rates directly
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> rate_model = RateLevel(
    ...     dm=dm,
    ...     target="interest_rate"
    ... )
    >>> 
    >>> # Convert rate predictions to adjusted rate predictions
    >>> rate_preds = pd.Series([0.05, 0.052, 0.048], name="interest_rate")
    >>> p0 = pd.Timestamp("2023-12-31")
    >>> adjusted_rates = rate_model.predict_base(rate_preds, p0)
    """
    
    def __init__(self, dm: DataManager, target: str):
        """
        Initialize RateLevel model type.
        
        Parameters
        ----------
        dm : DataManager
            DataManager instance
        target : str
            Name of rate target variable
        """
        # For RateLevel, target and target_base are the same
        super().__init__(dm, target, target_base=target)

    def predict_base(
        self,
        y_pred: pd.Series,
        p0: pd.Timestamp
    ) -> pd.Series:
        """
        Convert rate predictions using difference approach for comparability.
        
        Parameters
        ----------
        y_pred : pd.Series
            Series of predicted rates
        p0 : pd.Timestamp
            Starting date for conversion (uses rate at this date)
            
        Returns
        -------
        pd.Series
            Series of adjusted rate predictions
            
        Example
        -------
        >>> # Rate predictions: 5.0%, 5.2%, 4.8%
        >>> rate_preds = pd.Series([0.05, 0.052, 0.048])
        >>> p0 = pd.Timestamp("2023-12-31")
        >>> # If rate at p0 is 4.5%:
        >>> # Differences from prediction: [0.05-0.05, 0.052-0.05, 0.048-0.052]
        >>> # Result: [4.5%, 4.7%, 4.2%]
        >>> adjusted = rate_model.predict_base(rate_preds, p0)
        """
        if y_pred.empty:
            return pd.Series([], name=self.target)
            
        # Get the jumpoff rate
        base_data = self.dm.internal_data[self.target]
        
        if p0 not in base_data.index:
            raise ValueError(f"P0 date {p0} not found in target variable data")
            
        jumpoff_rate = base_data.loc[p0]
        
        # Calculate differences in predicted rates
        rate_diffs = y_pred.diff()
        
        # Apply cumulative differences starting from jumpoff rate
        result = pd.Series(index=y_pred.index, name=self.target, dtype=float)
        current_rate = jumpoff_rate
        
        for i, date in enumerate(y_pred.index):
            if i == 0:
                # For first prediction, use the difference from jumpoff rate to first prediction
                current_rate = jumpoff_rate + (y_pred.iloc[0] - jumpoff_rate)
            else:
                # For subsequent predictions, apply the difference
                current_rate = current_rate + rate_diffs.iloc[i]
            result.loc[date] = current_rate
            
        return result


class BalanceLevel(TimeModelType):
    """
    Model type for balance level modeling.
    
    Models the balance itself without transformation from base variable.
    For more comparable results, converts the modeled target to growth rates
    first, then applies them back to the actual jumpoff balance level.
    
    Formula: final_result[t] = final_result[t-1] * growth(predicted_balance)[t]
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance
    target : str
        Name of the balance target variable (also serves as base variable)
        
    Example
    -------
    >>> # Model account balances directly
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> balance_model = BalanceLevel(
    ...     dm=dm,
    ...     target="account_balance"
    ... )
    >>> 
    >>> # Convert balance predictions to adjusted balance predictions
    >>> balance_preds = pd.Series([10000, 10500, 11000], name="account_balance")
    >>> p0 = pd.Timestamp("2023-12-31")
    >>> adjusted_balances = balance_model.predict_base(balance_preds, p0)
    """
    
    def __init__(self, dm: DataManager, target: str):
        """
        Initialize BalanceLevel model type.
        
        Parameters
        ----------
        dm : DataManager
            DataManager instance
        target : str
            Name of balance target variable
        """
        # For BalanceLevel, target and target_base are the same
        super().__init__(dm, target, target_base=target)

    def predict_base(
        self,
        y_pred: pd.Series,
        p0: pd.Timestamp
    ) -> pd.Series:
        """
        Convert balance predictions using growth rate approach for comparability.
        
        Parameters
        ----------
        y_pred : pd.Series
            Series of predicted balances
        p0 : pd.Timestamp
            Starting date for conversion (uses balance at this date)
            
        Returns
        -------
        pd.Series
            Series of adjusted balance predictions
            
        Example
        -------
        >>> # Balance predictions: 10000, 10500, 11000
        >>> balance_preds = pd.Series([10000, 10500, 11000])
        >>> p0 = pd.Timestamp("2023-12-31")
        >>> # If balance at p0 is 9500:
        >>> # Growth rates: [10000/10000, 10500/10000, 11000/10500]
        >>> # Result: [9500, 9975, 10472.5]
        >>> adjusted = balance_model.predict_base(balance_preds, p0)
        """
        if y_pred.empty:
            return pd.Series([], name=self.target)
            
        # Get the jumpoff balance
        base_data = self.dm.internal_data[self.target]
        
        if p0 not in base_data.index:
            raise ValueError(f"P0 date {p0} not found in target variable data")
            
        jumpoff_balance = base_data.loc[p0]
        
        # Calculate growth rates from predicted balances
        growth_rates = y_pred.pct_change()
        
        # Apply compound growth starting from jumpoff balance
        result = pd.Series(index=y_pred.index, name=self.target, dtype=float)
        current_balance = jumpoff_balance
        
        for i, date in enumerate(y_pred.index):
            if i == 0:
                # For first prediction, use the ratio of first prediction to jumpoff
                if jumpoff_balance != 0:
                    current_balance = jumpoff_balance * (y_pred.iloc[0] / jumpoff_balance)
                else:
                    current_balance = y_pred.iloc[0]
            else:
                # For subsequent predictions, apply the growth rate
                if not pd.isna(growth_rates.iloc[i]):
                    current_balance = current_balance * (1 + growth_rates.iloc[i])
                else:
                    current_balance = y_pred.iloc[i]
            result.loc[date] = current_balance
            
        return result


class Ratio(TimeModelType):
    """
    Model type for ratio modeling.
    
    Models the ratio of one level variable as percentage of another exposure
    level variable. Converts ratio predictions back to level values by
    multiplying with the exposure variable levels.
    
    Formula: final_result[t] = exposure_level[t] * predicted_ratio[t]
    
    Parameters
    ----------
    dm : DataManager
        DataManager instance
    target : str
        Name of the ratio target variable
    target_base : str
        Name of the base level variable of interest (numerator)
    target_exposure : str
        Name of the exposure level variable (denominator)
        
    Example
    -------
    >>> # Model outflow ratio = outflow_balance / eligible_balance
    >>> dm = DataManager(internal_loader, mev_loader)
    >>> ratio_model = Ratio(
    ...     dm=dm,
    ...     target="outflow_ratio",
    ...     target_base="outflow_balance",
    ...     target_exposure="eligible_balance"
    ... )
    >>> 
    >>> # Convert ratio predictions to balance predictions
    >>> ratio_preds = pd.Series([0.15, 0.18, 0.12], name="outflow_ratio")
    >>> p0 = pd.Timestamp("2023-12-31")
    >>> balance_preds = ratio_model.predict_base(ratio_preds, p0)
    """
    
    def __init__(self, dm: DataManager, target: str, target_base: str, target_exposure: str):
        """
        Initialize Ratio model type.
        
        Parameters
        ----------
        dm : DataManager
            DataManager instance
        target : str
            Name of ratio target variable
        target_base : str
            Name of base level variable (numerator, required)
        target_exposure : str
            Name of exposure level variable (denominator, required)
        """
        if not target_base:
            raise ValueError("Ratio model type requires target_base parameter")
        if not target_exposure:
            raise ValueError("Ratio model type requires target_exposure parameter")
        super().__init__(dm, target, target_base, target_exposure)

    def predict_base(
        self,
        y_pred: pd.Series,
        p0: pd.Timestamp
    ) -> pd.Series:
        """
        Convert ratio predictions to level predictions using exposure variable.
        
        Parameters
        ----------
        y_pred : pd.Series
            Series of predicted ratios
        p0 : pd.Timestamp
            Starting date for conversion (not directly used for jumpoff,
            but used to validate date availability)
            
        Returns
        -------
        pd.Series
            Series of predicted levels for the base variable
            
        Example
        -------
        >>> # Ratio predictions: 15%, 18%, 12%
        >>> ratio_preds = pd.Series([0.15, 0.18, 0.12])
        >>> # If eligible_balance at those dates: [10000, 11000, 12000]
        >>> # Result: [1500, 1980, 1440]
        >>> levels = ratio_model.predict_base(ratio_preds, p0)
        """
        if y_pred.empty:
            return pd.Series([], name=self.target_base)
            
        # Get exposure variable data
        exposure_data = self.dm.internal_data[self.target_exposure]
        
        # Check that we have exposure variable data for prediction dates
        missing_dates = y_pred.index.difference(exposure_data.index)
        if not missing_dates.empty:
            warnings.warn(
                f"Exposure variable '{self.target_exposure}' missing for dates: {missing_dates}. "
                "These predictions will be NaN."
            )
        
        # Apply ratio formula: base = exposure * ratio
        result = pd.Series(index=y_pred.index, name=self.target_base, dtype=float)
        
        for date in y_pred.index:
            if date in exposure_data.index:
                exposure_level = exposure_data.loc[date]
                result.loc[date] = exposure_level * y_pred.loc[date]
            else:
                result.loc[date] = np.nan
                
        return result 