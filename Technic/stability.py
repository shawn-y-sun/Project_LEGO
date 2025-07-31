# =============================================================================
# module: stability.py
# Purpose: Model stability testing including Walk Forward Test
# Dependencies: typing, numpy, pandas, matplotlib, .model.ModelBase
# =============================================================================

from typing import Type, Dict, List, Union, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from .model import ModelBase

class ModelStabilityTest(ABC):
    """
    Abstract base class for model stability testing.
    
    This class defines the interface for model stability testing implementations.
    Subclasses should implement specific stability testing methodologies.
    
    Parameters
    ----------
    model_cls : Type[ModelBase]
        ModelBase subclass to use for creating model instances.
    dm : DataManager
        DataManager instance with stability testing configurations.
    specs : List[Union[str, Dict[str, Any]]]
        Feature specifications to pass to model instances.
    target : str
        Name of the target variable in the internal data.
    model_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to model_cls constructor.
    """
    
    def __init__(
        self,
        model_cls: Type[ModelBase],
        dm: Any,
        specs: List[Union[str, Dict[str, Any]]],
        target: str,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ModelStabilityTest instance.
        
        Parameters
        ----------
        model_cls : Type[ModelBase]
            ModelBase subclass to use for creating model instances.
        dm : DataManager
            DataManager instance with stability testing capability.
        specs : List[Union[str, Dict[str, Any]]]
            Feature specifications for model training.
        target : str
            Name of target variable in internal data.
        model_kwargs : Dict[str, Any], optional
            Additional arguments for model constructor.
        """
        self.model_cls = model_cls
        self.dm = dm
        self.specs = specs
        self.target = target
        self.model_kwargs = model_kwargs or {}
    
    @abstractmethod
    def show_all(self) -> None:
        """
        Display comprehensive stability test results.
        
        This method should provide a complete view of all stability test results
        including plots and summary tables. The specific implementation depends
        on the stability testing methodology.
        """
        pass

class WalkForwardTest(ModelStabilityTest):
    """
    Walk Forward Test for model stability analysis.
    
    This class manages model training and testing across multiple pseudo-out-of-sample
    periods to assess model stability and performance degradation over time.
    
    The Walk Forward Test works by:
    1. Creating multiple training windows with progressively shorter in-sample periods
    2. Training models on each reduced training set
    3. Testing on the corresponding pseudo out-of-sample periods
    4. Comparing performance across different training windows
    
    Parameters
    ----------
    model_cls : Type[ModelBase]
        ModelBase subclass to use for creating model instances (e.g., OLS).
    dm : DataManager
        DataManager instance with pseudo-out-of-sample configurations.
        Must have poos_dms property available.
    specs : List[Union[str, Dict[str, Any]]]
        Feature specifications to pass to model instances.
        Same format as used in ModelBase classes.
    target : str
        Name of the target variable in the internal data.
    model_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to model_cls constructor.
        Common examples: sample='in', outlier_idx=[], model_type=None.
        
    Examples
    --------
    Basic Usage:
    >>> from .model import OLS
    >>> from .data import DataManager
    >>> 
    >>> # Setup DataManager with Walk Forward Test periods
    >>> dm = DataManager(internal_loader, mev_loader, poos_periods=[3, 6, 12])
    >>> 
    >>> # Create Walk Forward Test instance
    >>> wft = WalkForwardTest(
    ...     model_cls=OLS,
    ...     dm=dm,
    ...     specs=['GDP', 'UNRATE', 'CPI'],
    ...     target='balance',
    ...     model_kwargs={'sample': 'in', 'outlier_idx': []}
    ... )
    >>> 
    >>> # Access final model (automatically fitted)
    >>> final_model = wft.final_model
    >>> print(f"Final model R²: {final_model.rsquared:.3f}")
    >>> 
    >>> # Access Walk Forward models (automatically fitted)
    >>> wf_models = wft.wf_models
    >>> print(f"Walk Forward models: {list(wf_models.keys())}")
    >>> 
    >>> # Compare performance across models
    >>> for name, model in wf_models.items():
    ...     print(f"{name} R²: {model.rsquared:.3f}")
    
    Advanced Usage with Model Configuration:
    >>> # With specific model configuration
    >>> wft = WalkForwardTest(
    ...     model_cls=OLS,
    ...     dm=dm,
    ...     specs=['GDP', 'UNRATE', diff('CPI')],
    ...     target='charge_offs',
    ...     model_kwargs={
    ...         'sample': 'in',
    ...         'outlier_idx': ['2020-03-31', '2020-04-30'],
    ...         'model_type': RateLevel,
    ...         'target_base': 'loans'
    ...     }
    ... )
    >>> 
    >>> # Access model end dates
    >>> print(f"Final model in-sample end: {wft.model_in_sample_end}")
    >>> print(f"POOS in-sample ends: {wft.poos_in_sample_end}")
    
    Notes
    -----
    - Models are created and automatically fitted when properties are accessed
    - All models use the same specs and model_kwargs for consistency
    - Walk Forward models use progressively shorter training periods
    - Final model uses the full original training period for comparison
    - No explicit fitting required - models are ready to use when accessed
    - Date formatting follows frequency: 'yyyy-mm' for monthly, 'yyyy-Qq' for quarterly
    - Implements ModelStabilityTest abstract interface
    
    See Also
    --------
    ModelStabilityTest : Abstract base class for model stability testing
    DataManager.poos_dms : Creates pseudo-out-of-sample DataManager instances
    ModelBase : Base class for all statistical models
    """
    
    def __init__(
        self,
        model_cls: Type[ModelBase],
        dm: Any,
        specs: List[Union[str, Dict[str, Any]]],
        target: str,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Walk Forward Test instance.
        
        Parameters
        ----------
        model_cls : Type[ModelBase]
            ModelBase subclass to use for creating model instances.
        dm : DataManager
            DataManager instance with poos_dms capability.
        specs : List[Union[str, Dict[str, Any]]]
            Feature specifications for model training.
        target : str
            Name of target variable in internal data.
        model_kwargs : Dict[str, Any], optional
            Additional arguments for model constructor.
        """
        super().__init__(model_cls, dm, specs, target, model_kwargs)
        
        # Validate inputs
        if not hasattr(dm, 'poos_dms'):
            raise ValueError("DataManager must have poos_dms property for Walk Forward Testing")
        
        # Cache for model instances
        self._final_model: Optional[ModelBase] = None
        self._wf_models: Optional[Dict[str, ModelBase]] = None
    
    @property
    def final_model(self) -> ModelBase:
        """
        Get the final model trained on the original full training period.
        
        This model serves as the baseline for comparison with Walk Forward models.
        It uses the original DataManager with the full in-sample period for training.
        The model is automatically fitted when accessed.
        
        Returns
        -------
        ModelBase
            Fitted model instance trained on the original full training period.
            
        Example
        -------
        >>> final_model = wft.final_model
        >>> print(f"Final model R²: {final_model.rsquared:.3f}")
        >>> print(f"Training period: {final_model.dm.in_sample_idx.min()} to {final_model.dm.in_sample_idx.max()}")
        """
        if self._final_model is None:
            self._final_model = self.model_cls(
                dm=self.dm,
                specs=self.specs,
                target=self.target,
                **self.model_kwargs
            )
            # Automatically fit the model
            self._final_model.fit()
        return self._final_model
    
    @property  
    def wf_models(self) -> Dict[str, ModelBase]:
        """
        Get Walk Forward models trained on progressively shorter periods.
        
        Creates and automatically fits model instances for each pseudo-out-of-sample 
        DataManager, with keys formatted as 'WF1', 'WF2', 'WF3', etc. The order 
        follows the poos_periods list in the DataManager.
        
        Returns
        -------
        Dict[str, ModelBase]
            Dictionary mapping Walk Forward model names to fitted model instances.
            Keys: 'WF1', 'WF2', 'WF3', etc.
            Values: Fitted ModelBase instances using adjusted DataManagers.
            
        Example
        -------
        >>> wf_models = wft.wf_models
        >>> print(f"Available models: {list(wf_models.keys())}")
        >>> 
        >>> # All models are already fitted
        >>> for name, model in wf_models.items():
        ...     print(f"{name} - Training end: {model.dm.in_sample_end}")
        ...     print(f"{name} - R²: {model.rsquared:.3f}")
        >>> 
        >>> # Access specific model
        >>> wf1_model = wf_models['WF1']
        >>> wf1_performance = wf1_model.out_perf_measures
        """
        if self._wf_models is None:
            self._wf_models = {}
            
            # Get pseudo-out-of-sample DataManagers
            poos_dms = self.dm.poos_dms
            
            # Create models in the order of poos_periods
            for i, period in enumerate(self.dm.poos_periods, 1):
                poos_key = f'poos_dm_{period}'
                if poos_key in poos_dms:
                    model_name = f'WF{i}'
                    poos_dm = poos_dms[poos_key]
                    
                    model = self.model_cls(
                        dm=poos_dm,
                        specs=self.specs,
                        target=self.target,
                        **self.model_kwargs
                    )
                    # Automatically fit the model
                    model.fit()
                    self._wf_models[model_name] = model
        
        return self._wf_models
    
    @property
    def model_in_sample_end(self) -> str:
        """
        Get the in-sample end date of the final model as a formatted string.
        
        Returns the in_sample_end from the DataManager in a format appropriate
        for the frequency: 'yyyy-mm' for monthly data, 'yyyy-Qq' for quarterly data.
        
        Returns
        -------
        str
            Formatted in-sample end date string.
            
        Example
        -------
        >>> wft = WalkForwardTest(model_cls=OLS, dm=dm, specs=specs, target='balance')
        >>> print(wft.model_in_sample_end)  # '2024-03' for monthly, '2024-Q1' for quarterly
        """
        in_sample_end = self.dm.in_sample_end
        if in_sample_end is None:
            return "Not set"
        
        if self.dm.freq == 'M':
            return in_sample_end.strftime('%Y-%m')
        elif self.dm.freq == 'Q':
            quarter = (in_sample_end.month - 1) // 3 + 1
            return f"{in_sample_end.year}-Q{quarter}"
        else:
            return str(in_sample_end)
    
    @property
    def poos_in_sample_end(self) -> List[str]:
        """
        Get the in-sample end dates for each POOS period as formatted strings.
        
        Returns a list of formatted in-sample end dates for each pseudo-out-of-sample
        period, following the same format as model_in_sample_end. The order matches
        the poos_periods list in the DataManager.
        
        Returns
        -------
        List[str]
            List of formatted in-sample end date strings for each POOS period.
            
        Example
        -------
        >>> wft = WalkForwardTest(model_cls=OLS, dm=dm, specs=specs, target='balance')
        >>> print(wft.poos_in_sample_end)  # ['2023-12', '2023-09', '2023-06'] for monthly
        >>> print(wft.poos_in_sample_end)  # ['2023-Q4', '2023-Q3', '2023-Q2'] for quarterly
        """
        poos_dms = self.dm.poos_dms
        poos_end_dates = []
        
        for period in self.dm.poos_periods:
            poos_key = f'poos_dm_{period}'
            if poos_key in poos_dms:
                poos_dm = poos_dms[poos_key]
                in_sample_end = poos_dm.in_sample_end
                
                if in_sample_end is None:
                    poos_end_dates.append("Not set")
                elif self.dm.freq == 'M':
                    poos_end_dates.append(in_sample_end.strftime('%Y-%m'))
                elif self.dm.freq == 'Q':
                    quarter = (in_sample_end.month - 1) // 3 + 1
                    poos_end_dates.append(f"{in_sample_end.year}-Q{quarter}")
                else:
                    poos_end_dates.append(str(in_sample_end))
            else:
                poos_end_dates.append("Not available")
        
        return poos_end_dates
    
    @property
    def in_sample_start(self) -> str:
        """
        Get the in-sample start date of the DataManager as a formatted string.
        
        Returns the in_sample_start from the DataManager in a format appropriate
        for the frequency: 'yyyy-mm' for monthly data, 'yyyy-Qq' for quarterly data.
        
        Returns
        -------
        str
            Formatted in-sample start date string.
            
        Example
        -------
        >>> wft = WalkForwardTest(model_cls=OLS, dm=dm, specs=specs, target='balance')
        >>> print(wft.in_sample_start)  # '2015-06' for monthly, '2015-Q2' for quarterly
        """
        # Get in-sample start from the internal loader
        in_sample_start = getattr(self.dm._internal_loader, 'in_sample_start', None)
        if in_sample_start is None:
            return "Not set"
        
        if self.dm.freq == 'M':
            return in_sample_start.strftime('%Y-%m')
        elif self.dm.freq == 'Q':
            quarter = (in_sample_start.month - 1) // 3 + 1
            return f"{in_sample_start.year}-Q{quarter}"
        else:
            return str(in_sample_start)
    
    @property
    def param_tbl(self) -> pd.DataFrame:
        """
        Get parameter coefficients table for all models.
        
        Creates a DataFrame with parameter coefficients from the final model and
        all Walk Forward models. Excludes periodical dummies (M:1, Q:1, etc.)
        from the index. Column names use formatted in-sample end dates.
        
        Returns
        -------
        pd.DataFrame
            Parameter coefficients table with:
            - Index: Parameter names (excluding periodical dummies)
            - Columns: Formatted in-sample end dates
            - Values: Parameter coefficients from each model
            
        Example
        -------
        >>> param_table = wft.param_tbl
        >>> print(param_table)
        # Output:
        #           2024-03  2023-12  2023-09  2023-06
        # GDP        0.245    0.251    0.238    0.242
        # UNRATE    -0.123   -0.118   -0.125   -0.120
        # CPI        0.089    0.092    0.085    0.088
        """
        # Get final model parameters
        final_params = self.final_model.params
        
        # Filter out periodical dummies (M:1, Q:1, etc.)
        param_names = [name for name in final_params.index 
                      if not (':' in name and name.split(':')[1].isdigit())]
        
        # Create DataFrame with final model coefficients
        param_data = {self.model_in_sample_end: [final_params.get(name, np.nan) for name in param_names]}
        
        # Add Walk Forward model coefficients
        for i, (model_name, model) in enumerate(self.wf_models.items()):
            col_name = self.poos_in_sample_end[i]
            param_data[col_name] = [model.params.get(name, np.nan) for name in param_names]
        
        return pd.DataFrame(param_data, index=param_names)
    
    @property
    def param_pct_chg_tbl(self) -> pd.DataFrame:
        """
        Get parameter percentage change table for Walk Forward models.
        
        Creates a DataFrame showing percentage changes in parameter coefficients
        for Walk Forward models relative to the final model. Excludes periodical
        dummies and does not include the final model column.
        
        Returns
        -------
        pd.DataFrame
            Parameter percentage change table with:
            - Index: Parameter names (excluding periodical dummies)
            - Columns: Formatted in-sample end dates for Walk Forward models
            - Values: Decimal changes from final model coefficients (e.g., 0.0245 for 2.45%)
            
        Example
        -------
        >>> pct_chg_table = wft.param_pct_chg_tbl
        >>> print(pct_chg_table)
        # Output:
        #           2023-12  2023-09  2023-06
        # GDP        0.0245  -0.0286  -0.0122
        # UNRATE     0.0407   0.0163   0.0244
        # CPI        0.0337  -0.0449  -0.0112
        """
        # Get final model parameters as baseline
        final_params = self.final_model.params
        
        # Filter out periodical dummies
        param_names = [name for name in final_params.index 
                      if not (':' in name and name.split(':')[1].isdigit())]
        
        # Calculate percentage changes for each Walk Forward model
        pct_chg_data = {}
        
        for i, (model_name, model) in enumerate(self.wf_models.items()):
            col_name = self.poos_in_sample_end[i]
            pct_changes = []
            
            for name in param_names:
                final_coef = final_params.get(name, np.nan)
                wf_coef = model.params.get(name, np.nan)
                
                if pd.notna(final_coef) and pd.notna(wf_coef) and final_coef != 0:
                    pct_change = (wf_coef - final_coef) / final_coef
                    pct_changes.append(pct_change)
                else:
                    pct_changes.append(np.nan)
            
            pct_chg_data[col_name] = pct_changes
        
        return pd.DataFrame(pct_chg_data, index=param_names)
    
    @property
    def p_value_tbl(self) -> pd.DataFrame:
        """
        Get p-values table for all models.
        
        Creates a DataFrame with p-values for each parameter across all models.
        Excludes periodical dummies from the index. Column names use formatted
        in-sample end dates.
        
        Returns
        -------
        pd.DataFrame
            P-values table with:
            - Index: Parameter names (excluding periodical dummies)
            - Columns: Formatted in-sample end dates
            - Values: P-values from each model
            
        Example
        -------
        >>> p_value_table = wft.p_value_tbl
        >>> print(p_value_table)
        # Output:
        #           2024-03  2023-12  2023-09  2023-06
        # GDP        0.001    0.002    0.003    0.001
        # UNRATE     0.045    0.052    0.038    0.041
        # CPI        0.123    0.118    0.135    0.127
        """
        # Get final model p-values
        final_pvalues = self.final_model.pvalues
        
        # Filter out periodical dummies
        param_names = [name for name in final_pvalues.index 
                      if not (':' in name and name.split(':')[1].isdigit())]
        
        # Create DataFrame with final model p-values
        pvalue_data = {self.model_in_sample_end: [final_pvalues.get(name, np.nan) for name in param_names]}
        
        # Add Walk Forward model p-values
        for i, (model_name, model) in enumerate(self.wf_models.items()):
            col_name = self.poos_in_sample_end[i]
            pvalue_data[col_name] = [model.pvalues.get(name, np.nan) for name in param_names]
        
        return pd.DataFrame(pvalue_data, index=param_names)
    
    @property
    def model_perf_tbl(self) -> pd.DataFrame:
        """
        Get model performance comparison table.
        
        Creates a DataFrame comparing performance metrics across all models:
        in-sample R-squared, in-sample RMSE, and out-of-sample RMSE.
        
        Returns
        -------
        pd.DataFrame
            Model performance table with columns:
            - 'In-Sample End': Formatted in-sample end dates
            - 'PoOS Period': POOS periods (0 for final model, actual periods for WF models)
            - 'IS R-Square': In-sample R-squared values
            - 'IS RMSE': In-sample RMSE values
            - 'OOS RMSE': Out-of-sample RMSE values (NaN for final model)
            
        Example
        -------
        >>> perf_table = wft.model_perf_tbl
        >>> print(perf_table)
        # Output:
        #   In-Sample End  PoOS Period  IS R-Square  IS RMSE  OOS RMSE
        # 0      2024-03            0        0.856    0.234       NaN
        # 1      2023-12            3        0.842    0.241     0.256
        # 2      2023-09            6        0.838    0.245     0.261
        # 3      2023-06           12        0.831    0.248     0.268
        """
        # Prepare data for the table
        in_sample_ends = [self.model_in_sample_end] + self.poos_in_sample_end
        poos_periods = [0] + self.dm.poos_periods
        
        # Get performance metrics
        is_r_square = [self.final_model.rsquared]
        is_r_square.extend([model.rsquared for model in self.wf_models.values()])
        
        is_rmse = [self.final_model.in_perf_measures.get('RMSE', np.nan)]
        is_rmse.extend([model.in_perf_measures.get('RMSE', np.nan) for model in self.wf_models.values()])
        
        # OOS RMSE: NaN for final model, actual values for WF models
        oos_rmse = [np.nan]  # Final model has no OOS period
        oos_rmse.extend([model.out_perf_measures.get('RMSE', np.nan) for model in self.wf_models.values()])
        
        # Create DataFrame
        perf_data = {
            'In-Sample End': in_sample_ends,
            'PoOS Period': poos_periods,
            'IS R-Square': is_r_square,
            'IS RMSE': is_rmse,
            'OOS RMSE': oos_rmse
        }
        
        return pd.DataFrame(perf_data)
    
    def plot(self, figsize: tuple = (15, 10), save_path: str = None, show: bool = True) -> None:
        """
        Plot model performance for each Walk Forward model.
        
        Creates a grid of plots showing actual vs fitted (in-sample) vs predicted 
        (out-of-sample) values for each Walk Forward model. Each plot includes
        the in-sample period range in the title for easy identification.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (15, 10).
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        show : bool, optional
            Whether to display the plot. Default is True.
            
        Returns
        -------
        None
            The plot is displayed or saved as specified.
            
        Example
        -------
        >>> wft.plot(figsize=(18, 12))
        >>> 
        >>> # Save the plot without showing
        >>> wft.plot(save_path='walk_forward_performance.png', show=False)
        """
        # Close all existing figures to prevent duplicates
        plt.close('all')
        
        wf_models = self.wf_models
        n_models = len(wf_models)
        
        if n_models == 0:
            raise ValueError("No Walk Forward models available to plot")
        
        # Calculate grid dimensions (2 plots per row)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
        
        # Create a new figure with explicit control
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle different subplot configurations
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each Walk Forward model
        for i, (model_name, model) in enumerate(wf_models.items()):
            ax = axes[i]
            
            # Get model data
            try:
                # Get actual values from the full sample
                actual_data = model.dm.internal_data[model.target]
                
                # Get fitted values (in-sample)
                fitted_values = model.y_fitted_in
                
                # Get predicted values (out-of-sample)
                predicted_values = model.y_pred_out
                
                # Determine the plot range (from in-sample start to end of OOS)
                oos_end = model.dm.out_sample_idx.max() if len(model.dm.out_sample_idx) > 0 else model.dm.in_sample_end
                is_start = model.dm._internal_loader.in_sample_start
                
                # Filter actual data to plot range
                plot_mask = (actual_data.index >= is_start) & (actual_data.index <= oos_end)
                actual_plot_data = actual_data[plot_mask]
                
                # Plot actual values (only within the relevant range)
                ax.plot(actual_plot_data.index, actual_plot_data, 'k-', linewidth=2, 
                       label='Actual', alpha=0.8)
                
                # Plot fitted values (in-sample period)
                if len(fitted_values) > 0:
                    # Get fitted dates by aligning with fitted_values index
                    fitted_dates = fitted_values.index
                    ax.plot(fitted_dates, fitted_values, 'b-', linewidth=2, 
                           label='Fitted (IS)', alpha=0.8)
                
                # Plot predicted values (out-of-sample period)
                if len(predicted_values) > 0:
                    # Get predicted dates by aligning with predicted_values index
                    predicted_dates = predicted_values.index
                    ax.plot(predicted_dates, predicted_values, 'b--', linewidth=2, 
                           label='Predicted (OOS)', alpha=0.8)
                
                # Customize the plot
                poos_end_str = self.poos_in_sample_end[i]
                title = f'{model_name}: {self.in_sample_start} to {poos_end_str}'
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.tick_params(axis='x', labelsize=8, rotation=45)
                ax.tick_params(axis='y', labelsize=8)
                
                # Add vertical line at in-sample end
                in_sample_end_date = model.dm.in_sample_end
                if in_sample_end_date is not None:
                    ax.axvline(x=in_sample_end_date, color='red', linestyle=':', 
                             alpha=0.7, linewidth=1)
                
            except Exception as e:
                # Handle any plotting errors gracefully
                ax.text(0.5, 0.5, f'Error plotting {model_name}:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray'))
                ax.set_title(f'Walk Forward Model {model_name} - Error', fontsize=11)
        
        # Hide unused subplots
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        
        # Add overall title
        fig.suptitle('Walk Forward Test - Model Performance Comparison', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # Control display to prevent duplicates
        if show:
            plt.show()
        
        # Close the figure if not showing to free memory
        if not show:
            plt.close(fig)
    

    
    def show_all(self) -> None:
        """
        Display comprehensive Walk Forward Test results.
        
        Shows both visual and tabular results including:
        1. Performance plots for all Walk Forward models
        2. Parameter coefficients table
        3. Parameter percentage change table
        4. P-values table
        5. Model performance comparison table
        
        Example
        -------
        >>> wft.show_all()
        # Displays plots and prints all analysis tables
        """
        # print("=== Walk Forward Test - Comprehensive Results ===\n")
        
        # 1. Show plots
        print("1. Performance Plots:")
        self.plot()
        
        # 2. Show parameter coefficients table
        print("\n2. Parameter Coefficients Table:")
        print(self.param_tbl)
        
        # 3. Show parameter percentage change table
        print("\n3. Parameter Percentage Change Table:")
        print(self.param_pct_chg_tbl)
        
        # 4. Show p-values table
        print("\n4. P-Values Table:")
        print(self.p_value_tbl)
        
        # 5. Show model performance table (without index)
        print("\n5. Model Performance Comparison:")
        perf_table = self.model_perf_tbl
        print(perf_table.to_string(index=False))
        
        print("\n" + "="*60)
    
    def __repr__(self) -> str:
        """String representation of WalkForwardTest instance."""
        n_wf_models = len(self.dm.poos_periods)
        return f"WalkForwardTest(model_cls={self.model_cls.__name__}, n_models={n_wf_models + 1}, target='{self.target}')"