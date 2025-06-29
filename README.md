# Project LEGO

<div align="center">

![Project LEGO](https://img.shields.io/badge/Project-LEGO-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**A comprehensive Python framework for OLS-based financial modeling with advanced scenario analysis**

</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Core Components](#-core-components)
- [Scenario Analysis](#-scenario-analysis)
- [Usage Examples](#-usage-examples)
- [Contributing](#-contributing)
- [License](#-license)

## 📖 Overview

**Project LEGO** is a comprehensive Python framework designed for building, evaluating, and deploying OLS-based candidate models through a standardized pipeline. It specializes in financial modeling, particularly PPNR (Pre-Provision Net Revenue) analysis, with built-in support for advanced scenario analysis, model validation, and comprehensive reporting.

The framework follows a modular "LEGO-like" architecture where components can be easily assembled, modified, and extended to suit specific modeling needs. Recent enhancements include sophisticated scenario management, enhanced data handling for both time series and panel data, and integrated plotting capabilities.

## ✨ Features

- **Advanced Data Management**
  - Unified handling of internal data and macro-economic variables (MEVs)
  - Support for both time series and panel data structures
  - Flexible data loaders with configurable sampling periods
  - Automated data interpolation and feature engineering
  - Multi-source MEV integration with frequency handling

- **Comprehensive Scenario Analysis**
  - **ScenManager**: Dedicated scenario forecasting and analysis
  - Multi-scenario MEV support with 3-layer dictionary structure
  - Conditional and simple forecasting capabilities
  - Integrated scenario plotting with customizable visualizations
  - Forecast vs. actual comparison with period indicators

- **Enhanced Model Building**
  - Standardized OLS modeling pipeline with robust covariance handling
  - Flexible feature transformation framework (TSFM)
  - Support for conditional variables (CondVar) and lag operations
  - Comprehensive model validation suite with filtering capabilities
  - Automated outlier detection and robust standard error computation

- **Robust Testing & Validation**
  - Extensive statistical test suite (Normality, Stationarity, Autocorrelation, Heteroscedasticity)
  - Performance metrics (R², F-tests, VIF, Cointegration)
  - In-sample and out-of-sample analysis
  - Scenario impact assessment with filtering modes
  - Automated test result interpretation

- **Professional Reporting & Visualization**
  - Integrated scenario plotting with forecast and variable analysis
  - Performance visualization with customizable styling
  - Standardized model reporting with ReportSet aggregation
  - Excel-based export templates with PPNR-specific formats
  - Comprehensive segment-level reporting

- **Model Search & Selection**
  - Exhaustive model search capabilities
  - Automated feature combination testing
  - Performance-based model ranking
  - Top-N model selection with configurable criteria

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/shawn-y-sun/Project_LEGO.git
cd Project_LEGO

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites
- Python 3.7 or higher
- Core dependencies:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing
  - `statsmodels`: Statistical modeling
  - `matplotlib`: Data visualization
  - `openpyxl`: Excel file handling
  - `arch`: Time series analysis

## 🚀 Quick Start

```python
from TECHNIC.data import DataManager
from TECHNIC.segment import Segment
from TECHNIC.transform import TSFM
from TECHNIC.condition import CondVar
from TECHNIC.model import OLS
from TECHNIC.scenario import ScenManager

# 1. Initialize DataManager with enhanced data sources
dm = DataManager(
    internal_df=your_internal_df,
    model_mev_source={'model_rates.xlsx': 'BaseRates'},
    scen_mevs_source={
        'scenario_rates.xlsx': {
            'CCAR2024': {
                'Base': 'BaseRates',
                'Adverse': 'AdverseRates', 
                'Severe': 'SevereRates'
            }
        }
    },
    in_sample_end='2023-12-31',
    scen_in_sample_end='2022-12-31'
)

# 2. Advanced Feature Engineering
var_dfs = dm.build_search_vars(['GDP', 'CPI', 'UNEMPLOYMENT'])

# Create conditional variables
cond_var = CondVar(
    name='GDP_conditional',
    main_series=dm.mev_data['GDP'],
    cond_var=[dm.internal_data['target']],
    condition=lambda x: x > 0,
    if_true=lambda main, cond: main * 1.2,
    if_false=lambda main, cond: main * 0.8
)

# 3. Model Building with Scenario Analysis
seg = Segment(
    segment_id='SegmentA',
    target='target_variable',
    data_manager=dm,
    model_cls=OLS
)

seg.build_cm(
    cm_id='Model1',
    specs=[
        'base_variable',
        TSFM('GDP', transform_fn='GR', lag=1),
        TSFM('CPI', transform_fn='DIFF', lag=2),
        cond_var
    ],
    sample='both'
)

# 4. Comprehensive Reporting with Scenario Analysis
seg.show_report(
    cm_ids=['Model1'],
    report_sample='in',
    show_out=True,
    show_params=True,
    show_tests=True,
    show_scens=True,  # Enable scenario analysis
    scen_kwargs={'figsize': (8, 4), 'title_prefix': 'CCAR2024: '}
)

# 5. Direct Scenario Analysis
cm = seg.cms['Model1']
scen_mgr = cm.scen_manager_in

# Plot all scenario forecasts and variables
scen_mgr.plot_all(
    figsize=(8, 4),
    title_prefix="Stress Test: ",
    save_path="outputs/scenarios/"
)

# Access forecast data
forecast_dfs = scen_mgr.forecast_df
for scen_set, df in forecast_dfs.items():
    print(f"Scenario Set: {scen_set}")
    print(df.head())
```

## 🏗 Architecture

The package follows a modular architecture with enhanced scenario analysis capabilities:

```
Project_LEGO/
├─ TECHNIC/                # Core Python modules
│  ├─ __init__.py          # Package initializer
│  ├─ cm.py                # Candidate Model orchestration with scenario integration
│  ├─ data.py              # DataManager: enhanced data loading and feature engineering
│  ├─ internal.py          # InternalDataLoader with panel/time series support
│  ├─ mev.py               # MEVLoader with multi-source and scenario MEV handling
│  ├─ scenario.py          # ScenManager: comprehensive scenario analysis
│  ├─ transform.py         # TSFM: advanced feature transformations
│  ├─ condition.py         # CondVar: conditional variable framework
│  ├─ feature.py           # Feature engineering utilities
│  ├─ test.py              # Enhanced testing framework with filtering
│  ├─ model.py             # ModelBase & OLS with robust covariance
│  ├─ plot.py              # Advanced plotting utilities
│  ├─ report.py            # Comprehensive reporting with scenario support
│  ├─ segment.py           # Segment management with search capabilities
│  ├─ search.py            # Model search and selection framework
│  ├─ helper.py            # Utility functions and helpers
│  ├─ writer.py            # Excel writer utilities
│  └─ template.py          # Export templates
├─ support/                # Static support files
│  ├─ mev_map.xlsx         # MEV code → type/category mapping
│  └─ type_tsfm.yaml       # Type → TSFM mapping
├─ templates/              # Excel export templates
├─ requirements.txt        # Dependencies
└─ README.md              # Documentation
```

## 🔧 Core Components

### DataManager
- **Purpose**: Central data management with enhanced MEV and scenario support
- **Key Features**:
  - Multi-source data integration (internal, model MEVs, scenario MEVs)
  - Support for both time series and panel data structures
  - Flexible sampling period configuration
  - Advanced feature building with conditional variables
- **Key Methods**:
  - `build_features`: Create feature matrices for modeling
  - `build_search_vars`: Generate transformed variable sets for exploration
  - `apply_to_all`: Apply functions across all data sources
  - `apply_to_mevs`: MEV-specific transformations

### ScenManager
- **Purpose**: Comprehensive scenario analysis and forecasting
- **Key Features**:
  - Multi-scenario forecasting with conditional and simple modes
  - Integrated plotting for forecasts and variable analysis
  - Period-based data organization with P0 reference points
  - Support for both time series and panel data scenarios
- **Key Methods**:
  - `plot_forecasts`: Visualize scenario forecasts vs. fitted values
  - `plot_scenario_variables`: Plot individual variables across scenarios
  - `plot_all`: Comprehensive scenario visualization
- **Properties**:
  - `forecast_df`: Organized forecast data with period indicators
  - `y_scens`: Nested scenario forecast results

### Enhanced Model Framework
- **ModelBase**: Abstract base with scenario manager integration
- **OLS**: Advanced OLS implementation with:
  - Automatic robust covariance detection (HC1, HAC)
  - Comprehensive diagnostic testing
  - Integrated scenario analysis capabilities
  - Performance and parameter measurement

### Advanced Testing Framework
- **Test Categories**:
  - **Fit Tests**: R², Adjusted R², Error measures
  - **Significance Tests**: P-value, F-test, VIF analysis
  - **Residual Tests**: Normality, Stationarity, Autocorrelation, Heteroscedasticity
  - **Cointegration Tests**: Y-X cointegration analysis
- **Features**:
  - Configurable filtering modes (strict, moderate, lenient)
  - Automated test result interpretation
  - Integration with model building pipeline

### Comprehensive Reporting System
- **ReportSet**: Aggregate reporting across multiple models
- **Features**:
  - Performance comparison tables
  - Parameter significance analysis
  - Diagnostic test summaries
  - Integrated scenario analysis
  - Excel export capabilities
- **Visualization**:
  - Performance plots with actual vs. fitted/predicted
  - Scenario forecast visualization
  - Variable analysis across scenarios

## 🎯 Scenario Analysis

The framework provides comprehensive scenario analysis capabilities through the `ScenManager` class:

### Scenario Data Structure
```python
# 3-layer scenario MEV structure
scen_mevs = {
    'CCAR2024': {
        'Base': base_scenario_df,
        'Adverse': adverse_scenario_df,
        'Severe': severe_scenario_df
    },
    'ICAAP2024': {
        'Baseline': baseline_df,
        'Stress': stress_df
    }
}
```

### Scenario Forecasting
```python
# Access scenario manager from fitted model
scen_mgr = model.scen_manager

# Get organized forecast data
forecast_data = scen_mgr.forecast_df

# Plot comprehensive scenario analysis
figures = scen_mgr.plot_all(
    figsize=(8, 4),
    title_prefix="Stress Test: "
)
```

### Scenario Integration
- **Automatic Integration**: Scenario managers are automatically created during model building
- **CM Integration**: Accessible through `cm.scen_manager_in` and `cm.scen_manager_full`
- **Reporting Integration**: Scenario plots can be included in standard reports
- **Segment Integration**: Scenario analysis available at the segment level

## 📊 Usage Examples

### Basic Model Building
```python
# Simple model with transformations
seg.build_cm(
    cm_id='BasicModel',
    specs=['var1', TSFM('var2', transform_fn='LOG')],
    sample='both'
)
```

### Advanced Feature Engineering
```python
# Complex feature specifications
specs = [
    'base_var',
    TSFM('economic_indicator', transform_fn='GR', lag=1),
    CondVar(
        name='conditional_feature',
        main_series=data['main'],
        cond_var=[data['condition']],
        condition=lambda x: x > threshold,
        if_true=lambda m, c: m * 1.5,
        if_false=lambda m, c: m * 0.5
    )
]
```

### Model Search and Selection
```python
# Exhaustive model search
top_models = seg.search_cms(
    desired_pool=['GDP', 'CPI', 'UNEMPLOYMENT'],
    forced_in=['base_variable'],
    top_n=5,
    max_var_num=4,
    rank_weights=(1, 1, 1)
)
```

### Comprehensive Reporting
```python
# Full reporting with scenario analysis
seg.show_report(
    show_out=True,
    show_params=True, 
    show_tests=True,
    show_scens=True,
    scen_kwargs={'save_path': 'outputs/'}
)
```

For detailed usage examples and workflows, please refer to the Jupyter notebooks in the repository:
- `LEGO_Demo.ipynb`: Comprehensive framework demonstration
- `LEGO_ModuleTest.ipynb`: Individual module testing and validation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License © Shawn Y. Sun, Kexin Zhu