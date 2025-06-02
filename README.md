# Project LEGO

<div align="center">

![Project LEGO](https://img.shields.io/badge/Project-LEGO-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**A modular Python framework for OLS-based financial modeling pipelines**

</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Core Components](#-core-components)
- [Usage Examples](#-usage-examples)
- [Contributing](#-contributing)
- [License](#-license)

## 📖 Overview

**Project LEGO** is a comprehensive Python framework designed for building, evaluating, and deploying OLS-based candidate models through a standardized pipeline. It specializes in financial modeling, particularly PPNR (Pre-Provision Net Revenue) analysis, with built-in support for scenario analysis and model validation.

The framework follows a modular "LEGO-like" architecture where components can be easily assembled, modified, and extended to suit specific modeling needs.

## ✨ Features

- **Unified Data Management**
  - Seamless handling of internal data and macro-economic variables (MEVs)
  - Built-in support for multiple data sources and scenarios
  - Automated data interpolation and feature engineering

- **Advanced Model Building**
  - Standardized OLS modeling pipeline
  - Flexible feature transformation framework
  - Support for conditional variables and lag operations
  - Comprehensive model validation suite

- **Robust Testing & Validation**
  - Statistical tests (Normality, Stationarity, Significance)
  - Performance metrics (R², F-tests)
  - In-sample and out-of-sample analysis
  - Scenario impact assessment

- **Professional Reporting**
  - Automated performance visualization
  - Standardized model reporting
  - Excel-based export templates
  - PPNR-specific reporting formats

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
from TECHNIC.data      import DataManager
from TECHNIC.segment   import Segment
from TECHNIC.transform import TSFM
from TECHNIC.model     import OLS
from TECHNIC.test      import PPNR_OLS_TestSet
from TECHNIC.template  import PPNR_OLS_ExportTemplate

# 1. Initialize DataManager with data sources
dm = DataManager(
    internal_df=your_internal_df,
    model_mev_source={'model_rates.xlsx': 'BaseRates'},
    scen_mevs_source={
        'scenario_rates.xlsx': {
            'base':  'BaseRates',
            'adv':   'AdverseRates',
            'sev':   'SevereRates'
        }
    },
    in_sample_end='2023-12-31',
    scen_in_sample_end='2022-12-31'
)

# 2. Feature Engineering
var_dfs = dm.build_search_vars(['GDP', 'CPI'])
def spread(internal, mev):
    return (internal['Price'] - mev['NGDP']).rename('Price_minus_NGDP')
dm.apply_to_all(spread)

# 3. Model Building and Evaluation
seg = Segment(
    segment_id='SegmentA',
    target='y',
    data_manager=dm,
    model_cls=OLS,
    testset_cls=PPNR_OLS_TestSet
)

seg.build_cm(
    cm_id='Model1',
    specs=[
        'x1',
        TSFM('x2', transform_fn='GR', lag=1),
    ],
    sample='both'
)

# 4. Reporting
seg.show_report(
    cm_ids=['Model1'],
    report_sample='in',
    show_out=True,
    show_params=True,
    show_tests=True
)

# 5. Export Results
tmpl = PPNR_OLS_ExportTemplate(
    template_files=['templates/PPNR_OLS_Template.xlsx'],
    cms=seg.cms
)
tmpl.export({
    'templates/PPNR_OLS_Template.xlsx': 'outputs/SegmentA_Report.xlsx'
})
```

## 🏗 Architecture

The package follows a modular architecture with the following structure:

```
Project_LEGO/
├─ TECHNIC/                # Core Python modules
│  ├─ __init__.py          # Package initializer
│  ├─ cm.py                # Candidate Model orchestration
│  ├─ data.py              # DataManager: load, interpolate, engineer features
│  ├─ internal.py          # InternalDataLoader
│  ├─ mev.py               # MEVLoader for model & scenario MEVs
│  ├─ transform.py         # TSFM: feature transformations
│  ├─ conditional.py       # CondVar: conditional features
│  ├─ test.py              # ModelTestBase & specific tests
│  ├─ model.py             # ModelBase & OLS implementation
│  ├─ plot.py              # Plot utilities
│  ├─ report.py            # Model reporting framework
│  ├─ segment.py           # Segment management
│  ├─ writer.py            # Excel writer utilities
│  └─ template.py          # Export templates
├─ support/                # Static support files
│  ├─ mev_type.xlsx        # MEV code → type mapping
│  └─ type_tsfm.yaml       # Type → TSFM mapping
├─ templates/              # Excel export templates
├─ requirements.txt        # Dependencies
└─ README.md              # Documentation
```

## 🔧 Core Components

### DataManager
- **Purpose**: Central data management and feature engineering
- **Key Methods**:
  - `build_search_vars`: Create transformed variable sets
  - `apply_to_all`: Apply functions across all data sources
  - `apply_to_mevs`: MEV-specific transformations
  - `apply_to_internal`: Internal data transformations

### TSFM (Transformation)
- **Features**:
  - Dynamic naming system
  - String-based transform function lookup
  - Flexible lag operations
  - Type-based transformation mapping

### Testing Framework
- **Available Tests**:
  - `NormalityTest`: Distribution analysis
  - `StationarityTest`: Time series properties
  - `SignificanceTest`: Variable significance
  - `R2Test`: Model fit assessment
  - `FTest`: Model comparison

### Reporting System
- **Capabilities**:
  - Performance visualization
  - Parameter tables
  - Test result summaries
  - Excel-based reporting
  - PPNR-specific templates

## 📊 Usage Examples

For detailed usage examples and workflows, please refer to `Project_LEGO_Demo.ipynb` in the repository.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License © Shawn Y. Sun, Kexin Zhu