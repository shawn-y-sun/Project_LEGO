# Project LEGO

<div align="center">

![Project LEGO](https://img.shields.io/badge/Project-LEGO-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-Proprietary-black)
![Version](https://img.shields.io/badge/Version-Beta%20v2.3-orange)

**Build models like LEGO: a modular Python framework for automated search, rigorous evaluation, and scenario forecasting**

</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [LEGO‑Style Modular Architecture](#-lego-style-modular-architecture)
- [The LEGO Workflow](#-the-lego-workflow)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Demo Notebooks](#-demo-notebooks)
- [License](#-license)

## 📖 Overview

**Project LEGO** is a production‑grade framework for assembling econometric models through a consistent pipeline. Designed for financial modeling (PPNR focus), it combines:
- automated, exhaustive model search,
- comprehensive evaluation and diagnostics (fit, significance, residual tests, cointegration, stability), and
- integrated scenario forecasting —
all exposed via a small set of composable APIs that snap together like LEGO bricks.

## 🧱 LEGO‑Style Modular Architecture

The framework is designed like LEGO bricks: small, interchangeable components that snap together to build complete modeling workflows. Each component has standardized interfaces, so you can easily swap, extend, or combine them.

**🔧 Foundation Bricks (Data Layer):**
- **`InternalLoader`** (e.g., `TimeSeriesLoader`, `PanelLoader`) — loads and standardizes internal time‑series/panel data with sample splits
- **`MEVLoader`** — loads macro‑economic variables (MEVs) for both historical and scenario data
- **`DataManager`** — **combines InternalLoader + MEVLoader**, handles interpolation/aggregation, feature engineering

**🏗️ Orchestration Bricks (Modeling Layer):**
- **`Segment`** — manages a modeling sub‑project for a specific target variable
  - Takes: `DataManager` + `ModelBase` (e.g., `OLS`) + `ModelType` (optional)
  - Auto‑creates: `ModelSearch` instance (the "searcher")
- **`ModelSearch`** — exhaustive search engine that generates and evaluates model combinations
  - Produces: `CM` (Candidate Model) instances

**🔬 Analysis Bricks (CM Layer):**
Each `CM` (Candidate Model) contains multiple analysis modules:
- **`ScenManager`** — scenario forecasting and analysis
- **`StabilityTest`** (e.g., `WalkForwardTest`) — model stability validation  
- **`TestSet`** — comprehensive diagnostics (fit, significance, residual tests, cointegration)
- **Model instances** — fitted `ModelBase` objects (in‑sample, full‑sample)

**🎯 Feature Bricks (Transform Layer):**
- **`TSFM`**, **`CondVar`**, **`DumVar`** — declarative feature transforms that snap onto any variable

**🔄 Easy Extension (Just Like LEGO):**
```python
# 1. Snap on new transforms
my_transform = lambda x: np.log(x + 1)
tc.TSFM('GDP', my_transform)

# 2. Swap model engines  
class MyARModel(tc.ModelBase): ...
tc.Segment(..., model_cls=MyARModel)

# 3. Extend search logic
seg.search_cms(desired_pool=[...], custom_constraints=my_filter)

# 4. Add custom diagnostics
seg.build_cm('test', specs=[...], test_update_func=my_tests)
```

**The LEGO Magic:** Change one brick, everything else still works. The workflow stays consistent whether you're using OLS or future AR/VECM models, working with quarterly or monthly data, or adding custom transforms.

## 🔄 The LEGO Workflow

- **Phase 1) Data Setup & Initialization**: Load with `InternalLoader` (e.g., `TimeSeriesLoader`) + `MEVLoader`, then snap together via `DataManager` — handles interpolation/aggregation automatically.
- **Phase 2) Feature Analysis & Engineering**: Engineer features via `DataManager.apply_to_all()`. Create `Segment`. Use `Segment.explore_vars()` for visual exploration.
- **Phase 3) Automated Model Search**: `Segment` auto‑creates `ModelSearch`. Run `Segment.search_cms()` with driver pools (`TSFM`, `CondVar`, `DumVar('*')`) — produces ranked `CM` instances.
- **Phase 4) Candidate Evaluation & Selection**: Load saved models via `Segment.load_cms()`. Rank them using `Segment.rerank_cms()`. Deep-dive into champions using `Segment.show_report()`.
- **Phase 5) Documentation & Delivery**: Export via `Segment.export()` for consistent reporting and external Excel template.

## ✨ Features (by phase)

- **Phase 1 — Data Setup**:
  - Time‑series/panel loaders (`TimeSeriesLoader`, `PanelLoader`) with explicit in/out sample and `scen_p0`
  - `MEVLoader` for monthly/quarterly MEVs; auto Q↔M interpolation/aggregation; variable map + TSFM map
  - Three‑layer scenario ingestion and alignment (set → scenario → DataFrame)

- **Phase 2 — Feature Analysis**:
  - Broadcast feature engineering with `DataManager.apply_to_all()`; maintain metadata via `update_var_map()`
  - `Segment.explore_vars()` for plots and correlation rankings across transforms

- **Phase 3 — Automated Search**:
  - Automated search via `Segment.search_cms()` across driver pools (`TSFM`, `CondVar`, `DumVar('*')`, raw vars)
  - Constraints (expected‑signs, lags/periods, max‑vars), scoring and Top‑N selection

- **Phase 4 — Evaluation & Selection**:
  - Load and re-rank models from past searches via `Segment.load_cms()` and `rerank_cms()`
  - `Segment.show_report()` with performance summaries, parameter significance, residual diagnostics (normality, stationarity, autocorrelation, heteroscedasticity), and cointegration
  - Walk‑forward/POOS stability; integrated scenario plots and comparisons

- **Phase 5 — Delivery**:
  - `Segment.export()` to curated files; companion Excel template for presentation‑ready deliverables
  - Consistent plots, tables, and reproducible outputs

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

The demo notebooks (`DEMO_1` through `DEMO_4`) demonstrate the full pipeline. Below is a concise version following the core phases.

```python
import pandas as pd
import Technic as tc
from Technic import TSFM, DumVar

# ===========================================================================
# Phase 1: Data Setup & Initialization
# ===========================================================================
# 1. Load Internal Data
df_internal = pd.read_csv('Demo Data/housing_market.csv')
# Create target: forward-looking growth
df_internal['home_price_GR1'] = df_internal['home_price_index'].pct_change().shift(-1)

int_ldr = tc.TimeSeriesLoader(
    in_sample_start="2006-01-31",
    in_sample_end="2023-09-30",
    full_sample_end="2025-09-30",
    scen_p0="2023-09-30"
)
int_ldr.load(df_internal, date_col='date')

# 2. Load Macro Data (MEVs)
mev_ldr = tc.MEVLoader()
mev_ldr.load(pd.read_csv('Demo Data/macro_quarterly.csv'), date_col='observation_date')
mev_ldr.load(pd.read_csv('Demo Data/macro_monthly.csv'), date_col='observation_date')
# Load Scenarios (Base, Adverse, Severe)
mev_ldr.load_scens(
    {'Base': df_scen_base, 'Adv': df_scen_adv, 'Sev': df_scen_sev}, # Assume DFs loaded
    set_name='Scenario'
)

# 3. Create DataManager and Feature Engineering
dm = tc.DataManager(int_ldr, mev_ldr)

def new_features(df_mev, df_in):
    # Yield curve slopes
    df_mev['USYC10_2'] = df_mev['USGOV10Y'] - df_mev['USGOV2Y']
    # Credit spreads
    df_mev['USCORP_SPRD_BAA_AAA'] = df_mev['USCORPBBB10Y'] - df_mev['USCORPAA10Y']
    return df_mev, df_in

dm.apply_to_all(new_features)

# 4. Initialize Segment
seg = tc.Segment(
    segment_id='home_price_GR1',
    target='home_price_GR1',
    model_type=tc.Growth,
    target_base='home_price_index',
    data_manager=dm,
    model_cls=tc.OLS
)

# ===========================================================================
# Phase 2: EDA & Driver Selection
# ===========================================================================
# Visual exploration and correlation analysis
seg.explore_vars(['USCORPBBB10Y', 'USMORT30Y', 'USYC10_2'])

# ===========================================================================
# Phase 3: Automated Model Search
# ===========================================================================
# Define constraints
forced_in = [DumVar('M', categories=[2,3,4,5,10,11,12])] # Seasonal dummies
desired_pool = ['USCORPBBB10Y', 'USMORT30Y', 'USYC10_2', 'USPRIME']

# Run exhaustive search (saves models to disk)
seg.search_cms(
    forced_in=forced_in,
    desired_pool=desired_pool,
    max_var_num=3,
    max_lag=3
)

# ===========================================================================
# Phase 4: Candidate Evaluation & Selection
# ===========================================================================
# Load saved models
seg.load_cms() # Loads from the latest search

# Rank candidates using weighted business metrics
seg.rerank_cms(rank_weights=(1, 1, 2))

# Deep-dive report for the top champion (cm1)
if 'cm1' in seg.cms:
    seg.cms['cm1'].show_report(show_scens=True)

# ===========================================================================
# Phase 5: Documentation & Delivery
# ===========================================================================
# Export final results
seg.export(output_dir='outputs/home_price_GR1')
```

## 🧪 Demo Notebooks

For a complete, end‑to‑end walkthrough, follow the demo series:

1.  **`DEMO_1_Setup.ipynb`**: Data loading, feature engineering, and segment initialization.
2.  **`DEMO_2_Feature.ipynb`**: Exploratory data analysis (EDA) and driver pool selection.
3.  **`DEMO_3_Search.ipynb`**: Configuring and running the automated model search.
4.  **`DEMO_4_Evaluate.ipynb`**: Loading, ranking, and selecting candidate models.


## 📄 License

Proprietary software. All rights reserved.

Copyright © Shawn Y. Sun, Kexin Zhu. 

This software and its source code are licensed for internal use only under the terms in the accompanying `LICENSE` file. No redistribution, sublicensing, or commercial offering is permitted without prior written permission.
