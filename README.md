# Project LEGO

<div align="center">

![Project LEGO](https://img.shields.io/badge/Project-LEGO-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-Proprietary-black)
![Version](https://img.shields.io/badge/Version-Beta%20v2.0-orange)

**Build models like LEGO: a modular Python framework for automated search, rigorous evaluation, and scenario forecasting**

</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [LEGO‑Style Modular Architecture](#-lego-style-modular-architecture)
- [The LEGO Six‑Step Workflow](#-the-lego-sixstep-workflow)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start (Six Steps)](#-quick-start-six-steps)
- [Demo Notebook](#-demo-notebook)
- [License](#-license)

## 📖 Overview

**Project LEGO** is a production‑grade framework for assembling econometric models through a consistent six‑step pipeline. Designed for financial modeling (PPNR focus), it combines:
- automated, exhaustive model search,
- comprehensive evaluation and diagnostics (fit, significance, residual tests, cointegration, stability), and
- integrated scenario forecasting —
all exposed via a small set of composable APIs that snap together like LEGO bricks.

## 🧱 LEGO‑Style Modular Architecture

The framework is designed like LEGO bricks: small, interchangeable components that snap together to build complete modeling workflows. Each component has standardized interfaces, so you can easily swap, extend, or combine them.

**🔧 Foundation Bricks (Data Layer):**
- **`InternalLoader`** (e.g., `PPNRInternalLoader`, `PanelLoader`) — loads and standardizes internal time‑series/panel data with sample splits
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

**The LEGO Magic:** Change one brick, everything else still works. The six‑step workflow stays consistent whether you're using OLS or future AR/VECM models, working with quarterly or monthly data, or adding custom transforms.

## 🔄 The LEGO Six‑Step Workflow

- **1) Data Preprocessing**: Load with `InternalLoader` (e.g., `PPNRInternalLoader`) + `MEVLoader`, then snap together via `DataManager` — handles interpolation/aggregation automatically.
- **2) EDA & Driver Selection**: Create `Segment` (takes `DataManager` + `ModelBase` + `ModelType`). Use `Segment.explore_vars()` for visual exploration. Engineer features via `DataManager.apply_to_all()`.
- **3) Exhaustive Search**: `Segment` auto‑creates `ModelSearch` (the "searcher"). Run `Segment.search_cms()` with driver pools (`TSFM`, `CondVar`, `DumVar('*')`) — produces ranked `CM` instances.
- **4) Model Evaluation & Validation**: Each `CM` contains `ScenManager`, `StabilityTest`, `TestSet` modules. Use `Segment.show_report()` for comprehensive analysis across all `CM` instances.
- **5) Fine‑tune & Enhancement**: Build individual `CM` instances via `Segment.build_cm()` or re‑run search with refined pools and constraints.
- **6) Presentation & Documentation**: Export via `Segment.export()` — leverages each `CM`'s analysis modules for consistent reporting and external Excel template.

## ✨ Features (by step)

- **Step 1 — Data Preprocessing**:
  - Time‑series/panel loaders (`PPNRInternalLoader`, `PanelLoader`) with explicit in/out sample and `scen_p0`
  - `MEVLoader` for monthly/quarterly MEVs; auto Q↔M interpolation/aggregation; variable map + TSFM map
  - Three‑layer scenario ingestion and alignment (set → scenario → DataFrame)

- **Step 2 — EDA & Driver Selection**:
  - `Segment.explore_vars()` for plots and correlation rankings across transforms
  - Broadcast feature engineering with `DataManager.apply_to_all()`; maintain metadata via `update_var_map()`

- **Step 3 — Exhaustive Search**:
  - Automated search via `Segment.search_cms()` across driver pools (`TSFM`, `CondVar`, `DumVar('*')`, raw vars)
  - Constraints (expected‑signs, lags/periods, max‑vars), scoring and Top‑N selection

- **Step 4 — Evaluation & Validation**:
  - `Segment.show_report()` with performance summaries, parameter significance, residual diagnostics (normality, stationarity, autocorrelation, heteroscedasticity), and cointegration
  - Walk‑forward/POOS stability; integrated scenario plots and comparisons

- **Step 5 — Fine‑tune & Enhancement**:
  - Rapid iteration using `Segment.build_cm()` and refined search criteria; quick spec comparisons

- **Step 6 — Presentation & Documentation**:
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

## 🚀 Quick Start (Six Steps)

The notebook `LEGO_Demo.ipynb` demonstrates the full pipeline. Below is a concise version following the six‑step flow above.

```python
import Technic as tc

# 1) Data Preprocessing ---------------------------------------------
# Build internal loader and load historical + scenario internal data
int_ldr = tc.PPNRInternalLoader(
    in_sample_start='2020-06-30',
    in_sample_end='2023-12-31',
    full_sample_end='2023-12-31',
    scen_p0='2023-12-31'
)
int_ldr.load(source=your_internal_df, date_col='File_Date')
int_ldr.load_scens(
    source={'Base': internal_base_df, 'Adv': internal_adv_df, 'Sev': internal_sev_df},
    set_name='EWST_2024',
    date_col='File_Date'
)

# Load model and scenario MEVs
mev_ldr = tc.MEVLoader()
mev_ldr.load(source='path/to/model_mevs_qtr.xlsx', sheet='Historical Data > Enterprise')
mev_ldr.load(source=df_mev_mth)  # monthly historical from a DataFrame
mev_ldr.load_scens(
    source='path/to/EWST_2024.xlsx',
    scens={'Base': 'Baseline > Enterprise', 'Adv': 'EMST Adverse > Enterprise', 'Sev': 'EMST Severe > Enterprise'}
)

# 2) EDA & Driver Selection -----------------------------------------
dm = tc.DataManager(int_ldr, mev_ldr, poos_periods=[4, 8, 12])

# Optional feature engineering broadcast to model/scenario data
def new_features(df_mev, df_in):
    df_mev['T_1Y1M'] = df_mev['CAGOV12M'] - df_mev['CAGOV1M']
    df_in['Price_Inc'] = df_in['Term_Price'] - df_in['EDB_Price']
    return df_mev, df_in

dm.apply_to_all(new_features)

# Visual exploration and correlations
seg = tc.Segment(
    segment_id='EDB_TERM',
    target='EDB_Flow_rt',
    target_base='EDB_Outflow',
    target_exposure='Eligible_EDB_Bal',
    data_manager=dm,
    model_cls=tc.OLS
)
df_corr = seg.explore_vars(['T_1Y1M', 'CAGOV12M', 'CAONR'])

# 3) Exhaustive Search of Model Options ------------------------------
desired_pool = [tc.DumVar('*'), tc.TSFM('CAGOV12M', transform_fn='LV'), 'Price_Inc']
exp_sign_map = {'CAGOV12M': 1, 'CAONR': -1}
seg.search_cms(
    desired_pool=desired_pool,
    max_var_num=3,
    periods=[1, 2],
    exp_sign_map=exp_sign_map,
    top_n=10
)

# 4) Model Evaluation & Validation ----------------------------------
seg.show_report(show_params=True, show_tests=True, show_scens=True)
cm = next(iter(seg.cms.values()))
cm.testset_in.print_test_info()

# 5) Fine‑tune & Enhancement ----------------------------------------
# Build one more CM quickly from specs
seg.build_cm(
    cm_id='cm_new',
    specs=[DumVar('*'), TSFM('CAONR', transform_fn='LV'), 'Price_Inc']
)
seg.show_report(cm_ids=['cm_new'], show_stab=True)

# 6) Presentation & Documentation -----------------------------------
# Export segment results (CSV by default). Use the external Excel template as needed.
seg.export(output_dir='outputs/EDB_TERM')
```

## 🧪 Demo Notebook

- Open `LEGO_Demo.ipynb` for a complete, end‑to‑end walkthrough mirroring this README.


## 📄 License

Proprietary software. All rights reserved.

Copyright © Shawn Y. Sun, Kexin Zhu. 

This software and its source code are licensed for internal use only under the terms in the accompanying `LICENSE` file. No redistribution, sublicensing, or commercial offering is permitted without prior written permission.