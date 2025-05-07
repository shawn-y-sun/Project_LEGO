# Project LEGO

**Project LEGO** is a modular Python package for building, evaluating, reporting, and exporting OLS-based candidate models via a standardized pipeline.

---

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

> **Note:** Core dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `openpyxl`

---

## 🚀 Quickstart

```python
from TECHNIC.data      import DataManager
from TECHNIC.segment   import Segment
from TECHNIC.transform import TSFM
from TECHNIC.model     import OLS
from TECHNIC.test      import PPNR_OLS_TestSet
from TECHNIC.template  import PPNR_OLS_ExportTemplate

# 1. Prepare DataManager with internal + MEV sources
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

# 2. Use DataManager helper methods
# • build_search_vars → dict of transformed DataFrames per var
var_dfs = dm.build_search_vars(['GDP','CPI'])
# • apply_to_all → apply a two-arg fn (internal, mev_df) across all MEVs
def spread(internal, mev):
    return (internal['Price'] - mev['NGDP']).rename('Price_minus_NGDP')
dm.apply_to_all(spread)

# • apply_to_mevs → same two-arg fn but returns full DataFrame
def multi_feats(mev, internal):
    df = mev.copy()
    df['GDP-Price']     = df['GDP']    - internal['Price']
    df['Unemp-Price']   = df['Unemp']  - internal['Price']
    return df
dm.apply_to_mevs(multi_feats)

# • apply_to_internal → apply single-arg fn to internal_data
def add_macro(internal):
    internal['Real_GDP'] = internal['GDP'] / (1 + internal['Inflation'])
dm.apply_to_internal(add_macro)

# 3. Create a Segment, build and compare CMs
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
        TSFM('x2', transform_fn=lambda s: s.pct_change(), max_lag=1),
    ],
    sample='both'
)

# • show_report: aggregate in/out perf, params, tests
seg.show_report(
    cm_ids=['Model1'],
    report_sample='in',
    show_out=True,
    show_params=True,
    show_tests=True,
    perf_kwargs={'digits': 3}
)

# • explore_vars: visualize each var vs target (line or scatter)
seg.explore_vars(['x1','x2_pct_change_L1'], plot_type='line')

# 4. Export all CMs in the segment to Excel
tmpl = PPNR_OLS_ExportTemplate(
    template_files=['templates/PPNR_OLS_Template.xlsx'],
    cms=seg.cms
)
tmpl.export({
    'templates/PPNR_OLS_Template.xlsx': 'outputs/SegmentA_Report.xlsx'
})
```

---

## 📂 Package Structure

```
Project_LEGO/
├─ TECHNIC/                # Core Python package modules
│  ├─ __init__.py          # Package initializer
│  ├─ cm.py                # Candidate Model orchestration (build → fit → report)
│  ├─ data.py              # DataManager: load, interpolate, and engineer features
│  ├─ internal.py          # InternalDataLoader: handles raw internal data
│  ├─ mev.py               # MEVLoader: loads model & scenario MEV tables
│  ├─ transform.py         # TSFM: feature transformation manager
│  ├─ conditional.py       # CondVar: conditional variable generator
│  ├─ test.py              # TestSetBase & specific test implementations
│  ├─ model.py             # ModelBase & OLS model subclass
│  ├─ plot.py              # Plotting utilities (performance, diagnostics)
│  ├─ report.py            # ModelReportBase & OLSReport
│  ├─ segment.py           # Segment: manage multiple CMs & exploration
│  ├─ writer.py            # Writer utilities for Excel output
│  └─ template.py          # Export templates (e.g. PPNR_OLS_ExportTemplate)
├─ support/                # Static support files for DataManager
│  ├─ mev_type.xlsx        # MEV code → type mapping
│  └─ type_tsfm.yaml       # Type → transform specification
├─ templates/              # Excel template files for exports
├─ requirements.txt        # Python dependencies
└─ README.md               # This README document
```

---

## 🔍 Module Highlights

### `mev.py` (MEVLoader)

* **Inputs:**

  * `model_mev_source: Dict[str, str]`
  * `scen_mevs_source: Dict[str, Dict[str, str]]`
  * Optional `load_and_preprocess` function (default provided)
* **`.load()`** populates:

  * `.model_mev` (DataFrame)
  * `.model_map` (code → name)
  * `.scen_mevs` (nested dict `{workbook_key: DataFrame}`)
  * `.scen_maps` (nested dict `{workbook_key: {code → name}}`)
* **`.apply_to_all(fn)`** applies a function to every MEV DataFrame.

### `data.py` (DataManager)

* **Constructor inputs** aligned with `MEVLoader`:

  * `model_mev_source` / `scen_mevs_source`
* **Properties:**

  * `.internal_in`, `.internal_out`
  * `.model_in`, `.model_out`
  * `.scen_mevs` (trimmed by `scen_in_sample_end`)
* **`build_indep_vars(specs)`** supports raw strings, `TSFM`, and `CondVar` instances.

### `transform.py` (TSFM)

* **`TSFM(feature, transform_fn, max_lag, exp_sign)`**
* **`apply_transform()`** returns a named `pd.Series` with `.name = <feature>_<fn_name>_L<lag>`.

### `conditional.py` (CondVar)

* **`CondVar(main_var, cond_var, cond_fn, cond_fn_kwargs)`**
* **`apply()`** returns a Series named `<main>_<cond_fn_name>`.

### `test.py` 

* **`NormalityTest`**, **`StationarityTest`**: run multiple metrics per test, return nested dicts
* **`TestSetBase`**: flattens multiple tests, `.test_results` and `.all_passed()`
* **`PPNR_OLS_TestSet`**: defaults to α=0.05.

### `model.py` (ModelBase & OLS)

* **`ModelBase`**

  * Core attrs: `X, y, X_out, y_out, testset_cls, report_cls`
  * `@property report` → instantiates report\_cls
  * `@property tests`  → builds `TestSetBase` on residuals
* **`OLS`**: `.fit()`, `.predict()`, `.y_fitted_in`, plus measure properties.

### `cm.py` (CM)

* **`.build(sample='in'|'full'|'both')`** → fits models with `testset_cls` & `report_cls`
* **Properties:** `report_in`, `report_full`, `tests_in`, `tests_full`
* **`show_report()`** delegates to in-sample and full-sample reports.

### `report.py` (ModelReportBase & OLSReport)

* **`show_perf_tbl()`**
* **`show_test_tbl()`**: general flattening of arbitrary test metrics
* **Plot methods** via injected functions.

### `template.py` (ExportTemplateBase)

* **`PPNR_OLS_ExportTemplate`**: maps CM outputs to Excel via `Val` and `ValueWriter`

---

## 🏗️ New Utility Methods

### DataManager (`data.py`)

* **`build_search_vars(specs, mev_type_map=…, type_tsfm_map=…) → Dict[str, DataFrame]`**
  Builds a DataFrame per variable (raw + transformed features), warning on any unknown types.

* **`apply_to_all(fn)`**
  Applies a two-arg function `fn(internal_df, mev_df)` to **all** MEV tables (model + scenarios).

  * If `fn` returns a `Series`/`DataFrame`, merges it into each MEV table.
  * If `fn` returns `None`, assumes `fn` mutated `mev_df` in place.

* **`apply_to_mevs(fn)`**
  Similar to `apply_to_all`, but expects `fn(mev_df, internal_df)` and merges the returned DataFrame into each MEV table only.

* **`apply_to_internal(fn)`**
  Runs `fn(internal_df)` on the internal dataset.

  * If `fn` returns a `Series`/`DataFrame`, merges back into `internal_data`.
  * If `None`, assumes in-place mutation.

* **Support files**

  * `support/mev_type.xlsx` → loaded on import into `MEV_TYPE_MAP`, must exist or raises.
  * `support/type_tsfm.yaml` → loaded on import into `TYPE_TSFM_MAP`, must exist or raises.

---

### Transformations (`transform.py`)

* Generalized transforms:

  * **`DF(series, periods=1)`** – difference over `periods`.
  * **`GR(series, periods=1)`** – growth‐rate over `periods`.
  * **Alias**: `DF2 = partial(DF, periods=2)`, `DF3`, `GR2`, `GR3`.
* **TSFM** name logic inspects `periods` or `window` args (from `partial`), appending suffix (e.g. `PDI_GR2`).

---

### Segment Exploration (`segment.py`)

* **`explore_vars(vars_list, plot_type='line')`**

  * Builds feature‐DataFrames for each variable via `build_search_vars`.
  * **Line plots**: dual‐axis time‐series (`target` left, feature right), no axis labels, legend included.
  * **Scatter plots**: feature vs target, blue dots, no legend, no axis labels.
  * Grid layout: 3 subplots per row; subplot title = feature name; figure title dynamic (e.g. “Time Series: x1 vs y”).

---

### Reporting Across Models (`report.py` & `segment.py`)

* **`ReportSet.show_report(...)`**
  Aggregates over multiple `ReportBase` objects to:

  1. Show in‐sample performance table
  2. Optionally show out‐of‐sample table
  3. Plot combined performance
  4. Optionally show per‐model parameter tables
  5. Optionally show per‐model test results

* **`Segment.show_report(...)`** delegates to `ReportSet`, letting you compare any subset of CMs, pick `'in'` or `'full'` sample, and toggle parameters/tests.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development, testing, and pull requests.

---

## 📄 License

MIT License © Shawn Y. Sun, Kexin Zhu
