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

> **Note:** Core dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `openpyxl`, `arch`

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

# 3. Create a Segment, build and compare candidate models
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

# 4. Show report: in-sample, out-of-sample, params, tests
seg.show_report(
    cm_ids=['Model1'],
    report_sample='in',
    show_out=True,
    show_params=True,
    show_tests=True
)

# 5. Export to Excel
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
├─ TECHNIC/                # Core Python modules
│  ├─ __init__.py          # Package initializer
│  ├─ cm.py                # Candidate Model orchestration
│  ├─ data.py              # DataManager: load, interpolate, engineer features
│  ├─ internal.py          # InternalDataLoader
│  ├─ mev.py               # MEVLoader for model & scenario MEVs
│  ├─ transform.py         # TSFM: feature transformations
│  ├─ conditional.py       # CondVar: conditional features
│  ├─ test.py              # ModelTestBase & specific tests (Normality, Stationarity, Significance, R2, F)
│  ├─ model.py             # ModelBase & OLS subclass with testset_func
│  ├─ plot.py              # Plot utilities (performance, test visuals)
│  ├─ report.py            # ModelReportBase & OLS_ModelReport (tables & plots)
│  ├─ segment.py           # Segment: manage & compare CMs
│  ├─ writer.py            # Excel writer utilities
│  └─ template.py          # ExportTemplateBase & PPNR_OLS_ExportTemplate
├─ support/                # Static support files
│  ├─ mev_type.xlsx        # MEV code → type mapping
│  └─ type_tsfm.yaml       # Type → TSFM mapping
├─ templates/              # Excel export templates
├─ requirements.txt        # Dependencies
└─ README.md               # This file
```

---

## 🔍 Highlights & Utility Methods

* **DataManager**: `build_search_vars`, `apply_to_all`, `apply_to_mevs`, `apply_to_internal`
* **TSFM**: dynamic naming, string‐based `transform_fn` lookup, `lag` support.
* **CondVar**: generate conditional variables easily.
* **TestSet**: standardized framework with `NormalityTest`, `StationarityTest`, `SignificanceTest`, `R2Test`, `FTest`.
* **ModelBase & OLS**: unified `fit()`, `predict()`, integrated via `ppnr_ols_testset_func`.
* **Plot**: `ols_plot_perf_set` for combined IS/OOS comparisons with correct actual concat logic.
* **Reporting**: `ModelReportBase.show_*_tbl()` prints dataframes; `show_test_tbl()` consolidates tests.
* **ExportTemplate**: PPNR‐specific Excel reports via `PPNR_OLS_ExportTemplate`.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License © Shawn Y. Sun, Kexin Zhu