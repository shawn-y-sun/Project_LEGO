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
from TECHNIC.segment      import Segment
from TECHNIC.data         import DataManager
from TECHNIC.transform    import TSFM
from TECHNIC.conditional  import CondVar
from TECHNIC.model        import OLS
from TECHNIC.test         import TestSetBase
from TECHNIC.template     import PPNR_OLS_ExportTemplate

# 1. Prepare DataManager with internal data + MEV sources
dm = DataManager(
    internal_df=your_internal_df,
    model_mev_source={'model_rates.xlsx': 'BaseRates'},
    scen_mevs_source={
        'scenario_rates.xlsx': {
            'base': 'BaseRates',
            'adv':  'AdverseRates',
            'sev':  'SevereRates'
        }
    },
    in_sample_end='2023-12-31',
    scen_in_sample_end='2022-12-31'
)

# 2. Create a Segment and build Candidate Models (CMs)
seg = Segment(
    segment_id='SegmentA',
    target='y',
    data_manager=dm,
    model_cls=OLS,
    testset_cls=PPNR_OLS_TestSet,
    report_cls=None           # or OLSReport if you want immediate charting
)

# build both in-sample and full-sample models (default)
cm = seg.build_cm(
    model_id='Model1',
    specs=[
      'x1',
      TSFM(feature='x2', transform_fn=lambda s: s.pct_change(), max_lag=1),
      CondVar(
        main_var='x3',
        cond_var='x4',
        cond_fn=zero_if_exceeds,
        cond_fn_kwargs={'threshold': 10}
      )
    ],
    sample='both'
)

# inspect
print(cm)                   # e.g. "y~C+x1+x2_pct_change_L1+x3_zero_if_exceeds"
df_perf_in = cm.report_in.show_perf_tbl()
df_tests_in = cm.tests_in   # DataFrame of in-sample test results

# 3. Export all CMs in the segment to Excel
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
├─ TECHNIC/
│  ├─ __init__.py           # package entrypoint
│  ├─ cm.py                 # CM: orchestrates build → fit → tests → report
│  ├─ data.py               # DataManager: loads internal & MEV, build_indep_vars()
│  ├─ internal.py           # InternalDataLoader: raw data & period dummies
│  ├─ mev.py                # MEVLoader: loads model_mev_source & scen_mevs_source
│  ├─ transform.py          # TSFM: feature-transform manager
│  ├─ conditional.py        # CondVar: conditional variable generator
│  ├─ test.py               # ModelTestBase, NormalityTest, StationarityTest
│  ├─ model.py              # ModelBase (.report, .tests), OLS subclass
│  ├─ plot.py               # ols_model_perf_plot, ols_model_test_plot, ols_seg_perf_plot
│  ├─ report.py             # ModelReportBase, OLSReport (general show_test_tbl)
│  ├─ segment.py            # Segment: manage multiple CMs + build_cm()
│  ├─ writer.py             # Val, ValueWriter, SheetWriter, WorkbookWriter
│  └─ template.py           # ExportTemplateBase, PPNR_OLS_ExportTemplate
├─ templates/               # Excel template files
├─ requirements.txt
└─ README.md                # this file
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

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development, testing, and pull requests.

---

## 📄 License

MIT License © Shawn Y. Sun, Kexin Zhu
