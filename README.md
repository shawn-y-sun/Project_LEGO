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
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows

# Install dependencies
pip install -r requirements.txt
``` 

> **Note:** Core dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `openpyxl`

---

## 🚀 Quickstart

```python
from TECHNIC.segment import Segment
from TECHNIC.datamgr import DataManager
from TECHNIC.model import OLS
from TECHNIC.measure import OLS_Measures
from TECHNIC.export_template import PPNR_OLS_ExportTemplate

# 1. Prepare DataManager
dm = DataManager(data_frame)

# 2. Create Segment and build CMs
seg = Segment(
    segment_id='SegmentA',
    target='y',
    data_manager=dm,
    model_cls=OLS,
    measure_cls=OLS_Measures,
    report_cls=None  # use export_template for output
)
cm = seg.build_cm('Model1', specs=['x1', 'x2'])
# ... build other CMs

# 3. Export to Excel using PPNR OLS template
tmpl = PPNR_OLS_ExportTemplate(
    template_files=['templates/PPNR_OLS_Template.xlsx'],
    cms=seg.cms
)
tmpl.export({'templates/PPNR_OLS_Template.xlsx': 'outputs/SegmentA_Report.xlsx'})
```  

---

## 📂 Package Structure

```
Project_LEGO/
├─ TECHNIC/
│  ├─ __init__.py           # Package entrypoint
│  ├─ cm.py                 # Candidate-Model orchestration (CM)
│  ├─ datamgr.py            # DataManager: load & prepare features/target
│  ├─ internal.py           # InternalDataLoader: raw data prep & dummies
│  ├─ mev.py                # MEVLoader: load mean excess variation data
│  ├─ measure.py            # MeasureBase & OLS_Measures: performance & tests
│  ├─ model.py              # ModelBase & OLS: regression templates
│  ├─ plot.py               # ols_model_perf_plot, ols_model_test_plot, ols_seg_perf_plot
│  ├─ report.py             # ModelReportBase & OLSReport
│  ├─ segment.py            # Segment orchestration of CMs
│  ├─ transform.py          # TSFM wrapper for feature transforms
│  ├─ writer.py             # Val, ValueWriter, SheetWriter, TemplateLoader, WorkbookWriter, TemplateWriter
│  ├─ template.py           # ExportTemplateBase & PPNR_OLS_ExportTemplate
│  └─ segment.py            # Optional: specialized exporters (if any)
├─ templates/               # Excel template files
├─ requirements.txt
└─ README.md                # Project overview and usage (this file)
```

---

## 🔍 Module Highlights

### `cm.py`  
- **`CM`**: orchestrates building, fitting, measuring, and reporting of single candidate-models.  

### `datamgr.py`  
- **`DataManager`**: loads raw data, builds feature (`X`) and target (`y`) DataFrames.  

### `measure.py`  
- **`MeasureBase`**: abstract base for performance and diagnostic measures.  
- **`OLS_Measures`**: computes R², MAE, RMSE, Jarque–Bera, VIF, etc.  

### `plot.py`  
- **`ols_model_perf_plot`**, **`ols_model_test_plot`**, **`ols_seg_perf_plot`**: visual diagnostics for single-model and multi-model comparisons.  

### `report.py`  
- **`ModelReportBase`** & **`OLSReport`**: text and table reporting for in- and out-of-sample performance, parameters, and tests.  
- **`OLS_SegmentReport`**: aggregating multiple CMs across a segment.  

### `segment.py`  
- **`Segment`**: manage a group of candidate models (`CM`), build them, and integrate with export/report templates.  

### `transform.py`  
- **`TSFM`**: sequential feature-transform wrapper.  

### `writer.py`  
- **`Val`**, **`ValueWriter`**, **`SheetWriter`**, **`TemplateLoader`**, **`WorkbookWriter`**, **`TemplateWriter`**: core utilities for exporting arbitrary Python data (scalars, lists, Series, DataFrames, dicts) into Excel templates.  

### `template.py`  
- **`ExportTemplateBase`**: abstract driver for Excel-based workflows.  
- **`PPNR_OLS_ExportTemplate`**: sample implementation mapping CM outputs into a PPNR OLS workbook.  

---

## 📄 License

MIT License © Shawn Y. Sun, Kexin Zhu