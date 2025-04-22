# Project LEGO

**Project LEGO** is a modular Python package for building, evaluating, and reporting on linear regression (OLS) models within a standardized pipeline. It streamlines data loading, feature transformation, model fitting, performance measurement, diagnostic testing, and report generation.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/shawn-y-sun/Project_LEGO.git
cd Project_LEGO

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** No external dependencies beyond standard data science stack (`pandas`, `numpy`, `statsmodels`, `matplotlib`).

---

## 🚀 Quickstart

```python
from TECHNIC.cm import CM
from TECHNIC.model import OLS
from TECHNIC.measure import OLS_Measures
from TECHNIC.report import OLSReport

# 1. Load & prepare data
dm = CM.data_manager  # assume DataManager instance handles loading
X, y = dm.load_features(), dm.load_target()

# 2. Initialize and build candidate-model (in/out split, fit)
cm = CM(model_id='my_model', model=OLS(target='y'), target='y')
cm.build(X, y, in_sample_end='2020-12-31')

# 3. Generate performance & diagnostics
measures = OLS_Measures(cm.model_in, cm.X_in, cm.y_in, cm.X_out, cm.y_out, cm.y_pred_out)
report = OLSReport(measures)
report.show_report(show_out=True, show_tests=True)
```  

---

## 📂 Package Structure

```
Project_LEGO/
├─ TECHNIC/
│  ├─ __init__.py       # Package entrypoint
│  ├─ cm.py             # Candidate‐Model orchestration (CM)
│  ├─ datamgr.py        # DataManager: build indep/dep variables, transforms
│  ├─ internal.py       # InternalDataLoader: standardize & dummy vars
│  ├─ measure.py        # MeasureBase & OLS_Measures: performance & tests
│  ├─ mev.py            # MEVLoader: load mean excess variation data
│  ├─ model.py          # ModelBase & OLS: regression templates
│  ├─ plot.py           # ols_perf_plot & ols_test_plot visualizations
│  ├─ report.py         # ModelReportBase & OLSReport: tables & plots
│  ├─ segment.py        # (Optional) segmentation utilities
│  └─ transform.py      # custom TSFM wrapper and transform functions
├─ requirements.txt     # Python dependencies
└─ README.md            # Project overview and usage (this file)
```

---

## 🔍 Module Highlights

### `cm.py`  
- **`CM`** class coordinates data loading, splitting (in vs out of sample), model fitting, and result storage.

### `model.py`  
- **`ModelBase`** (abstract) and **`OLS`** (concrete) templates for regression models.  
- `.fit()` updates internal attributes and returns `self`; `.predict()` wraps `statsmodels` predictions.

### `measure.py`  
- **`MeasureBase`** collects performance, filtering, and testing functions.  
- **`OLS_Measures`** implements R², MAE, RMSE, Jarque–Bera tests, and VIF calculations.

### `plot.py`  
- **`ols_perf_plot`**: actual vs. fitted/predicted lines + absolute-error bars.  
- **`ols_test_plot`**: residuals vs. fitted scatter.

### `report.py`  
- **`ModelReportBase`** (abstract) defines table/diagnostic plotting methods.  
- **`OLSReport`** renders performance tables, parameter summaries, and calls plotting functions.

---

## 🤝 Contributing

1. **Fork** the repo and create a feature branch.  
2. **Implement** your changes, with unit tests if applicable.  
3. **Open a Pull Request** against `main`, describing your additions.

Please adhere to the existing code style and add documentation for new functionality.

---

## 📄 License

MIT License © Shawn Y. Sun

