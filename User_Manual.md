# Project LEGO User Manual

Welcome to the Project LEGO user manual. This guide walks you through installation, core components, and typical workflows for building, evaluating, and reporting OLS models.

---

## 1. Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/shawn-y-sun/Project_LEGO.git
   cd Project_LEGO
   ```
2. **Set up environment** (optional but recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Package Structure

```
Project_LEGO/
├─ TECHNIC/
│  ├─ cm.py           # Candidate‐Model orchestration (CM)
│  ├─ datamgr.py      # DataManager: building X/y and transforms
│  ├─ internal.py     # InternalDataLoader: standardize index, produce dummies
│  ├─ mev.py          # MEVLoader: loading MEV (Macro Economic Variable) data
│  ├─ measure.py      # MeasureBase & OLS_Measures: metrics & tests
│  ├─ model.py        # ModelBase & OLS: regression templates
│  ├─ plot.py         # ols_perf_plot & ols_test_plot
│  ├─ report.py       # ModelReportBase & OLSReport
│  ├─ segment.py      # (Optional) segmentation utilities
│  └─ transform.py    # TSFM wrapper for feature transforms
├─ requirements.txt
└─ README.md
```

---

## 3. Core Components

### 3.1 Data Loading & Preparation

#### `InternalDataLoader`
- **Location**: `TECHNIC/internal.py`
- **Purpose**: Load raw data, standardize index, automatically add period dummy variables (e.g., Q1–Q4, M1–M12).
- **Key Methods**:
  - `load(path)`: reads data and stores to `self._internal_data`.
  - `_add_period_dummies()`: creates and concatenates dummy columns.

#### `MEVLoader`
- **Location**: `TECHNIC/mev.py`
- **Purpose**: Load and organize Mean Excess Variation workbooks and sheets per scenario (`scen_*` attributes).
- **Usage**: Instantiate with `scen_maps`, call `load()` to populate `scen_mevs`.

### 3.2 Data Management

#### `DataManager`
- **Location**: `TECHNIC/datamgr.py`
- **Purpose**: Build independent (`X`) and target (`y`) variables, apply feature transforms, flatten nested structures.
- **Key Features**:
  - `build_indep_vars(transform_fn)`: takes a function to generate features.
  - `flatten()`: collapse multi-index columns.
  - Suffix logic to tag scenario features.

### 3.3 Feature Transforms

#### `TSFM` Wrapper
- **Location**: `TECHNIC/transform.py` and `TECHNIC/tsfm.py`
- **Purpose**: Compose or wrap multiple transform functions into a single pipeline.
- **Typical Usage**:
  ```python
  from TECHNIC.transform import TSFM
  tsfm = TSFM([fn1, fn2, ...])
  X_transformed = tsfm.fit_transform(X)
  ```

### 3.4 Modeling

#### `ModelBase` (Abstract)
- **Location**: `TECHNIC/model.py`
- **Purpose**: Define interface for regression models (.fit, .predict).
- **Signature**:
  ```python
  class ModelBase(ABC):
      def __init__(self, model_id: str, target: str):
          ...
      @abstractmethod
      def fit(self, X, y): ...
      @abstractmethod
      def predict(self, X): ...
  ```

#### `OLS` (Concrete)
- **Subclass of**: `ModelBase`
- **Behavior**:
  - `.fit(X, y)`: fits `statsmodels.OLS`; stores `self.results`, returns `self`.
  - `.predict(X)`: uses `self.results.predict(X)`.

### 3.5 Measure Calculators

#### `MeasureBase` (Abstract)
- **Location**: `TECHNIC/measure.py`
- **Purpose**: Collect performance, filtering, and test functions.
- **Key Properties**:
  - `filter_measures`: apply filtering functions (e.g., max p-value).
  - `in_perf_measures`: in-sample metrics dict.
  - `out_perf_measures`: out-of-sample metrics dict (empty if no OOS).
  - `test_measures`: diagnostic tests (Jarque–Bera, VIF).
  - `param_measures`: parameter-level metrics (coef, pvalue, vif, std).

#### `OLS_Measures`
- Implements R², adjusted R², ME, MAE, RMSE for in/out sample.
- Testing: Jarque–Bera statistic & p-value, VIF per variable.

### 3.6 Reporting

#### `ModelReportBase` (Abstract)
- **Location**: `TECHNIC/report.py`
- **Purpose**: Define interface for showing tables and plots.
- **Abstract Methods**:
  - `show_perf_tbl()`
  - `show_out_perf_tbl()`
  - `show_test_tbl()`
  - `show_params_tbl()`
  - `plot_perf(**kwargs)`
  - `plot_tests(**kwargs)`

#### `OLSReport`
- Implements all `show_...` methods returning nicely formatted `pandas.DataFrame` or printed tables.
- `plot_perf()`: invokes `ols_perf_plot` with in/out data.
- `plot_tests()`: invokes `ols_test_plot`.
- `show_report()`: orchestrates the sequence: performance tables, parameter table, plots, and diagnostics.

### 3.7 Plotting Utilities

#### `ols_perf_plot`
- **Purpose**: Time-series comparison of actual vs. fitted (solid) & predicted (dashed) values, with absolute error bars at 70% transparency.
- **Inputs**: `model`, `X`, `y_full`, optional `X_out`, `y_pred_out`.

#### `ols_test_plot`
- **Purpose**: Residuals vs. fitted scatter plot with horizontal zero line.

---

## 4. Example Workflow

```python
from TECHNIC.internal import InternalDataLoader
from TECHNIC.datamgr import DataManager
from TECHNIC.model import OLS
from TECHNIC.measure import OLS_Measures
from TECHNIC.report import OLSReport

# 1) Load internal data
idl = InternalDataLoader(path='data/internal.csv')
idl.load()

# 2) Build features & target
dm = DataManager(idl.internal_data)
X = dm.build_indep_vars(lambda df: df[['feature1', 'feature2']])
y = dm.internal_data['target']

# 3) Instantiate CM and fit
cm = CM(model_id='example', model=OLS('target'), target='target')
cm.build(X, y, in_sample_end='2021-12-31')

# 4) Compute measures & report
measures = OLS_Measures(
    cm.model_in, cm.X_in, cm.y_in,
    cm.X_out, cm.y_out, cm.y_pred_out
)
report = OLSReport(measures)
report.show_report(show_out=True, show_tests=True)
```

---

## 5. Extending the Pipeline

- **Custom Models**: Subclass `ModelBase`, implement `.fit()` & `.predict()`.
- **Additional Metrics**: Extend `MeasureBase` or subclass to add new `perf_funcs` or `test_funcs`.
- **New Plots**: Provide custom plotting functions to a new `Report` subclass.

---

## 6. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and pull requests.

---

## 7. License

MIT License © Shawn Y. Sun, Kexin Zhu

