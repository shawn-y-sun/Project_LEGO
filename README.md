# Project LEGO

A modular Python package to streamline and standardize the end‑to‑end model development workflow for time‑series banking products. It provides data loaders, feature transforms, modeling templates, measure calculators, and a candidate‐model orchestrator—all designed for flexibility and extensibility.

---

## Features

- **Data loading & preprocessing**  
  - `InternalDataLoader`: ingest CSV/Excel or raw DataFrame, standardize to month‑ or quarter‑end index, add quarter & month dummies.  
  - `MEVLoader`: load “model” and “scen” MEV sheets from one or more workbooks, preprocess dates, extract code→name maps, and apply batch transforms.  
  - `DataManager`: glue loader outputs together, interpolate MEV to match internal frequency, build custom feature sets, split in‑sample vs. out‑of‑sample, and trim scen tables by cutoff.

- **Feature transformations**  
  - Pre‑defined transforms (LV, MMGR, MM, QQGR, YYGR, rolling averages, divergences, etc.) in `features/transforms.py`.  
  - `TSFM` manager to apply arbitrary transform functions + generate lagged features + track naming and sign expectations.

- **Model templates & measures**  
  - `ModelBase` abstract class & `OLS` implementation (stores coefficients, p‑values, VIFs).  
  - `MeasureBase` abstract class + `OLSMesures` (filtering, performance, assumption tests).  

- **Candidate‐Model orchestrator**  
  - `CM` class ties together `DataManager`, a `ModelBase` subclass, and a `MeasureBase` subclass.  
  - Build and split design matrices, fit in‑sample and full models, compute & export performance/testing tables.

---

## Installation

```bash
git clone https://github.com/shawn-y-sun/Project_LEGO.git
cd Project_LEGO
pip install -e .