# **The LEGO Playbook**

## **1\. Introduction to Project LEGO**

**Project LEGO** is a modular framework designed to streamline the end-to-end econometric modeling process. By decoupling data management, feature engineering, and model estimation into distinct "blocks," LEGO enables teams to build consistent, reproducible, and rigorously tested models rapidly.

The core philosophy replaces ad-hoc scripting with a standardized assembly line: **Load** ![][image1] **Transform** \-\> **Search** \-\> **Validate.**

### **1.1 Recommended Modeling Procedure**

The following workflow outlines the high-level procedure for developing models using the LEGO framework, ensuring every candidate model undergoes rigorous testing and validation.

#### **Phase 1: Data Setup & Initialization**

**1\. Data Preparation & Loading**

* **Concept:** Unified ingestion of internal portfolio data and macroeconomic variables (MEVs).  
* **Goal:** Create a frequency-aligned, sample-consistent dataset before modeling begins.

**2\. Segment Definition**

* **Concept:** Explicit definition of the modeling scope, including target variables and model architecture (e.g., OLS vs. Time Series).  
* **Goal:** Establish clear boundaries and objectives for the specific model.

#### **Phase 2: Feature Analysis & Engineering**

**3\. Feature Engineering**

* **Concept:** Creation of predictive drivers capturing economic nuances, such as regime signals or complex ratios.  
* **Goal:** Enhance raw data with domain-specific features to improve interpretability and performance.

**4\. Exploratory Analysis**

* **Concept:** Statistical analysis of potential drivers against the target variable.  
* **Goal:** Narrow the universe of variables to a high-quality "Driver Pool" for automated search.

#### **Phase 3: Automated Model Search**

**5\. Automated Search & Basic Validation**

* **Concept:** Systematic testing of driver combinations and transformations.  
* **Goal:** Identify the universe of statistically valid models passing basic assumptions (e.g., stationarity, significance, multicollinearity) without manual trial-and-error.

#### **Phase 4: Candidate Evaluation & Selection**

**6\. Filtering & Ranking**

* **Concept:** Application of strict project-specific business logic to rank statistically valid models.  
* **Goal:** Reduce the valid model pool to a "Shortlist" of top candidates aligning with business intuition.

**7\. Champion Selection**

* **Concept:** Deep-dive review of shortlisted models against comprehensive criteria (stress testing, economic rationale).  
* **Goal:** Select the final **Champion Model** and backup candidates.

**🔄 The Iterative Loop**

If no satisfactory champion is found in **Phase 4**, the process returns to **Phase 2** to refine features or adjust the driver pool.

#### **Phase 5: Documentation & Delivery**

**8\. Final Delivery**

* **Concept:** Generation of standardized documentation and artifacts.  
* **Goal:** Produce the final model package (specifications, metrics, charts) for validation and deployment.

### **1.2 Modules & Implementation Guide**

This section details how to utilize specific LEGO modules and methods to execute the procedure described above.

#### **Phase 1: Data Setup & Initialization**

**1\. Data Loading (DataLoader & MEVLoader)**

* **Classes:** Technic.data.DataLoader, Technic.data.MEVLoader, Technic.data.DataManager  
* **Action:** Instantiate loaders for internal and macro data.  
* **Implementation:** Create DataLoader and MEVLoader instances, then combine them into a single DataManager object. This object acts as the central source of truth, automatically handling sample splitting and frequency alignment.

**2\. Segment Initialization (Segment)**

* **Class:** Technic.segment.Segment  
* **Action:** Create a Segment object.  
* **Implementation:** Pass the DataManager and configuration parameters (target variable, model type via Technic.modeltype.ModelType) to the constructor. This object orchestrates all subsequent search and analysis.

#### **Phase 2: Feature Analysis & Engineering**

**3\. Feature Engineering (DataManager)**

* **Method:** Technic.data.DataManager.apply\_to\_all  
* **Action:** Define and apply custom transformation logic.  
* **Implementation:** Write a Python function accepting portfolio and mev dataframes to add new columns (e.g., interaction terms). Use .apply\_to\_all() on the DataManager instance to execute this logic across the dataset.

**4\. Exploratory Data Analysis (Segment)**

* **Method:** Technic.segment.Segment.explore\_var  
* **Action:** Run correlation and trend analysis.  
* **Implementation:** Use .explore\_var() on the Segment instance to generate plots and statistics, aiding the selection of the "Desired Driver Pool."

#### **Phase 3: Automated Model Search**

**5\. Automated Search (ModelSearch)**

* **Method:** Technic.segment.Segment.search\_cms  
* **Action:** Execute the exhaustive search algorithm.  
* **Implementation:** Call search\_cms() on the Segment instance.  
  * **Inputs:** Provide the desired\_driver\_pool, optional forced\_in variables, and constraints (e.g., max\_vars).  
  * **Process:** Triggers Technic.search.ModelSearch to iterate combinations, fit models, run basic tests (VIF, p-values), and **save valid CM (Candidate Model) objects to disk**.

#### **Phase 4: Candidate Evaluation & Selection**

**6\. Candidate Loading & Filtering (Segment)**

* **Method:** Technic.segment.Segment.load\_cms  
* **Action:** Reload saved models.  
* **Implementation:** Use .load\_cms() on the Segment instance to retrieve models saved during Phase 3\.  
  * **Batch Selection:** You can load specific batches of surviving models from past search rounds by simply providing the search\_id. This ID corresponds to the folder name located in Segment/\<segment\_id\>/cms, allowing for easy comparison across different experimental runs.

**7\. Ranking & Evaluation (Segment & CM)**

* **Methods:** Technic.segment.Segment.rerank\_cms, Technic.cm.CM attributes  
* **Action:** Filter, rank, and select the champion.  
* **Implementation:**  
  * **Filter/Rank:** Define a custom filter function (e.g., lambda cm: cm.rsquared \> 0.6) and pass it to .rerank\_cms() along with rank\_weights (e.g., {'aic': 0.5, 'mape': 0.5}).  
  * **Evaluate:** Inspect top-ranked Technic.cm.CM objects directly (reviewing model\_report, diagnostic plots, and forecasts) to make the final decision.

#### **Phase 5: Documentation & Delivery**

**8\. Champion Delivery (Segment)**

* **Method:** Technic.segment.Segment.export  
* **Action:** Export final artifacts.  
* **Implementation:** Use .export() on the Segment instance to aggregate selected champion/backup models into a standardized template for reporting.

## **2\. System Architecture & Module Interaction**

This section explains the interaction between LEGO modules, providing a mental model for efficient usage.

### **2.1 The LEGO Object Hierarchy**

Project LEGO uses a hierarchical structure where objects pass information downstream:

**DataManager** \-\> **Segment** \-\> **CM (Candidate Model)**

#### **1\. The Foundation: DataManager**

* **Role:** Centralized data repository.  
* **Interaction:** Ingests data via loaders; serves as the **Single Source of Truth**; instantiated once and passed to every modeling segment.  
* **Efficiency Tip:** Perform heavy data lifting (joins, cleaning) here, not inside a Segment.

#### **2\. The Orchestrator: Segment**

* **Role:** Modeling scope definition (e.g., "Auto Loans \- Default Rate").  
* **Interaction:** Consumes DataManager; holds configuration (Target, Model Type); executes search (search\_cms) and analysis (explore\_var).  
* **Efficiency Tip:** Multiple Segment objects can share one DataManager.

#### **3\. The Result Unit: CM (Candidate Model)**

* **Role:** Encapsulated object representing a single, fully-fitted model.  
* **Sub-Module Hierarchy:**  
  * **ScenarioManager:** Forecasts (Base, Adverse, Severe).  
  * **ModelReport:** Diagnostic plots, summary tables.  
  * **TestSet:** Validation tests (Stationarity, Residuals, Stability).  
* **Interaction:** Produced by Segment search; persistable (saved/loaded).  
* **Efficiency Tip:** Filtering queries nested attributes within these sub-modules (e.g., cm.test\_set.stability.p\_value).

### **2.2 Module Interaction Flow**

1. **Preparation (DataManager):** Loaders ingest raw files; DataManager consolidates them and runs feature engineering.  
2. **Configuration (Segment):** Segment initializes with DataManager and defines the problem space.  
3. **Execution (ModelSearch & ModelBase):**  
   * Segment triggers ModelSearch.  
   * Search engine instantiates CM objects wrapping specific ModelBase implementations (e.g., OLS).  
   * Engine triggers CM.fit(), executing regression and embedded TestSet validation.  
   * **Outcome:** Valid CM objects are saved locally; failed ones are dumped.  
4. **Decision (CM & Segment):** Segment reloads CM objects; advanced filtering is applied; final selection is exported.

### **2.3 Model-Specific Architecture (ModelBase)**

The CM object wraps model-specific logic handled by subclasses of ModelBase.

* **CM (Wrapper):** Manages lifecycle (training, testing, saving). Model-agnostic.  
* **ModelBase (Engine):** Defines mathematical logic (e.g., OLS, AR, ECM). Performs fitting and prediction.  
* **Key Implementation:** OLS handles standard linear regression.  
* **Extensibility:** New model types require a pretest function (fast-fail checks), a testset function (model-specific diagnostics), and a ModelReportBase subclass (visualization).

## **3\. Feature Engineering & Analysis: Best Practices**

Careful preparation determines input quality for the automated search.

### **3.1 Feature Engineering Principles**

* **Structural Features:** While automation handles standard transformations (Lags, Growth), modelers must define structural features like **Regime Signals** (e.g., Recession flags), **Interaction Terms**, and **Custom Ratios**.  
* **Data Integrity:** Ensure features are robust (e.g., stationarity for OLS) and consistent across historical/projected data.

### **3.2 Feature Analysis Guidelines**

* **Univariate Analysis:** Use .explore\_var() to filter drivers based on correlation, visual alignment with the target, and economic logic.  
* **Driver Pool Selection:**  
  * **Quality:** Don't feed all MEVs into the search.  
  * **Diversity:** Select from different categories (Labor, Housing, Rates).  
  * **Quantity:** Aim for **15-20 high-quality drivers**. A focused pool yields better results than a "kitchen sink" approach.

## **4\. Automated Model Search: Mechanism & Best Practices**

The automated search (segment.search\_cms()) uses the ModelSearch class to rigorously explore model combinations.

### **4.1 Mechanism: How It Works**

1. **Input Configuration:** User defines Driver Pools (Desired/Must-Have), Transformations, and Constraints (Structural/Test thresholds).  
2. **Feature Validity Pre-test:** Engine filters invalid features (NaNs, non-stationary) to reduce load.  
3. **Combination Generation:** Engine generates equations based on variable combinations, transformations, and constraint enforcement.  
4. **Fitting & Basic Validation:**  
   * Models are fitted using the ModelBase engine (e.g., OLS).  
   * **Dynamic Validation:** Specific tests (Significance, VIF, etc.) defined by the base model are executed.  
   * **Outcome:** Only models passing all gates are saved as CM objects.  
5. **Ranking & Initial Preview:** Models are sorted by default or custom rank\_weights to provide an immediate "Leaderboard."

### **4.2 Strategic Usage**

* **"Garbage In, Garbage Out":** The search optimizes *known* good drivers; it doesn't discover meaning in random data. Spend 80% of effort on Feature Analysis.  
* **The "Dry Run" Rule:** Always run a small test search (2-3 drivers) first to verify the pipeline (data integrity, transformations, saving) before a full run.  
* **The "Forced-In" Lever:** Use forced\_in variables to reduce combinatorial space, but remember they must still pass statistical significance tests.  
* **Iterative Refinement (Wide Net vs. Deep Dive):**  
  1. **Driver Pool Preparation:** Start with a "Desired Pool" of key drivers identified during Feature Analysis.  
  2. **Search Configuration:** Determine the optimal setup for the automated search.  
     * *Structure:* Set maximum variable numbers (max\_vars), maximum lag depth, and transformation spans.  
     * *Thresholds:* Fine-tune pass/fail gates (e.g., adjusting P-value or VIF limits) to balance rigorousness with yield.  
  3. **Efficiency Check:** After configuration, the function will estimate the number of combinations. If the count is excessively high:  
     * *Action:* Refine the setup. Reduce the number of desired variables (select only the most significant ones), move highly-confident drivers to the forced\_in pool, or lower the max\_vars limit.  
  4. **Phased Approach:**  
     * *Early/Middle Stages:* Run broader searches to gain insights on potential drivers from a wide pool. The goal here is discovery.  
     * *Late Stage:* Once the list is narrowed to highly predictive drivers, run a truly "exhaustive" search to select the "best from excellent."

### **4.3 Caveats**

* **Overfitting:** A statistically perfect model may lack economic meaning. Manual Champion Evaluation is mandatory.  
* **Computational Explosion:** Keep max\_vars reasonable (2-3) and use category constraints to prevent exponential runtimes.

## **5\. Candidate Selection & Feedback Loop**

### **5.1 Selection Workflow**

#### **1\. The Loading-Filtering-Ranking Pipeline**

To begin selection, load the saved models back into the Segment object and simultaneously apply filters and ranking logic.

* **Method:** segment.rerank\_cms(func\_filter, rank\_weights)  
* **Action:** This single command executes a dual-process pipeline:  
  1. **Filter:** It iterates through every loaded model and applies your custom boolean function (func\_filter). Models returning False are immediately discarded from consideration.  
  2. **Rank:** The survivors are then scored and sorted based on your weighted preferences (rank\_weights) for specific metrics (e.g., AIC, MAPE).  
* **Outcome:** A refined, ordered list of candidates ready for manual inspection.

#### **2\. Champion Evaluation (Manual)**

Select the top 5-10 "survivors" from the ranked list for deep-dive inspection.

* **Visual Check:** Plot the actual vs. predicted values. Does the model capture the key turning points (e.g., the 2008 or 2020 crises)?  
* **Residual Analysis:** Are the errors truly random, or is there a pattern indicating a missing variable?  
* **Economic Rationale:** Does the combination of drivers make theoretical sense? (e.g., combining "Unemployment" and "GDP" might be redundant due to high correlation).

### **5.2 Guidelines for Effective Filtering**

Effective filtering transforms technical outputs into business solutions.

#### **1\. Unified Filtering Logic (The "One-Pass" Rule)**

Currently, all advanced filters must be consolidated into **one single boolean function**. You cannot apply filters sequentially in multiple steps.

* **Best Practice:** Define a master function that combines all your logic with and operators.  
* **Example:**  
  def master\_filter(cm):  
      \# 1\. Stability Check (Example Attribute)  
      \# Note: Check actual codebase for exact attribute paths (e.g., cm.test\_set.chow.p\_value)  
      is\_stable \= cm.test\_set.chow.p\_value \> 0.05

      \# 2\. Scenario Check (Example Attribute)  
      \# Note: Forecasts are typically stored in cm.model\_report.prediction  
      sev\_loss \= cm.model\_report.prediction\['Severe'\]\['projection'\].sum()  
      base\_loss \= cm.model\_report.prediction\['Base'\]\['projection'\].sum()  
      is\_logical \= sev\_loss \> base\_loss

      \# 3\. Sign Check (Example Attribute)  
      \# Note: Coefficients are typically in cm.model.params  
      is\_positive \= cm.model.params.get('Unemployment', 0\) \> 0

      return is\_stable and is\_logical and is\_positive

#### **2\. Dynamic Weighting Strategy**

Modelers should adjust rank\_weights dynamically based on the observed weaknesses of the candidate pool.

* **Scenario A (Overfitting):** If surviving models have high In-Sample ![][image2] but poor Out-of-Sample performance:  
  * *Action:* Increase the weight of Out-of-Sample MAPE significantly (e.g., 70%).  
* **Scenario B (Instability):** If forecasts are volatile:  
  * *Action:* Increase the weight of stability metrics or penalized criteria like BIC.  
* **Principle:** Use the weights to "pull" the ranking towards the specific characteristic your current batch of models is lacking.

### **5.3 Feedback Loop (Iterative Improvement)**

If the search does not yield a satisfactory champion, the results themselves offer valuable clues for the next iteration. Do not simply restart blindly; analyze the failure patterns.

* **Dominant Drivers:** If a driver appears in 90% of top models, consider "forcing" it to explore other variables more deeply.  
* **Missing Drivers:** If an expected driver is absent, check for collinearity or try different transformations.  
* **Transformation Patterns:** If specific lags (e.g., Lag 4\) dominate, narrow the transformation scope to focus on that window.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAoklEQVR4XmNgGAWjYBACKSkpWSDoVlBQ4ECXIwvIycmVgzC6OFlAXFxcTF5efr+MjIwZuhxZAGQQ0HVHgLQKioSoqCgP0CZJMnAw0MBHQAM54YYBBSpAgmTgZ0AD/wPpeCS3kQ6A4cYNNGQh0LA+dDmSANAQVyBejeI9MgELyEVAgzzQJUgGQEOkgYZtBiZeEXQ5cgAr0EAhIM2ILjEKBhgAALE4LFqqfgmMAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAZCAYAAADaILXQAAABpElEQVR4Xu1UPUsDURBMkICCIIckQj7uThuxsrASLCxiYaGNVmoh5CeIjZ2IjX9A0DQWUUwaC60sklLwD9ik0MY+YGFjnPH2hc0mYrwqRQaGe7uzO/fe5l0SiWFGEAQ7vu9fgGfZbHbe6rEBwyK4xrXneVN40RPYMGXxkM/nV2FWcjFedIX4Vdd0gOJ1KXjB8w28kyM7HqMsadp+4nQ6Pcldc/dGjwBhAQbbeFbAdqFQ2GcsuRJYx7rGEdhejgf8QM2W1TpQO2hbjWAeJs1MJjPDmC9CXAWLtrYHKJqDwTvYshqQFPNn0GMC8Sm47HSc9kTVdwPipjOwGvKz1MADSaVo7kaH3iPE111NGrITGpSNlPKjH/WcayZ4AaRW0/ZFUCP5BCtids8mPB9zudy07RkYeiRupgTWe5Kv6vp/IfhlJDBd8qNr1v8D+Qv6CoK7WvOjz/wrtrmadwuzXdSaOxFqmjo/MNB4KQa1MAzHXZ5r5B/kRA3ckAnU3HJUur8vuEvuVpo1b1wNjFY4c5l7GTxM9P7HxAevIcw3ZMdjVh9hhCHGN7EFgx8h0TpiAAAAAElFTkSuQmCC>