# 📘 Employee Performance Predictor: Complete Project Guide

This guide explains everything about the project: the concept, design choices, every phase, the simulation, how to run it, and how to fix problems.

**Contents**
- [A. Project explanation](#a-project-explanation)
- [B. Tech stack options](#b-tech-stack-options)
- [C. Architecture](#c-architecture)
- [Implementation plan (10 phases)](#implementation-plan-10-phases)
- [Folder structure explained](#folder-structure-explained)
- [Installation guide](#installation-guide)
- [D. Code walkthrough (file by file)](#d-code-walkthrough-file-by-file)
- [E. Virtual simulation](#e-virtual-simulation)
- [How to run & expected output](#how-to-run--expected-output)
- [Future improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

---

## A. Project explanation

### What is Employee Performance Prediction?

**Simple explanation (for beginners)**
Think of a cricket team selector. They look at a player's past scores, fitness, practice hours and attitude to guess how well that player will do next season. Employee performance prediction does the same for a company. It looks at an employee's training, projects, attendance, feedback and satisfaction, and a computer program guesses whether they will be a **Low**, **Medium** or **High** performer. The program learns these patterns from past employees.

**Technical explanation (for interviews)**
It is a **supervised multi-class classification** problem. Each employee is a row of features *X* (demographics, tenure, compensation, behavioural and engagement metrics), and the label *y* is the performance band derived from appraisal scores. We train a model *f(X) → y* on historical records, validate it with stratified cross-validation, and use it to produce class probabilities for current or new employees. A business rule layer then maps those probabilities to HR interventions. Because HR decisions affect people, the model also needs to be **interpretable** (feature importance) and **fair** (no protected attributes).

### Why companies need it

| Who | Pain point | How prediction helps |
|---|---|---|
| **HR** | Finds problems only at the annual review | Early-warning list months in advance |
| **Managers** | Appraisals are subjective and biased by recent events | Objective, data-based second opinion |
| **Business leaders** | Can't see talent strength across departments | Department-level heat map of performance risk |
| **Finance** | Training budget is spread evenly, not where it's needed | Targets training spend where it raises performance most |

### How companies use it

1. **Identifying high performers**: employees predicted *High* with strong confidence go into the talent pool.
2. **Predicting low performance**: employees predicted *Low* get support early (mentoring, a Performance Improvement Plan, a workload review) instead of being surprised at appraisal time.
3. **Promotion decisions**: the model's probability is one input alongside the manager's view, which reduces bias.
4. **Training & development planning**: feature importance shows *which levers matter*. For example, if training hours drive performance, the business case for training budget gets stronger.
5. **Retention strategies**: low job satisfaction and burnout signals overlap with attrition risk. High performers with low satisfaction are the people a company most risks losing.

### End-to-end workflow

```
DATA ──► PREPROCESSING ──► ANALYSIS ──► MODEL ──► PREDICTION ──► INSIGHTS ──► HR DECISIONS
 │            │                │           │           │              │              │
 HR records   clean, fix,      charts,     train &     Low/Med/High   drivers,       promote,
 (synthetic)  engineer         patterns,   compare 4   + probability  dept. view,    train, support,
              features         drivers     models                     action lists   review workload
```

---

## B. Tech stack options

| | **Option A: Easy** | **Option B: Intermediate** ✅ | **Option C: Advanced** |
|---|---|---|---|
| **Tools** | Excel / Google Sheets, or Python + Pandas in one notebook | Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn (Pipelines), joblib, Streamlit, Git | Everything in B + XGBoost/LightGBM, SHAP, MLflow, FastAPI, Docker, PostgreSQL, cloud deploy |
| **ML models** | Logistic Regression only | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (compared with CV) | Tuned gradient boosting, neural nets, model monitoring |
| **Structure** | Single notebook | Modular `src/` package + `main.py` + dashboard | Microservices, CI/CD, tests, containers |
| **Difficulty** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Time** | 1–2 days | 5–7 days | 3–4 weeks |
| **Interview impact** | Low: looks like coursework | **High: shows real engineering habits** | Very high, but hard to finish and explain well |

### ✅ Selected: Option B (Intermediate)

Why it's the best choice for a student:
- It uses the **exact tools listed in most Data Analyst / Data Scientist job descriptions**.
- **Modular code** (`src/`) and **sklearn Pipelines** show production thinking without heavy DevOps.
- **Comparing 4 models with cross-validation** gives strong interview talking points.
- The **Streamlit dashboard** gives you something visual to demo live.
- You can finish it in about a week **and explain every line**, which matters more than complexity.

---

## C. Architecture

### Input → Process → Output

| Stage | What | Where in code |
|---|---|---|
| **Input** | Employee dataset: age, experience, department, education, job level, salary, training hours, projects, monthly hours, absences, last rating, peer feedback, years since promotion, satisfaction, work-life balance | `data/raw/employee_data.csv` |
| **Process 1** | Data cleaning: duplicates, text, outliers, missing values | `src/preprocessing.py` |
| **Process 2** | Feature engineering: salary per level, projects per year, training per year, overtime | `src/preprocessing.py` |
| **Process 3** | Model training: 4 models, 5-fold CV, best one saved | `src/train.py` |
| **Process 4** | Prediction: class + probability + HR rule | `src/predict.py` |
| **Output** | Predictions, HR action lists, insights report, dashboard | `outputs/`, `images/`, `app.py` |

### Text-based architecture diagram

```
                         ┌─────────────────────────────────────────┐
                         │           main.py (orchestrator)        │
                         └─────────────────────────────────────────┘
                                              │
   ┌──────────────────┐   CSV    ┌────────────▼───────────┐   CSV   ┌────────────────┐
   │ data_generator   │ ───────► │ preprocessing          │ ──────► │ eda            │──► images/01-06
   │ (virtual company)│ data/raw │  clean_data()          │ data/   └────────────────┘
   └──────────────────┘          │  engineer_features()   │ processed
                                 └────────────┬───────────┘
                                              │ DataFrame
                                 ┌────────────▼───────────┐
                                 │ train                   │
                                 │  split 80/20 stratified │
                                 │  4 models × 5-fold CV   │──► outputs/model_comparison.csv
                                 │  fit best → .joblib     │──► models/best_model.joblib
                                 └────────────┬───────────┘
                                              │ Pipeline(prep → model)
                                 ┌────────────▼───────────┐
                                 │ evaluate                │──► metrics.json, report.txt
                                 │  test set only          │──► images/07-09
                                 └────────────┬───────────┘
                                              │
            ┌──────────────────┐ ┌────────────▼───────────┐
            │ new_employees.csv│►│ predict                 │──► new_employee_predictions.csv
            └──────────────────┘ │  probabilities + rules  │
                                 └────────────┬───────────┘
                                              │
                                 ┌────────────▼───────────┐
                                 │ insights                │──► hr_insights_report.md, action lists
                                 │                         │──► images/10_hr_dashboard.png
                                 └────────────┬───────────┘
                                              ▼
                                 ┌─────────────────────────┐
                                 │ app.py (Streamlit)      │  ◄── HR user
                                 └─────────────────────────┘
```

### Data flow explained

1. **Generation:** `data_generator.py` creates 2,000 employees and writes a deliberately messy CSV.
2. **Cleaning:** duplicates are dropped (25), department text is standardised, impossible ages and training hours are set to missing, and all 350 missing cells are filled.
3. **Feature engineering:** 4 business-meaningful columns are added.
4. **Split:** 80% for training and 20% held back as the "exam" the model never sees.
5. **Preprocessing inside the Pipeline:** numbers are scaled and categories one-hot encoded. This is fitted **only on training data**, which prevents data leakage.
6. **Model selection:** each candidate is scored by 5-fold CV and the best F1-macro wins.
7. **Evaluation:** the winning pipeline is scored once on the test set.
8. **Scoring & rules:** every employee gets a predicted band, a confidence and an HR action.
9. **Delivery:** HR reads the report or dashboard and acts.

---

## Implementation plan (10 phases)

### Phase 1: Setup
- **What:** Install Python, create the project folder and virtual environment, install libraries, `git init`.
- **Why:** An isolated environment means the project runs the same on any machine.
- **Expected output:** `pip list` shows pandas, scikit-learn and the other libraries; the folder structure exists.
- **Common mistakes:** Installing into global Python; forgetting to activate the venv; Python not on PATH (Windows).

### Phase 2: Data creation / loading
- **What:** Run `python -m src.data_generator`.
- **Why:** There is no real HR data, so we simulate a realistic company where the true performance formula is known.
- **Expected output:** `data/raw/employee_data.csv` (2,025 rows × 20 columns) and `new_employees.csv`.
- **Common mistakes:** Making data too perfect (100% accuracy looks fake); not fixing a random seed (results change every run).

### Phase 3: Data cleaning
- **What:** Remove duplicates, standardise text, handle outliers, fill missing values.
- **Why:** "Garbage in, garbage out." Real HR exports are always messy.
- **Expected output:** Cleaning report showing 25 duplicates removed, 10 invalid values fixed, and missing values going from 350 to 0.
- **Common mistakes:** Filling with the mean when outliers exist (use the median); dropping too many rows; cleaning *after* splitting inconsistently.

### Phase 4: Exploratory Data Analysis
- **What:** Class balance, department comparison, correlations, driver box plots, overtime effect.
- **Why:** Understand the data *before* modelling and find the business story.
- **Expected output:** 6 charts in `images/` plus console summaries.
- **Common mistakes:** Making charts with no question behind them; not writing down what each chart shows.

### Phase 5: Feature engineering
- **What:** `salary_per_level`, `projects_per_year`, `training_per_year`, `overtime_hours`.
- **Why:** Ratios make employees comparable. 10 projects in 1 year is very different from 10 projects in 10 years.
- **Expected output:** `projects_per_year` turns out to be a top-5 driver.
- **Common mistakes:** **Data leakage**, meaning using `performance_score` (the source of the label) as a feature, which gives a fake 100% accuracy.

### Phase 6: Model building
- **What:** Build 4 Pipelines (preprocessor + model), compare them with 5-fold stratified CV, and train the best.
- **Why:** You never know in advance which model fits best, and comparing them is industry practice.
- **Expected output:** A comparison table, with Logistic Regression winning (F1 ≈ 0.76).
- **Common mistakes:** Judging models on training accuracy; not stratifying an imbalanced target.

### Phase 7: Evaluation
- **What:** Accuracy, precision, recall, F1, confusion matrix and permutation importance on the test set.
- **Why:** Accuracy alone hides which classes are confused. For HR, missing a Low performer (low recall on Low) is the costly error.
- **Expected output:** Test accuracy around 76%, with the confusion matrix showing that most errors are between neighbouring classes (Low↔Medium, Medium↔High).
- **Common mistakes:** Tuning on the test set; reporting only accuracy.

### Phase 8: Insights
- **What:** Convert predictions into action lists and a plain-English report.
- **Why:** Companies pay for *decisions*, not models.
- **Expected output:** `hr_insights_report.md`, `promotion_candidates.csv`, `at_risk_employees.csv`.
- **Common mistakes:** Ending the project at "accuracy = 76%" with no business recommendation.

### Phase 9: Visualisation
- **What:** A one-page PNG dashboard plus an interactive Streamlit app.
- **Why:** Managers read dashboards, not notebooks.
- **Expected output:** `images/10_hr_dashboard.png` and a live app at `localhost:8501`.
- **Common mistakes:** Cluttered charts; no titles or units.

### Phase 10: GitHub upload
- **What:** README, screenshots, clean commits, push. See [GITHUB_GUIDE.md](GITHUB_GUIDE.md).
- **Why:** The GitHub repo is your proof of work for recruiters.
- **Expected output:** A public repo with a README that renders charts.
- **Common mistakes:** One giant "final" commit; no README; committing `venv/`.

---

## Folder structure explained

| Folder / file | Purpose |
|---|---|
| `data/raw/` | Original generated data (messy) + new employees to score. Never edited by hand. |
| `data/processed/` | Cleaned + feature-engineered data used for training. |
| `notebooks/` | Jupyter walkthrough for exploring step by step and for demos. |
| `src/` | Reusable Python modules, one responsibility each. |
| `models/` | Saved trained pipeline (`best_model.joblib`), loaded by predict and app. |
| `outputs/` | Machine-readable results: metrics, reports, predictions, action lists. |
| `images/` | All charts, used in the README and for screenshots. |
| `docs/` | Project guide, GitHub guide, interview prep. |
| `main.py` | Runs the entire pipeline with one command. |
| `app.py` | Streamlit dashboard. |
| `requirements.txt` | Library list so anyone can reproduce your environment. |
| `.gitignore` | Stops `venv/`, `__pycache__/` and similar from being uploaded. |

---

## Installation guide

### Step 1: Install Python
- **Windows:** Download Python 3.10+ from python.org. During setup, **tick "Add python.exe to PATH"**.
- **Mac:** `brew install python` or use the python.org installer.
- Check it worked: `python --version` (Mac may need `python3 --version`).

### Step 2: Get the project
```bash
git clone https://github.com/SNischayPrasad/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor
```

### Step 3: Create a virtual environment

| | Windows (PowerShell / CMD) | Mac / Linux |
|---|---|---|
| Create | `python -m venv venv` | `python3 -m venv venv` |
| Activate | `venv\Scripts\activate` | `source venv/bin/activate` |
| Deactivate | `deactivate` | `deactivate` |

When it's active, you'll see `(venv)` at the start of your prompt.

> **Windows PowerShell error "running scripts is disabled"?** Run this once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Step 4: Install libraries
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify
```bash
python -c "import pandas, sklearn, matplotlib, seaborn; print('All good!')"
```

---

## D. Code walkthrough (file by file)

### `src/config.py`
The single source of truth for paths, random seed, test size and **feature lists**. The comments explain why `performance_score` is excluded (leakage).

### `src/data_generator.py`
- `generate_employee_data()` builds employees with realistic dependencies: experience depends on age, job level on experience, and salary on level × department × education.
- It computes a hidden **latent performance** value as a weighted sum of standardised drivers, plus random noise.
- `performance_score = 60 + 12 × (latent + noise)`, clipped to 0–100.
- Bottom 25% → **Low**, middle 50% → **Medium**, top 25% → **High**.
- `add_real_world_noise()` injects missing values, outliers (age 999, 9999 training hours), messy text and 25 duplicate rows.
- `generate_new_employees()` creates 10 unlabeled people for the prediction demo.

### `src/preprocessing.py`
- `clean_data()`: dedupe → standardise text → invalid values to NaN → median/mode fill → prints a cleaning report.
- `engineer_features()`: adds the 4 ratio features.
- `build_preprocessor()`: `ColumnTransformer` with `StandardScaler` for numbers and `OneHotEncoder(handle_unknown="ignore")` for categories. It lives *inside* the model Pipeline, so the same transformation is applied at training and prediction time.

### `src/eda.py`
Six charts, each answering one question: class balance, department mix, correlations, driver box plots, score distribution, and overtime vs performance.

### `src/train.py`
- `split_data()`: stratified 80/20 split.
- `compare_models()`: 5-fold `StratifiedKFold` CV, scoring accuracy and F1-macro.
- `train_best()`: refits the winner on all training data and saves `{pipeline, model_name, features}` with joblib.
- `class_weight="balanced"` keeps the minority classes (Low/High) from being ignored.

### `src/evaluate.py`
Test-set metrics, `classification_report`, a confusion matrix heatmap, a model comparison chart, and **permutation importance**. Permutation importance shuffles one feature at a time and measures the F1 drop, which works for any model.

### `src/predict.py`
- `predict_employees()` applies the same cleaning and features, then returns the predicted class, confidence and per-class probabilities.
- `recommend()` adds a **rule layer**: High + ≥70% confidence → promotion candidate, Low → PIP + mentor. It also adds personalised flags for low training, high absences, low satisfaction and burnout.
- CLI: `python -m src.predict --input yourfile.csv`.

### `src/insights.py`
Scores the whole workforce, then writes the action list, promotion candidates, at-risk list, department summary, a Markdown report with data-driven findings, and the one-page dashboard PNG.

### `main.py`
Calls every step in order with clear banners and timing.

### `app.py`
Streamlit app with 4 tabs: Overview (KPIs + charts), single-employee prediction form, batch CSV upload with download, and action lists.

### `notebooks/employee_performance_walkthrough.ipynb`
The same pipeline split into cells. Good for learning and for screen-sharing in interviews.

---

## E. Virtual simulation

### How the virtual company works

**NovaTech Solutions** is a fictional IT services company with **2,000 employees in 7 departments**.

| Step | What happens | Real-world equivalent |
|---|---|---|
| **1. Hire employees** | Random age 21–59; experience ≈ 0.8 × (age − 21); tenure ≤ experience | HRIS employee master data |
| **2. Assign roles** | Job level from experience (more years → Senior/Lead) | Org structure |
| **3. Set pay** | Base by level × department factor × education bonus ± 10% | Payroll |
| **4. Record behaviour** | Training hours, projects, monthly hours, absences | LMS, project tools, attendance system |
| **5. Collect feedback** | Peer feedback (1–5), satisfaction and work-life balance surveys (1–5) | 360° feedback, engagement surveys |
| **6. Calculate performance** | Hidden weighted formula + random noise → score 0–100 | Annual appraisal |
| **7. Make data messy** | Missing values, typos, duplicates | Real HR exports |
| **8. AI learns** | The model sees only the features + label, **never the formula** | Training on history |
| **9. AI predicts** | New employees get a band + probability | Mid-year early warning |
| **10. HR acts** | Rules turn predictions into actions | Talent review meeting |

### How the performance score is calculated (the hidden formula)

```
latent =  0.30 × training_hours        (z-score)
        + 0.30 × projects per year      (capped at 4)
        + 0.35 × peer_feedback_score
        + 0.30 × job_satisfaction
        + 0.15 × work_life_balance
        − 0.30 × absences
        + 0.10 × education level
        + 0.10 × experience (capped at 15 yrs)
        − 0.25 × burnout penalty        (hours above 220/month)
        − 0.10 × years since promotion

performance_score = 60 + 12 × (latent + random_noise)     → clipped 0–100
category: bottom 25% = Low | middle 50% = Medium | top 25% = High
```

The **random noise** stands for things no dataset captures, like personal life, luck or team dynamics. It is why no model can reach 100%, which keeps the results realistic.

> 💡 **Great interview point:** "Because I built the simulation, I knew the true drivers. The model's feature importance found peer feedback, absences, satisfaction, projects per year and training as the top 5, which matches the formula. That shows the model learned real signal rather than noise."

### How the AI predicts
1. The new employee's row is cleaned and the engineered features are added.
2. The saved Pipeline scales and encodes the row exactly as it did in training.
3. Logistic Regression computes a score for each class and converts them into probabilities (softmax).
4. The highest probability becomes the prediction, and that probability is the confidence.
5. The rule layer adds the HR recommendation.

### How HR would use this system (a day in the life)
1. **Quarterly:** HR exports the latest employee data to CSV.
2. Uploads it on the **Batch upload** tab and downloads predictions.
3. Opens **Action lists**: the at-risk list goes to managers with a suggested support plan, and promotion candidates go to the talent review.
4. Leadership looks at the **department summary** and moves training budget to Finance and Support, which have the highest share predicted Low.
5. Managers hold the 1:1s. **A human makes every final decision.**
6. After the appraisal cycle, actual results are added to the data and the model is **retrained**.

### Step-by-step simulation run

```
Step 1 → python main.py                    (generates company, 2,025 messy rows)
Step 2 → cleaning report printed           (25 dupes, 10 invalid values, 350 missing → 0)
Step 3 → 6 EDA charts saved                (images/01–06)
Step 4 → 4 models compared                 (LogReg 0.763 F1 wins)
Step 5 → test evaluation                   (76.0% accuracy, confusion matrix)
Step 6 → HR insights generated             (report + action lists + dashboard)
Step 7 → 10 new employees predicted        (with recommendations)
Step 8 → streamlit run app.py              (interactive demo)
Final  → HR decision: promote / train / support / review workload
```

### What to capture as proof

| # | Screenshot | How |
|---|---|---|
| 1 | Project folder structure | VS Code explorer sidebar |
| 2 | Raw messy data | `data/raw/employee_data.csv` opened in Excel/VS Code (show the 9999 / blank cells) |
| 3 | Cleaning report | Terminal output of Step 2–3 |
| 4 | EDA charts | `images/01`–`06` (already saved) |
| 5 | Model comparison | Terminal output of Step 5 + `images/08` |
| 6 | Classification report | Terminal Step 6 / `outputs/classification_report.txt` |
| 7 | Confusion matrix | `images/07` |
| 8 | Feature importance | `images/09` |
| 9 | New-employee predictions | Terminal Step 8 |
| 10 | HR dashboard | `images/10_hr_dashboard.png` |
| 11 | Streamlit app (each tab) | Browser screenshots of `localhost:8501` |
| 12 | "DONE" banner | Final terminal line with accuracy |

Save the manual screenshots in `images/screenshots/` with names like `01_folder_structure.png`.

---

## How to run & expected output

| Command | What it does | Time |
|---|---|---|
| `python main.py` | Full pipeline | ~1–2 min |
| `python -m src.data_generator` | Only generate data | 2 s |
| `python -m src.preprocessing` | Only clean data | 2 s |
| `python -m src.eda` | Only charts | 10 s |
| `python -m src.train` | Only train | ~1 min |
| `python -m src.predict` | Predict new employees (needs a trained model) | 2 s |
| `streamlit run app.py` | Dashboard in browser | — |
| `jupyter notebook notebooks/` | Walkthrough notebook | — |

**Expected console output (abridged):**
```
STEP 1/8  Generating synthetic employee data
[data] Generated 2,025 rows x 20 columns
STEP 2-3/8  Cleaning data & engineering features
        duplicates_removed      : 25
        missing_before          : 350
        missing_after           : 0
STEP 5/8  Training & comparing models
[train] Logistic Regression  accuracy=0.758  F1-macro=0.763
[train] Decision Tree        accuracy=0.554  F1-macro=0.554
[train] Random Forest        accuracy=0.674  F1-macro=0.678
[train] Gradient Boosting    accuracy=0.707  F1-macro=0.695
[train] Best model: Logistic Regression
STEP 6/8  Evaluating best model on test set
[eval] Accuracy : 0.760
DONE in ~100s | Best model: Logistic Regression | Test accuracy: 76.0%
```
Results are reproducible because of `RANDOM_STATE = 42`. Tiny differences can occur between library versions.

---

## Future improvements

| Upgrade | What to do | Skill shown |
|---|---|---|
| **Real HR dataset** | Swap in the IBM HR Analytics dataset (Kaggle, has `PerformanceRating`) and adapt the `config.py` feature lists | Working with real data |
| **Attrition prediction** | Add a second model predicting `will_leave` (binary), and combine it with performance into a 9-box "performance × flight risk" grid | Multi-model systems |
| **Explainability** | SHAP values per employee ("why was I predicted Low?") | Responsible AI |
| **Better models** | GridSearchCV / Optuna tuning, XGBoost, LightGBM | Model optimisation |
| **Deep learning** | A small Keras MLP for comparison. Note that on tabular data, tree models usually still win | Knowing when *not* to use DL |
| **Deploy dashboard** | Streamlit Community Cloud (free) gives a live link for your resume | Deployment |
| **Real-time system** | FastAPI `/predict` endpoint + Docker + database; retrain monthly with MLflow tracking | MLOps |
| **Fairness audit** | Compare prediction rates across gender/age groups (added only for auditing) | Ethics |
| **Testing & CI** | `pytest` unit tests + GitHub Actions | Software engineering |

---

## Troubleshooting

| Error / problem | Cause | Fix |
|---|---|---|
| `'python' is not recognized` | Python not on PATH | Reinstall and tick "Add to PATH", or use `py` instead of `python` |
| `ModuleNotFoundError: No module named 'pandas'` | Libraries not installed / venv not active | Activate the venv, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Running from the wrong folder | `cd` to the project root; run `python -m src.predict`, not `python src/predict.py` |
| `FileNotFoundError: ... best_model.joblib` | Model not trained yet | Run `python main.py` first |
| `FileNotFoundError: employee_data_clean.csv` | Running a single step out of order | Run `python main.py` once |
| PowerShell: `running scripts is disabled` | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Charts don't pop up | They are saved to files, not shown (Agg backend) | Open the `images/` folder |
| `streamlit: command not found` | Not installed | `pip install streamlit`, or run `python -m streamlit run app.py` |
| Streamlit shows "No trained model found" | `main.py` not run | Run `python main.py`, then refresh |
| Training is slow | CV runs 20 model fits | Normal (~1–2 min). Lower `N_EMPLOYEES` in config for a quick test |
| `TypeError ... width` in Streamlit | Streamlit older than 1.50 | `pip install --upgrade streamlit` |
| `git push` rejected | Remote has commits (e.g. a README created on GitHub) | `git pull origin main --rebase`, then push |
| `UnicodeDecodeError` reading CSV | Excel saved it in another encoding | `pd.read_csv(path, encoding="latin-1")` |
| Accuracy is 100% | Data leakage | Make sure `performance_score` isn't in `ALL_FEATURES` |
