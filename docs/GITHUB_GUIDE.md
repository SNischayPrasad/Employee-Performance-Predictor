# 🐙 GitHub Upload & Proof-Building Guide

## F1. Repository details

| Field | Value |
|---|---|
| **Repo name** | `Employee-Performance-Predictor` |
| **Description** | `End-to-end HR analytics & ML project: predicts employee performance (Low/Medium/High) with scikit-learn and turns predictions into HR actions. Includes synthetic company simulation, EDA, 4-model comparison, insights report and Streamlit dashboard.` |
| **Topics / tags** | `machine-learning` `data-analytics` `hr-analytics` `people-analytics` `python` `scikit-learn` `pandas` `data-visualization` `classification` `streamlit` `eda` `employee-performance` |
| **Visibility** | Public |
| **Website field** | Your Streamlit Cloud link (once deployed) |

## F2. Step-by-step upload

### Step 1: One-time Git setup
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Step 2: Create the repo on GitHub
1. Go to https://github.com/new
2. Repository name: `Employee-Performance-Predictor`
3. Paste the description above
4. **Public**
5. **Do NOT tick** "Add a README" (we already have one; ticking it causes a push conflict)
6. Click **Create repository**

### Step 3: Connect your local folder
From the project folder:
```bash
git init                      # skip if the repo is already initialised
git branch -M main
git remote add origin https://github.com/SNischayPrasad/Employee-Performance-Predictor.git
git remote -v                 # check the link
```

### Step 4: Commit & push
```bash
git add .
git status                    # make sure venv/ and __pycache__/ are NOT listed
git commit -m "Initial commit: project structure and pipeline"
git push -u origin main
```
GitHub will ask you to sign in. Use the browser login, or a **Personal Access Token** instead of your password (GitHub → Settings → Developer settings → Tokens).

### Step 5: Add topics
On the repo page → ⚙️ next to **About** → paste the topics → Save.

### Step 6: Upload screenshots
1. Put manual screenshots in `images/screenshots/` (e.g. `01_folder_structure.png`, `11_streamlit_predict.png`).
2. Reference them in the README: `![Streamlit app](images/screenshots/11_streamlit_predict.png)`
3. Commit:
```bash
git add images/screenshots README.md
git commit -m "docs: add Streamlit and terminal screenshots"
git push
```

### Step 7: Pin it
GitHub profile → **Customize your pins** → select this repo.

### Step 8 (optional): Deploy the dashboard for free
1. https://share.streamlit.io → sign in with GitHub
2. New app → choose the repo → main file `app.py` → Deploy
3. Because `models/` and `outputs/` are committed, the app works straight away
4. Put the URL in the repo's **Website** field and on your resume

## F3. Good commit message style

Format: `type: short description` (imperative mood, under ~60 characters)

| Type | Use for | Example |
|---|---|---|
| `feat` | New functionality | `feat: add Random Forest and Gradient Boosting comparison` |
| `data` | Data generation/cleaning | `data: inject missing values and outliers into synthetic data` |
| `fix` | Bug fix | `fix: prevent years_at_company exceeding experience` |
| `docs` | README/guides | `docs: add results table and screenshots to README` |
| `refactor` | Code tidy-up | `refactor: move paths into config.py` |
| `viz` | Charts | `viz: add confusion matrix and feature importance plots` |

❌ Avoid: `update`, `final`, `final2`, `changes`, `asdf`

## F4. Day-by-day proof-building plan

Pushing a little each day builds a **green contribution graph** and a believable history.

| Day | Work | Files | Commit message(s) | Proof screenshot |
|---|---|---|---|---|
| **Day 1: Setup** | Folder structure, venv, requirements, .gitignore, README skeleton, config | `requirements.txt`, `.gitignore`, `src/config.py`, `README.md` | `chore: initial project structure and requirements` | Folder tree in VS Code; `pip list` |
| **Day 2: Dataset** | Synthetic company generator + noise injection | `src/data_generator.py`, `data/raw/` | `data: add synthetic HR data generator (2,000 employees)` · `data: inject real-world noise (missing, outliers, dupes)` | CSV preview showing messy cells |
| **Day 3: Cleaning + EDA** | Cleaning, feature engineering, 6 charts, notebook | `src/preprocessing.py`, `src/eda.py`, `notebooks/`, `images/01-06` | `feat: add data cleaning and feature engineering` · `viz: add EDA charts` | Cleaning report; 2–3 best charts |
| **Day 4: Model** | Train/test split, 4-model CV comparison, save best | `src/train.py`, `models/` | `feat: compare 4 classifiers with stratified 5-fold CV` | Model comparison terminal output |
| **Day 5: Results** | Evaluation, feature importance, predictions, insights, dashboard PNG | `src/evaluate.py`, `src/predict.py`, `src/insights.py`, `main.py`, `outputs/`, `images/07-10` | `feat: add evaluation, confusion matrix, feature importance` · `feat: add prediction with HR recommendations` · `feat: generate HR insights report and dashboard` | Classification report; confusion matrix; dashboard |
| **Day 6: Dashboard** | Streamlit app | `app.py` | `feat: add Streamlit HR dashboard` | Each app tab |
| **Day 7: Polish** | Final README, docs, screenshots, deploy | `README.md`, `docs/`, `images/screenshots/` | `docs: complete README with results and screenshots` | Repo home page; LinkedIn post |

> **Tip:** The project is already complete in this folder. To follow the daily plan, add and commit files in the groups shown above, one group per day, instead of committing everything at once.

## F5. Exactly what to screenshot

| Screenshot | What should be visible |
|---|---|
| **Dataset preview** | `df.head()` in the notebook, or the CSV in VS Code, with a few blank/9999 cells |
| **Cleaning report** | `duplicates_removed: 25 … missing_after: 0` |
| **Graphs** | Drivers box plot, correlation heatmap, department chart, overtime chart |
| **Model accuracy** | Terminal comparison of the 4 models + test-set classification report |
| **Confusion matrix** | `images/07_confusion_matrix.png` |
| **Prediction output** | Terminal table of 10 new employees with recommendations |
| **Final result** | `DONE … Test accuracy: 76.0%` banner + HR dashboard PNG + Streamlit predict tab |

## F6. LinkedIn post template

> 🚀 Just finished my project: **Employee Performance Predictor using Data Analytics**
>
> I simulated a 2,000-employee company, cleaned messy HR data, compared 4 ML models, and built a Streamlit dashboard that tells HR *who to promote, who to train, and who needs support*.
>
> 📊 76% accuracy on unseen employees (3-class)
> 🔍 Top drivers: peer feedback, absences, job satisfaction, productivity, training
> 💡 Surprise: simple Logistic Regression beat Random Forest, and it's more explainable for HR
>
> Tech: Python · Pandas · scikit-learn · Seaborn · Streamlit
> 🔗 GitHub: <link>
>
> #DataScience #MachineLearning #HRAnalytics #Python #StudentProject
