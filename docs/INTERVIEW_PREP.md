# 🎤 Interview Preparation: Employee Performance Predictor

## G1. 30-second elevator pitch

> "I built an end-to-end HR analytics project that predicts whether an employee will be a low, medium or high performer. Since real HR data is confidential, I simulated a 2,000-employee company with realistic messy data. I cleaned it, engineered features like projects per year, compared four classification models with cross-validation, and got 76% accuracy on unseen data with Logistic Regression. The key part is that I turned predictions into HR actions: promotion candidates, support plans and burnout flags, shown in a Streamlit dashboard."

---

## G2. How to explain it to an HR / non-technical interviewer

Use the **Problem → Solution → Impact** story, with no jargon:

1. **Problem:** "Companies usually find out someone is struggling only at the annual review, when it's too late to help. Promotions are also often based on gut feeling."
2. **Solution:** "I built a tool that looks at things like training, attendance, peer feedback and job satisfaction, and gives an early signal of who is likely to be a top performer and who might need support."
3. **What HR gets:** "A list of promotion-ready people, a list of people who need mentoring or training, and a department view showing where to spend the training budget."
4. **Finding:** "Employees with the most training were more than three times as likely to be high performers. Working more than about 220 hours a month actually *lowered* performance, which is a burnout signal."
5. **Responsibility:** "The tool supports managers; it doesn't replace them. I deliberately excluded things like gender, and every flagged case should be reviewed by a person."

## G3. How to explain it to a technical interviewer

Walk through the pipeline and **name your decisions and trade-offs**:

1. **Data:** "A synthetic generator with correlated features, such as level depending on experience and salary on level × department. The label comes from a hidden linear formula plus Gaussian noise, so the ceiling is below 100%. I injected 350 missing cells, outliers and 25 duplicates."
2. **Cleaning:** "Invalid values become NaN and are then imputed with the median for numeric columns (robust to skew) and the mode for categoricals."
3. **Leakage control:** "`performance_score` is excluded because the label is derived from it. Scaling and encoding live inside an sklearn `Pipeline` with a `ColumnTransformer`, so they're fitted only on training folds."
4. **Modelling:** "Stratified 80/20 split, then 5-fold stratified CV across Logistic Regression, Decision Tree, Random Forest and Gradient Boosting. I used `class_weight='balanced'` because the classes are 25/50/25. The selection metric was macro-F1, so Low and High count as much as Medium."
5. **Result:** "Logistic Regression won, with 0.763 CV F1 and 76% test accuracy. That makes sense because the true signal is additive, and trees overfit the noise. Errors are mostly between adjacent classes."
6. **Interpretability:** "Permutation importance recovered the true top drivers, which validates that the model learned signal."
7. **Productionising:** "The model is saved with joblib. `predict.py` reuses the same cleaning code, adds a rule layer for HR actions, and is exposed through a CLI and Streamlit."

---

## G4. Ten interview questions with strong answers

### 1. Why did you use synthetic data? Isn't that a weakness?
"Real HR data is highly confidential, so students can't get it. I turned that into an advantage: because I designed the data-generating process, I could **validate** the model. The feature importance matched the true drivers. I also added realistic noise and data-quality problems so the cleaning wasn't trivial. The pipeline is data-agnostic, so swapping in the IBM HR Kaggle dataset only means changing the feature lists in `config.py`."

### 2. Why is 76% accuracy good enough?
"Three points. First, random guessing on a 3-class problem gives ~33%, and always predicting 'Medium' gives 50%, so 76% is a large lift. Second, I built irreducible noise into the data, just like real performance has luck and personal factors, so 100% isn't possible. Third, most errors are between *neighbouring* classes, like Medium vs High. The model almost never confuses Low with High, and that would be the costly mistake. For an early-warning tool reviewed by humans, that's useful."

### 3. Why did Logistic Regression beat Random Forest?
"The underlying relationship is mostly linear and additive. Each factor pushes performance up or down independently. Logistic Regression's assumptions match that, so it has low bias *and* low variance. Random Forest has to approximate smooth linear effects with step functions and fits some of the noise. It's a good reminder that more complex isn't automatically better, and HR also prefers a model whose coefficients they can explain."

### 4. What is data leakage and how did you avoid it?
"Leakage is when information that wouldn't be available at prediction time gets into training, which gives fake-high scores. I avoided two kinds. **Target leakage:** I excluded `performance_score`, because the label is computed from it. **Train/test contamination:** scaling and encoding sit inside a Pipeline, so in cross-validation they're fitted only on the training folds."

### 5. Why F1-macro instead of accuracy?
"The classes are imbalanced (25% Low, 50% Medium, 25% High). Accuracy can look good while the model ignores the minority classes, and those are exactly the ones HR cares about: Low means intervene, High means promote. Macro-F1 averages F1 across classes equally, so each class counts the same."

### 6. Explain precision vs recall in this business context.
"For the **Low** class, recall means: of all employees who really will underperform, how many did we catch? Our recall is 82%. Precision means: of everyone we flagged as Low, how many really were? That's 73%. For an early-warning system I'd favour **recall**, because missing someone who needed support is worse than offering mentoring to someone who turns out fine. For **promotions**, I'd favour precision, which is why I only list High predictions with at least 70% confidence."

### 7. How did you handle missing values and outliers?
"First I found impossible values, like age 999 or 9,999 training hours, using domain rules, and set them to missing. Then I imputed numeric columns with the **median**, because it's robust to skew, and categoricals with the **mode**. Only about 1% of cells were missing, so imputing was better than dropping rows. There's also an imputer inside the Pipeline as a safety net for new data at prediction time."

### 8. What features did you engineer, and why?
"Four ratio features. `projects_per_year`, because raw project count is biased toward long-tenured people. `training_per_year`, for the same reason. `salary_per_level`, which shows whether someone is paid well for their level. And `overtime_hours`, meaning hours above a standard month, to capture burnout. `projects_per_year` became a top-5 driver, ahead of raw `projects_completed`."

### 9. How would you deploy this in a real company?
"Short term: the Streamlit app on an internal server, with HR uploading a quarterly CSV. Production: wrap `predict_employees()` in a FastAPI endpoint, containerise it with Docker, connect it to the HRIS database, track experiments and model versions with MLflow, schedule retraining after each appraisal cycle, and monitor for **data drift**, for example if a new policy changes the training hours distribution. Before go-live I'd also run a fairness audit and agree with HR and Legal how predictions may and may not be used."

### 10. What are the ethical risks, and how did you address them?
"The main risks are bias, self-fulfilling prophecies and over-reliance. I **excluded protected attributes** like gender and age-based categories as model inputs. Even so, proxies can leak bias, so in production I'd audit prediction rates across groups. The output is framed as **decision support**: every flag goes to a human, and Low predictions trigger *support* (mentoring, training), not punishment. I'd also be transparent with employees about what data is used."

---

## G5. Bonus rapid-fire questions

| Question | Short answer |
|---|---|
| What is cross-validation? | Splitting the training data into k folds and training k times, each time validating on a different fold. It gives a more reliable score than one split. |
| Why stratify the split? | So train and test keep the same 25/50/25 class ratio. |
| What does StandardScaler do? | Rescales each feature to mean 0 and std 1, so salary (in lakhs) doesn't dominate satisfaction (1–5) in linear models. |
| Why one-hot encoding? | Models need numbers. One-hot avoids implying a fake order (e.g. Sales < HR). |
| What is permutation importance? | Shuffle one feature and measure how much the score drops. A bigger drop means a more important feature. It works with any model. |
| What is `class_weight='balanced'`? | Gives errors on rarer classes a bigger penalty, so the model doesn't ignore them. |
| Overfitting: how would you detect it? | Train score much higher than CV/test score. The Decision Tree showed this. |
| What would you do with more time? | Real dataset, SHAP explanations, hyper-parameter tuning, attrition model, deployment. |
| Biggest challenge? | Making the synthetic data realistic: not too clean (100% accuracy looks fake) and not pure noise. I tuned the noise level until the results were believable. |
| Why Streamlit? | It's the fastest way to turn Python into an interactive app that non-technical HR users can use. |

## G6. Resume bullet points

- Built an end-to-end **HR analytics ML pipeline** in Python (Pandas, scikit-learn) predicting employee performance (Low/Medium/High) for a simulated 2,000-employee company, reaching **76% accuracy / 0.76 macro-F1** on unseen data.
- Cleaned messy HR data (350 missing values, outliers, duplicates) and engineered productivity features. Compared **4 classifiers with stratified 5-fold CV** inside leakage-safe sklearn Pipelines.
- Translated predictions into **HR actions** (promotion candidates, support plans, burnout flags) and delivered an interactive **Streamlit dashboard**. Identified that high-training employees were **3.5× more likely** to be top performers.
