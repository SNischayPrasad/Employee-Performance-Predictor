"""
app.py
------
Interactive HR dashboard (optional, Phase 9).

    pip install streamlit
    python main.py          # train the model first
    streamlit run app.py    # opens http://localhost:8501

Tabs:
  1. Overview      - KPIs + charts from the analysis
  2. Predict       - enter one employee's details, get a prediction
  3. Batch upload  - upload a CSV of employees, download predictions
  4. Action lists  - promotion candidates & at-risk employees
"""
import json

import pandas as pd
import streamlit as st

from src import config
from src.predict import load_model, predict_employees

st.set_page_config(page_title="Employee Performance Predictor", page_icon="📊", layout="wide")
st.title("📊 Employee Performance Predictor")
st.caption("HR analytics demo on simulated data — predictions support, not replace, manager judgement.")


@st.cache_resource
def get_bundle():
    return load_model()


try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error("No trained model found. Run `python main.py` first, then refresh this page.")
    st.stop()

metrics = json.loads((config.OUTPUTS_DIR / "metrics.json").read_text())
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict one employee", "Batch upload", "Action lists"])

# ---------------------------------------------------------------- Overview
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", metrics["model"])
    c2.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
    c3.metric("Test F1-macro", f"{metrics['test_f1_macro']:.3f}")
    st.image(str(config.IMAGES_DIR / "10_hr_dashboard.png"), width="stretch")
    col_a, col_b = st.columns(2)
    col_a.image(str(config.IMAGES_DIR / "07_confusion_matrix.png"))
    col_b.image(str(config.IMAGES_DIR / "09_feature_importance.png"))

# ---------------------------------------------------------------- Single prediction
with tab2:
    st.subheader("Enter employee details")
    a, b, c = st.columns(3)
    with a:
        department = st.selectbox("Department", ["Sales", "Engineering", "HR", "Finance", "Marketing", "Operations", "Support"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"], index=1)
        job_level = st.selectbox("Job level", ["Junior", "Mid", "Senior", "Lead"], index=1)
        work_mode = st.selectbox("Work mode", ["Office", "Hybrid", "Remote"])
        age = st.slider("Age", 21, 60, 30)
    with b:
        years_experience = st.slider("Total experience (yrs)", 0.0, 35.0, 6.0, 0.5)
        years_at_company = st.slider("Years at company", 0.0, 30.0, 3.0, 0.5)
        monthly_salary = st.number_input("Monthly salary (INR)", 10000, 400000, 60000, 1000)
        training_hours = st.slider("Training hours / year", 0, 160, 35)
        projects_completed = st.slider("Projects completed", 0, 40, 5)
    with c:
        avg_monthly_hours = st.slider("Avg monthly hours", 120, 280, 176)
        absences = st.slider("Absences (days / year)", 0, 20, 4)
        last_rating = st.slider("Last year's rating (1-5)", 1, 5, 3)
        peer_feedback_score = st.slider("Peer feedback (1-5)", 1.0, 5.0, 3.5, 0.1)
        years_since_promotion = st.slider("Years since promotion", 0.0, 15.0, 1.5, 0.5)
        job_satisfaction = st.slider("Job satisfaction (1-5)", 1, 5, 3)
        work_life_balance = st.slider("Work-life balance (1-5)", 1, 5, 3)

    if st.button("Predict performance", type="primary"):
        row = pd.DataFrame([{
            "employee_id": "MANUAL", "age": age, "department": department, "education": education,
            "job_level": job_level, "work_mode": work_mode, "years_experience": years_experience,
            "years_at_company": min(years_at_company, years_experience),
            "monthly_salary": monthly_salary, "training_hours": training_hours,
            "projects_completed": projects_completed, "avg_monthly_hours": avg_monthly_hours,
            "absences": absences, "last_rating": last_rating, "peer_feedback_score": peer_feedback_score,
            "years_since_promotion": years_since_promotion, "job_satisfaction": job_satisfaction,
            "work_life_balance": work_life_balance,
        }])
        res = predict_employees(row, bundle).iloc[0]
        icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[res["predicted_category"]]
        st.markdown(f"### {icon} Predicted: **{res['predicted_category']}** performer "
                    f"({res['confidence']:.0%} confidence)")
        for cls in ["Low", "Medium", "High"]:
            p = float(res[f"prob_{cls.lower()}"])
            st.progress(p, text=f"{cls}: {p:.0%}")
        st.info(f"**HR recommendation:** {res['hr_recommendation']}")

# ---------------------------------------------------------------- Batch upload
with tab3:
    st.write("Upload a CSV with the same columns as `data/raw/new_employees.csv`.")
    up = st.file_uploader("Employee CSV", type="csv")
    if up is not None:
        result = predict_employees(pd.read_csv(up), bundle)
        show = ["employee_id", "department", "job_level", "predicted_category", "confidence", "hr_recommendation"]
        st.dataframe(result[show], width="stretch")
        st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------------------------------------------- Action lists
with tab4:
    left, right = st.columns(2)
    left.subheader("🏆 Promotion candidates")
    left.dataframe(pd.read_csv(config.OUTPUTS_DIR / "promotion_candidates.csv").head(50), width="stretch")
    right.subheader("⚠️ Employees needing support")
    right.dataframe(pd.read_csv(config.OUTPUTS_DIR / "at_risk_employees.csv").head(50), width="stretch")
    st.subheader("Department summary")
    st.dataframe(pd.read_csv(config.OUTPUTS_DIR / "department_summary.csv"), width="stretch")
