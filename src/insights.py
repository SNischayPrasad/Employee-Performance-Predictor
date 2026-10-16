"""
insights.py
-----------
Phase 8: Turn model output into HR decisions.

Scores EVERY current employee and produces:
  * outputs/hr_action_list.csv     -> one row per employee with an action
  * outputs/promotion_candidates.csv
  * outputs/at_risk_employees.csv
  * outputs/department_summary.csv
  * outputs/hr_insights_report.md  -> plain-English summary for managers
  * images/10_hr_dashboard.png     -> one-page visual summary
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.predict import predict_employees

COLORS = {"Low": "#e76f51", "Medium": "#e9c46a", "High": "#2a9d8f"}


def build_insights(df: pd.DataFrame, bundle, feature_importance: pd.DataFrame, metrics: dict):
    scored = predict_employees(df.drop(columns=[config.TARGET, "performance_score"]), bundle)

    keep = ["employee_id", "department", "job_level", "training_hours", "absences",
            "job_satisfaction", "predicted_category", "confidence", "hr_recommendation"]
    scored[keep].to_csv(config.OUTPUTS_DIR / "hr_action_list.csv", index=False)

    promo = (scored[(scored.predicted_category == "High") & (scored.confidence >= 0.7)]
             .sort_values("confidence", ascending=False))
    promo[keep].to_csv(config.OUTPUTS_DIR / "promotion_candidates.csv", index=False)

    at_risk = (scored[scored.predicted_category == "Low"].sort_values("prob_low", ascending=False))
    at_risk[keep].to_csv(config.OUTPUTS_DIR / "at_risk_employees.csv", index=False)

    dept = (scored.groupby("department")
            .agg(employees=("employee_id", "count"),
                 pct_high=("predicted_category", lambda s: round((s == "High").mean() * 100, 1)),
                 pct_low=("predicted_category", lambda s: round((s == "Low").mean() * 100, 1)),
                 avg_training_hours=("training_hours", "mean"),
                 avg_satisfaction=("job_satisfaction", "mean"))
            .round(1).sort_values("pct_low", ascending=False))
    dept.to_csv(config.OUTPUTS_DIR / "department_summary.csv")

    # Data-driven facts for the report
    low_train = scored.training_hours < scored.training_hours.quantile(0.25)
    high_train = scored.training_hours > scored.training_hours.quantile(0.75)
    pct_high_low_train = (scored[low_train].predicted_category == "High").mean() * 100
    pct_high_high_train = (scored[high_train].predicted_category == "High").mean() * 100
    top_feats = feature_importance.head(5)["feature"].tolist()

    report = f"""# HR Insights Report - NovaTech Solutions (simulated)

**Model:** {metrics['model']} | **Test accuracy:** {metrics['test_accuracy']:.1%} | **Test F1-macro:** {metrics['test_f1_macro']:.3f}

## Workforce snapshot ({len(scored):,} employees scored)
| Predicted category | Employees | Share |
|---|---|---|
""" + "\n".join(
        f"| {c} | {(scored.predicted_category == c).sum()} | {(scored.predicted_category == c).mean():.1%} |"
        for c in config.CLASS_ORDER) + f"""

## Key findings
1. **Top 5 performance drivers:** {", ".join(top_feats)}.
2. **Training pays off:** employees in the top 25% of training hours are predicted High
   performers {pct_high_high_train:.0f}% of the time vs {pct_high_low_train:.0f}% for the bottom 25%.
3. **Department with most at-risk staff:** {dept.index[0]} ({dept.iloc[0]['pct_low']}% predicted Low).
4. **Department with most top talent:** {dept.sort_values('pct_high', ascending=False).index[0]}
   ({dept['pct_high'].max()}% predicted High).

## Recommended HR actions
- **{len(promo)} promotion candidates** (High with >=70% confidence) -> `outputs/promotion_candidates.csv`
- **{len(at_risk)} employees need support** (predicted Low) -> `outputs/at_risk_employees.csv`
- Raise training budget for departments with low average training hours.
- Run engagement 1:1s for employees with satisfaction <= 2.
- Review workload for anyone averaging > 220 hours/month (burnout lowers performance).

> Predictions support, not replace, manager judgement. Review every flagged case with a human.
"""
    (config.OUTPUTS_DIR / "hr_insights_report.md").write_text(report, encoding="utf-8")
    print("[insights] saved outputs/hr_insights_report.md (+ action list, candidates, at-risk, dept summary)")

    plot_dashboard(scored, dept, feature_importance, metrics, len(promo), len(at_risk))
    return scored


def plot_dashboard(scored, dept, fi, metrics, n_promo, n_risk):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("HR Performance Dashboard - NovaTech Solutions (simulated data)", fontsize=15, weight="bold")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4])

    # KPI tiles
    kpis = [("Employees scored", f"{len(scored):,}"),
            ("Model accuracy", f"{metrics['test_accuracy']:.1%}"),
            ("Promotion candidates", str(n_promo)),
            ("Need support", str(n_risk))]
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    for i, (label, val) in enumerate(kpis):
        y = 0.88 - i * 0.25
        ax.text(0.02, y, val, fontsize=20, weight="bold", color="#264653")
        ax.text(0.02, y - 0.09, label, fontsize=10, color="#555")

    # Predicted category pie
    ax = fig.add_subplot(gs[0, 1])
    counts = scored.predicted_category.value_counts().reindex(config.CLASS_ORDER)
    ax.pie(counts, labels=counts.index, autopct="%1.0f%%", colors=[COLORS[c] for c in counts.index],
           startangle=90, wedgeprops={"width": 0.45})
    ax.set_title("Predicted performance mix")

    # Feature importance
    ax = fig.add_subplot(gs[0, 2])
    top = fi.head(6).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#e76f51")
    ax.set_title("Top performance drivers")
    ax.tick_params(labelsize=8)

    # Department stacked bar
    ax = fig.add_subplot(gs[1, :2])
    ct = pd.crosstab(scored.department, scored.predicted_category, normalize="index")[config.CLASS_ORDER] * 100
    ct.sort_values("High").plot(kind="barh", stacked=True, color=[COLORS[c] for c in config.CLASS_ORDER], ax=ax)
    ax.set_title("Predicted performance by department (%)")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=8)

    # Training vs prob_high
    ax = fig.add_subplot(gs[1, 2])
    bins = pd.cut(scored.training_hours, [-1, 15, 30, 45, 60, 1000], labels=["<15", "15-30", "30-45", "45-60", "60+"])
    (scored.groupby(bins, observed=True).prob_high.mean() * 100).plot(kind="bar", color="#2a9d8f", ax=ax)
    ax.set_title("Avg. chance of High performance\nby training hours")
    ax.set_ylabel("%")
    ax.set_xlabel("Training hours / year")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / "10_hr_dashboard.png", dpi=120)
    plt.close(fig)
    print("[insights] saved images/10_hr_dashboard.png")
