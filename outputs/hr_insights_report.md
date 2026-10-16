# HR Insights Report - NovaTech Solutions (simulated)

**Model:** Logistic Regression | **Test accuracy:** 76.0% | **Test F1-macro:** 0.763

## Workforce snapshot (2,000 employees scored)
| Predicted category | Employees | Share |
|---|---|---|
| Low | 590 | 29.5% |
| Medium | 830 | 41.5% |
| High | 580 | 29.0% |

## Key findings
1. **Top 5 performance drivers:** peer_feedback_score, absences, job_satisfaction, projects_per_year, training_hours.
2. **Training pays off:** employees in the top 25% of training hours are predicted High
   performers 49% of the time vs 14% for the bottom 25%.
3. **Department with most at-risk staff:** Finance (34.9% predicted Low).
4. **Department with most top talent:** Marketing
   (37.6% predicted High).

## Recommended HR actions
- **425 promotion candidates** (High with >=70% confidence) -> `outputs/promotion_candidates.csv`
- **590 employees need support** (predicted Low) -> `outputs/at_risk_employees.csv`
- Raise training budget for departments with low average training hours.
- Run engagement 1:1s for employees with satisfaction <= 2.
- Review workload for anyone averaging > 220 hours/month (burnout lowers performance).

> Predictions support, not replace, manager judgement. Review every flagged case with a human.
