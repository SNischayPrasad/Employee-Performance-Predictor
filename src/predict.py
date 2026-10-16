"""
predict.py
----------
Use the saved model to predict performance for NEW employees and turn
each prediction into an HR recommendation.

Run on its own:
    python -m src.predict                       # uses data/raw/new_employees.csv
    python -m src.predict --input my_file.csv   # your own file
"""
import argparse

import joblib
import pandas as pd

from src import config
from src.preprocessing import clean_data, engineer_features


def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python main.py` first.")
    return joblib.load(config.MODEL_PATH)


def recommend(row: pd.Series) -> str:
    """Rule-based HR action built on top of the ML prediction."""
    pred, conf = row["predicted_category"], row["confidence"]
    actions = []
    if pred == "High":
        actions.append("Promotion / leadership-track candidate" if conf >= 0.7
                       else "Strong performer - recognise & retain")
    elif pred == "Medium":
        actions.append("Targeted upskilling to move into High band")
    else:
        actions.append("Performance Improvement Plan + mentor")

    # Personalised reasons, based on the features the model found important
    if row["training_hours"] < 20:
        actions.append("increase training hours")
    if row["absences"] >= 8:
        actions.append("check attendance / wellbeing")
    if row["job_satisfaction"] <= 2:
        actions.append("1:1 engagement conversation")
    if row["avg_monthly_hours"] > 220:
        actions.append("burnout risk - review workload")
    return "; ".join(actions)


def predict_employees(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """Clean + engineer features, then predict category and probabilities."""
    bundle = bundle or load_model()
    pipe = bundle["pipeline"]
    data = engineer_features(clean_data(df, verbose=False))

    proba = pipe.predict_proba(data[config.ALL_FEATURES])
    classes = list(pipe.classes_)
    out = data.copy()
    out["predicted_category"] = pipe.predict(data[config.ALL_FEATURES])
    out["confidence"] = proba.max(axis=1).round(3)
    for i, c in enumerate(classes):
        out[f"prob_{c.lower()}"] = proba[:, i].round(3)
    out["hr_recommendation"] = out.apply(recommend, axis=1)
    return out


def main(input_path=None) -> pd.DataFrame:
    input_path = input_path or config.NEW_EMPLOYEES_PATH
    new = pd.read_csv(input_path)
    result = predict_employees(new)
    out_path = config.OUTPUTS_DIR / "new_employee_predictions.csv"
    result.to_csv(out_path, index=False)

    show = ["employee_id", "department", "job_level", "predicted_category", "confidence", "hr_recommendation"]
    print("\n[predict] Predictions for new employees:")
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(result[show].to_string(index=False))
    print(f"\n[predict] Saved -> {out_path.relative_to(config.ROOT_DIR)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict employee performance")
    parser.add_argument("--input", help="CSV with the same columns as data/raw/new_employees.csv")
    args = parser.parse_args()
    main(args.input)
