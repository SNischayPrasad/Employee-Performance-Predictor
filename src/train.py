"""
train.py
--------
Phase 6: Model building.

1. Split data into train (80%) and test (20%) -- stratified so each split
   keeps the same Low/Medium/High ratio.
2. Compare 4 classification models with 5-fold cross-validation.
3. Pick the best one (by macro F1 score), retrain it on the full training
   set and save it to models/best_model.joblib.
"""
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.preprocessing import build_preprocessor


def get_models() -> dict:
    """The candidate models, from simplest to most powerful."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                class_weight="balanced", random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=3, random_state=rs),
    }


def split_data(df: pd.DataFrame):
    X = df[config.ALL_FEATURES]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def compare_models(X_train, y_train) -> pd.DataFrame:
    """5-fold cross-validation for every model. Returns a results table."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["accuracy", "f1_macro"], n_jobs=-1)
        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1_macro": scores["test_f1_macro"].mean(),
            "cv_f1_std": scores["test_f1_macro"].std(),
        })
        print(f"[train] {name:<20} accuracy={rows[-1]['cv_accuracy']:.3f}  "
              f"F1-macro={rows[-1]['cv_f1_macro']:.3f} (+/- {rows[-1]['cv_f1_std']:.3f})")
    return pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)


def train_best(results: pd.DataFrame, X_train, y_train) -> Pipeline:
    best_name = results.loc[0, "model"]
    print(f"[train] Best model: {best_name}")
    pipe = Pipeline([("prep", build_preprocessor()), ("model", get_models()[best_name])])
    pipe.fit(X_train, y_train)
    joblib.dump({"pipeline": pipe, "model_name": best_name, "features": config.ALL_FEATURES},
                config.MODEL_PATH)
    print(f"[train] Saved model -> {config.MODEL_PATH.relative_to(config.ROOT_DIR)}")
    return pipe


def main(df: pd.DataFrame | None = None):
    config.make_dirs()
    if df is None:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"[train] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    results = compare_models(X_train, y_train)
    results.round(4).to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)
    pipe = train_best(results, X_train, y_train)
    return pipe, results, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
