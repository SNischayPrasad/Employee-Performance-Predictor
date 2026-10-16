"""
preprocessing.py
----------------
Phase 3 (Data cleaning) + Phase 5 (Feature engineering).

clean_data()      -> fixes duplicates, text, outliers, missing values
engineer_features -> creates new, more meaningful columns
build_preprocessor-> scikit-learn transformer that scales numbers and
                     one-hot encodes categories (fitted ONLY on training data)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return a cleaned copy of the raw HR data."""
    df = df.copy()
    report = {"rows_before": len(df)}

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()
    report["duplicates_removed"] = report["rows_before"] - len(df)

    # 2. Fix inconsistent text: " sales " / "SALES" -> "Sales"
    df["department"] = df["department"].str.strip().str.title()
    df["department"] = df["department"].replace({"Hr": "HR"})

    # 3. Fix impossible values (outliers caused by typing mistakes)
    #    We set them to NaN, then they get filled like other missing values.
    bad_age = ~df["age"].between(18, 70)
    df.loc[bad_age, "age"] = np.nan
    bad_training = df["training_hours"] > 500
    df.loc[bad_training, "training_hours"] = np.nan
    report["invalid_age_fixed"] = int(bad_age.sum())
    report["invalid_training_fixed"] = int(bad_training.sum())

    # 4. Fill missing values
    #    numbers -> median (robust to outliers), text -> most common value
    report["missing_before"] = int(df.isna().sum().sum())
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    report["missing_after"] = int(df.isna().sum().sum())

    df["age"] = df["age"].astype(int)
    report["rows_after"] = len(df)

    if verbose:
        print("[clean] Cleaning report:")
        for k, v in report.items():
            print(f"        {k:<24}: {v}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features that capture business meaning."""
    df = df.copy()
    level_rank = df["job_level"].map({"Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4})

    # Salary relative to job level -> is this person paid well for their level?
    df["salary_per_level"] = (df["monthly_salary"] / level_rank).round(0)
    # Productivity: projects per year at the company
    df["projects_per_year"] = (df["projects_completed"] / (df["years_at_company"] + 1)).round(2)
    # Learning intensity
    df["training_per_year"] = (df["training_hours"] / (df["years_at_company"] + 1)).round(2)
    # Overtime above a standard 176-hour month (22 days x 8 h)
    df["overtime_hours"] = (df["avg_monthly_hours"] - 176).clip(lower=0)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Scaling + encoding, packaged so it can live inside a model Pipeline."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # safety net for new data
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, config.NUMERIC_FEATURES),
        ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
    ])


def main() -> pd.DataFrame:
    config.make_dirs()
    raw = pd.read_csv(config.RAW_DATA_PATH)
    df = engineer_features(clean_data(raw))
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"[clean] Saved cleaned data ({len(df):,} rows) -> {config.PROCESSED_DATA_PATH}")
    return df


if __name__ == "__main__":
    main()
