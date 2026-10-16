"""
config.py
---------
Central place for every path and setting used in the project.
Change a value here and the whole pipeline picks it up.
"""
from pathlib import Path

# ---- Folder paths ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "employee_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "employee_data_clean.csv"
NEW_EMPLOYEES_PATH = DATA_DIR / "raw" / "new_employees.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGES_DIR = ROOT_DIR / "images"

# ---- Simulation settings --------------------------------------------------
N_EMPLOYEES = 2000          # size of the virtual company
RANDOM_STATE = 42           # makes every run reproducible
TEST_SIZE = 0.2             # 80% train / 20% test

# ---- Target column --------------------------------------------------------
TARGET = "performance_category"           # Low / Medium / High
CLASS_ORDER = ["Low", "Medium", "High"]

# ---- Feature groups -------------------------------------------------------
# NOTE: employee_id and the raw performance_score are NOT features.
# performance_score is what the target is built from, so using it would be
# "data leakage" (the model would be cheating).
NUMERIC_FEATURES = [
    "age",
    "years_experience",
    "years_at_company",
    "monthly_salary",
    "training_hours",
    "projects_completed",
    "avg_monthly_hours",
    "absences",
    "last_rating",
    "peer_feedback_score",
    "years_since_promotion",
    "job_satisfaction",
    "work_life_balance",
    # engineered features (created in preprocessing.py)
    "salary_per_level",
    "projects_per_year",
    "training_per_year",
    "overtime_hours",
]

CATEGORICAL_FEATURES = [
    "department",
    "education",
    "job_level",
    "work_mode",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for folder in [
        RAW_DATA_PATH.parent,
        PROCESSED_DATA_PATH.parent,
        MODELS_DIR,
        OUTPUTS_DIR,
        IMAGES_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
