"""
data_generator.py
-----------------
VIRTUAL COMPANY SIMULATION

We do not have real HR data, so we simulate a company called "NovaTech
Solutions" with 2,000 employees. Each employee gets realistic attributes,
and a hidden "true" performance score is calculated from those attributes
plus random noise (because real performance is never 100% predictable).

We also deliberately inject real-world data problems:
  * missing values   (HR forms left blank)
  * duplicate rows   (same record entered twice)
  * outliers         (typing mistakes, e.g. 9999 training hours)
  * inconsistent text ("sales", " Sales ", "SALES")
so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from src import config

DEPARTMENTS = ["Sales", "Engineering", "HR", "Finance", "Marketing", "Operations", "Support"]
DEPT_WEIGHTS = [0.18, 0.28, 0.07, 0.10, 0.10, 0.15, 0.12]
EDUCATION = ["High School", "Bachelor", "Master", "PhD"]
EDU_WEIGHTS = [0.10, 0.55, 0.30, 0.05]
JOB_LEVELS = ["Junior", "Mid", "Senior", "Lead"]
WORK_MODES = ["Office", "Hybrid", "Remote"]

# Base monthly salary (INR) per job level
BASE_SALARY = {"Junior": 35000, "Mid": 60000, "Senior": 95000, "Lead": 140000}
DEPT_SALARY_FACTOR = {
    "Sales": 0.95, "Engineering": 1.15, "HR": 0.85, "Finance": 1.05,
    "Marketing": 0.95, "Operations": 0.90, "Support": 0.80,
}


def _job_level_from_experience(years: np.ndarray, rng) -> np.ndarray:
    """More experience -> more likely to be at a higher level."""
    levels = []
    for y in years:
        if y < 3:
            p = [0.85, 0.15, 0.0, 0.0]
        elif y < 7:
            p = [0.25, 0.60, 0.15, 0.0]
        elif y < 12:
            p = [0.05, 0.35, 0.50, 0.10]
        else:
            p = [0.0, 0.15, 0.50, 0.35]
        levels.append(rng.choice(JOB_LEVELS, p=p))
    return np.array(levels)


def generate_employee_data(n: int = config.N_EMPLOYEES, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic but realistic HR dataset."""
    rng = np.random.default_rng(seed)

    # ---- 1. Basic profile -------------------------------------------------
    age = rng.integers(21, 60, n)
    # experience can't exceed (age - 21)
    years_experience = np.clip(rng.normal((age - 21) * 0.8, 2.5), 0, age - 21).round(1)
    years_at_company = np.clip(rng.uniform(0, 1, n) * years_experience, 0, None).round(1)
    department = rng.choice(DEPARTMENTS, n, p=DEPT_WEIGHTS)
    education = rng.choice(EDUCATION, n, p=EDU_WEIGHTS)
    job_level = _job_level_from_experience(years_experience, rng)
    work_mode = rng.choice(WORK_MODES, n, p=[0.40, 0.40, 0.20])

    # ---- 2. Salary --------------------------------------------------------
    base = np.array([BASE_SALARY[l] for l in job_level])
    dept_factor = np.array([DEPT_SALARY_FACTOR[d] for d in department])
    edu_bonus = pd.Series(education).map({"High School": 0.9, "Bachelor": 1.0, "Master": 1.1, "PhD": 1.2}).values
    monthly_salary = (base * dept_factor * edu_bonus * rng.normal(1, 0.10, n)).round(-2)

    # ---- 3. Work behaviour ------------------------------------------------
    training_hours = np.clip(rng.gamma(3, 12, n), 0, 160).round(0)
    avg_monthly_hours = np.clip(rng.normal(175, 22, n), 120, 280).round(0)
    absences = rng.poisson(4, n)
    projects_completed = np.clip(rng.poisson(2 + years_at_company * 0.9), 0, 40)
    years_since_promotion = np.clip(rng.exponential(2.2, n), 0, years_at_company).round(1)
    job_satisfaction = rng.integers(1, 6, n)            # 1-5 survey score
    work_life_balance = rng.integers(1, 6, n)           # 1-5 survey score
    peer_feedback_score = np.clip(rng.normal(3.5, 0.7, n), 1, 5).round(1)

    # ---- 4. Hidden "true" performance score (0-100) ------------------------
    # This formula is the "ground truth" of our virtual company.
    # The ML model never sees this formula -- it has to LEARN it from data.
    z = lambda x: (x - x.mean()) / x.std()          # standardise helper
    edu_score = pd.Series(education).map({"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 2.5}).values

    latent = (
        0.30 * z(training_hours)
        + 0.30 * z(np.minimum(projects_completed / (years_at_company + 1), 4))
        + 0.35 * z(peer_feedback_score)
        + 0.30 * z(job_satisfaction)
        + 0.15 * z(work_life_balance)
        - 0.30 * z(absences)
        + 0.10 * z(edu_score)
        + 0.10 * z(np.minimum(years_experience, 15))
        - 0.25 * np.maximum(avg_monthly_hours - 220, 0) / 20   # burnout penalty
        - 0.10 * z(years_since_promotion)
    )
    noise = rng.normal(0, 0.30, n)                    # the unpredictable part
    performance_score = np.clip(60 + 12 * (latent + noise), 0, 100).round(1)

    # Last year's rating (1-5) is correlated with this year's performance,
    # but not identical -- people improve or decline.
    last_rating = np.clip(np.round(3 + (latent + rng.normal(0, 0.9, n)) * 0.9), 1, 5).astype(int)

    # ---- 5. Turn the score into a category (the TARGET) --------------------
    # Bottom ~25% = Low, middle ~50% = Medium, top ~25% = High
    low_cut, high_cut = np.percentile(performance_score, [25, 75])
    performance_category = np.where(
        performance_score < low_cut, "Low",
        np.where(performance_score >= high_cut, "High", "Medium"),
    )

    df = pd.DataFrame({
        "employee_id": [f"EMP{str(i).zfill(5)}" for i in range(1, n + 1)],
        "age": age,
        "department": department,
        "education": education,
        "job_level": job_level,
        "work_mode": work_mode,
        "years_experience": years_experience,
        "years_at_company": years_at_company,
        "monthly_salary": monthly_salary,
        "training_hours": training_hours,
        "projects_completed": projects_completed,
        "avg_monthly_hours": avg_monthly_hours,
        "absences": absences,
        "last_rating": last_rating,
        "peer_feedback_score": peer_feedback_score,
        "years_since_promotion": years_since_promotion,
        "job_satisfaction": job_satisfaction,
        "work_life_balance": work_life_balance,
        "performance_score": performance_score,
        "performance_category": performance_category,
    })
    return df


def add_real_world_noise(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Make the clean data messy, like real HR exports."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    # 1) Missing values (~3-5% in some columns)
    for col, frac in [("training_hours", 0.04), ("peer_feedback_score", 0.05),
                      ("monthly_salary", 0.03), ("education", 0.02), ("job_satisfaction", 0.03)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2) Outliers / typing mistakes
    idx = rng.choice(n, 6, replace=False)
    df.loc[idx, "training_hours"] = 9999
    idx = rng.choice(n, 4, replace=False)
    df.loc[idx, "age"] = rng.choice([150, 2, 999], 4)

    # 3) Inconsistent text formatting
    idx = rng.choice(n, 40, replace=False)
    df.loc[idx, "department"] = df.loc[idx, "department"].map(
        lambda d: rng.choice([d.lower(), d.upper(), f" {d} "])
    )

    # 4) Duplicate rows
    dupes = df.sample(25, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_new_employees(n: int = 10, seed: int = 7) -> pd.DataFrame:
    """A small batch of 'new' employees (no labels) for the prediction demo."""
    df = generate_employee_data(n=200, seed=seed).sample(n, random_state=seed)
    df["employee_id"] = [f"NEW{str(i).zfill(3)}" for i in range(1, n + 1)]
    return df.drop(columns=["performance_score", "performance_category"]).reset_index(drop=True)


def main() -> pd.DataFrame:
    config.make_dirs()
    df = add_real_world_noise(generate_employee_data())
    df.to_csv(config.RAW_DATA_PATH, index=False)
    generate_new_employees().to_csv(config.NEW_EMPLOYEES_PATH, index=False)
    print(f"[data] Generated {len(df):,} rows x {df.shape[1]} columns -> {config.RAW_DATA_PATH}")
    print(f"[data] New-employee batch for prediction  -> {config.NEW_EMPLOYEES_PATH}")
    return df


if __name__ == "__main__":
    main()
