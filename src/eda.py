"""
eda.py
------
Phase 4: Exploratory Data Analysis.
Answers business questions with charts and saves them to images/.
"""
import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = {"Low": "#e76f51", "Medium": "#e9c46a", "High": "#2a9d8f"}


def _save(fig, name: str) -> None:
    path = config.IMAGES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[eda]  saved {path.relative_to(config.ROOT_DIR)}")


def summary(df: pd.DataFrame) -> None:
    print("\n[eda] Dataset shape:", df.shape)
    print("[eda] Target distribution (%):")
    print((df[config.TARGET].value_counts(normalize=True) * 100).round(1).to_string())
    print("\n[eda] Average performance score by department:")
    print(df.groupby("department")["performance_score"].mean().sort_values(ascending=False).round(1).to_string())


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=config.TARGET, order=config.CLASS_ORDER, hue=config.TARGET,
                  palette=PALETTE, legend=False, ax=ax)
    ax.set_title("How many employees are Low / Medium / High performers?")
    ax.set_xlabel("Performance category")
    ax.set_ylabel("Employees")
    for c in ax.containers:
        ax.bar_label(c)
    _save(fig, "01_target_distribution.png")


def plot_department_performance(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct = pd.crosstab(df["department"], df[config.TARGET], normalize="index")[config.CLASS_ORDER] * 100
    ct.sort_values("High").plot(kind="barh", stacked=True, color=[PALETTE[c] for c in config.CLASS_ORDER], ax=ax)
    ax.set_title("Performance mix by department (%)")
    ax.set_xlabel("% of employees")
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, "02_department_performance.png")


def plot_correlation(df):
    cols = ["age", "years_experience", "monthly_salary", "training_hours", "projects_completed",
            "avg_monthly_hours", "absences", "last_rating", "peer_feedback_score",
            "years_since_promotion", "job_satisfaction", "work_life_balance", "performance_score"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation between employee attributes")
    _save(fig, "03_correlation_heatmap.png")


def plot_drivers(df):
    """Box plots of the strongest drivers vs. performance category."""
    drivers = ["training_hours", "peer_feedback_score", "absences", "job_satisfaction"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, col in zip(axes, drivers):
        sns.boxplot(data=df, x=config.TARGET, y=col, order=config.CLASS_ORDER, hue=config.TARGET,
                    palette=PALETTE, legend=False, ax=ax)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
    fig.suptitle("Key drivers: what separates high and low performers?", y=1.03)
    _save(fig, "04_performance_drivers.png")


def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["performance_score"], bins=40, kde=True, color="#264653", ax=ax)
    ax.set_title("Distribution of performance score (0-100)")
    _save(fig, "05_score_distribution.png")


def plot_overtime_effect(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.cut(df["avg_monthly_hours"], [0, 160, 180, 200, 220, 300],
                  labels=["<160", "160-180", "180-200", "200-220", "220+"])
    df.groupby(bins, observed=True)["performance_score"].mean().plot(kind="bar", color="#457b9d", ax=ax)
    ax.set_title("Working more hours does NOT always mean better performance")
    ax.set_xlabel("Average monthly hours")
    ax.set_ylabel("Avg performance score")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "06_overtime_vs_performance.png")


def main(df: pd.DataFrame | None = None) -> None:
    config.make_dirs()
    if df is None:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    summary(df)
    plot_target_distribution(df)
    plot_department_performance(df)
    plot_correlation(df)
    plot_drivers(df)
    plot_score_distribution(df)
    plot_overtime_effect(df)


if __name__ == "__main__":
    main()
