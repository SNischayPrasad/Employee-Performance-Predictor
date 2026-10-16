"""
screenshots.py
--------------
Creates "proof" images for GitHub / LinkedIn / Instagram:

  images/screenshots/01_dataset_preview.png    raw messy data table
  images/screenshots/02_cleaning_report.png    terminal-style cleaning output
  images/screenshots/03_model_results.png      terminal-style model comparison + test report
  images/screenshots/04_new_predictions.png    terminal-style prediction output
  images/social/instagram_*.png                1080x1080 carousel slides

Run after `python main.py`:
    python -m src.screenshots
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src import config
from src.preprocessing import clean_data

SHOTS = config.IMAGES_DIR / "screenshots"
SOCIAL = config.IMAGES_DIR / "social"

BG, FG, GREEN, YELLOW, BLUE, GREY = "#1e1e2e", "#cdd6f4", "#a6e3a1", "#f9e2af", "#89b4fa", "#6c7086"
TEAL, CORAL, SAND, NAVY = "#2a9d8f", "#e76f51", "#e9c46a", "#264653"


# ---------------------------------------------------------------- terminal look
def terminal_image(lines, path, title="Terminal - python main.py", width=13):
    """Render text lines as a dark terminal window."""
    height = 0.9 + 0.235 * len(lines)
    fig = plt.figure(figsize=(width, height), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # window bar
    bar = 0.45 / height
    ax.add_patch(plt.Rectangle((0, 1 - bar), 1, bar, color="#313244"))
    for i, c in enumerate(["#f38ba8", "#f9e2af", "#a6e3a1"]):
        ax.add_patch(plt.Circle((0.015 + i * 0.018, 1 - bar / 2), 0.0055, color=c, transform=ax.transAxes))
    ax.text(0.5, 1 - bar / 2, title, color=GREY, ha="center", va="center", fontsize=10, family="monospace")
    step = 0.235 / height
    for i, line in enumerate(lines):
        color = FG
        if line.startswith("$"): color = GREEN
        elif "=====" in line or "STEP" in line or "DONE" in line: color = BLUE
        elif "Best model" in line or "Accuracy" in line or "F1-macro :" in line: color = YELLOW
        ax.text(0.015, 1 - bar - step * (i + 0.9), line, color=color, fontsize=10.5,
                family="monospace", va="center")
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[shots] saved {path.relative_to(config.ROOT_DIR)}")


def _log_section(log, start, end):
    s = log.index(start)
    e = log.index(end, s + 1) if end else len(log)
    return log[s:e]


def terminal_shots(log_path):
    log = [l.rstrip() for l in open(log_path, encoding="utf-8").read().splitlines()]
    cleaning = ["$ python main.py", ""] + [l for l in _log_section(log, "  STEP 1/8  Generating synthetic employee data",
                                                                   "  STEP 4/8  Exploratory data analysis") if not l.startswith("====")]
    terminal_image(cleaning, SHOTS / "02_cleaning_report.png")

    model = [l for l in _log_section(log, "  STEP 5/8  Training & comparing models",
                                     "[eval] saved images/07_confusion_matrix.png") if not l.startswith("====")]
    terminal_image(model, SHOTS / "03_model_results.png")

    pred = [l for l in _log_section(log, "  STEP 8/8  Predicting performance for new employees", None)
            if not l.startswith("====")]
    terminal_image(pred, SHOTS / "04_new_predictions.png", width=19)


# ---------------------------------------------------------------- dataset table
def dataset_preview():
    raw = pd.read_csv(config.RAW_DATA_PATH)
    # mix of normal rows and deliberately messy rows (missing / 9999 / bad text)
    bad_text = raw.department != raw.department.str.strip().str.title().replace({"Hr": "HR"})
    picks = [raw[raw.notna().all(axis=1) & ~bad_text & (raw.training_hours < 500)].head(4),
             raw[raw.training_hours > 500].head(2), raw[bad_text].head(2),
             raw[raw.peer_feedback_score.isna()].head(1), raw[raw.monthly_salary.isna()].head(1),
             raw[raw.education.isna()].head(1), raw[raw.age > 70].head(1)]
    sample = pd.concat(picks).drop_duplicates(subset="employee_id")
    cols = ["employee_id", "age", "department", "education", "job_level", "years_experience",
            "monthly_salary", "training_hours", "projects_completed", "absences",
            "peer_feedback_score", "job_satisfaction", "performance_category"]

    def fmt(v):
        if pd.isna(v):
            return "NaN"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return f"'{v}'" if isinstance(v, str) and v != v.strip() else str(v)

    tbl = sample[cols].map(fmt)

    fig, ax = plt.subplots(figsize=(19, 6.8))
    ax.axis("off")
    ax.set_title(f"Raw synthetic HR dataset - {len(raw):,} rows x {raw.shape[1]} columns "
                 "(note the NaN, 9999 and inconsistent text: real-world mess to clean)",
                 fontsize=13, weight="bold", loc="left")
    t = ax.table(cellText=tbl.values, colLabels=[c.replace("_", "\n") for c in cols],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.1)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", weight="bold", fontsize=9)
            cell.set_height(cell.get_height() * 1.6)
        else:
            txt = cell.get_text().get_text()
            if txt in ("NaN", "9999") or txt.startswith("'") or (c == 1 and int(txt) > 70)                     or (c == 2 and txt not in ("Sales", "Engineering", "HR", "Finance", "Marketing", "Operations", "Support")):
                cell.set_facecolor("#ffe1d6")
    fig.tight_layout()
    path = SHOTS / "01_dataset_preview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[shots] saved {path.relative_to(config.ROOT_DIR)}")


# ---------------------------------------------------------------- instagram carousel
def _slide(num, total):
    fig = plt.figure(figsize=(7.2, 7.2), dpi=150)  # 1080 x 1080 px
    fig.patch.set_facecolor("#fbf8f3")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.94, 0.04, f"{num}/{total}", ha="right", fontsize=11, color=GREY)
    ax.text(0.06, 0.04, "Employee Performance Predictor", fontsize=10, color=GREY)
    return fig, ax


def _save_slide(fig, name):
    path = SOCIAL / name
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[shots] saved {path.relative_to(config.ROOT_DIR)}")


def _image_in(fig, img_path, rect):
    a = fig.add_axes(rect); a.axis("off")
    a.imshow(plt.imread(img_path))


def instagram_carousel():
    m = json.loads((config.OUTPUTS_DIR / "metrics.json").read_text())
    fi = pd.read_csv(config.OUTPUTS_DIR / "feature_importance.csv")
    total = 6

    # 1 - cover
    fig, ax = _slide(1, total)
    ax.add_patch(FancyBboxPatch((0.06, 0.12), 0.88, 0.76, boxstyle="round,pad=0.02", color=NAVY))
    ax.text(0.12, 0.76, "Can AI predict", fontsize=30, color="white", weight="bold")
    ax.text(0.12, 0.67, "who will be a", fontsize=30, color="white", weight="bold")
    ax.text(0.12, 0.58, "top performer?", fontsize=30, color=SAND, weight="bold")
    ax.text(0.12, 0.44, "I built an HR analytics + ML project\nto find out.", fontsize=15, color="#e0e0e0", linespacing=1.5)
    ax.text(0.12, 0.22, "Python  ·  Pandas  ·  scikit-learn  ·  Streamlit", fontsize=12, color=SAND)
    ax.text(0.12, 0.16, "Swipe  →", fontsize=14, color="white", weight="bold")
    _save_slide(fig, "instagram_01_cover.png")

    # 2 - problem & approach
    fig, ax = _slide(2, total)
    ax.text(0.06, 0.88, "The problem", fontsize=24, weight="bold", color=NAVY)
    ax.text(0.06, 0.74, "Companies discover low performance\nat the yearly review, when it's too late.\n"
            "Promotions often run on gut feeling.", fontsize=14, color="#333", linespacing=1.5)
    ax.text(0.06, 0.60, "What I built", fontsize=24, weight="bold", color=NAVY)
    steps = ["Simulated a 2,000-employee company", "Cleaned messy HR data (350 missing values)",
             "Compared 4 machine-learning models", "Predicted Low / Medium / High performers",
             "Turned predictions into HR actions"]
    for i, s in enumerate(steps):
        y = 0.51 - i * 0.075
        ax.add_patch(plt.Circle((0.09, y + 0.008), 0.022, color=TEAL))
        ax.text(0.09, y + 0.008, str(i + 1), ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(0.14, y, s, fontsize=14, color="#333", va="center")
    _save_slide(fig, "instagram_02_problem.png")

    # 3 - results
    fig, ax = _slide(3, total)
    ax.text(0.06, 0.88, "The results", fontsize=24, weight="bold", color=NAVY)
    for i, (val, lab) in enumerate([(f"{m['test_accuracy']:.0%}", "accuracy on\nunseen employees"),
                                    ("4", "models\ncompared"), ("2,000", "employees\nanalysed")]):
        x = 0.06 + i * 0.30
        ax.add_patch(FancyBboxPatch((x, 0.62), 0.26, 0.18, boxstyle="round,pad=0.01", color="white", ec="#e5e0d8"))
        ax.text(x + 0.13, 0.74, val, ha="center", fontsize=28, weight="bold", color=TEAL)
        ax.text(x + 0.13, 0.655, lab, ha="center", fontsize=10.5, color="#555", linespacing=1.3)
    _image_in(fig, config.IMAGES_DIR / "08_model_comparison.png", [0.06, 0.08, 0.88, 0.50])
    _save_slide(fig, "instagram_03_results.png")

    # 4 - drivers
    fig, ax = _slide(4, total)
    ax.text(0.06, 0.88, "What drives performance?", fontsize=24, weight="bold", color=NAVY)
    top = fi.head(5).iloc[::-1]
    names = {"peer_feedback_score": "Peer feedback", "absences": "Absences", "job_satisfaction": "Job satisfaction",
             "projects_per_year": "Projects per year", "training_hours": "Training hours"}
    b = fig.add_axes([0.32, 0.38, 0.6, 0.42]); b.set_facecolor("#fbf8f3")
    b.barh([names.get(f, f) for f in top.feature], top.importance, color=CORAL)
    b.spines[["top", "right", "bottom"]].set_visible(False); b.set_xticks([])
    b.tick_params(labelsize=13, left=False)
    b.set_yticks(range(5)); b.set_yticklabels([names.get(f, f) for f in top.feature]); b.yaxis.set_tick_params(pad=8)
    ax.text(0.06, 0.25, "Employees with the most training were\n3.5x more likely to be top performers.",
            fontsize=15, color=NAVY, weight="bold", linespacing=1.5)
    ax.text(0.06, 0.13, "Working 220+ hours a month LOWERED performance (burnout).", fontsize=12.5, color="#555")
    _save_slide(fig, "instagram_04_drivers.png")

    # 5 - dashboard
    fig, ax = _slide(5, total)
    ax.text(0.06, 0.88, "From prediction to action", fontsize=24, weight="bold", color=NAVY)
    ax.text(0.06, 0.81, "Promote · Train · Support · Review workload", fontsize=14, color=TEAL, weight="bold")
    _image_in(fig, config.IMAGES_DIR / "10_hr_dashboard.png", [0.04, 0.12, 0.92, 0.66])
    _save_slide(fig, "instagram_05_dashboard.png")

    # 6 - lesson / CTA
    fig, ax = _slide(6, total)
    ax.add_patch(FancyBboxPatch((0.06, 0.12), 0.88, 0.76, boxstyle="round,pad=0.02", color=TEAL))
    ax.text(0.12, 0.76, "Biggest lesson", fontsize=26, color="white", weight="bold")
    ax.text(0.12, 0.55, "The simplest model\n(Logistic Regression)\nbeat Random Forest.", fontsize=21,
            color="white", linespacing=1.4)
    ax.text(0.12, 0.38, "Complex ≠ better. Explainable wins in HR.", fontsize=14, color=SAND, weight="bold")
    ax.text(0.12, 0.22, "Full code on GitHub, link in bio", fontsize=14, color="white")
    ax.text(0.12, 0.16, "Save this for your next project idea", fontsize=12, color="#e8f5f3")
    _save_slide(fig, "instagram_06_lesson.png")


def main(log_path=None):
    SHOTS.mkdir(parents=True, exist_ok=True)
    SOCIAL.mkdir(parents=True, exist_ok=True)
    dataset_preview()
    if log_path:
        terminal_shots(log_path)
    instagram_carousel()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log", help="saved output of `python main.py > run_log.txt`")
    main(p.parse_args().log)
