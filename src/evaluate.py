"""
evaluate.py
-----------
Phase 7: Evaluation on the 20% TEST set the model has never seen.

Produces:
  * accuracy, precision, recall, F1 per class
  * confusion matrix image
  * model comparison chart
  * feature importance (permutation importance -> works for any model)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

from src import config


def evaluate(pipe, X_test, y_test, model_name: str) -> dict:
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, labels=config.CLASS_ORDER, digits=3)

    print(f"\n[eval] ===== Test-set results: {model_name} =====")
    print(f"[eval] Accuracy : {acc:.3f}")
    print(f"[eval] F1-macro : {f1:.3f}\n")
    print(report)

    # Save metrics so README / dashboard can show them
    metrics = {"model": model_name, "test_accuracy": round(acc, 4), "test_f1_macro": round(f1, 4),
               "test_rows": int(len(y_test))}
    (config.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Model: {model_name}\nAccuracy: {acc:.3f}\nF1-macro: {f1:.3f}\n\n{report}")

    plot_confusion_matrix(y_test, y_pred, model_name)
    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred, labels=config.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / "07_confusion_matrix.png", dpi=120)
    plt.close(fig)
    print("[eval] saved images/07_confusion_matrix.png")


def plot_model_comparison(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    r = results.sort_values("cv_f1_macro")
    ax.barh(r["model"], r["cv_f1_macro"], xerr=r["cv_f1_std"], color="#2a9d8f")
    for i, (v, sd) in enumerate(zip(r["cv_f1_macro"], r["cv_f1_std"])):
        ax.text(v + sd + 0.015, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cross-validated F1-macro (higher is better)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / "08_model_comparison.png", dpi=120)
    plt.close(fig)
    print("[eval] saved images/08_model_comparison.png")


def plot_feature_importance(pipe, X_test, y_test, top_n: int = 12) -> pd.DataFrame:
    """Permutation importance: shuffle one column at a time and measure how
    much the score drops. A big drop = the model relies on that feature."""
    result = permutation_importance(pipe, X_test, y_test, n_repeats=10, scoring="f1_macro",
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp.round(4).to_csv(config.OUTPUTS_DIR / "feature_importance.csv", index=False)

    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="#e76f51")
    ax.set_xlabel("Drop in F1 when feature is shuffled")
    ax.set_title("What drives employee performance? (top features)")
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / "09_feature_importance.png", dpi=120)
    plt.close(fig)
    print("[eval] saved images/09_feature_importance.png")
    return imp
