"""
Model training, evaluation, and comparison for hate speech classification.

Provides:
  - Feature loading utilities
  - Model factory with compatibility checking
  - Single-model training + evaluation
  - Full experiment grid (feature x model matrix)
  - Hyperparameter tuning wrappers
  - Visualization: confusion matrices, ROC curves, summary heatmaps

All functions consume the feature files produced by Step 1 (features/ dir).
"""

import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
)
from sklearn.utils.class_weight import compute_sample_weight


# ===========================================================================
# Constants & registry
# ===========================================================================

RANDOM_STATE = 42

FEATURE_REGISTRY = {
    "bow":        {"file_train": "bow_train.npz",        "file_test": "bow_test.npz",
                   "type": "sparse", "non_negative": True},
    "tfidf":      {"file_train": "tfidf_train.npz",      "file_test": "tfidf_test.npz",
                   "type": "sparse", "non_negative": True},
    "char_tfidf": {"file_train": "char_tfidf_train.npz", "file_test": "char_tfidf_test.npz",
                   "type": "sparse", "non_negative": True},
    "glove":      {"file_train": "glove_train.npy",      "file_test": "glove_test.npy",
                   "type": "dense",  "non_negative": False},
    "distilbert": {"file_train": "distilbert_train.npy",  "file_test": "distilbert_test.npy",
                   "type": "dense",  "non_negative": False},
}

MODEL_KEYS = ["mnb", "gnb", "lr", "svm", "knn", "rf", "gb"]

FEATURE_ORDER = ["bow", "tfidf", "char_tfidf", "glove", "distilbert"]

DISPLAY_NAMES = {
    # features
    "bow": "BoW", "tfidf": "TF-IDF", "char_tfidf": "Char TF-IDF",
    "glove": "GloVe 200d", "distilbert": "DistilBERT 768d",
    # models
    "mnb": "MultinomialNB", "gnb": "GaussianNB",
    "lr": "Logistic Regression", "svm": "LinearSVC",
    "knn": "k-NN", "rf": "Random Forest", "gb": "Gradient Boosting",
}


# ===========================================================================
# Feature loading
# ===========================================================================

def load_features(features_dir) -> dict:
    """
    Load all 5 feature sets + labels from disk.

    Returns
    -------
    dict  — keys per feature ('bow', ..., 'distilbert') each mapping to
            {'train': array, 'test': array}, plus top-level 'y_train', 'y_test'.
    """
    features_dir = Path(features_dir)
    data = {}

    for key, info in FEATURE_REGISTRY.items():
        if info["type"] == "sparse":
            data[key] = {
                "train": sp.load_npz(str(features_dir / info["file_train"])),
                "test":  sp.load_npz(str(features_dir / info["file_test"])),
            }
        else:
            data[key] = {
                "train": np.load(str(features_dir / info["file_train"])),
                "test":  np.load(str(features_dir / info["file_test"])),
            }

    data["y_train"] = np.load(str(features_dir / "y_train.npy"))
    data["y_test"]  = np.load(str(features_dir / "y_test.npy"))

    print(f"Loaded {len(FEATURE_REGISTRY)} feature sets + labels from {features_dir}")
    return data


# ===========================================================================
# Model compatibility & factory
# ===========================================================================

def is_compatible(model_key: str, feature_key: str) -> bool:
    """Check whether a model–feature combination is valid."""
    info = FEATURE_REGISTRY[feature_key]
    if model_key == "mnb":
        return info["non_negative"]          # sparse non-negative only
    if model_key == "gnb":
        return info["type"] == "dense"       # dense only (memory)
    if model_key == "gb":
        return info["type"] == "dense"       # dense only (speed)
    return True


def get_model(model_key: str, feature_key: str, class_weight: str = "balanced"):
    """
    Return an unfitted sklearn estimator with sensible defaults.

    Parameters
    ----------
    model_key    : one of MODEL_KEYS
    feature_key  : one of FEATURE_ORDER
    class_weight : 'balanced' or None

    Returns
    -------
    sklearn estimator (unfitted), or None if incompatible
    """
    if not is_compatible(model_key, feature_key):
        return None

    ftype = FEATURE_REGISTRY[feature_key]["type"]

    if model_key == "mnb":
        return MultinomialNB(alpha=1.0, class_prior=[0.5, 0.5])

    if model_key == "gnb":
        return GaussianNB(priors=[0.5, 0.5])

    if model_key == "lr":
        solver = "liblinear" if ftype == "sparse" else "lbfgs"
        return LogisticRegression(
            C=1.0, max_iter=1000, solver=solver,
            class_weight=class_weight, random_state=RANDOM_STATE,
        )

    if model_key == "svm":
        base = LinearSVC(
            C=1.0, max_iter=5000, class_weight=class_weight,
            random_state=RANDOM_STATE, dual="auto",
        )
        return CalibratedClassifierCV(base, cv=3)

    if model_key == "knn":
        metric = "cosine" if ftype == "sparse" else "minkowski"
        return KNeighborsClassifier(
            n_neighbors=5, metric=metric, n_jobs=-1,
        )

    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1,
        )

    if model_key == "gb":
        return GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model_key: {model_key}")


# ===========================================================================
# Train & evaluate a single model
# ===========================================================================

def _ensure_dense(X):
    """Convert sparse matrix to dense if needed."""
    if sp.issparse(X):
        return X.toarray()
    return X


def train_and_evaluate(
    model,
    X_train, y_train,
    X_test,  y_test,
    model_key: str = "",
    feature_key: str = "",
) -> dict:
    """
    Fit *model* on (X_train, y_train), predict on X_test, compute metrics.

    Returns
    -------
    dict with metric values, predictions, confusion matrix, etc.
    """
    model_name   = DISPLAY_NAMES.get(model_key, model_key)
    feature_name = DISPLAY_NAMES.get(feature_key, feature_key)

    # Some models need dense input
    needs_dense = model_key in ("gnb", "gb")
    Xtr = _ensure_dense(X_train) if needs_dense else X_train
    Xte = _ensure_dense(X_test)  if needs_dense else X_test

    # Compute sample_weight for GB (no native class_weight)
    fit_kwargs = {}
    if model_key == "gb":
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, y_train, **fit_kwargs)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(Xte)

    # Probability estimates (for ROC)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(Xte)[:, 1]
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-toxic", "Toxic"])

    return {
        "model_key":   model_key,
        "feature_key": feature_key,
        "model_name":   model_name,
        "feature_name": feature_name,
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_toxic":        f1_score(y_test, y_pred, average="binary", zero_division=0),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
        "classification_report": report,
        "train_time": train_time,
    }


# ===========================================================================
# Full experiment grid
# ===========================================================================

def run_experiment_grid(
    data: dict,
    model_keys: list = None,
    feature_keys: list = None,
    class_weight: str = "balanced",
):
    """
    Train all compatible model x feature combinations.

    Parameters
    ----------
    data         : output of load_features()
    model_keys   : list of model identifiers (default: all 7)
    feature_keys : list of feature identifiers (default: all 5)
    class_weight : passed to get_model()

    Returns
    -------
    (results_df, fitted_models)
        results_df    : DataFrame with one row per experiment
        fitted_models : dict {(model_key, feature_key): fitted_model}
    """
    if model_keys is None:
        model_keys = MODEL_KEYS
    if feature_keys is None:
        feature_keys = FEATURE_ORDER

    y_train, y_test = data["y_train"], data["y_test"]

    combos = [
        (mk, fk) for mk in model_keys for fk in feature_keys
        if is_compatible(mk, fk)
    ]
    print(f"Running {len(combos)} experiments ...\n")

    results = []
    fitted_models = {}

    for i, (mk, fk) in enumerate(combos, 1):
        mname = DISPLAY_NAMES[mk]
        fname = DISPLAY_NAMES[fk]
        print(f"[{i:2d}/{len(combos)}] {mname:20s} + {fname:15s} ", end="", flush=True)

        model = get_model(mk, fk, class_weight=class_weight)
        res = train_and_evaluate(
            model,
            data[fk]["train"], y_train,
            data[fk]["test"],  y_test,
            model_key=mk, feature_key=fk,
        )
        fitted_models[(mk, fk)] = model
        results.append(res)
        print(f"→ F1={res['f1_macro']:.4f}  ({res['train_time']:.1f}s)")

    results_df = pd.DataFrame(results)
    print(f"\nDone. Best F1-macro: {results_df['f1_macro'].max():.4f}")
    return results_df, fitted_models


# ===========================================================================
# Visualization helpers
# ===========================================================================

def _save_or_show(fig, save_path=None):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_results_heatmap(results_df: pd.DataFrame, metric: str = "f1_macro",
                         save_path: str = None) -> None:
    """Pivot-table heatmap: features (rows) x models (cols), coloured by *metric*."""
    pivot = results_df.pivot_table(
        index="feature_name", columns="model_name", values=metric, aggfunc="first",
    )
    # Reorder rows / cols by registry order
    row_order = [DISPLAY_NAMES[k] for k in FEATURE_ORDER if DISPLAY_NAMES[k] in pivot.index]
    col_order = [DISPLAY_NAMES[k] for k in MODEL_KEYS if DISPLAY_NAMES[k] in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=pivot.min().min() - 0.02,
    )
    label = metric.replace("_", " ").title()
    ax.set_title(f"Model × Feature Comparison — {label}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Feature Representation")
    ax.set_xlabel("Classifier")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrices(results_df: pd.DataFrame, top_n: int = 6,
                            metric: str = "f1_macro",
                            save_path: str = None) -> None:
    """Plot confusion matrices for the top-N models ranked by *metric*."""
    top = results_df.nlargest(top_n, metric)
    nrows = (top_n + 2) // 3
    ncols = min(top_n, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, (_, row) in enumerate(top.iterrows()):
        ax = axes[idx]
        cm = row["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-toxic", "Toxic"],
            yticklabels=["Non-toxic", "Toxic"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{row['model_name']}\n+ {row['feature_name']}  (F1={row['f1_macro']:.3f})",
            fontsize=10, fontweight="bold",
        )

    # Hide unused axes
    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices — Top Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curves(results_df: pd.DataFrame, y_test: np.ndarray,
                    top_n: int = 8, metric: str = "f1_macro",
                    save_path: str = None) -> None:
    """Overlay ROC curves for the top-N models that have probability estimates."""
    top = results_df.nlargest(top_n * 2, metric)  # over-select in case some lack y_prob

    fig, ax = plt.subplots(figsize=(9, 7))
    plotted = 0

    for _, row in top.iterrows():
        if row["y_prob"] is None or plotted >= top_n:
            continue
        fpr, tpr, _ = roc_curve(y_test, row["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                label=f"{row['model_name']} + {row['feature_name']} (AUC={roc_auc:.3f})")
        plotted += 1

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Top Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_top_models_bar(results_df: pd.DataFrame, top_n: int = 10,
                        metric: str = "f1_macro",
                        save_path: str = None) -> None:
    """Horizontal bar chart of top-N models ranked by *metric*."""
    top = results_df.nlargest(top_n, metric).iloc[::-1]  # ascending for barh

    fig, ax = plt.subplots(figsize=(10, 0.6 * top_n + 1))
    labels = [f"{r['model_name']} + {r['feature_name']}" for _, r in top.iterrows()]
    values = top[metric].values
    colors = sns.color_palette("YlOrRd_r", top_n)

    bars = ax.barh(range(top_n), values, color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    label = metric.replace("_", " ").title()
    ax.set_xlabel(label)
    ax.set_title(f"Top {top_n} Models by {label}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, values.max() * 1.08)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ===========================================================================
# Hyperparameter tuning
# ===========================================================================

def get_param_grid(model_key: str) -> dict:
    """Return a hyperparameter search grid for *model_key*."""
    grids = {
        "mnb": {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
        "gnb": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        "lr":  {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
        "svm": {
            "estimator__C": [0.01, 0.1, 1.0, 10.0],
            "estimator__loss": ["hinge", "squared_hinge"],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
        },
        "rf": {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 50, None],
            "min_samples_split": [2, 5, 10],
        },
        "gb": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    }
    return grids.get(model_key, {})


def tune_model(
    model_key: str,
    feature_key: str,
    X_train, y_train,
    X_test,  y_test,
    class_weight: str = "balanced",
    cv: int = 5,
) -> dict:
    """
    Tune a model–feature combination via GridSearchCV (or RandomizedSearchCV
    for large grids like RF / GB).

    Returns
    -------
    dict with 'best_params', 'best_cv_score', 'test_results'.
    """
    model = get_model(model_key, feature_key, class_weight=class_weight)
    param_grid = get_param_grid(model_key)

    needs_dense = model_key in ("gnb", "gb")
    Xtr = _ensure_dense(X_train) if needs_dense else X_train
    Xte = _ensure_dense(X_test)  if needs_dense else X_test

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Use RandomizedSearchCV for large grids
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    fit_params = {}
    if model_key == "gb":
        fit_params["sample_weight"] = compute_sample_weight("balanced", y_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_combos > 30:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=skf,
                scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=skf,
                scoring="f1_macro", n_jobs=-1,
            )
        search.fit(Xtr, y_train, **fit_params)

    best_model = search.best_estimator_
    res = train_and_evaluate(
        best_model, X_train, y_train, X_test, y_test,
        model_key=model_key, feature_key=feature_key,
    )

    return {
        "best_params":   search.best_params_,
        "best_cv_score": search.best_score_,
        "test_results":  res,
    }


# ===========================================================================
# SMOTE experiment helper
# ===========================================================================

def run_smote_experiment(
    data: dict,
    model_keys: list = None,
    feature_keys: list = None,
) -> pd.DataFrame:
    """
    Re-run selected experiments with SMOTE oversampling on training data.
    Requires ``imbalanced-learn`` (``pip install imbalanced-learn``).
    """
    from imblearn.over_sampling import SMOTE

    if model_keys is None:
        model_keys = ["lr", "svm", "rf"]
    if feature_keys is None:
        feature_keys = ["bow", "tfidf", "char_tfidf"]

    y_train, y_test = data["y_train"], data["y_test"]
    smote = SMOTE(random_state=RANDOM_STATE)

    combos = [
        (mk, fk) for mk in model_keys for fk in feature_keys
        if is_compatible(mk, fk)
    ]
    print(f"Running {len(combos)} SMOTE experiments ...\n")
    results = []

    for i, (mk, fk) in enumerate(combos, 1):
        mname = DISPLAY_NAMES[mk]
        fname = DISPLAY_NAMES[fk]
        print(f"[{i:2d}/{len(combos)}] {mname:20s} + {fname:15s} (SMOTE) ", end="", flush=True)

        X_tr = data[fk]["train"]
        # SMOTE requires dense for sparse input
        if sp.issparse(X_tr):
            X_res, y_res = smote.fit_resample(X_tr.toarray(), y_train)
        else:
            X_res, y_res = smote.fit_resample(X_tr, y_train)

        model = get_model(mk, fk, class_weight=None)  # no class_weight — SMOTE handles it
        res = train_and_evaluate(
            model,
            X_res if not sp.issparse(X_tr) else sp.csr_matrix(X_res),
            y_res,
            data[fk]["test"], y_test,
            model_key=mk, feature_key=fk,
        )
        results.append(res)
        print(f"→ F1={res['f1_macro']:.4f}  ({res['train_time']:.1f}s)")

    return pd.DataFrame(results)
