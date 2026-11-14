import argparse
import logging
import sys
from pathlib import Path

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")  # works without a display
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.exceptions import NotFittedError
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"Missing dependency: {exc.name}. Install with:\n"
        "  pip install scikit-learn pandas numpy matplotlib joblib"
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("iris")

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"
RANDOM_STATE = 42
OUTPUT_DIR = Path("iris_output")


class DataError(Exception):
    """Raised when the dataset is missing, malformed or unusable."""


# Data loading & validation
def load_from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if not path.is_file():
        raise DataError(f"Not a file: {path}")
    if path.suffix.lower() != ".csv":
        log.warning("File '%s' does not have a .csv extension.", path.name)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DataError(f"CSV file is empty: {path}")
    except pd.errors.ParserError as exc:
        raise DataError(f"Could not parse CSV: {exc}")
    except PermissionError:
        raise DataError(f"No permission to read: {path}")
    except UnicodeDecodeError:
        raise DataError("CSV encoding not supported (expected UTF-8).")
    return df


def load_from_sklearn() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].map(dict(enumerate(data.target_names)))
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DataError("Dataset has no rows.")

    # Normalise column names (strip whitespace)
    df = df.rename(columns=lambda c: str(c).strip())

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise DataError(
            f"Missing required columns: {missing_cols}. Found: {list(df.columns)}"
        )

    df = df[FEATURES + [TARGET]].copy()

    # Coerce features to numeric; invalid entries become NaN
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        log.warning("Dropped %d row(s) with missing/invalid values.", n_before - len(df))

    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        log.warning("Dropped %d duplicate row(s).", n_before - len(df))

    if (df[FEATURES] <= 0).any().any():
        bad = int((df[FEATURES] <= 0).any(axis=1).sum())
        log.warning("Removing %d row(s) with non-positive measurements.", bad)
        df = df[(df[FEATURES] > 0).all(axis=1)]

    df[TARGET] = df[TARGET].astype(str).str.strip()

    counts = df[TARGET].value_counts()
    if counts.size < 2:
        raise DataError("Need at least 2 classes to train a classifier.")
    if counts.min() < 5:
        raise DataError(
            f"Too few samples in class '{counts.idxmin()}' ({counts.min()}); need >= 5."
        )
    return df.reset_index(drop=True)


def get_data(args) -> pd.DataFrame:
    if args.use_sklearn:
        log.info("Loading built-in scikit-learn Iris dataset.")
        return validate_and_clean(load_from_sklearn())

    csv_path = Path(args.csv)
    try:
        log.info("Loading data from %s", csv_path)
        return validate_and_clean(load_from_csv(csv_path))
    except DataError as exc:
        log.error("%s", exc)
        log.info("Falling back to the built-in scikit-learn Iris dataset.")
        return validate_and_clean(load_from_sklearn())


# Exploration
def explore(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60 + "\nDATA OVERVIEW\n" + "=" * 60)
    print(f"Shape: {df.shape}")
    print("\nClass distribution:\n", df[TARGET].value_counts().to_string())
    print("\nSummary statistics:\n", df[FEATURES].describe().round(2).to_string())

    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, col in zip(axes.ravel(), FEATURES):
            for sp, grp in df.groupby(TARGET):
                ax.hist(grp[col], alpha=0.6, label=sp, bins=15)
            ax.set_title(col)
        axes[0, 0].legend()
        fig.suptitle("Feature distributions by species")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        for sp, grp in df.groupby(TARGET):
            ax.scatter(grp["PetalLengthCm"], grp["PetalWidthCm"], label=sp, alpha=0.8)
        ax.set_xlabel("Petal length (cm)")
        ax.set_ylabel("Petal width (cm)")
        ax.set_title("Petal length vs width")
        ax.legend()
        fig.savefig(OUTPUT_DIR / "petal_scatter.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting must never kill the pipeline
        log.warning("Could not create exploration plots: %s", exc)


# Modelling
def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
        ),
        "SVM (RBF)": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(random_state=RANDOM_STATE))]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def split_data(df: pd.DataFrame):
    X = df[FEATURES].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError as exc:
        raise DataError(f"Could not split data: {exc}")
    log.info("Train size: %d | Test size: %d", len(X_tr), len(X_te))
    return X_tr, X_te, y_tr, y_te, encoder


def compare_models(models, X_tr, y_tr) -> dict:
    n_splits = int(min(5, np.bincount(y_tr).min()))
    if n_splits < 2:
        raise DataError("Not enough samples per class for cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 60 + f"\nMODEL COMPARISON ({n_splits}-fold CV on training set)\n" + "=" * 60)
    scores = {}
    for name, model in models.items():
        try:
            s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
            scores[name] = s.mean()
            print(f"{name:<22} {s.mean():.4f} (+/- {s.std():.4f})")
        except Exception as exc:
            log.error("Model '%s' failed during cross-validation: %s", name, exc)
    if not scores:
        raise RuntimeError("All models failed during cross-validation.")
    return scores


def evaluate(model, name, X_te, y_te, encoder) -> float:
    if not hasattr(model, "predict"):
        raise TypeError("Object is not a valid model.")
    try:
        y_pred = model.predict(X_te)
    except NotFittedError:
        raise RuntimeError("Model was not fitted before evaluation.")

    acc = accuracy_score(y_te, y_pred)
    print("\n" + "=" * 60 + f"\nTEST SET EVALUATION - {name}\n" + "=" * 60)
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_te, y_pred, target_names=encoder.classes_, digits=3))

    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix:\n", cm)
    try:
        disp = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
        disp.plot(cmap="Blues", xticks_rotation=30)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
        plt.close()
    except Exception as exc:
        log.warning("Could not save confusion matrix plot: %s", exc)
    return acc


def feature_importance(model) -> None:
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(clf, "feature_importances_"):
        print("\nFeature importances:")
        for f, v in sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1]):
            print(f"  {f:<15} {v:.3f}")


# Prediction helper with input validation
def predict_species(model, encoder, sepal_len, sepal_wid, petal_len, petal_wid) -> str:
    try:
        values = [float(v) for v in (sepal_len, sepal_wid, petal_len, petal_wid)]
    except (TypeError, ValueError):
        raise ValueError("All measurements must be numeric.")
    if any(not np.isfinite(v) for v in values):
        raise ValueError("Measurements must be finite numbers.")
    if any(v <= 0 or v > 20 for v in values):
        raise ValueError("Measurements must be between 0 and 20 cm.")
    pred = model.predict(np.array([values]))
    return str(encoder.inverse_transform(pred)[0])


def save_model(model, encoder) -> None:
    try:
        path = OUTPUT_DIR / "iris_model.joblib"
        joblib.dump({"model": model, "encoder": encoder, "features": FEATURES}, path)
        log.info("Saved trained model to %s", path)
    except (OSError, PermissionError) as exc:
        log.warning("Could not save model: %s", exc)


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Iris flower classification")
    p.add_argument("--csv", default="Iris.csv", help="Path to Iris CSV file")
    p.add_argument("--use-sklearn", action="store_true", help="Use built-in dataset")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output directory: %s", exc)
        return 1

    try:
        df = get_data(args)
        explore(df)

        X_tr, X_te, y_tr, y_te, encoder = split_data(df)
        models = build_models()
        scores = compare_models(models, X_tr, y_tr)

        best_name = max(scores, key=scores.get)
        log.info("Best model by cross-validation: %s", best_name)
        best = models[best_name]
        best.fit(X_tr, y_tr)

        test_acc = evaluate(best, best_name, X_te, y_te, encoder)
        feature_importance(best)
        save_model(best, encoder)

        print("\n" + "=" * 60 + "\nSAMPLE PREDICTIONS\n" + "=" * 60)
        samples = [(5.1, 3.5, 1.4, 0.2), (6.0, 2.9, 4.5, 1.5), (6.9, 3.1, 5.8, 2.1)]
        for s in samples:
            print(f"{s} -> {predict_species(best, encoder, *s)}")
        try:
            predict_species(best, encoder, "abc", 1, 1, 1)
        except ValueError as exc:
            print(f"Invalid input handled correctly: {exc}")

        print(f"\nFinal test accuracy: {test_acc:.2%}")
        print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
        return 0

    except DataError as exc:
        log.error("Data problem: %s", exc)
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
    except MemoryError:
        log.error("Out of memory.")
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
    return 1


if __name__ == "__main__":
    sys.exit(main())
