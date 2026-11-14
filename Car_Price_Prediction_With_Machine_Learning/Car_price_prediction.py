import argparse
import json
import logging
import re
import sys
from pathlib import Path

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency: {exc.name}. Install with:\n"
             "  pip install pandas numpy scikit-learn matplotlib joblib")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("carprice")

RANDOM_STATE = 42
TARGET = "Selling_Price"
REQUIRED = ["Car_Name", "Year", "Selling_Price", "Present_Price", "Driven_kms",
            "Fuel_Type", "Selling_type", "Transmission", "Owner"]
NUM_FEATURES = ["Car_Age", "Log_Kms", "Kms_Per_Year", "Log_Present_Price", "Owner"]
OHE_FEATURES = ["Fuel_Type", "Selling_type", "Transmission", "Vehicle_Type"]
TE_FEATURES = ["Make"]  # brand goodwill, target-encoded inside the CV folds (no leakage)
ALL_FEATURES = NUM_FEATURES + OHE_FEATURES + TE_FEATURES
MIN_ROWS = 30


class DataError(Exception):
    """Raised when data is missing, malformed or unusable."""


# Brand / vehicle-type extraction (feature engineering from Car_Name)
CAR_MODELS = {
    "Maruti": ["ciaz", "sx4", "ertiga", "swift", "alto", "ritz", "wagon r", "dzire",
               "vitara brezza", "s cross", "ignis", "800", "baleno", "omni"],
    "Honda": ["city", "brio", "jazz", "amaze"],
    "Toyota": ["corolla", "fortuner", "innova", "etios", "camry", "land cruiser"],
    "Hyundai": ["verna", "i20", "grand i10", "i10", "eon", "xcent", "creta", "elantra"],
}
TWO_WHEELER_PREFIXES = {
    "royal enfield": "Royal Enfield", "bajaj": "Bajaj", "hero": "Hero", "tvs": "TVS",
    "yamaha": "Yamaha", "ktm": "KTM", "hyosung": "Hyosung", "um": "UM",
    "mahindra": "Mahindra", "suzuki": "Suzuki", "activa": "Honda", "honda": "Honda",
}
_MODEL_LOOKUP = sorted(((m, b) for b, ms in CAR_MODELS.items() for m in ms),
                       key=lambda t: -len(t[0]))  # longest match first


def parse_name(name: str):
    """Return (brand, vehicle_type) from a free-text car name."""
    n = re.sub(r"\s+", " ", str(name).strip().lower())
    for model, brand in _MODEL_LOOKUP:
        if n == model or n.startswith(model + " "):
            return brand, "Car"
    for prefix, brand in TWO_WHEELER_PREFIXES.items():
        if n == prefix or n.startswith(prefix + " "):
            return brand, "TwoWheeler"
    return "Other", "Unknown"


def engineer(df: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    """Stateless feature engineering, used for both training and prediction."""
    out = df.copy()
    parsed = out["Car_Name"].map(parse_name)
    out["Brand"] = parsed.map(lambda t: t[0])
    out["Vehicle_Type"] = parsed.map(lambda t: t[1])
    out["Make"] = out["Brand"] + "_" + out["Vehicle_Type"]  # Honda car != Honda scooter
    out["Car_Age"] = (ref_year - out["Year"]).clip(lower=0)
    out["Log_Kms"] = np.log1p(out["Driven_kms"])
    out["Kms_Per_Year"] = out["Driven_kms"] / (out["Car_Age"] + 1)
    out["Log_Present_Price"] = np.log1p(out["Present_Price"])
    return out


# Loading & cleaning
def find_csv(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent / p.name
    if alt.exists():
        log.warning("'%s' not found; using %s", p, alt)
        return alt
    raise DataError(f"CSV file not found: {p}")


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataError(f"Not a file: {path}")
    err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            err = exc
        except pd.errors.EmptyDataError:
            raise DataError(f"CSV file is empty: {path.name}")
        except pd.errors.ParserError as exc:
            raise DataError(f"Cannot parse {path.name}: {exc}")
        except PermissionError:
            raise DataError(f"No permission to read {path.name}")
    else:
        raise DataError(f"Cannot decode {path.name}: {err}")
    if df.empty:
        raise DataError("CSV has no rows.")
    return df


def clean(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    df = df[REQUIRED]

    for c in ("Car_Name", "Fuel_Type", "Selling_type", "Transmission"):
        df[c] = df[c].astype(str).str.strip()
    for c in ("Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.replace({"": np.nan, "nan": np.nan}).dropna()
    if len(df) < n0:
        log.warning("Dropped %d row(s) with missing/non-numeric values.", n0 - len(df))

    bad = (df["Selling_Price"] <= 0) | (df["Present_Price"] <= 0) | (df["Driven_kms"] < 0) \
        | (df["Owner"] < 0) | ~df["Year"].between(1980, current_year + 1)
    if bad.any():
        log.warning("Dropped %d row(s) with impossible values (price <= 0, negative kms/owners, "
                    "year outside 1980-%d).", int(bad.sum()), current_year + 1)
        df = df[~bad]

    n0 = len(df)
    df = df.drop_duplicates()
    if len(df) < n0:
        log.warning("Dropped %d duplicate row(s).", n0 - len(df))

    for c in ("Year", "Owner"):
        df[c] = df[c].astype(int)
    if len(df) < MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows after cleaning; need at least {MIN_ROWS}.")
    return df.reset_index(drop=True)


def report_outliers(df: pd.DataFrame) -> None:
    """Flag (but keep) extreme values: they are real cars, not errors."""
    for col in ("Selling_Price", "Driven_kms"):
        lg = np.log1p(df[col])
        q1, q3 = lg.quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((lg < q1 - 3 * iqr) | (lg > q3 + 3 * iqr)).sum())
        if n:
            log.info("%d extreme value(s) in %s kept (log-scale IQR rule); tree models handle them.", n, col)


def unknown_names(df: pd.DataFrame) -> None:
    unk = df.loc[df["Brand"] == "Other", "Car_Name"].unique()
    if len(unk):
        log.warning("%d car name(s) not in the brand map, grouped as 'Other': %s",
                    len(unk), ", ".join(map(str, unk[:8])))


# Modelling
def make_target_encoder() -> TargetEncoder:
    """Goodwill encoder; handles the cv/random_state API change in scikit-learn 1.9."""
    import sklearn
    major_minor = tuple(int(x) for x in re.findall(r"\d+", sklearn.__version__)[:2])
    if major_minor >= (1, 9):
        return TargetEncoder(target_type="continuous",
                             cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    return TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)


def make_pipeline(estimator) -> TransformedTargetRegressor:
    """Preprocessing + model, predicting log(1+price) and returning real prices."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), OHE_FEATURES),
        ("goodwill", make_target_encoder(), TE_FEATURES),
    ])
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def candidate_models() -> dict:
    return {
        "Ridge Regression": make_pipeline(Ridge(alpha=1.0)),
        "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
        "Gradient Boosting": make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


PARAM_GRIDS = {
    "Random Forest": {"regressor__model__max_depth": [None, 8, 14],
                      "regressor__model__min_samples_leaf": [1, 2, 4]},
    "Gradient Boosting": {"regressor__model__n_estimators": [150, 300],
                          "regressor__model__learning_rate": [0.05, 0.1],
                          "regressor__model__max_depth": [2, 3]},
    "Ridge Regression": {"regressor__model__alpha": [0.1, 1.0, 10.0]},
}


def compare_models(models, X, y):
    k = int(min(5, len(X)))
    cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    print("\n" + "=" * 64 + f"\nMODEL COMPARISON ({k}-fold CV on training data)\n" + "=" * 64)
    for name, model in models.items():
        try:
            r = cross_validate(model, X, y, cv=cv,
                               scoring={"r2": "r2", "mae": "neg_mean_absolute_error"})
            rows[name] = {"R2": r["test_r2"].mean(), "R2_std": r["test_r2"].std(),
                          "MAE": -r["test_mae"].mean()}
            print(f"{name:<20} R2 = {rows[name]['R2']:.4f} (+/- {rows[name]['R2_std']:.3f})"
                  f" | MAE = {rows[name]['MAE']:.3f} lakh")
        except Exception as exc:
            log.error("Model '%s' failed during cross-validation: %s", name, exc)
    if not rows:
        raise RuntimeError("All candidate models failed during cross-validation.")
    return rows


def tune(name, model, X, y):
    grid = PARAM_GRIDS.get(name)
    if not grid:
        return model.fit(X, y)
    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(model, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(X, y)
        log.info("Best parameters: %s", {k.split("__")[-1]: v for k, v in gs.best_params_.items()})
        return gs.best_estimator_
    except Exception as exc:
        log.warning("Tuning failed (%s); using default parameters instead.", exc)
        return model.fit(X, y)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedianAPE_%": float(np.median(np.abs(y_true - y_pred) / y_true) * 100),
    }


# Charts (each isolated so one failure never stops the pipeline)
def safe_chart(func, *args):
    try:
        func(*args)
    except Exception as exc:
        log.warning("Chart '%s' skipped: %s", func.__name__, exc)
        plt.close("all")


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    log.info("Saved chart: %s", name)


def chart_price_distribution(df, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df[TARGET], bins=40, color="steelblue")
    ax[0].set(title="Selling price (heavily right-skewed)", xlabel="Lakh INR")
    ax[1].hist(np.log1p(df[TARGET]), bins=40, color="seagreen")
    ax[1].set(title="log(1 + price): far more symmetric", xlabel="log(1 + lakh INR)")
    _save(fig, out, "01_price_distribution.png")


def chart_age_vs_price(df, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for vt, g in df.groupby("Vehicle_Type"):
        ax.scatter(g["Car_Age"], g[TARGET], alpha=0.6, s=22, label=f"{vt} (n={len(g)})")
    ax.set_yscale("log")
    ax.set(title="Price falls with age; cars and two-wheelers sit in different price bands",
           xlabel="Age (years)", ylabel="Selling price (lakh INR, log scale)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out, "02_age_vs_price.png")


def chart_goodwill(df, out):
    d = df[df["Present_Price"] > 0].assign(Retention=lambda x: x[TARGET] / x["Present_Price"])
    g = d.groupby("Make").agg(Retention=("Retention", "median"), n=("Retention", "size"))
    g = g[g["n"] >= 5].sort_values("Retention")
    if g.empty:
        raise DataError("No make has >= 5 listings.")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(g.index, g["Retention"] * 100, color="darkorange")
    ax.set(title="Brand goodwill: median resale value as % of showroom price",
           xlabel="Selling price / present price (%)")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out, "03_brand_goodwill.png")


def chart_model_comparison(cv_rows, out):
    names = list(cv_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [cv_rows[n]["R2"] for n in names], color="slateblue")
    ax.set(title="Cross-validated R2 by model", ylabel="R2", ylim=(min(0.5, min(cv_rows[n]["R2"] for n in names) - 0.05), 1))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "04_model_comparison.png")


def chart_actual_vs_pred(y_true, y_pred, out):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set(title="Test set: actual vs predicted price", xlabel="Actual (lakh INR)", ylabel="Predicted (lakh INR)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out, "05_actual_vs_predicted.png")


def chart_residuals(y_true, y_pred, out):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_pred, res, alpha=0.7)
    ax[0].axhline(0, color="r", ls="--")
    ax[0].set(title="Residuals vs predicted", xlabel="Predicted (lakh INR)", ylabel="Actual - predicted")
    ax[1].hist(res, bins=20, color="gray")
    ax[1].set(title="Residual distribution", xlabel="Error (lakh INR)")
    _save(fig, out, "06_residuals.png")


def chart_importance(imp, out):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="teal")
    ax.set(title="Permutation importance on test data", xlabel="Increase in MAE (lakh INR) when the feature is shuffled")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out, "07_feature_importance.png")


# Prediction with input validation
def predict_price(bundle: dict, car: dict) -> float:
    """Predict the resale price (lakh INR) for one car; raises ValueError on bad input."""
    need = ["Car_Name", "Year", "Present_Price", "Driven_kms", "Fuel_Type",
            "Selling_type", "Transmission", "Owner"]
    missing = [k for k in need if k not in car]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    try:
        row = {
            "Car_Name": str(car["Car_Name"]).strip(),
            "Year": int(car["Year"]),
            "Present_Price": float(car["Present_Price"]),
            "Driven_kms": float(car["Driven_kms"]),
            "Fuel_Type": str(car["Fuel_Type"]).strip(),
            "Selling_type": str(car["Selling_type"]).strip(),
            "Transmission": str(car["Transmission"]).strip(),
            "Owner": int(car["Owner"]),
        }
    except (TypeError, ValueError):
        raise ValueError("Year, Owner, Present_Price and Driven_kms must be numeric.")
    ref = bundle["ref_year"]
    if not row["Car_Name"]:
        raise ValueError("Car_Name cannot be empty.")
    if not 1980 <= row["Year"] <= ref + 1:
        raise ValueError(f"Year must be between 1980 and {ref + 1} (the data's latest year is {ref}).")
    if not np.isfinite(row["Present_Price"]) or row["Present_Price"] <= 0:
        raise ValueError("Present_Price must be a positive number.")
    if not np.isfinite(row["Driven_kms"]) or row["Driven_kms"] < 0:
        raise ValueError("Driven_kms must be zero or more.")
    if row["Owner"] < 0:
        raise ValueError("Owner cannot be negative.")
    for field, allowed in bundle["categories"].items():
        canon = {a.lower(): a for a in allowed}  # accept 'petrol', 'CNG', 'cng', ...
        if row[field].lower() not in canon:
            raise ValueError(f"{field} must be one of {sorted(allowed)}; got '{row[field]}'.")
        row[field] = canon[row[field].lower()]
    X = engineer(pd.DataFrame([row]), ref)[ALL_FEATURES]
    return float(bundle["model"].predict(X)[0])


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Car price prediction")
    p.add_argument("--csv", default="Car_Price_Data.csv")
    p.add_argument("--out", default="car_price_output")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search")
    p.add_argument("--predict-json", help="JSON with one car's details to price after training")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        log.error("--test-size must be between 0.05 and 0.5.")
        return 1
    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output folder '%s': %s", out, exc)
        return 1

    try:
        #load & clean
        raw = read_csv_safely(find_csv(args.csv))
        df = clean(raw, current_year=pd.Timestamp.now().year)
        ref_year = int(df["Year"].max())
        df = engineer(df, ref_year)
        unknown_names(df)
        report_outliers(df)
        log.info("Clean data: %d rows | listing years %d-%d | reference year %d",
                 len(df), df["Year"].min(), ref_year, ref_year)

        print("\n" + "=" * 64 + "\nDATA OVERVIEW\n" + "=" * 64)
        print("Vehicle types:", df["Vehicle_Type"].value_counts().to_dict())
        print("Brands (top 8):", df["Brand"].value_counts().head(8).to_dict())
        print(df[[TARGET, "Present_Price", "Driven_kms", "Car_Age"]].describe().round(2).to_string())

        safe_chart(chart_price_distribution, df, out)
        safe_chart(chart_age_vs_price, df, out)
        safe_chart(chart_goodwill, df, out)

        #split
        X, y = df[ALL_FEATURES], df[TARGET]
        strat = df["Vehicle_Type"] if df["Vehicle_Type"].value_counts().min() >= 2 else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=strat)
        except ValueError as exc:
            raise DataError(f"Could not split data: {exc}")
        log.info("Train: %d | Test: %d", len(X_tr), len(X_te))

        #compare, tune, fit
        cv_rows = compare_models(candidate_models(), X_tr, y_tr)
        safe_chart(chart_model_comparison, cv_rows, out)
        best_name = min(cv_rows, key=lambda n: cv_rows[n]["MAE"])
        log.info("Best by cross-validated MAE: %s", best_name)
        base = candidate_models()[best_name]
        model = base.fit(X_tr, y_tr) if args.no_tune else tune(best_name, base, X_tr, y_tr)

        #evaluate
        pred = np.clip(model.predict(X_te), 0, None)
        m = metrics(y_te, pred)
        print("\n" + "=" * 64 + f"\nTEST SET EVALUATION - {best_name}\n" + "=" * 64)
        print(f"R2 score        : {m['R2']:.4f}")
        print(f"MAE             : {m['MAE']:.3f} lakh INR (average error)")
        print(f"RMSE            : {m['RMSE']:.3f} lakh INR (penalises big misses)")
        print(f"Median % error  : {m['MedianAPE_%']:.1f}%")
        by_type = {}
        for vt in X_te["Vehicle_Type"].unique():
            mask = (X_te["Vehicle_Type"] == vt).values
            if mask.sum() >= 3:
                by_type[vt] = metrics(y_te.values[mask], pred[mask])
                print(f"  {vt:<11} (n={mask.sum():>2}) MAE = {by_type[vt]['MAE']:.3f} | "
                      f"R2 = {by_type[vt]['R2']:.3f}")

        safe_chart(chart_actual_vs_pred, y_te.values, pred, out)
        safe_chart(chart_residuals, y_te.values, pred, out)
        try:
            pi = permutation_importance(model, X_te, y_te, n_repeats=20,
                                        random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
            imp = pd.Series(pi.importances_mean, index=X_te.columns)
            print("\nMost important features (rise in test MAE, lakh INR, when shuffled):")
            print(imp.sort_values(ascending=False).round(4).to_string())
            safe_chart(chart_importance, imp, out)
        except Exception as exc:
            log.warning("Feature importance skipped: %s", exc)

        #save
        bundle = {
            "model": model, "ref_year": ref_year, "features": ALL_FEATURES,
            "categories": {c: set(df[c]) for c in ("Fuel_Type", "Selling_type", "Transmission")},
            "test_metrics": m,
        }
        try:
            joblib.dump(bundle, out / "car_price_model.joblib")
            log.info("Saved model to %s", out / "car_price_model.joblib")
            pd.DataFrame({"Actual": y_te.values, "Predicted": pred.round(3)},
                         index=X_te.index).to_csv(out / "test_predictions.csv")
        except OSError as exc:
            log.warning("Could not save model/predictions: %s", exc)

        #sample predictions
        print("\n" + "=" * 64 + "\nSAMPLE PREDICTIONS (lakh INR)\n" + "=" * 64)
        samples = [
            {"Car_Name": "city", "Year": 2015, "Present_Price": 9.5, "Driven_kms": 40000,
             "Fuel_Type": "Petrol", "Selling_type": "Dealer", "Transmission": "Manual", "Owner": 0},
            {"Car_Name": "fortuner", "Year": 2014, "Present_Price": 30.6, "Driven_kms": 60000,
             "Fuel_Type": "Diesel", "Selling_type": "Dealer", "Transmission": "Automatic", "Owner": 0},
            {"Car_Name": "Bajaj Pulsar 150", "Year": 2013, "Present_Price": 0.8, "Driven_kms": 30000,
             "Fuel_Type": "Petrol", "Selling_type": "Individual", "Transmission": "Manual", "Owner": 0},
        ]
        for s in samples:
            print(f"{s['Car_Name']:<18} {s['Year']}  ->  {predict_price(bundle, s):.2f}")
        try:
            predict_price(bundle, {**samples[0], "Year": "abc"})
        except ValueError as exc:
            print(f"Invalid input handled correctly: {exc}")

        if args.predict_json:
            try:
                car = json.loads(args.predict_json)
                print(f"\nYour car -> predicted price: {predict_price(bundle, car):.2f} lakh INR")
            except json.JSONDecodeError as exc:
                log.error("--predict-json is not valid JSON: %s", exc)
            except ValueError as exc:
                log.error("Prediction input problem: %s", exc)

        print(f"\nOutputs saved in: {out.resolve()}")
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
