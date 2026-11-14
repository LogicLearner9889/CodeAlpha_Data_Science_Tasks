import argparse
import json
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
    from scipy import stats
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import (KFold, RepeatedKFold, cross_val_predict,
                                         cross_validate, train_test_split)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency: {exc.name}. Install with:\n"
             "  pip install pandas numpy scipy scikit-learn matplotlib joblib")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("sales")

RANDOM_STATE = 42
CHANNELS = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
INTERACTION = "TV_x_Radio"
TV_SQ, RADIO_SQ = "TV_sq", "Radio_sq"
ALL_TERMS = CHANNELS + [TV_SQ, RADIO_SQ, INTERACTION]
MIN_ROWS = 30
ALPHA = 0.05


class DataError(Exception):
    """Raised when data is missing, malformed or unusable."""


# Loading & cleaning
def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"CSV file not found: {path}")
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # match required columns case-insensitively
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for need in CHANNELS + [TARGET]:
        if need.lower() in lower:
            rename[lower[need.lower()]] = need
    df = df.rename(columns=rename)
    missing = [c for c in CHANNELS + [TARGET] if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    ignored = [c for c in df.columns if c not in CHANNELS + [TARGET]]
    if ignored:
        log.info("Ignoring non-model columns (e.g. row index): %s", ignored)
    df = df[CHANNELS + [TARGET]]

    for c in CHANNELS + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < n0:
        log.warning("Dropped %d row(s) with missing/non-numeric/infinite values.", n0 - len(df))

    neg = (df[CHANNELS + [TARGET]] < 0).any(axis=1)
    if neg.any():
        log.warning("Dropped %d row(s) with negative spend or sales.", int(neg.sum()))
        df = df[~neg]

    n0 = len(df)
    df = df.drop_duplicates()
    if len(df) < n0:
        log.warning("Dropped %d duplicate row(s).", n0 - len(df))

    if len(df) < MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows after cleaning; need at least {MIN_ROWS}.")
    for c in CHANNELS:
        if df[c].nunique() < 3:
            raise DataError(f"'{c}' has fewer than 3 distinct values; cannot model its effect.")
    if df[TARGET].std() == 0:
        raise DataError("Sales is constant; nothing to predict.")
    return df.reset_index(drop=True)


def flag_outliers(df: pd.DataFrame) -> None:
    """Report (but keep) values beyond 3 IQRs; they are plausible spend levels."""
    for c in CHANNELS + [TARGET]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 3 * iqr) | (df[c] > q3 + 3 * iqr)).sum())
        if n:
            log.info("%d extreme value(s) in %s kept (beyond 3 IQR).", n, c)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering: synergy between TV and Radio (interaction) and
    diminishing returns (squared terms)."""
    out = df.copy()
    out[INTERACTION] = out["TV"] * out["Radio"]
    out[TV_SQ] = out["TV"] ** 2
    out[RADIO_SQ] = out["Radio"] ** 2
    return out


# Small OLS implementation (coefficients, p-values, CIs) - no statsmodels needed
def fit_ols(df: pd.DataFrame, terms: list) -> dict:
    X = np.column_stack([np.ones(len(df))] + [df[t].values for t in terms])
    y = df[TARGET].values
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise DataError("Regression matrix is singular (perfectly collinear features).")
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        raise DataError("Not enough rows for the number of model terms.")
    s2 = float(resid @ resid) / dof
    cov = s2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    crit = stats.t.ppf(0.975, dof)
    ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    names = ["Intercept"] + list(terms)
    table = pd.DataFrame({"coef": beta, "std_err": se, "t": t, "p_value": p,
                          "ci_low": beta - crit * se, "ci_high": beta + crit * se}, index=names)
    return {"terms": list(terms), "names": names, "beta": beta, "cov": cov, "dof": dof,
            "table": table, "r2": r2, "adj_r2": 1 - (1 - r2) * (len(y) - 1) / dof,
            "resid_std": float(np.sqrt(s2)), "crit": crit}


def select_terms(df: pd.DataFrame) -> list:
    """Backward elimination at 5% significance. A main effect is kept while any term built
    from it (its square or the interaction) is kept, so the model stays well-formed."""
    terms = list(ALL_TERMS)
    while True:
        res = fit_ols(df, terms)
        p = res["table"]["p_value"].drop("Intercept")

        def droppable(t):
            if t == "TV":
                return not ({TV_SQ, INTERACTION} & set(terms))
            if t == "Radio":
                return not ({RADIO_SQ, INTERACTION} & set(terms))
            return True
        cand = p[[t for t in terms if droppable(t)]]
        if cand.empty or cand.max() <= ALPHA:
            return terms
        terms = [t for t in terms if t != cand.idxmax()]
        if not terms:
            raise DataError("No feature is statistically significant.")


def ols_predict(res: dict, df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(df))] + [df[t].values for t in res["terms"]])
    return X @ res["beta"]


def marginal_effect(res: dict, channel: str, at: dict):
    """d Sales / d channel at the given spend levels, with a 95% CI (delta method)."""
    names, c = res["names"], np.zeros(len(res["names"]))
    if channel in names:
        c[names.index(channel)] = 1.0
    sq = {"TV": TV_SQ, "Radio": RADIO_SQ}.get(channel)
    if sq in names:
        c[names.index(sq)] = 2 * at[channel]
    if INTERACTION in names and channel in ("TV", "Radio"):
        other = "Radio" if channel == "TV" else "TV"
        c[names.index(INTERACTION)] = at[other]
    eff = float(c @ res["beta"])
    se = float(np.sqrt(c @ res["cov"] @ c))
    return eff, eff - res["crit"] * se, eff + res["crit"] * se


# Modelling
def cols_model(cols, estimator) -> Pipeline:
    sel = ColumnTransformer([("keep", "passthrough", list(cols))], remainder="drop")
    return Pipeline([("select", sel), ("model", estimator)])


def candidate_models() -> dict:
    base = ["TV", "Radio"]
    return {
        "Linear: TV+Radio+Newspaper": cols_model(CHANNELS, LinearRegression()),
        "Linear: TV+Radio": cols_model(base, LinearRegression()),
        "Linear: TV+Radio+TVxRadio": cols_model(base + [INTERACTION], LinearRegression()),
        "Linear: all + TVxRadio": cols_model(CHANNELS + [INTERACTION], LinearRegression()),
        "Linear: TV+Radio+TV^2+TVxRadio": cols_model(base + [TV_SQ, INTERACTION], LinearRegression()),
        "Ridge poly(2), 3 channels": Pipeline([
            ("select", ColumnTransformer([("keep", "passthrough", CHANNELS)], remainder="drop")),
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0))]),
        "Random Forest": cols_model(CHANNELS, RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE)),
        "Gradient Boosting": cols_model(CHANNELS, GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)),
    }


def compare_models(models, X, y) -> pd.DataFrame:
    n_splits = int(min(5, len(X)))
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=3, random_state=RANDOM_STATE)
    rows = {}
    print("\n" + "=" * 70 + f"\nMODEL COMPARISON ({n_splits}-fold x3 repeated CV, training data)\n" + "=" * 70)
    for name, model in models.items():
        try:
            r = cross_validate(model, X, y, cv=cv, scoring={
                "r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"})
            rows[name] = {"R2": r["test_r2"].mean(), "RMSE": -r["test_rmse"].mean(),
                          "RMSE_sd": r["test_rmse"].std(), "MAE": -r["test_mae"].mean()}
            print(f"{name:<28} R2 = {rows[name]['R2']:.4f} | RMSE = {rows[name]['RMSE']:.3f} "
                  f"(+/- {rows[name]['RMSE_sd']:.3f}) | MAE = {rows[name]['MAE']:.3f}")
        except Exception as exc:
            log.error("Model '%s' failed during cross-validation: %s", name, exc)
    if not rows:
        raise RuntimeError("All candidate models failed during cross-validation.")
    return pd.DataFrame(rows).T


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))}

# Budget optimisation (on the interpretable OLS model, within observed ranges)
def optimise_budget(res: dict, df: pd.DataFrame, budget: float, step: float = 0.5):
    lo = {c: float(df[c].min()) for c in CHANNELS}
    hi = {c: float(df[c].max()) for c in CHANNELS}
    if budget < sum(lo.values()) or budget > sum(hi.values()):
        raise DataError(f"Budget {budget:.1f} is outside the range the data covers "
                        f"({sum(lo.values()):.1f} to {sum(hi.values()):.1f}).")
    tv = np.arange(lo["TV"], hi["TV"] + step, step)
    rad = np.arange(lo["Radio"], hi["Radio"] + step, step)
    T, R = np.meshgrid(tv, rad, indexing="ij")
    N = budget - T - R
    ok = (N >= lo["Newspaper"]) & (N <= hi["Newspaper"])
    if not ok.any():
        raise DataError("No feasible allocation within observed ranges for this budget.")
    grid = pd.DataFrame({"TV": T[ok], "Radio": R[ok], "Newspaper": N[ok]})
    grid = engineer(grid)
    pred = ols_predict(res, grid)
    i = int(np.argmax(pred))
    best = grid.iloc[i]
    at_bound = [c for c in CHANNELS if np.isclose(best[c], hi[c], atol=step) or np.isclose(best[c], lo[c], atol=step)]
    return {"alloc": {c: float(best[c]) for c in CHANNELS}, "sales": float(pred[i]), "at_bound": at_bound}


# Prediction with validation
def predict_sales(bundle: dict, tv, radio, newspaper) -> dict:
    """Predict sales (thousand units) for one spend mix; raises ValueError on bad input."""
    try:
        vals = {"TV": float(tv), "Radio": float(radio), "Newspaper": float(newspaper)}
    except (TypeError, ValueError):
        raise ValueError("TV, Radio and Newspaper spend must be numeric.")
    for k, v in vals.items():
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{k} spend must be a finite number >= 0.")
    warns = []
    for k, v in vals.items():
        lo, hi = bundle["ranges"][k]
        if not lo <= v <= hi:
            warns.append(f"{k}={v:g} is outside the training range ({lo:g}-{hi:g}); "
                         "the forecast is an extrapolation and less reliable.")
    X = engineer(pd.DataFrame([vals]))
    pred = float(bundle["model"].predict(X)[0])
    if warns:  # an extrapolation: the cross-validated error band no longer applies
        return {"prediction": max(pred, 0.0), "low": None, "high": None, "warnings": warns}
    low, high = pred + bundle["resid_q"][0], pred + bundle["resid_q"][1]
    return {"prediction": max(pred, 0.0), "low": max(low, 0.0), "high": max(high, 0.0), "warnings": warns}


# Charts (each isolated so one failure never stops the pipeline)
def safe_step(label, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except DataError as exc:
        log.error("%s skipped: %s", label, exc)
    except Exception as exc:
        log.warning("%s skipped due to unexpected error: %s", label, exc)
        plt.close("all")
    return None


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    log.info("Saved chart: %s", name)


def chart_scatter(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, c in zip(axes, CHANNELS):
        ax.scatter(df[c], df[TARGET], alpha=0.65, s=20)
        k, b = np.polyfit(df[c], df[TARGET], 1)
        xs = np.linspace(df[c].min(), df[c].max(), 50)
        ax.plot(xs, k * xs + b, "r-")
        ax.set(title=f"{c} (r = {df[c].corr(df[TARGET]):.2f})", xlabel=f"{c} spend ($k)")
    axes[0].set_ylabel("Sales (thousand units)")
    _save(fig, out, "01_spend_vs_sales.png")


def chart_corr(df, out):
    cols = CHANNELS + [TARGET]
    cm = df[cols].corr().values
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, out, "02_correlation.png")


def chart_models(cv, best, out):
    cv = cv.sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["seagreen" if n == best else "lightsteelblue" for n in cv.index]
    ax.barh(cv.index, cv["RMSE"], xerr=cv["RMSE_sd"], color=colors)
    ax.set(title="Cross-validated error by model (lower is better)", xlabel="RMSE (thousand units)")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out, "03_model_comparison.png")


def chart_test(y_true, pred, out):
    res = np.asarray(y_true) - np.asarray(pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(y_true, pred, alpha=0.7)
    lim = [min(np.min(y_true), np.min(pred)) - 1, max(np.max(y_true), np.max(pred)) + 1]
    ax[0].plot(lim, lim, "r--", label="Perfect")
    ax[0].set(title="Test set: actual vs predicted", xlabel="Actual sales", ylabel="Predicted sales")
    ax[0].legend()
    ax[1].scatter(pred, res, alpha=0.7)
    ax[1].axhline(0, color="r", ls="--")
    ax[1].set(title="Residuals", xlabel="Predicted sales", ylabel="Actual - predicted")
    _save(fig, out, "04_test_performance.png")


def chart_response(res, df, out):
    mean = df[CHANNELS].mean()
    xs = np.linspace(df["TV"].min(), df["TV"].max(), 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    for q, ls in ((0.1, ":"), (0.5, "-"), (0.9, "--")):
        r = float(df["Radio"].quantile(q))
        g = pd.DataFrame({"TV": xs, "Radio": r, "Newspaper": mean["Newspaper"]})
        g = engineer(g)
        ax.plot(xs, ols_predict(res, g), ls, label=f"Radio = {r:.0f}k ({int(q * 100)}th pct)")
    ax.set(title="Predicted sales vs TV spend at different Radio levels",
           xlabel="TV spend ($k)", ylabel="Predicted sales (thousand units)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out, "05_response_curves.png")


def chart_marginal(effects, out):
    names = list(effects)
    vals = [effects[n][0] * 1000 for n in names]
    err = [[(effects[n][0] - effects[n][1]) * 1000 for n in names],
           [(effects[n][2] - effects[n][0]) * 1000 for n in names]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, vals, yerr=err, capsize=6, color=["#1f77b4", "#ff7f0e", "#7f7f7f"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Extra units per extra $1,000 at average spend (95% CI)",
           ylabel="Units per $1,000")
    ax.text(0.99, 0.97, "An interval crossing 0 means no proven effect", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "06_marginal_returns.png")


def chart_allocation(cur, opt, budget, out):
    x = np.arange(len(CHANNELS))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - 0.2, [cur[c] for c in CHANNELS], 0.4, label="Current average mix")
    ax.bar(x + 0.2, [opt[c] for c in CHANNELS], 0.4, label="Model-optimal mix")
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set(title=f"Allocation of a ${budget:.0f}k budget", ylabel="Spend ($k)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "07_budget_allocation.png")


# Report
def build_report(ctx: dict) -> str:
    L = ["# Sales prediction and advertising impact", ""]
    tbl, full = ctx["ols"]["table"], ctx["ols_full"]["table"]
    cv = ctx["cv"]
    best, test = ctx["best"], ctx["test"]

    L += ["## Forecast model", "",
          f"- Best model by cross-validated error: **{best}** "
          f"(CV R2 {cv.loc[best, 'R2']:.3f}, RMSE {cv.loc[best, 'RMSE']:.2f}).",
          f"- Held-out test performance: R2 **{test['R2']:.3f}**, MAE {test['MAE']:.2f}, "
          f"RMSE {test['RMSE']:.2f} thousand units "
          f"(average sales are {ctx['mean_sales']:.1f}).",
          f"- Typical 90% prediction band: about {ctx['resid_q'][0]:+.1f} to {ctx['resid_q'][1]:+.1f} "
          "thousand units around a forecast.", ""]

    add, inter = cv.get("Linear: TV+Radio"), cv.get("Linear: TV+Radio+TVxRadio")
    terms, mean = ctx["ols"]["terms"], ctx["mean_spend"]
    L += ["## How advertising affects sales", ""]
    if add is not None and inter is not None and INTERACTION in terms:
        L.append(f"- **TV and Radio reinforce each other.** Allowing for this synergy lifts cross-validated "
                 f"R2 from {add['R2']:.3f} (additive model) to {inter['R2']:.3f}: the more you spend on one, "
                 f"the more the other pays off.")
    bl = ctx["by_level"]
    if TV_SQ in terms:
        coef = tbl.loc[TV_SQ, "coef"]
        L.append(f"- **TV shows {'diminishing' if coef < 0 else 'increasing'} returns** "
                 f"(squared-spend term {coef:+.5f}, p = {tbl.loc[TV_SQ, 'p_value']:.4f}). "
                 f"Extra units per extra $1,000 on TV: {bl['TV units/$1k'].iloc[0]:.0f} at low TV spend, "
                 f"{bl['TV units/$1k'].iloc[1]:.0f} at median, {bl['TV units/$1k'].iloc[2]:.0f} at high spend.")
    if INTERACTION in terms:
        L.append(f"- **Radio pays off more when TV spend is high:** its return per $1,000 rises from "
                 f"{bl['Radio units/$1k'].iloc[0]:.0f} units at low TV spend to "
                 f"{bl['Radio units/$1k'].iloc[2]:.0f} at high TV spend.")
    if RADIO_SQ not in terms:
        L.append("- **No significant saturation was detected for Radio** within the observed range "
                 "(data are sparse at the highest Radio spend, so this is not proof it never saturates).")
    for ch in ("TV", "Radio"):
        e, lo, hi = ctx["effects"][ch]
        L.append(f"- **{ch}:** about **{e * 1000:.0f} extra units per extra $1,000** at average spend "
                 f"(95% CI {lo * 1000:.0f} to {hi * 1000:.0f}).")
    e, lo, hi = ctx["effects"]["Newspaper"]
    pn = full.loc["Newspaper", "p_value"]
    if "Newspaper" in terms:
        L.append(f"- **Newspaper** has a statistically significant effect: {e * 1000:.0f} extra units per "
                 f"$1,000 (p = {pn:.3f}).")
    else:
        L.append(f"- **Newspaper adds no reliable effect** once TV and Radio are included: the full-model "
                 f"estimate is {e * 1000:.0f} units per $1,000 with a 95% CI of {lo * 1000:.0f} to "
                 f"{hi * 1000:.0f}, which includes zero (p = {pn:.2f}). Its raw correlation with sales "
                 f"({ctx['corr_news']:.2f}) mostly reflects that Newspaper spend moves together with Radio "
                 f"(correlation {ctx['corr_news_radio']:.2f}).")
    L += ["", "Selected regression terms and statistics:", "", "```", tbl.round(5).to_string(), "```", ""]

    L += ["## What-if scenarios (from the average spend mix)", "", "```",
          ctx["scenarios"].round(2).to_string(), "```", ""]

    o = ctx["opt"]
    L += ["## Budget optimisation", "",
          f"- For a ${ctx['budget']:.0f}k budget (the average total today is ${ctx['cur_total']:.0f}k): "
          f"current average mix predicts **{ctx['cur_sales']:.1f}** thousand units; the model-optimal mix "
          f"(TV {o['alloc']['TV']:.0f}k, Radio {o['alloc']['Radio']:.0f}k, "
          f"Newspaper {o['alloc']['Newspaper']:.0f}k) predicts **{o['sales']:.1f}** "
          f"({(o['sales'] / ctx['cur_sales'] - 1) * 100:+.0f}%)."]
    if o["at_bound"]:
        L.append(f"- The optimum sits at the edge of observed spend for: {', '.join(o['at_bound'])}. "
                 "The data cannot show what happens beyond that edge, so treat this as a direction "
                 "to test, not a guaranteed gain.")
    sc = ctx["scenarios"]
    mv = [i for i in sc.index if i.startswith("Move")]
    if mv:
        L.append(f"- A smaller, safer step: **{mv[0].lower()}** is predicted to change sales by "
                 f"{sc.loc[mv[0], 'Change']:+.1f} thousand units ({sc.loc[mv[0], 'Change_%']:+.0f}%).")
    L += ["", "```", ctx["alloc_table"].round(2).to_string(), "```", ""]

    eff = ctx["effects"]
    L += ["## Recommendations", "",
          f"- Radio currently returns about {eff['Radio'][0] / eff['TV'][0]:.1f}x more extra units per "
          f"dollar than TV at average spend ({eff['Radio'][0] * 1000:.0f} vs {eff['TV'][0] * 1000:.0f} per "
          f"$1,000), but TV is still essential: the two work together, and Radio's gain is only "
          f"proven up to the observed maximum (${ctx['radio_max']:.0f}k).",
          "- Do not increase Newspaper spend on the strength of this data; test reductions and redirect "
          "that money to Radio and TV." if "Newspaper" not in ctx["ols"]["terms"] else
          "- Keep all three channels in the plan, since each shows a significant effect.",
          "- Move toward the suggested mix gradually and measure results (A/B or regional tests) before "
          "committing the full budget.",
          "- Re-fit the model each quarter as new campaign data arrives.", "",
          "## Limitations", "",
          "- This is **observational data from 200 markets**, not a controlled experiment, so effects show "
          "association; other factors (price, seasonality, competitors) are not in the data.",
          "- There is **no date column**, so time-series forecasting and seasonality cannot be analysed, "
          "and **no customer-segment column**, so segment effects are not estimated.",
          "- Forecasts outside the observed spend ranges are extrapolations and should be treated with caution.",
          "- Units: spend in thousands of dollars, sales in thousands of units; revenue or profit "
          "per unit is needed to turn unit gains into financial ROI.", ""]
    return "\n".join(L)


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Sales prediction from advertising spend")
    p.add_argument("--csv", default="Advertising.csv")
    p.add_argument("--out", default="sales_output")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--budget", type=float, default=None,
                   help="Total budget in $k for optimisation (default: average total spend)")
    p.add_argument("--predict-json", help='JSON like {"TV":150,"Radio":25,"Newspaper":30}')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        log.error("--test-size must be between 0.05 and 0.5.")
        return 1
    if args.budget is not None and (not np.isfinite(args.budget) or args.budget <= 0):
        log.error("--budget must be a positive number.")
        return 1
    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output folder '%s': %s", out, exc)
        return 1

    try:
        #load & clean
        df = clean(read_csv_safely(Path(args.csv)))
        flag_outliers(df)
        log.info("Clean data: %d rows", len(df))
        try:
            df.to_csv(out / "cleaned_data.csv", index=False)
        except OSError as exc:
            log.warning("Could not save cleaned data: %s", exc)

        print("\n" + "=" * 70 + "\nDATA OVERVIEW\n" + "=" * 70)
        print(df.describe().round(2).to_string())
        print("\nCorrelation with Sales:\n", df.corr()[TARGET].drop(TARGET).round(3).to_string())
        safe_step("Chart 01", chart_scatter, df, out)
        safe_step("Chart 02", chart_corr, df, out)

        #feature engineering + split
        data = engineer(df)
        feats = list(ALL_TERMS)
        X, y = data[feats], data[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size,
                                                  random_state=RANDOM_STATE)
        log.info("Train: %d | Test: %d", len(X_tr), len(X_te))

        #compare models & choose
        cv = compare_models(candidate_models(), X_tr, y_tr)
        best = cv["RMSE"].idxmin()
        log.info("Best model by cross-validated RMSE: %s", best)
        safe_step("Chart 03", chart_models, cv, best, out)

        model = candidate_models()[best].fit(X_tr, y_tr)
        pred = np.clip(model.predict(X_te), 0, None)
        test = metrics(y_te, pred)
        print("\n" + "=" * 70 + f"\nTEST SET EVALUATION - {best}\n" + "=" * 70)
        print(f"R2   : {test['R2']:.4f}\nMAE  : {test['MAE']:.3f} thousand units"
              f"\nRMSE : {test['RMSE']:.3f} thousand units")
        safe_step("Chart 04", chart_test, y_te.values, pred, out)

        #final model on all data + prediction interval
        final = candidate_models()[best].fit(X, y)
        oof = cross_val_predict(candidate_models()[best], X, y,
                                cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        resid_q = [float(np.quantile(y - oof, 0.05)), float(np.quantile(y - oof, 0.95))]
        bundle = {"model": final, "features": feats, "resid_q": resid_q,
                  "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in CHANNELS},
                  "test_metrics": test, "model_name": best}
        try:
            joblib.dump(bundle, out / "sales_model.joblib")
            log.info("Saved model to %s", out / "sales_model.joblib")
        except OSError as exc:
            log.warning("Could not save model: %s", exc)

        #impact analysis (interpretable regression on all data)
        full = fit_ols(data, feats)
        terms = select_terms(data)
        res = fit_ols(data, terms)
        print("\n" + "=" * 70 + "\nREGRESSION (selected terms, all data)\n" + "=" * 70)
        print(res["table"].round(5).to_string())
        print(f"R2 = {res['r2']:.4f} | adj R2 = {res['adj_r2']:.4f} | residual SD = {res['resid_std']:.3f}")
        print("\nFull model p-values (before selection):")
        print(full["table"]["p_value"].round(4).to_string())
        try:
            res["table"].round(5).to_csv(out / "regression_coefficients.csv")
        except OSError as exc:
            log.warning("Could not save coefficients: %s", exc)

        mean = df[CHANNELS].mean().to_dict()
        # channels dropped by selection are described with the full model, so their
        # uncertainty (a CI that includes zero) is shown honestly instead of a fake 0
        effects = {c: marginal_effect(res if c in res["terms"] else full, c, mean) for c in CHANNELS}
        rows = {}
        for q in (0.1, 0.5, 0.9):
            at = dict(mean); at["TV"] = float(df["TV"].quantile(q))
            rows[f"TV at {int(q * 100)}th pct (${at['TV']:.0f}k)"] = {
                "TV units/$1k": marginal_effect(res, "TV", at)[0] * 1000,
                "Radio units/$1k": marginal_effect(res, "Radio", at)[0] * 1000}
        by_level = pd.DataFrame(rows).T
        print("\nMarginal effect at average spend (units per +$1,000):")
        for c, (e, lo, hi) in effects.items():
            print(f"  {c:<10} {e * 1000:7.1f}  (95% CI {lo * 1000:.1f} to {hi * 1000:.1f})")
        print("\nReturn per extra $1,000 at low/median/high TV spend (Radio at average):")
        print(by_level.round(1).to_string())
        safe_step("Chart 05", chart_response, res, data, out)
        safe_step("Chart 06", chart_marginal, effects, out)

        #scenarios
        base_row = engineer(pd.DataFrame([mean]))
        base_sales = float(ols_predict(res, base_row)[0])
        scen = {}
        def run(name, **chg):
            row = dict(mean)
            for c, mult in chg.items():
                row[c] = mean[c] * mult
            d = engineer(pd.DataFrame([row]))
            s = float(ols_predict(res, d)[0])
            scen[name] = {"Sales": s, "Change": s - base_sales, "Change_%": (s / base_sales - 1) * 100}
        run("Baseline (average spend)")
        run("TV +10%", TV=1.1)
        run("Radio +10%", Radio=1.1)
        run("TV +10% and Radio +10%", TV=1.1, Radio=1.1)
        if "Newspaper" in res["terms"]:
            run("Newspaper +10%", Newspaper=1.1)
            run("Newspaper cut to zero", Newspaper=0.0)
        move = min(mean["Newspaper"], 10.0)
        d = pd.DataFrame([{**mean, "Newspaper": mean["Newspaper"] - move, "Radio": mean["Radio"] + move}])
        d = engineer(d)
        s = float(ols_predict(res, d)[0])
        scen[f"Move ${move:.0f}k Newspaper -> Radio"] = {
            "Sales": s, "Change": s - base_sales, "Change_%": (s / base_sales - 1) * 100}
        scenarios = pd.DataFrame(scen).T
        print("\nWhat-if scenarios (thousand units):\n", scenarios.round(2).to_string())
        try:
            scenarios.round(3).to_csv(out / "scenarios.csv")
        except OSError as exc:
            log.warning("Could not save scenarios: %s", exc)

        #budget optimisation
        cur_total = float(sum(mean.values()))
        budget = args.budget if args.budget is not None else cur_total
        opt = safe_step("Budget optimisation", optimise_budget, res, df, budget)
        ctx_opt = None
        if opt:
            scale = budget / cur_total
            cur_mix = {c: mean[c] * scale for c in CHANNELS}
            cur_df = engineer(pd.DataFrame([cur_mix]))
            cur_sales = float(ols_predict(res, cur_df)[0])
            alloc_table = pd.DataFrame({"Current mix ($k)": cur_mix, "Optimal mix ($k)": opt["alloc"]})
            print(f"\nBUDGET OPTIMISATION for ${budget:.0f}k")
            print(alloc_table.round(1).to_string())
            print(f"Predicted sales: current mix {cur_sales:.2f} -> optimal {opt['sales']:.2f}")
            if opt["at_bound"]:
                log.warning("Optimum at edge of observed spend for %s: treat as a direction to test.",
                            ", ".join(opt["at_bound"]))
            safe_step("Chart 07", chart_allocation, cur_mix, opt["alloc"], budget, out)
            ctx_opt = dict(opt=opt, budget=budget, cur_total=cur_total, cur_sales=cur_sales,
                           alloc_table=alloc_table)

        #report
        if ctx_opt:
            report = safe_step("Report", build_report, dict(
                ols=res, ols_full=full, cv=cv, best=best, test=test, effects=effects,
                scenarios=scenarios, by_level=by_level, mean_spend=mean, radio_max=float(df["Radio"].max()), mean_sales=float(df[TARGET].mean()), resid_q=resid_q,
                corr_news=float(df["Newspaper"].corr(df[TARGET])),
                corr_news_radio=float(df["Newspaper"].corr(df["Radio"])), **ctx_opt))
            if report:
                try:
                    (out / "report.md").write_text(report, encoding="utf-8")
                    log.info("Saved report.md")
                except OSError as exc:
                    log.error("Could not write report: %s", exc)

        #sample / user predictions
        print("\n" + "=" * 70 + "\nSAMPLE FORECASTS (thousand units)\n" + "=" * 70)
        for spend in ({"TV": 150, "Radio": 25, "Newspaper": 30},
                      {"TV": 250, "Radio": 40, "Newspaper": 10},
                      {"TV": 50, "Radio": 5, "Newspaper": 60}):
            r = predict_sales(bundle, **{k.lower(): v for k, v in spend.items()})
            print(f"{spend} -> {r['prediction']:.1f}  (90% band {r['low']:.1f} to {r['high']:.1f})")
        try:
            predict_sales(bundle, "abc", 1, 1)
        except ValueError as exc:
            print(f"Invalid input handled correctly: {exc}")

        if args.predict_json:
            try:
                spend = json.loads(args.predict_json)
                r = predict_sales(bundle, spend["TV"], spend["Radio"], spend["Newspaper"])
                band = (f"(90% band {r['low']:.2f} to {r['high']:.2f})" if r["low"] is not None
                        else "(no reliable error band: outside the observed spend range)")
                print(f"\nYour spend mix -> {r['prediction']:.2f} thousand units {band}")
                for w in r["warnings"]:
                    log.warning(w)
            except json.JSONDecodeError as exc:
                log.error("--predict-json is not valid JSON: %s", exc)
            except (KeyError, TypeError):
                log.error("--predict-json must be an object with TV, Radio and Newspaper.")
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
