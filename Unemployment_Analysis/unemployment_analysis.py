import argparse
import logging
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency: {exc.name}. Install with:\n"
             "  pip install pandas numpy matplotlib")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("unemployment")

FILE_A = "Unemployment_in_India.csv"
FILE_B = "Unemployment_Rate_upto_11_2020.csv"

# canonical column names
UR, EMP, LFPR = "UnemploymentRate", "Employed", "LabourParticipation"
COLMAP = {
    "region": "Region", "state": "Region",
    "date": "Date", "frequency": "Frequency",
    "estimated unemployment rate (%)": UR,
    "estimated employed": EMP,
    "estimated labour participation rate (%)": LFPR,
    "area": "Area",
    "region.1": "Zone", "longitude": "Longitude", "latitude": "Latitude",
}
NUMERIC = [UR, EMP, LFPR]

# Covid phases (India's national lockdown began 25 March 2020)
PRE_END = pd.Timestamp("2020-03-31")
LOCK_START, LOCK_END = pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-31")


class DataError(Exception):
    """Raised when input data is missing, malformed or unusable."""


# Loading & cleaning
def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if not path.is_file():
        raise DataError(f"Not a file: {path}")
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):  # utf-8-sig strips the BOM
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
        except pd.errors.EmptyDataError:
            raise DataError(f"CSV is empty: {path.name}")
        except pd.errors.ParserError as exc:
            raise DataError(f"Cannot parse {path.name}: {exc}")
        except PermissionError:
            raise DataError(f"No permission to read {path.name}")
    else:
        raise DataError(f"Cannot decode {path.name}: {last_err}")
    if df.empty:
        raise DataError(f"CSV has no rows: {path.name}")
    return df


def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Standardise column names, types and remove unusable rows."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: COLMAP[c.lower()] for c in df.columns if c.lower() in COLMAP})

    missing = [c for c in ("Region", "Date", UR) if c not in df.columns]
    if missing:
        raise DataError(f"{name}: missing required columns {missing}. Found {list(df.columns)}")

    n0 = len(df)
    df = df.dropna(how="all")
    df = df.dropna(subset=["Region", "Date", UR])
    if len(df) < n0:
        log.warning("%s: dropped %d blank/incomplete rows.", name, n0 - len(df))

    # strip stray whitespace in text columns
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # dates: the files use dd-mm-yyyy
    parsed = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(df["Date"], errors="coerce", dayfirst=True))
    bad = int(parsed.isna().sum())
    if bad:
        log.warning("%s: dropped %d rows with unreadable dates.", name, bad)
    df["Date"] = parsed
    df = df.dropna(subset=["Date"])

    for c in [c for c in NUMERIC if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=[UR])
    if len(df) < n0:
        log.warning("%s: dropped %d rows with non-numeric unemployment values.", name, n0 - len(df))

    # plausibility checks
    out_of_range = ~df[UR].between(0, 100)
    if out_of_range.any():
        log.warning("%s: removed %d rows with unemployment outside 0-100%%.",
                    name, int(out_of_range.sum()))
        df = df[~out_of_range]

    key = [c for c in ("Region", "Date", "Area") if c in df.columns]
    dups = int(df.duplicated(key).sum())
    if dups:
        log.warning("%s: removed %d duplicate rows.", name, dups)
        df = df.drop_duplicates(key)

    if "Frequency" in df.columns:
        df["Frequency"] = df["Frequency"].replace({"M": "Monthly"})
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    if df.empty:
        raise DataError(f"{name}: no usable rows after cleaning.")
    log.info("%s: %d clean rows | %d regions | %s to %s", name, len(df),
             df["Region"].nunique(), df["Date"].min().date(), df["Date"].max().date())
    return df.reset_index(drop=True)


def load(path: Path, name: str):
    """Load+clean one file; return None (not crash) if it fails."""
    try:
        return clean(read_csv_safely(path), name)
    except DataError as exc:
        log.error("%s", exc)
        return None


# Analysis helpers
def phase(date: pd.Timestamp) -> str:
    if date <= PRE_END:
        return "Pre-Covid"
    if date <= LOCK_END:
        return "Lockdown"
    return "Post-lockdown"


def national_series(df: pd.DataFrame) -> pd.Series:
    """Simple mean of state rates per month (employment-weighting not possible
    because the labour force is not in the data)."""
    return df.groupby("Month")[UR].mean()


def covid_impact_by_state(a: pd.DataFrame) -> pd.DataFrame:
    d = a.assign(Phase=a["Date"].map(phase))
    t = d.pivot_table(index="Region", columns="Phase", values=UR, aggfunc="mean")
    for col in ("Pre-Covid", "Lockdown"):
        if col not in t.columns:
            raise DataError(f"Cannot compute Covid impact: no data for '{col}' phase.")
    incomplete = t[t["Pre-Covid"].isna() | t["Lockdown"].isna()].index.tolist()
    if incomplete:
        log.warning("Excluded from Covid-impact ranking (missing pre-Covid or lockdown data): %s",
                    ", ".join(incomplete))
    t = t.dropna(subset=["Pre-Covid", "Lockdown"]).copy()
    if t.empty:
        raise DataError("No state has both pre-Covid and lockdown data.")
    t.attrs["excluded"] = incomplete
    t["Increase_pp"] = t["Lockdown"] - t["Pre-Covid"]
    if "Post-lockdown" in t.columns:
        t["Recovery_pp"] = t["Lockdown"] - t["Post-lockdown"]
    return t.sort_values("Increase_pp", ascending=False)


def safe_step(label, func, *args, **kwargs):
    """Run one analysis step; a failure is logged and skipped, not fatal."""
    try:
        return func(*args, **kwargs)
    except DataError as exc:
        log.error("%s skipped: %s", label, exc)
    except Exception as exc:
        log.exception("%s failed unexpectedly: %s", label, exc)
    return None


# Charts
def save(fig, out: Path, name: str):
    try:
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
    finally:
        plt.close(fig)
    log.info("Saved chart: %s", name)


def shade_lockdown(ax):
    ax.axvspan(LOCK_START, LOCK_END, color="red", alpha=0.12, label="Lockdown (Apr-May 2020)")


def chart_national(a, b, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    sa = national_series(a)
    ax.plot(sa.index, sa.values, marker="o", label="File 1: May 2019 - Jun 2020")
    if b is not None:
        sb = national_series(b)
        ax.plot(sb.index, sb.values, marker="s", ls="--", label="File 2: Jan - Oct 2020")
    shade_lockdown(ax)
    ax.set(title="Average state unemployment rate over time", ylabel="Unemployment rate (%)")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, out, "01_national_trend.png")


def chart_rural_urban(a, out):
    if "Area" not in a.columns:
        raise DataError("No 'Area' column for rural/urban split.")
    fig, ax = plt.subplots(figsize=(11, 5))
    for area, g in a.groupby("Area"):
        s = national_series(g)
        ax.plot(s.index, s.values, marker="o", label=area)
    shade_lockdown(ax)
    ax.set(title="Rural vs urban unemployment", ylabel="Unemployment rate (%)")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, out, "02_rural_vs_urban.png")


def chart_state_impact(impact, out):
    top = impact.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top["Increase_pp"], color="firebrick")
    ax.set(title="Covid shock: rise in unemployment (lockdown vs pre-Covid avg)",
           xlabel="Percentage-point increase")
    ax.grid(axis="x", alpha=0.3)
    save(fig, out, "03_state_covid_impact.png")


def chart_heatmap(a, out):
    p = a.pivot_table(index="Region", columns="Month", values=UR, aggfunc="mean")
    p = p.loc[p.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(p.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index, fontsize=8)
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels([d.strftime("%b %y") for d in p.columns], rotation=60, fontsize=8)
    fig.colorbar(im, ax=ax, label="Unemployment rate (%)")
    ax.set_title("Unemployment rate by state and month")
    save(fig, out, "04_state_month_heatmap.png")


def chart_zones(b, out):
    if b is None or "Zone" not in b.columns:
        raise DataError("Zone data (file 2) not available.")
    fig, ax = plt.subplots(figsize=(11, 5))
    for z, g in b.groupby("Zone"):
        s = national_series(g)
        ax.plot(s.index, s.values, marker="o", label=z)
    shade_lockdown(ax)
    ax.set(title="Unemployment by geographic zone, 2020", ylabel="Unemployment rate (%)")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, out, "05_zone_trends.png")


def chart_seasonality(a, out):
    pre = a[a["Date"] <= PRE_END]
    if pre["Month"].nunique() < 6:
        raise DataError("Too little pre-Covid data for a seasonal profile.")
    m = pre.groupby("Month")[UR].mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(m)), m.values, color="steelblue")
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels([d.strftime("%b %y") for d in m.index], rotation=45)
    ax.axhline(m.mean(), color="k", ls="--", lw=1, label=f"Pre-Covid mean {m.mean():.1f}%")
    ax.set(title="Pre-Covid monthly profile (May 2019 - Mar 2020, single cycle)",
           ylabel="Unemployment rate (%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    save(fig, out, "06_pre_covid_monthly_profile.png")


def chart_lfpr(a, out):
    if LFPR not in a.columns:
        raise DataError("No labour participation column.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pre = a[a["Date"] <= PRE_END]
    axes[0].scatter(pre[LFPR], pre[UR], alpha=0.5, s=18)
    axes[0].set(title="Pre-Covid: participation vs unemployment",
                xlabel="Labour participation rate (%)", ylabel="Unemployment rate (%)")
    s = a.groupby("Month")[[UR, LFPR]].mean()
    axes[1].plot(s.index, s[LFPR], color="green", marker="o", label="Participation")
    ax2 = axes[1].twinx()
    ax2.plot(s.index, s[UR], color="firebrick", marker="s", label="Unemployment")
    axes[1].set(title="Participation fell as unemployment rose", ylabel="Participation (%)")
    ax2.set_ylabel("Unemployment (%)")
    axes[1].tick_params(axis="x", rotation=45)
    save(fig, out, "07_participation_vs_unemployment.png")


def chart_employed(a, out):
    if EMP not in a.columns:
        raise DataError("No employment column.")
    s = a.groupby("Month")[EMP].sum() / 1e6
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s.index, s.values, marker="o", color="teal")
    shade_lockdown(ax)
    ax.set(title="Total estimated employed (sum across state-area rows)", ylabel="Millions")
    ax.grid(alpha=0.3)
    save(fig, out, "08_employed_total.png")


# Insights / report
def build_report(a, b, impact) -> str:
    L = ["# Unemployment in India: Covid-19 impact analysis", ""]
    excluded = impact.attrs.get("excluded", []) if impact is not None else []
    sa = national_series(a)
    pre = sa[sa.index <= PRE_END]
    lock = sa[(sa.index >= LOCK_START) & (sa.index <= LOCK_END)]
    post = sa[sa.index > LOCK_END]

    L += ["## Key findings", ""]
    if len(pre) and len(lock):
        L.append(f"- **National shock:** the average state unemployment rate was "
                 f"{pre.mean():.1f}% before Covid (May 2019 - Mar 2020) and "
                 f"{lock.mean():.1f}% during the Apr-May 2020 lockdown "
                 f"(peak {lock.max():.1f}% in {lock.idxmax():%b %Y}), "
                 f"about {lock.mean() / pre.mean():.1f}x higher.")
    if len(post):
        L.append(f"- **Rebound:** by {post.index[-1]:%b %Y} the rate was {post.iloc[-1]:.1f}%, "
                 f"still {post.iloc[-1] - pre.mean():+.1f} pp versus the pre-Covid average.")
    if b is not None:
        sb = national_series(b)
        base_b = sb[sb.index <= PRE_END]
        base_txt = (f", against its own Jan-Mar 2020 average of {base_b.mean():.1f}%"
                    if len(base_b) else "")
        L.append(f"- **Later 2020 (file 2):** the rate peaked at {sb.max():.1f}% in "
                 f"{sb.idxmax():%b %Y} and was {sb.iloc[-1]:.1f}% by {sb.index[-1]:%b %Y}"
                 f"{base_txt}. File 2 shows the labour market recovering by October.")
    if impact is not None and len(impact):
        top = impact.head(5)
        L.append("- **Hardest-hit states (rise in pp, lockdown vs pre-Covid):** " +
                 ", ".join(f"{r} ({v:+.1f})" for r, v in top["Increase_pp"].items()) + ".")
        low = impact.tail(3)
        L.append("- **Smallest change (negative = fell):** " +
                 ", ".join(f"{r} ({v:+.1f})" for r, v in low["Increase_pp"].items()) + ".")
    if "Area" in a.columns:
        ar = a.assign(Phase=a["Date"].map(phase)).pivot_table(
            index="Area", columns="Phase", values=UR, aggfunc="mean")
        if {"Pre-Covid", "Lockdown"} <= set(ar.columns):
            parts = [f"{i}: {r['Pre-Covid']:.1f}% -> {r['Lockdown']:.1f}%" for i, r in ar.iterrows()]
            rise = (ar["Lockdown"] - ar["Pre-Covid"]).sort_values(ascending=False)
            rural_urban_note = (f"{rise.index[0]} areas saw the larger rise "
                                f"({rise.iloc[0]:+.1f} pp vs {rise.iloc[-1]:+.1f} pp for {rise.index[-1]}).")
            L.append("- **Rural vs urban:** " + "; ".join(parts) + ". " + rural_urban_note)
    if LFPR in a.columns:
        ms = a.groupby("Month")[LFPR].mean()
        L.append(f"- **Participation:** the average labour participation rate fell from "
                 f"{ms[ms.index <= PRE_END].mean():.1f}% pre-Covid to "
                 f"{ms[(ms.index >= LOCK_START) & (ms.index <= LOCK_END)].mean():.1f}% in the lockdown, "
                 f"consistent with many people leaving the labour force, not only losing jobs.")
    if b is not None and "Zone" in b.columns:
        z = b[(b["Date"] >= LOCK_START) & (b["Date"] <= LOCK_END)].groupby("Zone")[UR].mean()
        L.append("- **Zones during lockdown (file 2):** " +
                 ", ".join(f"{k} {v:.1f}%" for k, v in z.sort_values(ascending=False).items()) + ".")

    if len(pre) >= 6:
        slope = np.polyfit(np.arange(len(pre)), pre.values, 1)[0]
        direction = "upward" if slope > 0 else "downward"
        trend_txt = (f"- Before Covid the national rate moved {direction} by about {abs(slope):.2f} pp "
                     f"per month (see chart 06). That is a trend, not proof of a seasonal cycle.")
    else:
        trend_txt = "- Too few pre-Covid months to describe a trend."
    L += ["", "## Seasonal patterns (read with caution)", "",
          f"- The data holds only {len(pre)} pre-Covid months, one cycle, so true seasonality "
          "(recurring year-on-year patterns) **cannot be confirmed**.",
          trend_txt,
          "- Multi-year data (for example the same months in 2018 and 2021) is needed to separate "
          "seasonal effects such as harvest and festival hiring from the Covid shock.", ""]

    L += ["## Policy implications", "",
          "- **Targeted relief:** states with the largest jumps should get priority for income "
          "support and wage-subsidy schemes in future shocks.",
          "- **Both rural and urban need cover:** unemployment rose sharply in both; see the "
          "rural vs urban finding above for which was hit harder. Urban employment support "
          "for informal and migrant workers and scalable rural public works both deserve consideration.",
          "- **Watch participation, not only unemployment:** a falling participation rate hides "
          "discouraged workers, so policy should track both measures.",
          "- **Early-warning data:** monthly state-level data like this lets authorities spot "
          "stress within weeks instead of waiting for annual surveys.", "",
          "## Data notes and limitations", "",
          "- National figures are **simple averages of state rates**, not labour-force-weighted.",
          "- The two files overlap in Jan-Jun 2020 but have different structure "
          "(file 1 splits Rural/Urban; file 2 is overall by state), so their numbers are "
          "not merged or directly compared.",
          f"- Excluded from the state ranking for lacking pre-Covid or lockdown data: "
          f"{', '.join(excluded) if excluded else 'none'}. Blank rows in file 1 were removed.",
          "- A few state-months are extreme (for example Puducherry's lockdown figures); "
          "treat single-state spikes cautiously, as small samples can exaggerate swings.",
          "- Correlation here does not prove causation; the lockdown overlaps with other "
          "Covid effects (migration, supply disruption).", ""]
    return "\n".join(L)


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Unemployment analysis (India, Covid-19)")
    p.add_argument("--data-dir", default=".", help="Folder containing the two CSV files")
    p.add_argument("--file-a", default=FILE_A)
    p.add_argument("--file-b", default=FILE_B)
    p.add_argument("--out", default="unemployment_output", help="Output folder")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output folder '%s': %s", out, exc)
        return 1

    try:
        data_dir = Path(args.data_dir)
        a = load(data_dir / args.file_a, "File 1")
        b = load(data_dir / args.file_b, "File 2")
        if a is None and b is None:
            raise DataError("Neither input file could be loaded.")
        if a is None:
            log.warning("File 1 unavailable: Covid-impact analysis needs pre-Covid data, "
                        "so using file 2 only for what it supports.")
            a, b = b, None  # run the generic parts on whatever we have
        if b is None:
            log.warning("File 2 unavailable: zone and post-lockdown (to Oct 2020) analyses skipped.")

        # save cleaned data
        for df, name in ((a, "cleaned_primary.csv"), (b, "cleaned_secondary.csv")):
            if df is not None:
                try:
                    df.to_csv(out / name, index=False)
                except OSError as exc:
                    log.warning("Could not save %s: %s", name, exc)

        # quick exploration printout
        print("\n" + "=" * 60 + "\nDATA OVERVIEW\n" + "=" * 60)
        print(f"Rows: {len(a)} | Regions: {a['Region'].nunique()} | "
              f"{a['Date'].min():%b %Y} to {a['Date'].max():%b %Y}")
        print(a[[c for c in NUMERIC if c in a.columns]].describe().round(2).to_string())

        impact = safe_step("Covid impact by state", covid_impact_by_state, a)
        if impact is not None:
            print("\nTop 10 states by Covid-driven rise (pp):")
            print(impact[["Pre-Covid", "Lockdown", "Increase_pp"]].head(10).round(2).to_string())
            try:
                impact.round(2).to_csv(out / "covid_impact_by_state.csv")
            except OSError as exc:
                log.warning("Could not save impact table: %s", exc)

        safe_step("Chart 01", chart_national, a, b, out)
        safe_step("Chart 02", chart_rural_urban, a, out)
        if impact is not None:
            safe_step("Chart 03", chart_state_impact, impact, out)
        safe_step("Chart 04", chart_heatmap, a, out)
        safe_step("Chart 05", chart_zones, b, out)
        safe_step("Chart 06", chart_seasonality, a, out)
        safe_step("Chart 07", chart_lfpr, a, out)
        safe_step("Chart 08", chart_employed, a, out)

        report = safe_step("Report", build_report, a, b, impact)
        if report:
            try:
                (out / "report.md").write_text(report, encoding="utf-8")
                log.info("Saved report.md")
            except OSError as exc:
                log.error("Could not write report: %s", exc)
            print("\n" + report)

        print(f"\nAll outputs saved in: {out.resolve()}")
        return 0

    except DataError as exc:
        log.error("Data problem: %s", exc)
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
    return 1


if __name__ == "__main__":
    sys.exit(main())
