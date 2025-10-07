# src/q2_build_controls.py
import os, re, glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import argparse

OUT_DIR = os.getenv("OUT_DIR", "out")
DATA_DIR = "./data"
RANK_PATH = os.path.join(OUT_DIR, "country_sector_year_ranking.csv")
Q2_DIR = os.path.join(OUT_DIR, "q2")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(Q2_DIR, exist_ok=True)

# ---------- helpers for country codes ----------
ISO3_TO_ISO2 = {
    "USA":"US","CHN":"CN","JPN":"JP","KOR":"KR","CAN":"CA","AUS":"AU","DEU":"DE",
    "FRA":"FR","ESP":"ES","AUT":"AT","GBR":"GB","ITA":"IT","NLD":"NL","SWE":"SE",
    "NOR":"NO","FIN":"FI","DNK":"DK","CHE":"CH","BRA":"BR","IND":"IN","MEX":"MX",
    "ARG":"AR","ZAF":"ZA","RUS":"RU","TUR":"TR","IDN":"ID","SGP":"SG","MYS":"MY",
    "THA":"TH","VNM":"VN","PHL":"PH","SAU":"SA","ARE":"AE","ISR":"IL"
}
NAME_TO_ISO2 = {
    "United States":"US","United States of America":"US","China":"CN","Japan":"JP",
    "Korea, Rep.":"KR","Republic of Korea":"KR","South Korea":"KR",
    "Canada":"CA","Australia":"AU","Germany":"DE","France":"FR","Spain":"ES",
    "Austria":"AT","Netherlands":"NL","Italy":"IT","United Kingdom":"GB","UK":"GB",
    "Sweden":"SE","Norway":"NO","Finland":"FI","Denmark":"DK","Switzerland":"CH",
    "Brazil":"BR","India":"IN","Mexico":"MX","Argentina":"AR","South Africa":"ZA",
    "Russian Federation":"RU","Russia":"RU","Turkey":"TR","Indonesia":"ID","Israel":"IL",
    "Saudi Arabia":"SA","United Arab Emirates":"AE","Singapore":"SG","Malaysia":"MY",
    "Thailand":"TH","Viet Nam":"VN","Vietnam":"VN","Philippines":"PH"
}

def to_iso2(val):
    if pd.isna(val): return pd.NA
    s = str(val).strip()
    if len(s) == 2: return s.upper()
    if len(s) == 3: return ISO3_TO_ISO2.get(s.upper(), s.upper())
    return NAME_TO_ISO2.get(s, s)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join([str(x) for x in tup if str(x).lower() != "nan"]).strip()
            for tup in df.columns.values
        ]
    else:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- readers ----------
def read_any(path):
    if path.lower().endswith(".xlsx"):
        try:
            sheets = pd.read_excel(path, sheet_name=None)
            out = {}
            for name, df in sheets.items():
                if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
                    out[name] = normalize_columns(df)
            return out
        except Exception:
            return {}
    elif path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path)
            return {"_csv": normalize_columns(df)}
        except Exception:
            return {}
    return {}

# ---------- country-year harvesting (works even if no drivers exist) ----------
def guess_country_col(df):
    for c in df.columns:
        s = df[c].astype(str)
        mapped = s.map(to_iso2)
        if mapped.notna().mean() > 0.5:
            return c
    return None

def year_like_cols(df):
    return [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]

def longify_if_wide(df):
    country_col = guess_country_col(df)
    ycols = year_like_cols(df)
    if country_col and len(ycols) >= 3:
        out = df[[country_col] + ycols].copy()
        out = out.melt(id_vars=[country_col], value_vars=ycols,
                       var_name="year", value_name="_val")
        out = out.rename(columns={country_col: "country_raw"})
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["country"] = out["country_raw"].map(to_iso2)
        out = out.dropna(subset=["country","year"]).drop_duplicates(["country","year"])
        return out[["country","year"]]
    return None

def harvest_country_year(df):
    df = normalize_columns(df)
    # explicit columns
    cands_c = [c for c in df.columns if re.search(r"(country|iso2|iso-?2|name|nation|econom(y|ies))", c, re.I)]
    cands_y = [c for c in df.columns if re.fullmatch(r"year|yr|report(ing)?_?year|time_?year|calendar_?year", c, re.I)]
    if cands_c and cands_y:
        ccol, ycol = cands_c[0], cands_y[0]
        out = df[[ccol, ycol]].copy()
        out = out.rename(columns={ccol:"country_raw", ycol:"year"})
        out["country"] = out["country_raw"].map(to_iso2)
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["country","year"]).drop_duplicates(["country","year"])
        if not out.empty:
            return out[["country","year"]]
    # wide -> long
    return longify_if_wide(df)

def country_year_from_rankings():
    if os.path.exists(RANK_PATH):
        r = pd.read_csv(RANK_PATH, usecols=["country","year"]).dropna().drop_duplicates()
        if not r.empty:
            print("[q2_build_controls] using country-year pairs from ranking file as fallback.")
            r["year"] = r["year"].astype(int)
            return r
    return pd.DataFrame(columns=["country","year"])

def scan_and_merge_country_year():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")) + glob.glob(os.path.join(DATA_DIR, "*.csv")))
    merged = None
    seen = 0
    for p in files:
        sheets = read_any(p)
        for _, sdf in sheets.items():
            chunk = harvest_country_year(sdf)
            if chunk is not None and not chunk.empty:
                seen += 1
                merged = chunk if merged is None else pd.concat([merged, chunk], ignore_index=True)
        if sheets:
            print(f"[q2_build_controls] scanned {os.path.basename(p)} ({len(sheets)} sheet(s))")
    if merged is None or merged.empty:
        merged = country_year_from_rankings()
    if merged is None or merged.empty:
        raise ValueError("No (country,year) pairs found in ./data or ranking file.")
    merged = merged.drop_duplicates(["country","year"]).astype({"year":"int"})
    return merged

# ---------- optional drivers (placeholder hook; safe to be empty) ----------
def attach_drivers_if_possible(base):
    # Extend here later if you add real macro drivers.
    # For now, just return base (country, year).
    return base

# ---------- modeling (integrated Q2 drivers model) ----------
def prepare_lead(metric="LI_plus"):
    df = pd.read_csv(RANK_PATH)
    if metric not in df.columns:
        print(f"[q2] {metric} not found -> using LI_raw")
        metric = "LI_raw"
    g = df.groupby(["country","year"], as_index=False)[metric].mean()
    g = g.rename(columns={metric:"lead"})
    return g

def fe_ols(df):
    # dynamic set of regressors; build logs if available
    if "cons_pc_proxy" in df:
        df["log_cons_pc"] = np.log(pd.to_numeric(df["cons_pc_proxy"], errors="coerce").clip(lower=1e-6))
    if "population" in df:
        df["log_pop"] = np.log(pd.to_numeric(df["population"], errors="coerce").clip(lower=1))
    for c in ["trade_open","elec_ci","hddcdd"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    base_X = [v for v in ["log_cons_pc","log_pop","trade_open","elec_ci","hddcdd"] if v in df.columns and df[v].notna().sum() > 0]

    fallback_used = False
    if not base_X:
        df["time_trend"] = df["year"] - df["year"].mean()
        base_X = ["time_trend"]
        fallback_used = True
        print("[q2] WARNING: no drivers found. Using fallback regressor 'time_trend'.")

    use = ["lead","country","year"] + base_X
    dfm = df[use].dropna().copy()
    if dfm.empty:
        raise ValueError("Merged dataset empty after dropna; check inputs.")

    y = dfm["lead"].values
    X = sm.add_constant(dfm[base_X])

    d_country = pd.get_dummies(dfm["country"], prefix="c", drop_first=True)
    d_year    = pd.get_dummies(dfm["year"],    prefix="t", drop_first=True)
    X = pd.concat([X, d_country, d_year], axis=1)

    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="cluster", cov_kwds={"groups": dfm["country"]})
    return res, base_X, fallback_used, dfm

def export_results(res, base_X):
    with open(os.path.join(Q2_DIR, "drivers_model_summary.txt"), "w") as f:
        f.write(res.summary().as_text())

    rows = []
    for k in ["const"] + base_X:
        if k in res.params.index:
            ci = res.conf_int().loc[k]
            rows.append({
                "variable": k,
                "coef": float(res.params[k]),
                "se": float(res.bse[k]),
                "t": float(res.tvalues[k]),
                "p": float(res.pvalues[k]),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
            })
    coefs = pd.DataFrame(rows)
    coefs.to_csv(os.path.join(Q2_DIR, "drivers_coefs.csv"), index=False)

    plt.figure(figsize=(8,5))
    x = np.arange(len(coefs))
    plt.errorbar(x, coefs["coef"], yerr=1.96*coefs["se"], fmt="o")
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.xticks(x, coefs["variable"])
    plt.title("Drivers of Innovation Leadership (coef ±95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(Q2_DIR, "coefplot_q2.png"), dpi=220)
    plt.close()

def binscatter(df, xcol, ycol="lead", fname="binscatter.png", bins=20):
    z = df.dropna(subset=[xcol, ycol]).copy()
    if z.empty: return
    z[ycol+"_r"] = z[ycol] - z.groupby("country")[ycol].transform("mean") - z.groupby("year")[ycol].transform("mean") + z[ycol].mean()
    z[xcol+"_r"] = z[xcol] - z.groupby("country")[xcol].transform("mean") - z.groupby("year")[xcol].transform("mean") + z[xcol].mean()

    d = z[[xcol+"_r", ycol+"_r"]].dropna().sort_values(xcol+"_r")
    if d.empty: return
    q = min(bins, max(3, d.shape[0]//5))
    d["bin"] = pd.qcut(d[xcol+"_r"], q=q, duplicates="drop")
    g = d.groupby("bin", as_index=False).agg({xcol+"_r":"mean", ycol+"_r":"mean"})

    plt.figure(figsize=(6,4))
    plt.scatter(g[xcol+"_r"], g[ycol+"_r"])
    if len(g) >= 2:
        m, b = np.polyfit(g[xcol+"_r"], g[ycol+"_r"], 1)
        xs = np.linspace(g[xcol+"_r"].min(), g[xcol+"_r"].max(), 100)
        plt.plot(xs, m*xs+b)
    plt.title(f"FE-Residual Binscatter: {ycol} vs {xcol}")
    plt.xlabel(xcol); plt.ylabel(ycol)
    plt.tight_layout()
    plt.savefig(os.path.join(Q2_DIR, fname), dpi=220)
    plt.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="Also fit the drivers model and export plots.")
    ap.add_argument("--metric", default="LI_plus", choices=["LI_plus","LI_raw"])
    args = ap.parse_args()

    # 1) Build minimal controls (country, year, optional drivers)
    base = scan_and_merge_country_year()
    final = attach_drivers_if_possible(base)
    controls_csv = os.path.join(OUT_DIR, "controls_country_year.csv")
    final.to_csv(controls_csv, index=False)
    print(f"[q2_build_controls] wrote {controls_csv} (rows={len(final)}, cols={list(final.columns)})")

    if not args.fit:
        return

    # 2) Merge with leadership and fit the model
    lead = prepare_lead(args.metric)
    df = pd.merge(lead, final, on=["country","year"], how="inner")
    df.to_csv(os.path.join(Q2_DIR, "drivers_merged_input.csv"), index=False)
    print(f"[q2] merged rows: {len(df)}; columns: {list(df.columns)}")

    res, used, fallback, dfm = fe_ols(df)
    export_results(res, used)
    for x in used:
        if x != "const":
            binscatter(dfm, x, fname=f"binscatter_{x}.png")

    print(f"[q2] USED REGRESSORS: {used}  | FALLBACK_USED={fallback}")
    print(f"[q2] outputs -> {Q2_DIR}")

if __name__ == "__main__":
    main()
