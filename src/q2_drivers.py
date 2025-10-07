# src/q2_drivers.py
import os, argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

OUT_DIR = os.getenv("OUT_DIR", "out")
RANK_PATH = os.path.join(OUT_DIR, "country_sector_year_ranking.csv")
CTRL_PATH = os.path.join(OUT_DIR, "controls_country_year.csv")
Q2_DIR = os.path.join(OUT_DIR, "q2")
os.makedirs(Q2_DIR, exist_ok=True)

def prepare_lead(metric="LI_plus"):
    df = pd.read_csv(RANK_PATH)
    if metric not in df.columns:
        print(f"[q2] {metric} not found -> using LI_raw")
        metric = "LI_raw"
    g = df.groupby(["country","year"], as_index=False)[metric].mean()
    g = g.rename(columns={metric:"lead"})
    return g

def load_controls():
    if not os.path.exists(CTRL_PATH):
        raise FileNotFoundError(f"Missing {CTRL_PATH}. Run: python -m src.q2_build_controls")
    ctrl = pd.read_csv(CTRL_PATH)

    if "cons_pc_proxy" in ctrl:
        ctrl["log_cons_pc"] = np.log(pd.to_numeric(ctrl["cons_pc_proxy"], errors="coerce").clip(lower=1e-6))
    if "population" in ctrl:
        ctrl["log_pop"] = np.log(pd.to_numeric(ctrl["population"], errors="coerce").clip(lower=1))
    for c in ["trade_open","elec_ci","hddcdd"]:
        if c in ctrl:
            ctrl[c] = pd.to_numeric(ctrl[c], errors="coerce")
    return ctrl

def fe_ols(df):
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
        raise ValueError("Merged dataset has no rows after dropping NA. Check your controls file contents.")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="LI_plus", choices=["LI_plus","LI_raw"])
    args = ap.parse_args()

    lead = prepare_lead(args.metric)
    ctrl = load_controls()
    df = pd.merge(lead, ctrl, on=["country","year"], how="inner")
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
