# src/q3_run_model.py
# ---------------------------------------------------------------------
# Q3: Δln(CO2) on leadership (lagged LI preferred; contemporaneous LI fallback)
# - Country & year fixed-effects via two-way demeaning
# - Clustered SE by country
# - Robust I/O: always writes outputs to out/, with debug aids if needed
# ---------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

OUT = Path("out"); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data")

# ============================ Utilities ==============================
def _write_empty_outputs(reason: str, merged_panel: pd.DataFrame | None = None) -> None:
    """Always leave artifacts so you can inspect what's wrong."""
    # 1) results csv (empty schema)
    empty = pd.DataFrame(columns=["term", "coef", "se", "ci_low", "ci_high"])
    empty.to_csv(OUT / "q3_results.csv", index=False)

    # 2) panel csv (merged raw panel if provided)
    if merged_panel is not None:
        merged_panel.to_csv(OUT / "q3_panel.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "q3_panel.csv", index=False)

    # 3) coef plot (blank with message)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3))
    plt.text(0.5, 0.5, "No estimable sample", ha="center", va="center", transform=plt.gca().transAxes)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "q3_coefplot.png", dpi=300)

    # 4) summary
    (OUT / "model_summary.txt").write_text(
        f"Q3 estimation did not run.\nReason: {reason}\n"
        f"Artifacts written:\n"
        f" - {OUT/'q3_results.csv'}\n - {OUT/'q3_panel.csv'}\n - {OUT/'q3_coefplot.png'}\n"
        f"Inspect also: {OUT/'debug_coverage.txt'} and {OUT/'debug_emissions_preview.csv'} if present.\n"
    )

def _ensure_year(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    if "year" not in res.columns:
        for cand in ("Year", "YEAR", "year_x", "year_y"):
            if cand in res.columns:
                res = res.rename(columns={cand: "year"})
                break
    if "year" not in res.columns and res.index.name == "year":
        res = res.reset_index()
    if "year" not in res.columns:
        raise KeyError(f"Expected a 'year' column; got {list(res.columns)}")
    res["year"] = pd.to_numeric(res["year"], errors="coerce")
    return res

def _read_xwalk():
    p = DATA / "xwalk_iso2_iso3.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["iso2"] = df["iso2"].astype(str).str.upper().str.strip()
        df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
        return df
    return None
_XW = _read_xwalk()

def iso2_from_iso3(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    if _XW is not None:
        return s.map(_XW.set_index("iso3")["iso2"])
    try:
        import pycountry
        return s.map(lambda x: getattr(pycountry.countries.get(alpha_3=x), "alpha_2", np.nan))
    except Exception:
        return pd.Series(np.nan, index=s.index)

def iso2_from_name(s: pd.Series) -> pd.Series:
    def f(n):
        if not isinstance(n, str): return np.nan
        n = n.strip()
        aliases = {
            "United States":"US","United States of America":"US",
            "Russian Federation":"RU","Viet Nam":"VN",
            "Iran (Islamic Republic of)":"IR","Côte d’Ivoire":"CI","Cote d'Ivoire":"CI",
            "Korea, Rep.":"KR","Republic of Korea":"KR","Korea, Dem. People’s Rep.":"KP",
            "Hong Kong SAR, China":"HK","Macao SAR, China":"MO",
        }
        if n in aliases: return aliases[n]
        try:
            import pycountry
            return pycountry.countries.lookup(n).alpha_2
        except Exception:
            return np.nan
    return s.map(f)

# ============================ Loaders ================================
def load_emissions() -> pd.DataFrame:
    p = OUT / "q3_emissions_country_year.csv"
    if not p.exists():
        _write_empty_outputs(f"Missing {p}. Run: python -m src.q3_build_emissions")
        raise SystemExit(0)
    df = pd.read_csv(p)
    need = {"country_iso2", "year", "dln_co2"}
    if not need.issubset(df.columns):
        _write_empty_outputs(f"{p} must contain columns {sorted(need)}")
        raise SystemExit(0)
    df = df[list(need)].copy()
    df["country_iso2"] = df["country_iso2"].astype(str).str.upper()
    df = _ensure_year(df)
    return df

def _num(df, c):
    try: return pd.api.types.is_numeric_dtype(df[c])
    except Exception: return False

def _first_numeric(df, pats):
    for pat in pats:
        for c in df.columns:
            if re.search(pat, c.lower()) and _num(df, c):
                return c
    return None

def _z_by_year(df, col):
    g = df.groupby("year")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")

def load_li_country_year() -> pd.DataFrame:
    p = OUT / "country_sector_year_ranking.csv"
    if not p.exists():
        _write_empty_outputs(f"Missing {p}. Build Q1 rankings (python -m src.main rank).")
        raise SystemExit(0)
    raw = pd.read_csv(p)
    low = raw.rename(columns={c: c.lower() for c in raw.columns})
    if "year" not in low.columns:
        _write_empty_outputs("Q1 ranking file lacks 'year'")
        raise SystemExit(0)
    low = _ensure_year(low)

    # country id
    c_iso2 = next((c for c in low.columns if c in {"iso2","country_iso2","alpha2"}), None)
    c_iso3 = next((c for c in low.columns if c in {"iso3","country_iso3","alpha3","code"}), None)
    c_name = next((c for c in low.columns if c in {"country","country_name","name"}), None)
    if c_iso2: low["country_iso2"] = low[c_iso2].astype(str).str.upper()
    elif c_iso3: low["country_iso2"] = iso2_from_iso3(low[c_iso3])
    elif c_name: low["country_iso2"] = iso2_from_name(low[c_name])
    else:
        _write_empty_outputs("Q1 ranking file lacks country identifier (iso2/iso3/name)")
        raise SystemExit(0)

    # find a usable LI column; else compute from components
    cand = []
    for c in raw.columns:
        cl = c.lower()
        if (cl in {"li_plus_country_year","li_plus","li_country_year","li","leadership_index","li_cy"}
            or cl.endswith("_li") or cl.startswith("li_")
            or ("leadership" in cl and "index" in cl)):
            if _num(raw, c): cand.append(c)
    if cand:
        col = max(cand, key=lambda c: pd.Series(raw[c]).notna().sum())
        li_vals = raw[col]
        print(f"[LI] Using LI column: {col}")
    else:
        # compute z(P)+z(Q)+z(E) with any 2–3 components
        p_col = _first_numeric(raw, [r"\bp\b","quantity",r"patent[_\- ]?count","count_patents"])
        q_col = _first_numeric(raw, [r"\bq\b","quality",r"forward[_\- ]?cit",r"fw[_\- ]?cit","citations?"])
        e_col = _first_numeric(raw, [r"effectiv","elasticit","beta","theta","elasticity_component"])
        used = []; li_vals = 0
        tmp = low.copy()
        if p_col: tmp["_zp"] = _z_by_year(tmp.assign(**{p_col: raw[p_col]}), p_col); li_vals += tmp["_zp"]; used.append(p_col)
        if q_col: tmp["_zq"] = _z_by_year(tmp.assign(**{q_col: raw[q_col]}), q_col); li_vals += tmp["_zq"]; used.append(q_col)
        if e_col: tmp["_ze"] = _z_by_year(tmp.assign(**{e_col: raw[e_col]}), e_col); li_vals += tmp["_ze"]; used.append(e_col)
        if len(used) < 2:
            _write_empty_outputs("Could not find LI nor sufficient components to construct it (need ≥2 among P/Q/E).")
            raise SystemExit(0)
        print(f"[LI] Computed LI from components: {', '.join(used)} (z within year).")

    # aggregate to country–year
    group = ["country_iso2","year"]
    li_cy = (pd.concat([low[group], pd.Series(li_vals, name="LI")], axis=1)
             .groupby(group, as_index=False)
             .mean(numeric_only=True))
    li_cy = li_cy.dropna(subset=["country_iso2"])
    li_cy["country_iso2"] = li_cy["country_iso2"].astype(str).str.upper()
    return li_cy[["country_iso2","year","LI"]]

def load_controls_optional() -> tuple[pd.DataFrame | None, list[str]]:
    p = OUT / "controls_country_year_merged.csv"
    if not p.exists(): return None, []
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "country_iso2" not in df.columns:
        return None, []
    df["country_iso2"] = df["country_iso2"].astype(str).str.upper()
    df = _ensure_year(df)
    prefer = ["log_cons_pc","log_population","trade_open","elec_ci","hddcdd"]
    have = [c for c in prefer if c in df.columns]
    if not have: return None, []
    return df[["country_iso2","year"] + have], have

# ============================ Estimation ==============================
def cluster_ols(y, X, cluster):
    import statsmodels.api as sm
    model = sm.OLS(y, X)
    return model.fit(cov_type="cluster", cov_kwds={"groups": cluster})

def _demean_two_way(df: pd.DataFrame, cols: list[str]):
    """Add _tw columns: x_it - mean_i(x) - mean_t(x) + grand_mean(x)"""
    df = df.copy()
    df["year"] = df["year"].astype("int64")
    for col in cols:
        mean_i = df.groupby("country_iso2", observed=True)[col].transform("mean")
        mean_t = df.groupby(df["year"], observed=True)[col].transform("mean")  # group by VALUES
        grand  = float(df[col].mean())
        df[col + "_tw"] = df[col] - mean_i - mean_t + grand
    return df

def _coverage_table(panel: pd.DataFrame, label: str, controls: list[str]) -> str:
    def cov(col): return int(pd.Series(panel[col]).notna().sum()) if col in panel.columns else 0
    parts = [f"rows={len(panel)}", f"dln_co2={cov('dln_co2')}", f"{label}={cov(label)}"]
    for c in controls: parts.append(f"{c}={cov(c)}")
    return " | ".join(parts)

# ============================ Main ===================================
def main():
    # 1) Load pieces
    emis = load_emissions()                     # country_iso2, year, dln_co2
    li   = load_li_country_year()               # country_iso2, year, LI
    ctr, ctr_cols = load_controls_optional()    # optional

    emis["country_iso2"] = emis["country_iso2"].astype(str).str.upper()
    li["country_iso2"]   = li["country_iso2"].astype(str).str.upper()

    # Window info
    e_min, e_max = int(emis["year"].min()), int(emis["year"].max())
    l_min, l_max = int(li["year"].min()),   int(li["year"].max())
    print(f"[INFO] Emissions years: {e_min}-{e_max} | LI years: {l_min}-{l_max}")

    def build_and_estimate(li_like: pd.DataFrame, label: str):
        """Merge, auto-filter controls to those with data, run FE or return None with debug files."""
        # country intersection + window intersection
        common = sorted(set(emis["country_iso2"]).intersection(set(li_like["country_iso2"])))
        if not common:
            merged = emis.merge(li_like, on=["country_iso2","year"], how="outer")
            _write_empty_outputs("No common countries between emissions and LI.", merged_panel=merged)
            raise SystemExit(0)

        pe = emis[emis["country_iso2"].isin(common)].copy()
        pl = li_like[li_like["country_iso2"].isin(common)].copy()
        t_min = max(int(pe["year"].min()), int(pl["year"].min()))
        t_max = min(int(pe["year"].max()), int(pl["year"].max()))
        pe = pe[(pe["year"] >= t_min) & (pe["year"] <= t_max)].copy()
        pl = pl[(pl["year"] >= t_min) & (pl["year"] <= t_max)].copy()

        panel = pe.merge(pl, on=["country_iso2","year"], how="left")
        if ctr is not None:
            panel = panel.merge(ctr, on=["country_iso2","year"], how="left")

        print(f"[INFO] {label} overlap: years {t_min}-{t_max}, countries {len(common)}")
        print("[INFO] coverage:", _coverage_table(panel, label, ctr_cols))

        # write a preview for inspection always
        panel.head(2000).to_csv(OUT / "debug_emissions_preview.csv", index=False)

        # *** Only keep controls that actually have some data ***
        usable_ctrls = [c for c in ctr_cols if c in panel.columns and panel[c].notna().sum() > 0]

        x_cols = [label] + usable_ctrls
        need   = ["dln_co2"] + x_cols

        est = (panel.dropna(subset=need)
                     .sort_values(["country_iso2","year"])
                     .copy())

        if est.empty:
            dbg_cols = {"has_dln": panel["dln_co2"].notna().astype(int),
                        "has_li":  panel[label].notna().astype(int)}
            for c in usable_ctrls:
                dbg_cols[f"has_{c}"] = panel[c].notna().astype(int)
            dbg = (panel.assign(**dbg_cols)
                        .groupby("country_iso2")[list(dbg_cols.keys())]
                        .sum()
                        .sort_values("has_dln", ascending=False))
            (OUT / "debug_coverage.txt").write_text(
                f"Label: {label}\n"
                f"Years used: {t_min}-{t_max}\n"
                f"Countries overlapping: {len(common)}\n"
                f"Controls used (non-missing>0): {usable_ctrls}\n\n"
                f"Non-missing counts per country:\n{dbg.to_string()}\n"
            )
            return None, panel, x_cols

        est = _demean_two_way(est, need)
        y = est["dln_co2_tw"]
        X = est[[c + "_tw" for c in x_cols]]

        res = cluster_ols(y, X, cluster=est["country_iso2"])
        return res, est, x_cols

    # 2) Try lagged LI first
    li_lag = li.rename(columns={"year":"year_li"})
    li_lag["year"] = li_lag["year_li"] + 1
    li_lag = li_lag.drop(columns=["year_li"]).rename(columns={"LI":"LI_lag1"})
    res, est, x_cols = build_and_estimate(li_lag.rename(columns={"LI_lag1":"LI_use"}), "LI_use")

    used_label = "lagged"
    if res is None:
        print("[WARN] Lagged sample empty → trying contemporaneous LI.")
        res, est, x_cols = build_and_estimate(li.rename(columns={"LI":"LI_use"}), "LI_use")
        used_label = "contemporaneous"

    # 3) If still empty, write empty outputs and stop
    if res is None:
        _write_empty_outputs("No estimable rows after both lagged and contemporaneous attempts.",
                             merged_panel=None)
        print("[ERROR] No estimable sample. Wrote empty outputs and debug files.")
        return

    # 4) Export results
    coefs = (res.summary2().tables[1]
             .reset_index()
             .rename(columns={"index":"term","Coef.":"coef","Std.Err.":"se"}))
    coefs["ci_low"]  = coefs["coef"] - 1.96*coefs["se"]
    coefs["ci_high"] = coefs["coef"] + 1.96*coefs["se"]
    coefs.to_csv(OUT / "q3_results.csv", index=False)
    est.to_csv(OUT / "q3_panel.csv", index=False)

    # 5) Coefficient plot
    import matplotlib.pyplot as plt
    # label order for plot: LI first, then controls in the same order used
    ctrl_names = [c for c in x_cols[1:]]
    labels = (["LI (t-1)"] if used_label=="lagged" else ["LI (t)"]) + ctrl_names
    order  = ["LI_use_tw"] + [c + "_tw" for c in ctrl_names]
    plot_df = coefs[coefs["term"].isin(order)].copy()
    plot_df["label"] = [labels[order.index(t)] for t in plot_df["term"]]

    plt.figure(figsize=(5, max(3.0, 0.5 + 0.5*len(labels))))
    y_pos = np.arange(len(labels))
    xvals = plot_df.set_index("label").reindex(labels)
    plt.errorbar(xvals["coef"], y_pos,
                 xerr=[xvals["coef"] - xvals["ci_low"], xvals["ci_high"] - xvals["coef"]],
                 fmt="o", capsize=3)
    plt.yticks(y_pos, labels); plt.axvline(0, linestyle="--"); plt.xlabel("Coefficient (95% CI)")
    plt.tight_layout(); plt.savefig(OUT / "q3_coefplot.png", dpi=300)

    # 6) Write a summary text
    summary_txt = res.summary().as_text()
    (OUT / "model_summary.txt").write_text(
        f"Q3 model used: {'lagged LI (t-1)' if used_label=='lagged' else 'contemporaneous LI (t)'}\n"
        f"N countries: {est['country_iso2'].nunique()}\n"
        f"N rows: {len(est)}\n\n{summary_txt}\n"
    )

    print(summary_txt)
    print(f"\n[WRITE] {OUT/'q3_results.csv'}")
    print(f"[WRITE] {OUT/'q3_coefplot.png'}")
    print(f"[WRITE] {OUT/'q3_panel.csv'}")
    print(f"[WRITE] {OUT/'model_summary.txt'}")

if __name__ == "__main__":
    main()
