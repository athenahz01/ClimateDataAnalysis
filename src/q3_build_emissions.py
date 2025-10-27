# src/q3_build_emissions.py
from __future__ import annotations
import pandas as pd, numpy as np, re, sys
from pathlib import Path

DATA = Path("data")
OUT  = Path("out")
(DATA/"edgar").mkdir(parents=True, exist_ok=True)
(DATA/"owid").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ---------- ISO helpers (prefer local crosswalk; fallback to pycountry) ----------
XWALK = DATA / "xwalk_iso2_iso3.csv"
_xw = pd.read_csv(XWALK) if XWALK.exists() else None
if _xw is not None:
    _xw["iso2"] = _xw["iso2"].astype(str).str.upper().str.strip()
    _xw["iso3"] = _xw["iso3"].astype(str).str.upper().str.strip()

def iso2_from_iso3(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    if _xw is not None:
        return s.map(_xw.set_index("iso3")["iso2"])
    try:
        import pycountry
        return s.map(lambda x: getattr(pycountry.countries.get(alpha_3=x), "alpha_2", np.nan))
    except Exception:
        return pd.Series(np.nan, index=s.index)

def iso3_from_iso2(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    if _xw is not None:
        return s.map(_xw.set_index("iso2")["iso3"])
    try:
        import pycountry
        return s.map(lambda x: getattr(pycountry.countries.get(alpha_2=x), "alpha_3", np.nan))
    except Exception:
        return pd.Series(np.nan, index=s.index)

# ---------- EDGAR loader (very tolerant) ----------
def load_edgar() -> pd.DataFrame | None:
    files = sorted((DATA/"edgar").glob("*.xlsx"))
    if not files:
        return None
    path = files[0]
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"[EDGAR] Could not open {path}: {e}")
        return None

    # Pick likely sheet
    sheet = next((sn for sn in xl.sheet_names
                  if any(k in sn.lower() for k in ["country","national","emission","co2"])),
                 xl.sheet_names[0])
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    # identify id columns
    c_iso3 = next((c for c in df.columns if re.search(r"(^iso.*3$|country.*code.*3)", str(c), re.I)), None)
    c_iso2 = next((c for c in df.columns if re.search(r"(^iso.*2$|country.*code.*2)", str(c), re.I)), None)
    c_name = next((c for c in df.columns if re.fullmatch(r"(country|name)", str(c), re.I)), None)

    # pivot years if wide
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if year_cols:
        id_cols = [c for c in [c_iso3, c_iso2, c_name] if c is not None]
        long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="co2_val")
        long["year"] = long["year"].astype(int)
    else:
        c_year = next((c for c in df.columns if re.fullmatch(r"year", str(c), re.I)), None)
        c_val  = next((c for c in df.columns if re.search(r"(co2|co₂)", str(c), re.I)), None)
        if c_year is None or c_val is None:
            print("[EDGAR] Could not find year/CO2 columns.")
            return None
        long = df.rename(columns={c_year:"year", c_val:"co2_val"})
        long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")

    # map to iso2
    if c_iso3 and long[c_iso3].notna().any():
        long["iso3"] = long[c_iso3].astype(str).str.upper()
        long["country_iso2"] = iso2_from_iso3(long["iso3"])
    elif c_iso2 and long[c_iso2].notna().any():
        long["country_iso2"] = long[c_iso2].astype(str).str.upper()
        long["iso3"] = iso3_from_iso2(long["country_iso2"])
    elif c_name:
        try:
            import pycountry
            def to_iso3(n):
                if not isinstance(n, str): return np.nan
                alias = {"United States":"USA","Russian Federation":"RUS","Viet Nam":"VNM",
                         "Iran (Islamic Republic of)":"IRN","Côte d’Ivoire":"CIV","Cote d'Ivoire":"CIV",
                         "Korea, Rep.":"KOR","Korea, Dem. People’s Rep.":"PRK"}
                if n in alias: return alias[n]
                try: return pycountry.countries.lookup(n).alpha_3
                except Exception: return np.nan
            long["iso3"] = long[c_name].map(to_iso3)
            long["country_iso2"] = iso2_from_iso3(long["iso3"])
        except Exception:
            return None
    else:
        return None

    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["co2_val"] = pd.to_numeric(long["co2_val"], errors="coerce")

    # Heuristic units: EDGAR often reports ktCO2 → convert to MtCO2
    s = long["co2_val"]
    median = np.nanmedian(s.values)
    if np.isfinite(median) and 1e2 < median < 1e7:
        co2_mt = s / 1000.0
        unit = "kt→Mt (÷1000)"
    else:
        co2_mt = s
        unit = "assumed Mt"
    out = long.assign(co2_mt=co2_mt)[["country_iso2","iso3","year","co2_mt"]]
    out = out.dropna(subset=["country_iso2","year","co2_mt"]).copy()
    out["year"] = out["year"].astype(int)
    print(f"[EDGAR] Parsed {len(out):,} rows from {path.name} (unit: {unit}, sheet: {sheet}).")
    return out

# ---------- OWID loader (simple) ----------
def load_owid() -> pd.DataFrame | None:
    path = DATA/"owid"/"co2-data.csv"
    if not path.exists(): return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[OWID] Could not read {path}: {e}")
        return None
    # accept new/old schemas
    if "iso_code" in df.columns:
        df = df.rename(columns={"iso_code":"iso3"})
    if not {"iso3","year"}.issubset(df.columns):
        print("[OWID] Missing iso3/year columns.")
        return None
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["country_iso2"] = iso2_from_iso3(df["iso3"])
    co2_col = "co2" if "co2" in df.columns else None
    if co2_col is None:
        print("[OWID] Could not find 'co2' column.")
        return None
    keep = df.rename(columns={co2_col:"co2_mt"})[["country_iso2","iso3","year","co2_mt"]]
    keep["year"] = pd.to_numeric(keep["year"], errors="coerce").astype("Int64")
    keep["co2_mt"] = pd.to_numeric(keep["co2_mt"], errors="coerce")
    keep = keep.dropna(subset=["country_iso2","year","co2_mt"]).copy()
    keep["year"] = keep["year"].astype(int)
    print(f"[OWID] Parsed {len(keep):,} rows from {path.name} (MtCO2).")
    return keep

# ---------- Δln CO2 helper ----------
def dlog(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)  # non-positive → NaN (avoid log issues)
    return np.log(s) - np.log(s.shift(1))

def main():
    edgar = load_edgar()
    owid = load_owid()

    if edgar is None and owid is None:
        print("No emissions source found.\n"
              "Place an EDGAR .xlsx in data/edgar/  OR  OWID co2-data.csv in data/owid/")
        sys.exit(1)

    base = edgar if edgar is not None else owid
    src  = "EDGAR" if edgar is not None else "OWID"
    print(f"[INFO] Using {src} as emissions source.")

    base = base.sort_values(["country_iso2","year"]).reset_index(drop=True)
    base = base[(base["year"] >= 1960) & (base["year"] <= 2100)].copy()

    # country/year counts
    n_cty = base["country_iso2"].nunique()
    yr_min, yr_max = int(base["year"].min()), int(base["year"].max())
    print(f"[INFO] Coverage: {n_cty} countries, years {yr_min}-{yr_max}.")

    # Δln CO2 with aligned index
    base["dln_co2"] = base.groupby("country_iso2", group_keys=False)["co2_mt"].transform(dlog)

    outpath = OUT / "q3_emissions_country_year.csv"
    base.to_csv(outpath, index=False)
    print(f"[WRITE] {outpath}  ({len(base):,} rows)")
    print(base.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
