# src/q2_merge_drivers.py
# Q2 controls builder (robust):
# - Loads/derives a country–year scaffold (ISO-2, year). If the scaffold file is missing,
#   it builds one from the Q1 rankings.
# - Fetches WDI drivers via JSON API (stable) with local CSV fallbacks.
# - Fetches electricity CO2 intensity from OWID (with local grapher fallback).
# - Optionally merges HDD+CDD if you provide data/cckp/degree_days_country.csv.
#
# Output: out/controls_country_year_merged.csv
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

DATA = Path("data"); OUT = Path("out")
for p in [DATA/"wdi", DATA/"owid", DATA/"cckp", OUT]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- ISO helpers ----------------
XWALK = DATA/"xwalk_iso2_iso3.csv"
_xw = pd.read_csv(XWALK) if XWALK.exists() else None
if _xw is not None:
    _xw["iso2"]=_xw["iso2"].str.upper().str.strip()
    _xw["iso3"]=_xw["iso3"].str.upper().str.strip()

def iso3_from_iso2(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    if _xw is not None:
        return s.map(_xw.set_index("iso2")["iso3"])
    try:
        import pycountry
        return s.map(lambda x: getattr(pycountry.countries.get(alpha_2=x), "alpha_3", np.nan))
    except Exception:
        return pd.Series(np.nan, index=s.index)

def iso2_from_iso3(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    if _xw is not None:
        return s.map(_xw.set_index("iso3")["iso2"])
    try:
        import pycountry
        return s.map(lambda x: getattr(pycountry.countries.get(alpha_3=x), "alpha_2", np.nan))
    except Exception:
        return pd.Series(np.nan, index=s.index)

def iso2_from_name(s: pd.Series) -> pd.Series:
    def f(n):
        if not isinstance(n,str): return np.nan
        n=n.strip()
        aliases={"Korea, Rep.":"KR","Republic of Korea":"KR","Korea, Dem. People’s Rep.":"KP",
                 "Russian Federation":"RU","United States":"US","United States of America":"US",
                 "Viet Nam":"VN","Cote d'Ivoire":"CI","Côte d’Ivoire":"CI",
                 "Iran (Islamic Republic of)":"IR","Hong Kong SAR, China":"HK","Macao SAR, China":"MO"}
        if n in aliases: return aliases[n]
        try:
            import pycountry
            return pycountry.countries.lookup(n).alpha_2
        except Exception:
            return np.nan
    return s.map(f)

# ---------------- Load/build scaffold ----------------
def load_scaffold() -> pd.DataFrame:
    """
    Returns columns: country_iso2, iso3, year
    Uses out/controls_country_year.csv if present (flexible col names),
    else derives from out/country_sector_year_ranking.csv.
    """
    path = OUT/"controls_country_year.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df.rename(columns={c:c.lower() for c in df.columns})
        c_year = "year" if "year" in df.columns else None
        c_iso2 = next((c for c in df.columns if c in {"country_iso2","iso2","alpha2"}), None)
        c_iso3 = next((c for c in df.columns if c in {"country_iso3","iso3","alpha3","code"}), None)
        c_name = next((c for c in df.columns if c in {"country","country_name","name"}), None)
        if c_year is not None:
            df["year"]=df[c_year].astype(int)
        if c_iso2:   df["country_iso2"]=df[c_iso2].astype(str).str.upper()
        elif c_iso3: df["country_iso2"]=iso2_from_iso3(df[c_iso3])
        elif c_name: df["country_iso2"]=iso2_from_name(df[c_name])
        else:        df=None
        if df is not None and {"country_iso2","year"}.issubset(df.columns):
            df["iso3"]=iso3_from_iso2(df["country_iso2"])
            return df[["country_iso2","iso3","year"]].dropna()
    # Fallback: derive scaffold from Q1 rankings
    q1 = OUT/"country_sector_year_ranking.csv"
    if not q1.exists():
        raise RuntimeError("No scaffold and Q1 file missing. Run Q1 or provide out/controls_country_year.csv.")
    df = pd.read_csv(q1)
    df = df.rename(columns={c:c.lower() for c in df.columns})
    if "year" not in df.columns:
        raise RuntimeError("Q1 rankings lacks 'year'.")
    c_iso2 = next((c for c in df.columns if c in {"iso2","country_iso2","alpha2"}), None)
    c_iso3 = next((c for c in df.columns if c in {"iso3","country_iso3","alpha3","code"}), None)
    c_name = next((c for c in df.columns if c in {"country","country_name","name"}), None)
    if c_iso2:   df["country_iso2"]=df[c_iso2].astype(str).str.upper()
    elif c_iso3: df["country_iso2"]=iso2_from_iso3(df[c_iso3])
    elif c_name: df["country_iso2"]=iso2_from_name(df[c_name])
    else:        raise RuntimeError("Q1 rankings missing country identifier.")
    df["year"]=df["year"].astype(int)
    scaffold = df[["country_iso2","year"]].dropna().drop_duplicates()
    scaffold["iso3"]=iso3_from_iso2(scaffold["country_iso2"])
    return scaffold[["country_iso2","iso3","year"]]

# ---------------- WDI via JSON API (stable) ----------------
WDI_CODES = {
    "NY.GDP.PCAP.KD": "gdp_pc_const_2015usd",  # GDP per cap, constant 2015 US$
    "SP.POP.TOTL":    "population",
    "NE.TRD.GNFS.ZS": "trade_open_pct_gdp",
}
def fetch_wdi(ind_code: str) -> pd.DataFrame:
    import requests
    colname = WDI_CODES[ind_code]
    url = f"https://api.worldbank.org/v2/country/all/indicator/{ind_code}"
    params = {"format": "json", "per_page": 20000, "page": 1}
    frames = []
    try:
        while True:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2:
                break
            meta, data = payload[0], payload[1]
            if not data: break
            rows=[]
            for d in data:
                iso3 = (d.get("countryiso3code") or "").upper()
                year = d.get("date")
                val  = d.get("value")
                if iso3 and year is not None:
                    rows.append((iso3, int(year), val))
            if rows:
                frames.append(pd.DataFrame(rows, columns=["iso3","year",colname]))
            page = meta.get("page", params["page"])
            pages = meta.get("pages", params["page"])
            if page >= pages: break
            params["page"] = page + 1
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out = out.dropna(subset=["iso3","year"])
            out["iso3"]=out["iso3"].astype(str).str.upper(); out["year"]=out["year"].astype(int)
            return out[["iso3","year",colname]]
        raise RuntimeError("Empty response from WDI JSON API")
    except Exception as e:
        print(f"[WARN] WDI JSON fetch failed for {ind_code} ({e}). Trying local CSV fallback.")
        local = DATA/"wdi"/f"{ind_code}.csv"
        if not local.exists():
            raise RuntimeError(f"Missing local fallback {local}. Save CSV with columns iso3,year,value.")
        df = pd.read_csv(local).rename(columns={c.lower():c for c in df.columns})
        c_iso3 = next((c for c in df.columns if c.lower() in {"iso3","country code","countryiso3code"}), None)
        c_year = next((c for c in df.columns if c.lower() in {"year","date","time"}), None)
        c_val  = next((c for c in df.columns if c.lower() in {"value", colname.lower()}), None)
        df = df.rename(columns={c_iso3:"iso3", c_year:"year", c_val:colname})
        df["iso3"]=df["iso3"].astype(str).str.upper(); df["year"]=df["year"].astype(int)
        return df[["iso3","year",colname]]

def get_wdi_all() -> pd.DataFrame:
    parts = [fetch_wdi(code) for code in WDI_CODES]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["iso3","year"], how="outer")
    return out

# ---------------- OWID electricity CO2 intensity ----------------
def get_owid_elec_ci() -> pd.DataFrame:
    """
    Return columns: iso3, year, elec_ci_gco2_per_kwh.
    Tries multiple OWID grapher URLs; then data/owid/carbon-intensity-electricity.csv (grapher format).
    On total failure, returns an empty DF with the correct columns (pipeline continues with NaNs).
    """
    import pandas as pd

    def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
        # Accept either grapher format (Code,Year,value) or energy bundle format
        cols = set(df.columns)
        if {"Code", "Year", "value"} <= cols:
            g = df.rename(columns={"Code": "iso3", "Year": "year", "value": "elec_ci_gco2_per_kwh"})
        elif {"iso_code", "year", "electricity_co2_intensity"} <= cols:
            g = df.rename(columns={"iso_code": "iso3", "electricity_co2_intensity": "elec_ci_gco2_per_kwh"})
        else:
            return None
        g["iso3"] = g["iso3"].astype(str).str.upper()
        g["year"] = g["year"].astype(int)
        return g[["iso3", "year", "elec_ci_gco2_per_kwh"]]

    urls = [
        "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv",
        "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Carbon%20intensity%20of%20electricity/carbon-intensity-electricity.csv",
    ]
    for u in urls:
        try:
            df = pd.read_csv(u)
            norm = _normalize(df)
            if norm is not None:
                return norm
        except Exception:
            pass  # try next

    # Local fallback
    local = DATA / "owid" / "carbon-intensity-electricity.csv"
    if local.exists():
        try:
            df = pd.read_csv(local)
            norm = _normalize(df)
            if norm is not None:
                return norm
        except Exception:
            pass

    print("[WARN] Could not obtain electricity CI (OWID). Proceeding with NaNs.")
    return pd.DataFrame(columns=["iso3", "year", "elec_ci_gco2_per_kwh"])



# ---------------- Optional HDD + CDD ----------------
def get_hdd_cdd_optional() -> pd.DataFrame | None:
    path = DATA/"cckp"/"degree_days_country.csv"
    if not path.exists(): return None
    dd = pd.read_csv(path).rename(columns={c:c.lower() for c in path.read_text().splitlines()[0].split(",")}) if False else pd.read_csv(path)
    dd = dd.rename(columns={c:c.lower() for c in dd.columns})
    iso3 = next((c for c in dd.columns if c=="iso3"), None)
    year = next((c for c in dd.columns if c=="year"), None)
    hdd  = next((c for c in dd.columns if c in {"hdd_18c","hdd"}), None)
    cdd  = next((c for c in dd.columns if c in {"cdd_18c","cdd"}), None)
    if not all([iso3,year,hdd,cdd]):
        raise RuntimeError("degree_days_country.csv must have iso3,year,hdd_18c,cdd_18c (or hdd/cdd).")
    dd = dd.rename(columns={iso3:"iso3", year:"year", hdd:"hdd_18c", cdd:"cdd_18c"})
    dd["iso3"]=dd["iso3"].astype(str).str.upper(); dd["year"]=dd["year"].astype(int)
    dd["hdd_cdd_sum"] = dd["hdd_18c"].fillna(0) + dd["cdd_18c"].fillna(0)
    return dd[["iso3","year","hdd_18c","cdd_18c","hdd_cdd_sum"]]

# ---------------- Main ----------------
def main():
    scaffold = load_scaffold()  # country_iso2, iso3, year
    wdi = get_wdi_all()
    owid_ci = get_owid_elec_ci()
    dd = get_hdd_cdd_optional()

    merged = scaffold.merge(wdi, on=["iso3","year"], how="left") \
                     .merge(owid_ci, on=["iso3","year"], how="left")
    if dd is not None:
        merged = merged.merge(dd, on=["iso3","year"], how="left")

    # transforms for model
    merged["log_cons_pc"]   = np.log(merged["gdp_pc_const_2015usd"])
    merged["log_population"] = np.log(merged["population"])
    merged["trade_open"]    = merged["trade_open_pct_gdp"] / 100.0
    merged["elec_ci"]       = merged["elec_ci_gco2_per_kwh"]
    merged["hddcdd"]        = merged["hdd_cdd_sum"] if "hdd_cdd_sum" in merged.columns else np.nan

    keep = ["country_iso2","iso3","year","log_cons_pc","log_population","trade_open","elec_ci","hddcdd"]
    merged[keep].to_csv(OUT/"controls_country_year_merged.csv", index=False)
    print(f"Wrote {OUT/'controls_country_year_merged.csv'} with {len(merged):,} rows.")

if __name__ == "__main__":
    main()
