import os
from typing import List

import numpy as np
import pandas as pd
import re

from .io_utils import read_parquet, load_optional_table, ensure_output_dirs
from .schema import CANON
from .logger import get_logger


# ---------- ISO helpers ----------

# extend ISO2->ISO3 map (broad coverage)
ISO2_TO_ISO3 = {
    # OECD + large economies + common
    "US":"USA","JP":"JPN","FR":"FRA","DE":"DEU","CN":"CHN","KR":"KOR","GB":"GBR","IT":"ITA","ES":"ESP",
    "NL":"NLD","SE":"SWE","NO":"NOR","DK":"DNK","FI":"FIN","CH":"CHE","AT":"AUT","BE":"BEL","PL":"POL",
    "CZ":"CZE","PT":"PRT","IE":"IRL","CA":"CAN","MX":"MEX","BR":"BRA","ZA":"ZAF","RU":"RUS","TR":"TUR",
    "IN":"IND","ID":"IDN","AU":"AUS","NZ":"NZL","IL":"ISR","SG":"SGP","GR":"GRC","HU":"HUN","RO":"ROU",
    "AR":"ARG","CL":"CHL","CO":"COL","PE":"PER","TH":"THA","PH":"PHL","MY":"MYS","VN":"VNM","SA":"SAU",
    "AE":"ARE","EG":"EGY"
}

NAME_TO_ISO3 = {
    # english names (add more if your EDGAR uses others)
    "united states":"USA","united states of america":"USA","usa":"USA","u.s.a.":"USA","us":"USA","u.s.":"USA",
    "china":"CHN","people's republic of china":"CHN","pr china":"CHN",
    "japan":"JPN","korea, republic of":"KOR","south korea":"KOR","republic of korea":"KOR","korea":"KOR",
    "germany":"DEU","france":"FRA","united kingdom":"GBR","uk":"GBR","great britain":"GBR",
    "italy":"ITA","spain":"ESP","netherlands":"NLD","sweden":"SWE","norway":"NOR","denmark":"DNK","finland":"FIN",
    "switzerland":"CHE","austria":"AUT","belgium":"BEL","poland":"POL","czech republic":"CZE","portugal":"PRT",
    "ireland":"IRL","canada":"CAN","mexico":"MEX","brazil":"BRA","south africa":"ZAF","russian federation":"RUS",
    "russia":"RUS","turkey":"TUR","india":"IND","indonesia":"IDN","australia":"AUS","new zealand":"NZL",
    "israel":"ISR","singapore":"SGP","greece":"GRC","hungary":"HUN","romania":"ROU","argentina":"ARG",
    "chile":"CHL","colombia":"COL","peru":"PER","thailand":"THA","philippines":"PHL","malaysia":"MYS","vietnam":"VNM",
    "saudi arabia":"SAU","united arab emirates":"ARE","egypt":"EGY"
}

NON_SOVEREIGN_CODES = {"EP","WO","EA","AP","EM","OA","GC","GC**"}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def to_iso3_from_any(s: pd.Series) -> pd.Series:
    s = _norm(s)
    out = []
    for v in s:
        if not v or v.lower() in {"nan","none"}:
            out.append(np.nan); continue
        if len(v) == 3 and v.isupper():  # looks like ISO3 already
            out.append(v); continue
        if len(v) == 2 and v.isupper():  # ISO2
            out.append(ISO2_TO_ISO3.get(v, np.nan)); continue
        iso3 = NAME_TO_ISO3.get(v.lower(), np.nan)
        out.append(iso3)
    return pd.Series(out, index=s.index)


# ---------- metrics ----------

def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def build_lags(df: pd.DataFrame, vars: List[str] = ["P","Q"], lags: int = 5, group: List[str] = ["country","sector"]) -> pd.DataFrame:
    if "year" not in df.columns:
        raise ValueError("`year` column missing before building lags.")
    df = df.sort_values(group + ["year"])
    for v in vars:
        for L in range(1, lags+1):
            df[f"{v}_lag{L}"] = df.groupby(group, dropna=False)[v].shift(L)
    return df


# ---------- main ----------

def _reshape_emissions_wide(df: pd.DataFrame) -> pd.DataFrame:
    # Try to find a header row: a row with at least 5 cells that look like 4-digit years
    candidate = None
    for i in range(min(len(df), 30)):
        row = df.iloc[i]
        year_like = 0
        for v in row.tolist():
            s = str(v).strip()
            if s.isdigit() and 1960 <= int(float(s)) <= 2100:
                year_like += 1
        if year_like >= 5:
            candidate = i
            break
    if candidate is None:
        # fallback: assume first row is header
        df2 = df.copy()
        df2.columns = [str(c).strip() for c in df2.columns]
    else:
        df2 = df.iloc[candidate + 1 :].copy()
        headers = df.iloc[candidate].astype(str).apply(lambda x: x.strip()).tolist()
        df2.columns = headers
    df2.columns = [str(c).strip() for c in df2.columns]
    # pick a country-like column
    country_like = None
    for cand in ["country", "name", "nation", "territory", "area", "iso3", "iso", "code"]:
        for c in df2.columns:
            if cand == str(c).strip().lower():
                country_like = c
                break
        if country_like is not None:
            break
    if country_like is None:
        country_like = df2.columns[0]
    # detect year columns
    year_cols = []
    col_to_year = {}
    for c in df2.columns:
        if c == country_like:
            continue
        s = str(c).strip()
        # extract a 4-digit year anywhere in the header
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            y = int(m.group(0))
            if 1960 <= y <= 2100:
                year_cols.append(c)
                col_to_year[c] = y
        elif s.isdigit():
            y = int(float(s))
            if 1960 <= y <= 2100:
                year_cols.append(c)
                col_to_year[c] = y
    if not year_cols:
        raise ValueError("Unable to identify year columns in emissions table.")
    melted = df2.melt(id_vars=[country_like], value_vars=year_cols, var_name="_year_col", value_name="emissions")
    # map header text to numeric year
    melted["year"] = melted["_year_col"].map(col_to_year).astype(int)
    melted = melted.drop(columns=["_year_col"]).rename(columns={country_like: "country"})
    return melted

def prepare_features(out_dir: str) -> pd.DataFrame:
    # patents panel
    cst = read_parquet(os.path.join(out_dir, "cst_agg.parquet")).copy()
    cst[CANON["country"]] = _norm(cst[CANON["country"]])

    # drop non-sovereign patent-office codes, map to ISO3
    cst = cst[~cst[CANON["country"]].isin(NON_SOVEREIGN_CODES)]
    cst["iso3"] = to_iso3_from_any(cst[CANON["country"]])
    cst = cst.dropna(subset=["iso3"])

    # emissions
    emissions_path = os.getenv("CSV_EMISSIONS", "")
    emit = load_optional_table(emissions_path) if emissions_path else None
    if emit is not None and not getattr(emit, 'empty', False):
        logger = get_logger()
        try:
            df_e = emit.copy()
            df_e.columns = [str(c).strip() for c in df_e.columns]
            if "country" not in df_e.columns or "year" not in df_e.columns:
                df_e = _reshape_emissions_wide(df_e)
            df_e["iso3"] = to_iso3_from_any(df_e["country"])
            df_e = df_e.dropna(subset=["iso3"])
            df_e["year"] = df_e["year"].astype(int)

            if "emissions" in df_e.columns:
                em = pd.to_numeric(df_e["emissions"], errors="coerce").where(lambda s: s > 0)
                df_e["delta_CO2"] = np.log(em).groupby(df_e["iso3"]).diff()
            else:
                df_e["delta_CO2"] = np.nan

            # diagnostics
            ensure_output_dirs(out_dir)
            cst_keys = cst[["iso3","year"]].drop_duplicates()
            e_keys = df_e[["iso3","year"]].drop_duplicates()
            merged_keys = cst_keys.merge(e_keys, on=["iso3","year"], how="inner")
            with open(os.path.join(out_dir, "debug_coverage.txt"), "w", encoding="utf-8") as f:
                f.write(f"Patent iso3-year keys: {len(cst_keys):,}\n")
                f.write(f"EDGAR iso3-year keys: {len(e_keys):,}\n")
                f.write(f"Overlap iso3-year keys: {len(merged_keys):,}\n")

            cst = cst.merge(df_e[["iso3","year","delta_CO2"]], on=["iso3","year"], how="left")
        except Exception as e:
            logger.warning(f"Emissions parsing failed ({type(e).__name__}: {e}); proceeding without delta_CO2.")

    # Handle missing Q: create a simple proxy if all Q are NaN
    if cst["Q"].isna().all():
        logger = get_logger()
        logger.warning("All quality values are NaN; creating simple quality proxy based on patent count.")
        # Simple quality proxy: log(1 + P) normalized by sector-year
        cst["Q"] = np.log1p(cst["P"])
        cst["Q"] = cst.groupby([CANON["sector"], CANON["year"]])["Q"].transform(lambda x: (x - x.mean()) / x.std())
    
    # Build lags for P and Q
    cst = build_lags(cst, vars=["P","Q"], lags=int(os.getenv("MAX_LAG","5")), group=[CANON["country"], CANON["sector"]])
    return cst
