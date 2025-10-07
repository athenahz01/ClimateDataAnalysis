import os
import re
import time
from typing import Optional, Tuple, Dict, List

import pandas as pd


def ensure_output_dirs(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tmp_parts"), exist_ok=True)


def timestamp_suffix() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    engine = "pyarrow"
    try:
        df.to_parquet(path, engine=engine, index=False)
    except Exception:
        engine = "fastparquet"
        df.to_parquet(path, engine=engine, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def write_excel(df: pd.DataFrame, path: str, sheet_name: str = "Sheet1") -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_overwrite(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{timestamp_suffix()}{ext}"


# ------------------ NEW: Excel-or-CSV loader for emissions/controls ------------------

def load_optional_csv(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_optional_table(path: str) -> Optional[pd.DataFrame]:
    """Load CSV or Excel if path exists; return None if not provided or missing."""
    if not path:
        return None
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return _parse_edgar_booklet_excel(path)
    # Fallback: try CSV then Excel
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception:
            return None


def _detect_header_and_year_cols(df: pd.DataFrame) -> Tuple[int, str, Dict[str, int]]:
    """Return (header_row_index, country_col_name, mapping of column->year)."""
    # find header row
    header_row = 0
    for i in range(min(len(df), 50)):
        row = df.iloc[i]
        year_like = 0
        for v in row.tolist():
            s = str(v).strip()
            if re.search(r"(19|20)\d{2}", s):
                year_like += 1
        if year_like >= 5:
            header_row = i
            break
    # apply header
    headers = df.iloc[header_row].astype(str).apply(lambda x: x.strip()).tolist()
    df2 = df.iloc[header_row + 1 :].copy()
    df2.columns = headers
    # country-like column
    country_col = headers[0]
    for cand in ["country", "name", "nation", "territory", "area", "iso3", "iso", "code"]:
        for h in headers:
            if str(h).strip().lower() == cand:
                country_col = h
                break
        if country_col != headers[0]:
            break
    # year mapping
    col_to_year: Dict[str, int] = {}
    for h in headers:
        if h == country_col:
            continue
        s = str(h).strip()
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            y = int(m.group(0))
            if 1960 <= y <= 2100:
                col_to_year[h] = y
            continue
        if s.isdigit():
            y = int(float(s))
            if 1960 <= y <= 2100:
                col_to_year[h] = y
    return header_row, country_col, col_to_year


def _parse_edgar_booklet_excel(path: str) -> pd.DataFrame:
    """Parse EDGAR booklet Excel into tidy format: country, year, emissions."""
    sheets = pd.ExcelFile(path)
    for sheet in sheets.sheet_names:
        # quick sample read without headers
        sample = pd.read_excel(path, sheet_name=sheet, header=None, nrows=200)
        try:
            hrow, country_col, col_to_year = _detect_header_and_year_cols(sample)
        except Exception:
            continue
        if len(col_to_year) >= 5:
            full = pd.read_excel(path, sheet_name=sheet, header=None)
            # build with header
            headers = full.iloc[hrow].astype(str).apply(lambda x: x.strip()).tolist()
            data = full.iloc[hrow + 1 :].copy()
            data.columns = headers
            melt = data.melt(id_vars=[country_col], value_vars=list(col_to_year.keys()), var_name="_col", value_name="emissions")
            melt["year"] = melt["_col"].map(col_to_year).astype(int)
            melt = melt.drop(columns=["_col"]).rename(columns={country_col: "country"})
            return melt
    raise ValueError("Could not parse emissions Excel into tidy format; unrecognized layout.")


def _parse_edgar_booklet_excel(path: str) -> pd.DataFrame:
    """
    Heuristic parser for EDGAR GHG booklet Excel.
    - Prefer an ISO3-like country column if available (values look like 3-letter uppercase).
    - Otherwise fall back to a country/name column or the first column.
    - Detect year columns (1990..2099), melt to long.
    Returns columns: country, year, emissions
    Also writes a small preview to out/debug_emissions_preview.csv for sanity-check.
    """
    xls = pd.read_excel(path, sheet_name=None, header=0, engine="openpyxl")
    frames = []
    for _, df in xls.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)

        # candidates for a country column
        country_like = []
        for c in cols:
            lc = c.lower()
            if lc in ["iso3", "iso_3", "iso", "code", "country_code"]:
                country_like.append(c)
            if lc in ["country", "country name", "country_name", "name", "nation"]:
                country_like.append(c)
        if not country_like:
            country_like = [cols[0]]

        # choose the best country column:
        chosen_country = None
        for c in country_like:
            sample = df[c].astype(str).str.strip()
            # if looks like ISO3 (AAA) for a good share of non-null rows, prefer it
            share_iso3 = (sample.str.fullmatch(r"[A-Z]{3}")).mean()
            if share_iso3 >= 0.6:  # threshold
                chosen_country = c
                break
        if chosen_country is None:
            # fallback to first candidate
            chosen_country = country_like[0]

        # find year columns (strict 4-digit)
        year_cols = [c for c in cols if re.fullmatch(r"(19|20)\d{2}", str(c))]
        if not year_cols:
            continue

        keep = [chosen_country] + year_cols
        slim = df[keep].dropna(how="all")

        long = slim.melt(
            id_vars=[chosen_country],
            value_vars=year_cols,
            var_name="year",
            value_name="emissions",
        ).dropna(subset=["emissions"])

        long = long.rename(columns={chosen_country: "country"})
        long["country"] = long["country"].astype(str).str.strip()
        # coerce emissions numeric, allow commas
        long["emissions"] = pd.to_numeric(
            long["emissions"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        long = long.dropna(subset=["emissions"])
        long["year"] = long["year"].astype(int)
        frames.append(long)

    if not frames:
        raise ValueError("Could not find a sheet in EDGAR Excel with country + 4-digit year columns.")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["country", "year"], keep="last")

    # write a quick preview for debugging
    preview = out.groupby("country", as_index=False).agg(
        min_year=("year", "min"),
        max_year=("year", "max"),
        n_years=("year", "nunique"),
        ex=("emissions", "size"),
    ).sort_values("n_years", ascending=False).head(50)
    ensure_output_dirs(os.getenv("OUT_DIR", "out"))
    preview.to_csv(os.path.join(os.getenv("OUT_DIR", "out"), "debug_emissions_preview.csv"), index=False)

    return out
