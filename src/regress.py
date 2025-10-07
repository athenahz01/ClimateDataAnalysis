import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .schema import CANON
from .io_utils import write_text, write_csv
from .logger import get_logger


CONTROL_VARS = ["population", "trade_open", "carbon_intensity", "HDD", "CDD"]


def _try_build_model_df(df: pd.DataFrame, L: int, controls_present):
    """Return (model_df, p_lags, q_lags) using first L lags, or (None, _, _) if empty."""
    p_lags = [f"P_lag{i}" for i in range(1, L + 1)]
    q_lags = [f"Q_lag{i}" for i in range(1, L + 1)]
    need = ["delta_CO2", CANON["country"], "year"] + p_lags + q_lags + controls_present
    missing = [c for c in p_lags + q_lags if c not in df.columns]
    if missing:
        return None, p_lags, q_lags

    model_df = df[need].copy()
    # must have outcome and at least the selected lags
    model_df = model_df.dropna(subset=["delta_CO2"] + p_lags + q_lags)
    if model_df.empty:
        return None, p_lags, q_lags

    # create ln(1+P_lag)
    for c in p_lags:
        model_df[f"log1p_{c}"] = np.log1p(model_df[c])

    return model_df, p_lags, q_lags


def fit_delta_co2_model(df: pd.DataFrame, out_dir: str) -> Tuple[Optional[object], pd.DataFrame]:
    """
    ΔlnCO2_cst ~ Σ_{l=1..L} [ β_l * ln(1 + P_lag_l) + γ_l * Q_lag_l ] + controls + C(country) + C(year)
    Clustered SE by country. Automatically reduces L if data too sparse.
    """
    logger = get_logger()

    if "delta_CO2" not in df.columns:
        raise ValueError("delta_CO2 not found in dataframe")

    # Build formula
    lag_terms = []
    for L in range(1, 6):
        lag_terms.append(f"np.log(1 + P_lag{L})")
        lag_terms.append(f"Q_lag{L}")
    controls_present = [c for c in CONTROL_VARS if c in df.columns]
    formula = (
        "delta_CO2 ~ "
        + " + ".join(lag_terms + controls_present)
        + f" + C({CANON['country']}) + C({CANON['year']})"
    )

    # Drop rows with missing regressors
    required = ["delta_CO2"]
    for L in range(1, 6):
        required.extend([f"P_lag{L}", f"Q_lag{L}"])
    model_df = df.dropna(subset=required)
    if model_df.empty:
        logger.warning("No usable rows for regression after dropna; skipping model fit.")
        return None, pd.DataFrame(columns=["term", "coef", "se"])

    results = smf.ols(formula=formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df[CANON["country"]]})

    coefs = results.params.rename("coef").to_frame()
    coefs["se"] = results.bse
    coefs.reset_index(names=["term"], inplace=True)
    write_csv(coefs, os.path.join(out_dir, "model_coefs.csv"))
    logger.info("Saved model coefficients.")
    return results, coefs
