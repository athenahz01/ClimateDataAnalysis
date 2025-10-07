import os
from typing import Optional

import numpy as np
import pandas as pd

from .features import zscore
from .schema import CANON
from .io_utils import write_csv, write_excel
from .logger import get_logger


def _load_model_coefs(out_dir: str) -> Optional[pd.DataFrame]:
	path = os.path.join(out_dir, "model_coefs.csv")
	if os.path.exists(path):
		return pd.read_csv(path)
	return None


def compute_li(df: pd.DataFrame, use_model: bool, out_dir: str) -> pd.DataFrame:
	logger = get_logger()
	df = df.copy()
	# LI_raw
	df["LI_raw"] = zscore(df["P"].fillna(0)) + (zscore(df["Q"]) if "Q" in df.columns else 0)
	if use_model:
		coefs = _load_model_coefs(out_dir)
		if coefs is None:
			use_model = False
		else:
			beta = {}
			gamma = {}
			for L in range(1, 6):
				beta[f"P_lag{L}"] = coefs.loc[coefs["term"] == f"np.log(1 + P_lag{L})", "coef"].squeeze() if (coefs["term"] == f"np.log(1 + P_lag{L})").any() else 0.0
				gamma[f"Q_lag{L}"] = coefs.loc[coefs["term"] == f"Q_lag{L}", "coef"].squeeze() if (coefs["term"] == f"Q_lag{L}").any() else 0.0
			# compute elasticity term
			elas = 0.0
			for L in range(1, 6):
				elas += beta[f"P_lag{L}"] * np.log1p(df.get(f"P_lag{L}", 0).fillna(0))
				elas += gamma[f"Q_lag{L}"] * df.get(f"Q_lag{L}", 0).fillna(0)
			df["Elasticity"] = elas
			df["LI_plus"] = zscore(df["P"].fillna(0)) + (zscore(df["Q"]) if "Q" in df.columns else 0) + zscore(df["Elasticity"].fillna(0))
	else:
		logger.warning("Model not available; computing LI_raw only.")
	return df


def export_rankings(df: pd.DataFrame, out_dir: str) -> None:
	rank_df = df[[CANON["country"], CANON["sector"], CANON["year"], "LI_raw"] + (["LI_plus"] if "LI_plus" in df.columns else [])].copy()
	rank_df["LI_raw_rank"] = rank_df.groupby([CANON["sector"], CANON["year"]])["LI_raw"].rank(ascending=False, method="dense")
	if "LI_plus" in rank_df.columns:
		rank_df["LI_plus_rank"] = rank_df.groupby([CANON["sector"], CANON["year"]])["LI_plus"].rank(ascending=False, method="dense")

	write_csv(rank_df, os.path.join(out_dir, "country_sector_year_ranking.csv"))

	# Export top 20 by sector-year
	tops = []
	for (sector, year), g in rank_df.groupby([CANON["sector"], CANON["year"]]):
		g2 = g.nsmallest(20, columns=["LI_plus_rank"] if "LI_plus_rank" in g.columns else ["LI_raw_rank"]).copy()
		g2["sector"] = sector
		g2["year"] = year
		tops.append(g2)
	if tops:
		top_df = pd.concat(tops, ignore_index=True)
		write_excel(top_df, os.path.join(out_dir, "top_by_sector_year.xlsx"), sheet_name="top20")
