import argparse
import os
from dotenv import load_dotenv

from src.logger import get_logger
from src.io_utils import ensure_output_dirs, write_text
from src.etl_stream_excel import stream_aggregate_xlsx
from src.features import prepare_features
from src.regress import fit_delta_co2_model
from src.ranking import compute_li, export_rankings


def main():
	load_dotenv()
	logger = get_logger()
	out_dir = os.getenv("OUT_DIR", "out")
	ensure_output_dirs(out_dir)

	parser = argparse.ArgumentParser(description="Patent Ranking Project CLI")
	parser.add_argument("command", choices=["etl", "model", "rank", "all"], help="What to run")
	args = parser.parse_args()

	if args.command in ("etl", "all"):
		logger.info("Starting ETL streaming aggregation…")
		stream_aggregate_xlsx(os.getenv("XL_PATENTS", "data/patent_green_epo.xlsx"))
		logger.info("ETL completed.")

	if args.command in ("model", "all"):
		logger.info("Preparing features for regression…")
		features_df = prepare_features(out_dir)
		if "delta_CO2" in features_df.columns:
			logger.info("Fitting ΔlnCO2 model with FE and clustered SE…")
			results, coefs_df = fit_delta_co2_model(features_df, out_dir)
			write_text(os.path.join(out_dir, "model_summary.txt"), results.summary().as_text())
			logger.info("Model estimation completed.")
		else:
			logger.warning("delta_CO2 not found; skipping regression.")

	if args.command in ("rank", "all"):
		logger.info("Computing leadership indices and rankings…")
		features_df = prepare_features(out_dir)
		use_model = os.path.exists(os.path.join(out_dir, "model_coefs.csv")) and "delta_CO2" in features_df.columns
		rank_df = compute_li(features_df, use_model=use_model, out_dir=out_dir)
		export_rankings(rank_df, out_dir)
		logger.info("Ranking exports completed.")


if __name__ == "__main__":
	main()
