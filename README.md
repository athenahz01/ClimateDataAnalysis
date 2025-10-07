## Patent Ranking Project

Streaming ETL for very large Excel patent data, econometric estimation with fixed effects and clustered SEs, and leadership rankings by sector-year.

### Getting Started (3 minutes)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place input files under `data/`:
- `patent_green_epo.xlsx` (large xlsx; required)
- `oecd_patent_quality_country.xlsx` (optional)
- `emissions_controls.csv` (optional: country,sector,year,delta_CO2, controls)

3. Configure `.env` (copy `.env.example` to `.env` if needed).

4. Run end-to-end:
```bash
python main.py all
```

Outputs in `out/`:
- `cst_agg.parquet` aggregated counts and average quality
- `model_summary.txt` and `model_coefs.csv` if emissions provided
- `country_sector_year_ranking.csv` and `top_by_sector_year.xlsx`

### Commands
```bash
python main.py etl
python main.py model
python main.py rank
python main.py all
``` 