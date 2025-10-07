from typing import Dict, List

CANON = {
	"country": "country",
	"sector": "sector",
	"year": "year",
	"patent_id": "patent_id",
	"quality": "quality",
}

CANDIDATES: Dict[str, List[str]] = {
	"country": [
        "appln_auth", "publn_auth",
        "country", "assignee_country", "applicant_country",
        "orig_country", "residence_country",
        "iso", "iso3", "country_code", "appln_ctry"
    ],
	"sector": [
        "sector", "technology_field", "y02_sector", "cpc_sector",
        "tech_field"
    ],
	"year": [
        "year", "pub_year", "publication_year",
        "earliest_filing_date", "appln_filing_date",
        "earliest_publn_year", "appln_year", "filing_year"
    ],
	"patent_id": [
        "appln_id", "publication_number", "publn_nr",
        "doc_number", "patent_id"
    ],
	"quality": [
        "quality_index", "oecd_quality", "quality",
        "composite_index", "composite_index_4", "composite_index_6",
        "nb_citing_docdb_fam", "docdb_family_size", "granted"
    ],
}

NORMALIZE_MAP = str.maketrans({" ": "", "-": "", "_": "", "/": ""})


def normalize_header(s: str) -> str:
	return (s or "").strip().lower().translate(NORMALIZE_MAP)


def fuzzy_detect(header_row: List[str]) -> Dict[str, int]:
	norm_headers = [normalize_header(h) for h in header_row]
	out: Dict[str, int] = {}
	for key, opts in CANDIDATES.items():
		opts_norm = [normalize_header(o) for o in opts]
		for idx, h in enumerate(norm_headers):
			if h in opts_norm:
				out[key] = idx
				break
	return out
