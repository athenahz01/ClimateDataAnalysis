# src/viz_q2_coverage.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.getenv("OUT_DIR", "out")
Q2_DIR = os.path.join(OUT_DIR, "q2")
os.makedirs(Q2_DIR, exist_ok=True)

CTRL_CSV = os.path.join(OUT_DIR, "controls_country_year.csv")
RANK_CSV = os.path.join(OUT_DIR, "country_sector_year_ranking.csv")

def load_controls():
    df = pd.read_csv(CTRL_CSV)
    df = df.dropna(subset=["country","year"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["country","year"]].drop_duplicates()

def load_ranking_years():
    r = pd.read_csv(RANK_CSV, usecols=["country","year"]).dropna().copy()
    r["year"] = r["year"].astype(int)
    return r

def build_overlap(ctrl, rank, start=None, end=None):
    # restrict to ranking countries and years; optionally clamp year window
    df = pd.merge(ctrl, rank, on=["country","year"], how="inner")
    if start is None:
        start = rank["year"].min()
    if end is None:
        end = rank["year"].max()
    df = df[(df["year"] >= start) & (df["year"] <= end)]
    return df, start, end

def presence_pivot(df):
    df = df.assign(present=1)
    piv = df.pivot_table(index="country", columns="year",
                         values="present", aggfunc="max").fillna(0)
    piv = piv.sort_index(axis=1)
    return piv

def plot_heatmap(piv_top, out_path):
    years = list(piv_top.columns)
    plt.figure(figsize=(10, 6))
    plt.imshow(piv_top.values, aspect="auto")
    plt.colorbar(label="Presence")
    plt.yticks(range(len(piv_top.index)), piv_top.index)
    step = max(1, len(years)//15)
    ticks = list(range(0, len(years), step))
    plt.xticks(ticks, [years[i] for i in ticks], rotation=45)
    plt.title("Controls–Ranking overlap by country and year (Top N)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_bars(coverage_counts, out_path):
    cc = coverage_counts.iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(cc.index)), cc.values)
    plt.yticks(range(len(cc.index)), cc.index)
    plt.xlabel("Overlap years (with ranking window)")
    plt.title("Overlap coverage by country (Top N)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_countries_per_year(df, out_path, start, end):
    per_year = df.groupby("year")["country"].nunique().reindex(range(start, end+1), fill_value=0)
    plt.figure(figsize=(9, 4))
    plt.plot(per_year.index, per_year.values, marker="o", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("# countries with overlap")
    plt.title("Countries covered per year (overlap)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="Top N countries by overlap years (default: 10)")
    ap.add_argument("--start", type=int, default=None, help="Clamp start year (default: ranking min)")
    ap.add_argument("--end", type=int, default=None, help="Clamp end year (default: ranking max)")
    args = ap.parse_args()

    ctrl = load_controls()
    rank = load_ranking_years()

    overlap, start, end = build_overlap(ctrl, rank, args.start, args.end)
    if overlap.empty:
        raise ValueError("No overlap between controls and ranking. Check files or year window.")

    piv = presence_pivot(overlap)
    counts = piv.sum(axis=1).sort_values(ascending=False)
    top_ids = counts.head(args.top).index
    piv_top = piv.loc[top_ids]

    # Export a compact table for appendix
    piv_top.to_csv(os.path.join(Q2_DIR, f"overlap_coverage_top{args.top}_table.csv"))

    plot_heatmap(piv_top, os.path.join(Q2_DIR, f"overlap_heatmap_top{args.top}.png"))
    plot_bars(counts.loc[top_ids], os.path.join(Q2_DIR, f"overlap_bars_top{args.top}.png"))
    plot_countries_per_year(overlap, os.path.join(Q2_DIR, "overlap_countries_per_year.png"), start, end)

    print("[viz_q2_coverage] wrote:")
    print(" ", os.path.join(Q2_DIR, f"overlap_heatmap_top{args.top}.png"))
    print(" ", os.path.join(Q2_DIR, f"overlap_bars_top{args.top}.png"))
    print(" ", os.path.join(Q2_DIR, "overlap_countries_per_year.png"))
    print(" ", os.path.join(Q2_DIR, f"overlap_coverage_top{args.top}_table.csv"))

if __name__ == "__main__":
    main()
