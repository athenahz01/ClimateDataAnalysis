import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.getenv("OUT_DIR", "out")
RANK_PATH = os.path.join(OUT_DIR, "country_sector_year_ranking.csv")
FIG_DIR = os.path.join(OUT_DIR, "figs_q1")
os.makedirs(FIG_DIR, exist_ok=True)

def load_rank(metric: str) -> pd.DataFrame:
    df = pd.read_csv(RANK_PATH)
    if metric not in df.columns:
        # fallback if LI_plus missing
        if metric == "LI_plus":
            print("[viz_q1] LI_plus not found; falling back to LI_raw.")
            metric = "LI_raw"
    return df, metric

def top_countries_overall(df: pd.DataFrame, metric: str, top_n: int) -> pd.Index:
    winners = (
        df.groupby("country")[metric]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    return winners

def plot_timeseries(df: pd.DataFrame, metric: str, countries: pd.Index, fname: str):
    plt.figure(figsize=(12,6))
    for c in countries:
        g = df[df["country"] == c].sort_values("year")
        plt.plot(g["year"], g[metric], marker="o", label=c)
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title(f"Top {len(countries)} Countries in Climate Innovation Leadership ({metric})")
    plt.xlabel("Year"); plt.ylabel(metric)
    plt.legend(title="Country", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()

def plot_yearly_winners(df: pd.DataFrame, metric: str, fname: str):
    # one winner per year (averaging across sectors)
    yearly = (
        df.groupby(["country","year"], as_index=False)[metric].mean()
        .sort_values(["year", metric], ascending=[True, False])
    )
    winners = yearly.loc[yearly.groupby("year")[metric].idxmax()].reset_index(drop=True)
    winners.to_csv(os.path.join(FIG_DIR, "yearly_winners.csv"), index=False)

    plt.figure(figsize=(10,6))
    plt.scatter(winners["year"], winners[metric])
    for _, r in winners.iterrows():
        plt.text(r["year"], r[metric], r["country"], fontsize=8, ha="left", va="bottom")
    plt.title("Yearly Winner by Leadership Index")
    plt.xlabel("Year"); plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()

def plot_heatmap_country_year(df: pd.DataFrame, metric: str, countries: pd.Index, fname: str):
    # pivot to country x year matrix (avg across sectors)
    m = (
        df[df["country"].isin(countries)]
        .groupby(["country","year"], as_index=False)[metric].mean()
        .pivot(index="country", columns="year", values=metric)
        .sort_values(countries.name if hasattr(countries, "name") else None, ascending=False)
    )
    # ensure consistent order
    m = m.loc[list(countries)]
    plt.figure(figsize=(12, 0.6*len(countries) + 3))
    plt.imshow(m, aspect="auto")
    plt.colorbar(label=metric)
    plt.yticks(range(len(m.index)), m.index)
    plt.xticks(range(len(m.columns)), m.columns, rotation=90)
    plt.title(f"{metric} — Country x Year Heatmap (Top {len(countries)})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()
    m.to_csv(os.path.join(FIG_DIR, "country_year_heatmap_data.csv"))

def plot_sector_heatmap(df: pd.DataFrame, metric: str, countries: pd.Index, fname: str):
    # average across years within each (country, sector)
    m = (
        df[df["country"].isin(countries)]
        .groupby(["sector","country"], as_index=False)[metric].mean()
        .pivot(index="sector", columns="country", values=metric)
        .fillna(0)
    )
    plt.figure(figsize=(12, max(6, 0.4*len(m.index)+3)))
    plt.imshow(m, aspect="auto")
    plt.colorbar(label=metric)
    plt.yticks(range(len(m.index)), m.index)
    plt.xticks(range(len(m.columns)), m.columns, rotation=90)
    plt.title(f"{metric} — Sector Footprint by Country (Avg over years)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()
    m.to_csv(os.path.join(FIG_DIR, "sector_country_heatmap_data.csv"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="LI_plus", choices=["LI_plus","LI_raw"])
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    df, metric = load_rank(args.metric)
    countries = top_countries_overall(df, metric, args.top_n)

    # exports used in the slide
    plot_timeseries(df, metric, countries, f"leaders_timeseries_{metric}.png")
    plot_yearly_winners(df, metric, f"yearly_winner_{metric}.png")
    plot_heatmap_country_year(df, metric, countries, f"heatmap_country_year_{metric}.png")
    plot_sector_heatmap(df, metric, countries, f"heatmap_sector_{metric}.png")

    # write a compact “who’s leading” table for your appendix
    summary = (
        df.groupby("country")[metric]
        .agg(['mean','std','count'])
        .sort_values("mean", ascending=False)
        .head(args.top_n)
        .reset_index()
    )
    summary.to_csv(os.path.join(FIG_DIR, f"leaders_summary_{metric}.csv"), index=False)
    print(f"[viz_q1] Saved figures + tables to: {FIG_DIR}")

if __name__ == "__main__":
    main()
