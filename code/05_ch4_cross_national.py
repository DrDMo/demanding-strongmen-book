"""
05_ch4_cross_national.py
========================
Chapter 4: Cross-national distribution of AGP.

Produces:
  results/tables/ch4_agp_regional_means.csv       — AGP means by region × wave
  results/tables/ch4_agp_country_rankings.csv     — country rankings, latest wave
  results/tables/ch4_agp_trend_summary.csv        — H1 heterogeneity statistics
  results/figures/ch4_agp_distribution.png        — density plot by wave
  results/figures/ch4_agp_regional_trends.png     — regional trend lines

Author: Darin R. Molnar
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import f_oneway, bartlett

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC, TABLES, FIGS

# ── UN region mapping by COW country code ─────────────────────────────────────
# Approximate mapping: Eastern Europe & FSU / Western Europe & Offshoots /
# Latin America / Africa / Middle East & N.Africa / Asia-Pacific
REGION_MAP = {
    # Western Europe & offshoots
    **{c: "West" for c in [
        2,20,70,90,100,130,135,140,145,150,155,160,165,200,205,210,211,
        212,215,220,225,230,235,240,245,255,260,265,269,273,275,280,290,
        305,310,316,317,325,327,338,339,344,347,349,350,352,355,359,360,
        364,366,368,369,371,373,375,380,385,390,395,640,712,732
    ]},
    # Eastern Europe & FSU
    **{c: "EE/FSU" for c in [
        315,316,317,340,341,343,344,346,349,350,352,355,356,359,360,365,
        366,367,368,369,371,372,373,375,381,385,390,395,670,703,705,706,
        707,708,710,711,713,715,716,717,718,720,731,740,762,764
    ]},
    # Latin America
    **{c: "Latin America" for c in [
        41,42,51,52,53,54,55,56,57,58,60,70,80,90,91,92,93,94,95,100,
        101,110,115,130,135,140,145,150,155,160,165
    ]},
    # Africa
    **{c: "Africa" for c in [
        402,404,411,420,432,433,434,435,436,437,438,439,450,451,452,461,
        471,475,481,482,483,484,490,500,501,510,516,517,520,522,530,531,
        540,541,542,552,553,560,565,570,571,572,580,581,590,600
    ]},
    # Middle East & North Africa
    **{c: "MENA" for c in [
        600,615,616,620,625,630,640,645,651,652,660,663,666,670,678,679,680,
        690,694,696,698,699,700,701,702,703,704,705,706,770
    ]},
}


def assign_region(country_num):
    """Assign broad world-region using approximate COW code ranges."""
    c = int(country_num)
    if c in REGION_MAP:
        return REGION_MAP[c]
    if 2 <= c <= 165:
        return "Latin America"
    if 200 <= c <= 395:
        return "West"
    if 400 <= c <= 599:
        return "Africa"
    if 600 <= c <= 699:
        return "MENA"
    if 700 <= c <= 899:
        return "Asia-Pacific"
    return "Other"


WAVE_LABEL = {3: "Wave 3\n(~1997)", 4: "Wave 4\n(~2001)",
              5: "Wave 5\n(~2007)", 6: "Wave 6\n(~2013)", 7: "Wave 7\n(~2019)"}
REGION_ORDER = ["West", "EE/FSU", "Latin America", "MENA", "Africa", "Asia-Pacific"]
REGION_COLORS = {"West": "#1f77b4", "EE/FSU": "#d62728", "Latin America": "#2ca02c",
                 "MENA": "#ff7f0e", "Africa": "#9467bd", "Asia-Pacific": "#8c564b"}


if __name__ == "__main__":
    print("Chapter 4: Cross-national AGP distribution")
    print("=" * 55)

    agp = pd.read_parquet(os.path.join(PROC, "agp_country_wave.parquet"))
    agp["region"] = agp["country_num"].apply(assign_region)
    agp["wave_year"] = agp["wave_chron"].map({3:1997,4:2001,5:2007,6:2013,7:2019})

    # ── 1. H1: Heterogeneity test ──────────────────────────────────────────────
    print("\n── H1: Within-wave variance and heterogeneity ──────────")
    het_rows = []
    for wave in sorted(agp["wave_chron"].unique()):
        sub = agp[agp["wave_chron"] == wave]["agp_score"]
        het_rows.append({
            "wave": int(wave),
            "wave_year": int(agp[agp["wave_chron"] == wave]["wave_year"].iloc[0]),
            "n_countries": len(sub),
            "mean_agp": sub.mean(),
            "sd_agp": sub.std(),
            "min_agp": sub.min(),
            "max_agp": sub.max(),
            "range_agp": sub.max() - sub.min(),
            "cv_agp": sub.std() / abs(sub.mean()) if abs(sub.mean()) > 0.01 else np.nan,
        })
    het_df = pd.DataFrame(het_rows)
    print(het_df.to_string(index=False))

    # One-way ANOVA across regions within each wave (tests H1)
    print("\nOne-way ANOVA (region differences, latest wave):")
    w7 = agp[agp["wave_chron"] == 7]
    groups = [w7[w7["region"] == r]["agp_score"].values
              for r in REGION_ORDER if (w7["region"] == r).sum() > 2]
    F, p = f_oneway(*groups)
    print(f"  F({len(groups)-1}, {len(w7)-len(groups)}) = {F:.2f}, p = {p:.4f}")

    het_df.to_csv(os.path.join(TABLES, "ch4_agp_trend_summary.csv"), index=False)

    # ── 2. Regional means by wave ─────────────────────────────────────────────
    print("\n── Regional AGP means by wave ──────────────────────────")
    reg_wave = (agp.groupby(["region", "wave_chron"])
                   .agg(mean_agp=("agp_score", "mean"),
                        sd_agp=("agp_score", "std"),
                        n=("agp_score", "count"))
                   .reset_index())
    reg_wave["wave_year"] = reg_wave["wave_chron"].map({3:1997,4:2001,5:2007,6:2013,7:2019})
    print(reg_wave.to_string(index=False))
    reg_wave.to_csv(os.path.join(TABLES, "ch4_agp_regional_means.csv"), index=False)

    # ── 3. Country rankings, latest available wave ────────────────────────────
    latest = (agp.sort_values("wave_chron", ascending=False)
                 .drop_duplicates("country_num")
                 [["country_num", "wave_chron", "wave_year", "agp_score", "n", "region"]])
    latest = latest.sort_values("agp_score", ascending=False).reset_index(drop=True)
    latest.index += 1
    print(f"\n── Country rankings (latest wave, n={len(latest)}) ─────────────")
    print(latest.head(20).to_string())
    latest.to_csv(os.path.join(TABLES, "ch4_agp_country_rankings.csv"))

    # ── 4. Figure: AGP density by wave ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    for i, wave in enumerate([3, 4, 5, 6, 7]):
        sub = agp[agp["wave_chron"] == wave]["agp_score"].dropna()
        if len(sub) < 5:
            continue
        sub.plot.kde(ax=ax, color=colors[i], linewidth=2,
                     label=f"Wave {wave} (~{wave*6+1979})")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("AGP Score (standardised)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Cross-national AGP Distribution by Survey Wave", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-3, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "ch4_agp_distribution.png"), dpi=150)
    plt.close()
    print(f"\nFigure saved: ch4_agp_distribution.png")

    # ── 5. Figure: Regional trend lines ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for region in REGION_ORDER:
        sub = reg_wave[reg_wave["region"] == region].sort_values("wave_year")
        if len(sub) < 2:
            continue
        color = REGION_COLORS.get(region, "grey")
        ax.plot(sub["wave_year"], sub["mean_agp"], "o-",
                color=color, linewidth=2, markersize=6, label=region)
        ax.fill_between(sub["wave_year"],
                        sub["mean_agp"] - sub["sd_agp"],
                        sub["mean_agp"] + sub["sd_agp"],
                        color=color, alpha=0.1)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Survey wave year", fontsize=11)
    ax.set_ylabel("Mean AGP Score (±1 SD)", fontsize=11)
    ax.set_title("Regional AGP Trends, 1997–2019", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks([1997, 2001, 2007, 2013, 2019])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "ch4_agp_regional_trends.png"), dpi=150)
    plt.close()
    print("Figure saved: ch4_agp_regional_trends.png")
    print("\nChapter 4 complete.")
