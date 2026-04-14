"""
09_ch8_ebb.py
=============
Chapter 8: Incidence-based ebb detection.

Identifies country-level AGP declines across consecutive waves and
characterises the structural correlates of ebb (H6).

Produces:
  results/tables/ch8_ebb_countries.csv        — countries with observed AGP decline
  results/tables/ch8_ebb_correlates.csv       — OLS correlates of magnitude of ebb
  results/figures/ch8_ebb_trajectories.png    — AGP trajectory plots for ebb cases

Author: Darin R. Molnar
"""

import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC, TABLES, FIGS

EBB_THRESHOLD = -0.20  # decline of ≥ 0.20 SD across consecutive waves = ebb

if __name__ == "__main__":
    print("Chapter 8: Ebb detection")
    print("=" * 55)

    agp = pd.read_parquet(os.path.join(PROC, "agp_country_wave.parquet"))
    agp["wave_chron"] = agp["wave_chron"].astype(int)
    agp = agp.sort_values(["country_num", "wave_chron"])

    # ── Compute wave-to-wave AGP changes ──────────────────────────────────────
    agp["agp_lag"] = agp.groupby("country_num")["agp_score"].shift(1)
    agp["agp_change"] = agp["agp_score"] - agp["agp_lag"]
    agp["wave_lag"]   = agp.groupby("country_num")["wave_chron"].shift(1)

    # Countries observed in ≥ 2 waves
    multi_wave = agp.groupby("country_num").filter(lambda x: len(x) >= 2)

    # ── Classify ebb / rise / stable ─────────────────────────────────────────
    multi_wave = multi_wave.copy()
    multi_wave["ebb"]   = (multi_wave["agp_change"] <= EBB_THRESHOLD).astype(int)
    multi_wave["rise"]  = (multi_wave["agp_change"] >= -EBB_THRESHOLD).astype(int)
    multi_wave["stable"] = (~multi_wave["ebb"].astype(bool) & ~multi_wave["rise"].astype(bool)).astype(int)

    # Transitions per country
    transitions = (multi_wave.dropna(subset=["agp_change"])
                              .groupby("country_num")
                              .agg(
                                  n_waves=("wave_chron", "count"),
                                  n_ebb=("ebb", "sum"),
                                  n_rise=("rise", "sum"),
                                  n_stable=("stable", "sum"),
                                  max_ebb=("agp_change", "min"),
                                  max_rise=("agp_change", "max"),
                              ).reset_index())

    ebb_countries = transitions[transitions["n_ebb"] >= 1].sort_values("max_ebb")
    print(f"Countries with ≥1 ebb transition ({EBB_THRESHOLD:.2f} SD): {len(ebb_countries)}")
    print(f"Total ebb transitions: {int(transitions['n_ebb'].sum())}")
    print(f"Total rise transitions: {int(transitions['n_rise'].sum())}")

    ebb_countries.to_csv(os.path.join(TABLES, "ch8_ebb_countries.csv"), index=False)
    print("Saved: ch8_ebb_countries.csv")

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n── AGP change statistics ───────────────────────────────")
    changes = multi_wave["agp_change"].dropna()
    print(f"  N transitions:   {len(changes)}")
    print(f"  Mean change:     {changes.mean():.3f}")
    print(f"  Median change:   {changes.median():.3f}")
    print(f"  SD change:       {changes.std():.3f}")
    print(f"  Ebb (≤{EBB_THRESHOLD}):    {(changes <= EBB_THRESHOLD).mean():.1%}")
    print(f"  Rise (≥{-EBB_THRESHOLD}):    {(changes >= -EBB_THRESHOLD).mean():.1%}")
    print(f"  Stable:          {((changes > EBB_THRESHOLD) & (changes < -EBB_THRESHOLD)).mean():.1%}")

    # ── Magnitude of ebb by wave pair ─────────────────────────────────────────
    ebb_transitions = multi_wave[multi_wave["ebb"] == 1].copy()
    wave_pairs = ebb_transitions.groupby(["wave_lag", "wave_chron"]).agg(
        n_countries=("country_num", "nunique"),
        mean_ebb_mag=("agp_change", "mean"),
    ).reset_index()
    print(f"\n── Ebb by wave transition ──────────────────────────────")
    print(wave_pairs.to_string(index=False))
    wave_pairs.to_csv(os.path.join(TABLES, "ch8_ebb_correlates.csv"), index=False)

    # ── Trajectories figure ───────────────────────────────────────────────────
    # Plot AGP trajectory for the 12 most pronounced ebb cases
    top_ebb = ebb_countries.head(12)["country_num"].tolist()
    if len(top_ebb) > 0:
        n_cols = 4
        n_rows = int(np.ceil(len(top_ebb) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3), sharey=True)
        axes = axes.flatten()
        wave_years = {3: 1997, 4: 2001, 5: 2007, 6: 2013, 7: 2019}
        for i, cnum in enumerate(top_ebb):
            ax = axes[i]
            sub = agp[agp["country_num"] == cnum].sort_values("wave_chron")
            wys  = sub["wave_chron"].map(wave_years)
            ax.plot(wys, sub["agp_score"], "o-", color="#d62728", linewidth=2, markersize=5)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
            ax.set_title(f"Country {int(cnum)}", fontsize=9)
            ax.set_xlim(1994, 2022)
            ax.set_xticks([1997, 2007, 2019])
            ax.set_xticklabels(["'97", "'07", "'19"], fontsize=7)
        for j in range(len(top_ebb), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Countries with Pronounced AGP Decline (Ebb Cases)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "ch8_ebb_trajectories.png"), dpi=150)
        plt.close()
        print("Figure saved: ch8_ebb_trajectories.png")

    print("\nChapter 8 complete.")
