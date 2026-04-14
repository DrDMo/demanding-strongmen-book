"""
08_ch7_forecasting.py
=====================
Chapter 7: Ten-year autocratization risk forecasting.

Uses the primary hazard model (Ch6) to generate country-specific
10-year probability-of-onset forecasts from the latest observed AGP score.

Monte Carlo simulation propagates uncertainty in AGP scores and model
coefficients to produce posterior forecast distributions.

Produces:
  results/tables/ch7_forecasts_10yr.csv        — country 10-yr risk forecasts
  results/figures/ch7_forecast_top20.png       — top 20 highest-risk countries

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

N_SIM = 2500  # Monte Carlo draws

if __name__ == "__main__":
    print("Chapter 7: Ten-year autocratization risk forecasting")
    print("=" * 55)

    panel = pd.read_parquet(os.path.join(PROC, "hazard_panel.parquet"))
    base  = panel.dropna(subset=["agp_score", "onset"])

    # ── Fit primary model ─────────────────────────────────────────────────────
    model = smf.logit("onset ~ agp_score", data=base).fit(disp=False)
    beta0_hat = model.params["Intercept"]
    beta1_hat = model.params["agp_score"]
    cov_mat   = model.cov_params().values
    print(f"Primary model: β₀={beta0_hat:.3f}, β₁={beta1_hat:.3f}, "
          f"SE(β₁)={model.bse['agp_score']:.3f}")

    # ── Latest AGP score per country ──────────────────────────────────────────
    latest = (panel.sort_values("wave_chron", ascending=False)
                   .drop_duplicates("country_num")
                   [["country_num", "wave_chron", "wave_year",
                     "agp_score", "agp_score_se", "n_respondents"]])
    print(f"Countries with latest AGP: {len(latest)}")

    # ── Monte Carlo 10-year forecast ──────────────────────────────────────────
    # Per-wave onset probability: logistic(β₀ + β₁ * AGP)
    # 10-year probability (2 waves): 1 - (1 - p_wave)^2
    rng = np.random.default_rng(42)

    forecast_rows = []
    for _, row in latest.iterrows():
        agp_mu = row["agp_score"]
        agp_se = row["agp_score_se"] if pd.notna(row["agp_score_se"]) else 0.05

        # Draw coefficient uncertainty from multivariate normal
        coef_draws = rng.multivariate_normal([beta0_hat, beta1_hat],
                                             cov_mat, size=N_SIM)
        b0_sim = coef_draws[:, 0]
        b1_sim = coef_draws[:, 1]

        # Draw AGP uncertainty
        agp_sim = rng.normal(agp_mu, agp_se, size=N_SIM)

        # Per-wave logistic probability
        lp = b0_sim + b1_sim * agp_sim
        p_wave = 1 / (1 + np.exp(-lp))

        # 10-year (2-wave) cumulative onset probability
        p_10yr = 1 - (1 - p_wave) ** 2

        forecast_rows.append({
            "country_num":    row["country_num"],
            "wave_chron":     row["wave_chron"],
            "wave_year":      row["wave_year"],
            "agp_score":      agp_mu,
            "p10yr_mean":     p_10yr.mean(),
            "p10yr_median":   np.median(p_10yr),
            "p10yr_lo95":     np.percentile(p_10yr, 2.5),
            "p10yr_hi95":     np.percentile(p_10yr, 97.5),
            "p10yr_lo80":     np.percentile(p_10yr, 10),
            "p10yr_hi80":     np.percentile(p_10yr, 90),
            "n_sim":          N_SIM,
        })

    forecasts = pd.DataFrame(forecast_rows).sort_values("p10yr_mean", ascending=False)

    # Save
    forecasts.to_csv(os.path.join(TABLES, "ch7_forecasts_10yr.csv"), index=False)
    print(f"\nSaved: ch7_forecasts_10yr.csv ({len(forecasts)} countries)")
    print("\nTop 15 highest 10-year autocratization risk:")
    print(forecasts.head(15)[["country_num", "wave_chron", "agp_score",
                               "p10yr_mean", "p10yr_lo95", "p10yr_hi95"]].to_string(index=False))

    # ── Figure: top 20 forecast ───────────────────────────────────────────────
    top20 = forecasts.head(20).sort_values("p10yr_mean")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(top20)), top20["p10yr_mean"],
            xerr=[top20["p10yr_mean"] - top20["p10yr_lo80"],
                  top20["p10yr_hi80"] - top20["p10yr_mean"]],
            color="#d62728", alpha=0.7, capsize=3)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"Country {int(c)}" for c in top20["country_num"]], fontsize=8)
    ax.set_xlabel("10-year Pr(autocratization onset) [mean ± 80% CI]", fontsize=10)
    ax.set_title("Top 20 Highest 10-Year Autocratization Risk Countries", fontsize=11)
    ax.axvline(forecasts["p10yr_mean"].median(), color="grey",
               linestyle="--", linewidth=0.8, label="Median")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "ch7_forecast_top20.png"), dpi=150)
    plt.close()
    print("Figure saved: ch7_forecast_top20.png")
    print("\nChapter 7 complete.")
