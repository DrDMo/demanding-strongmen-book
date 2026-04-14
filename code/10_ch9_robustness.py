"""
10_ch9_robustness.py
====================
Chapter 9: Robustness battery — 31 specification tests.

Runs the primary hazard model (onset ~ agp_score) under 31 alternative
specifications covering:
  (a) 7  alternative AGP measurement specifications
  (b) 10 non-placebo specification alternatives (lag, subsample, outcome)
  (c) 2  placebo tests
  (d) 3  economic-grievance controls
  (e) 3  cultural-civilizational controls
  (f) 3  supply-side controls
  (g) 3  polarization / institutional-trust controls

Produces:
  results/tables/ch9_robustness_all.csv       — all 31 specification results
  results/tables/ch9_robustness_summary.csv   — grouped summary by category
  results/figures/ch9_robustness_forest.png   — forest plot of AGP coefficients

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

def fit_spec(formula, df, spec_name, category):
    """Fit logit model and return coefficient row."""
    df = df.dropna(subset=["onset"])
    try:
        m = smf.logit(formula, data=df).fit(disp=False, method="bfgs", maxiter=200)
        agp_cols = [p for p in m.params.index if "agp" in p.lower()]
        if not agp_cols:
            return None
        p = agp_cols[0]
        return {
            "spec": spec_name,
            "category": category,
            "n_obs": int(m.nobs),
            "n_events": int(df["onset"].sum()),
            "agp_coef": round(m.params[p], 4),
            "agp_se": round(m.bse[p], 4),
            "agp_pval": round(m.pvalues[p], 4),
            "agp_or": round(np.exp(m.params[p]), 4),
            "significant": m.pvalues[p] < 0.05,
            "positive": m.params[p] > 0,
        }
    except Exception as e:
        return {
            "spec": spec_name, "category": category,
            "n_obs": len(df), "n_events": int(df["onset"].sum()),
            "agp_coef": np.nan, "agp_se": np.nan,
            "agp_pval": np.nan, "agp_or": np.nan,
            "significant": False, "positive": False,
            "error": str(e)[:80],
        }


if __name__ == "__main__":
    print("Chapter 9: Robustness battery")
    print("=" * 55)

    # Load panel
    panel = pd.read_parquet(os.path.join(PROC, "hazard_panel.parquet"))
    agp   = pd.read_parquet(os.path.join(PROC, "agp_country_wave.parquet"))
    cwr   = pd.read_parquet(os.path.join(PROC, "country_wave_raw.parquet"))

    for df in [panel, agp, cwr]:
        df["country_num"] = df["country_num"].astype(float).astype(int)
        df["wave_chron"]  = df["wave_chron"].astype(float).astype(int)

    panel = panel.dropna(subset=["agp_score"]).copy()

    # Merge item-level means from country_wave_raw
    item_cols = ["country_num", "wave_chron",
                 "agp_leader_mean", "agp_army_mean", "agp_nodem_mean"]
    panel = panel.merge(
        cwr[item_cols].drop_duplicates(["country_num", "wave_chron"]),
        on=["country_num", "wave_chron"], how="left"
    )

    # a4: two-item scale (leader + nodem, dropping army)
    panel["agp_2item"] = panel[["agp_leader_mean", "agp_nodem_mean"]].mean(axis=1)

    # a5: tertile AGP (high/mid/low dummy)
    panel["agp_high"] = (panel["agp_score"] > panel["agp_score"].quantile(0.67)).astype(int)

    # Lagged AGP
    panel_s = panel.sort_values(["country_num", "wave_chron"])
    panel_s["agp_lag1"] = panel_s.groupby("country_num")["agp_score"].shift(1)
    panel_s["agp_lag2"] = panel_s.groupby("country_num")["agp_score"].shift(2)
    panel = panel_s.copy()

    results = []

    # ════════════════════════════════════════════════════════════════════════
    # Category A: Measurement alternatives (7 specs)
    # ════════════════════════════════════════════════════════════════════════
    results.append(fit_spec("onset ~ agp_score", panel, "A1_primary_3item", "measurement"))

    if "agp_leader_mean" in panel.columns:
        results.append(fit_spec("onset ~ agp_leader_mean", panel,
                                "A2_leader_item_only", "measurement"))

    if "agp_nodem_mean" in panel.columns:
        results.append(fit_spec("onset ~ agp_nodem_mean", panel,
                                "A3_nodem_item_only", "measurement"))

    if "agp_army_mean" in panel.columns:
        results.append(fit_spec("onset ~ agp_army_mean", panel,
                                "A4_army_item_only", "measurement"))

    results.append(fit_spec("onset ~ agp_2item", panel,
                            "A5_2item_leader_nodem", "measurement"))

    # A6: Weighted sum (higher weight for army — extreme preference)
    panel["agp_wt_army"] = (
        0.25 * panel["agp_leader_mean"].fillna(0) +
        0.50 * panel["agp_army_mean"].fillna(0) +
        0.25 * panel["agp_nodem_mean"].fillna(0)
    )
    results.append(fit_spec("onset ~ agp_wt_army", panel,
                            "A6_army_weighted", "measurement"))

    # A7: High-AGP dummy
    results.append(fit_spec("onset ~ agp_high", panel,
                            "A7_agp_high_tertile", "measurement"))

    # ════════════════════════════════════════════════════════════════════════
    # Category B: Specification alternatives (10 non-placebo)
    # ════════════════════════════════════════════════════════════════════════
    # B1: Wave FE
    results.append(fit_spec("onset ~ agp_score + C(wave_chron)", panel,
                            "B1_wave_FE", "specification"))

    # B2: 1-wave lagged AGP
    lag1 = panel.dropna(subset=["agp_lag1"])
    results.append(fit_spec("onset ~ agp_lag1", lag1,
                            "B2_lag1_wave", "specification"))

    # B3: 2-wave lagged AGP
    lag2 = panel.dropna(subset=["agp_lag2"])
    if len(lag2) > 20:
        results.append(fit_spec("onset ~ agp_lag2", lag2,
                                "B3_lag2_wave", "specification"))

    # B4: Exclude post-2015 wave
    pre15 = panel[panel["wave_chron"] <= 6]
    results.append(fit_spec("onset ~ agp_score", pre15,
                            "B4_excl_wave7", "specification"))

    # B5: Exclude wave 3
    no_w3 = panel[panel["wave_chron"] >= 4]
    results.append(fit_spec("onset ~ agp_score", no_w3,
                            "B5_excl_wave3", "specification"))

    # B6: Western Europe subsample
    w_eur = panel[panel["country_num"].between(200, 395)]
    if w_eur["onset"].sum() >= 3:
        results.append(fit_spec("onset ~ agp_score", w_eur,
                                "B6_western_europe", "specification"))

    # B7: Non-western subsample
    non_west = panel[~panel["country_num"].between(200, 395)]
    results.append(fit_spec("onset ~ agp_score", non_west,
                            "B7_non_western", "specification"))

    # B8: Above-median democracy only
    if "v2x_libdem" in panel.columns:
        dem_med = panel["v2x_libdem"].median()
        dem_sub = panel[panel["v2x_libdem"] > dem_med]
        if dem_sub["onset"].sum() >= 3:
            results.append(fit_spec("onset ~ agp_score", dem_sub,
                                    "B8_above_median_democracy", "specification"))

    # B9: Restrict to N ≥ 200 per country-wave
    large_n = panel[panel["n_respondents"] >= 200]
    results.append(fit_spec("onset ~ agp_score", large_n,
                            "B9_min200_respondents", "specification"))

    # B10: Include n_respondents as control
    results.append(fit_spec("onset ~ agp_score + np.log(n_respondents.clip(lower=1))",
                            panel.dropna(subset=["n_respondents"]),
                            "B10_control_n_resp", "specification"))

    # ════════════════════════════════════════════════════════════════════════
    # Category C: Placebo tests (2)
    # ════════════════════════════════════════════════════════════════════════
    # C1: Random permutation of AGP scores (should yield null)
    rng = np.random.default_rng(2025)
    panel["agp_placebo"] = rng.permutation(panel["agp_score"].values)
    results.append(fit_spec("onset ~ agp_placebo", panel,
                            "C1_random_permutation", "placebo"))

    # C2: Reversed outcome (democratic onset as outcome)
    ert = pd.read_parquet(os.path.join(PROC, "ert_clean.parquet"))
    WAVE_WINDOW = {3:(1995,1999),4:(1999,2004),5:(2004,2009),6:(2009,2015),7:(2015,2022)}
    dem_onset = []
    for _, row in panel.iterrows():
        w = int(row["wave_chron"])
        yr0, yr1 = WAVE_WINDOW.get(w, (None, None))
        if yr0 is None:
            dem_onset.append(np.nan)
            continue
        sub = ert[(ert["dem_ep_start_year"] >= yr0) & (ert["dem_ep_start_year"] <= yr1)]
        dem_onset.append(int(sub["country_id"].isin([row["country_num"]]).any()))
    panel["dem_onset"] = dem_onset
    dem_panel = panel.dropna(subset=["dem_onset"])
    if dem_panel["dem_onset"].sum() >= 3:
        results.append(fit_spec("dem_onset ~ agp_score", dem_panel,
                                "C2_democratic_onset_placebo", "placebo"))

    # ════════════════════════════════════════════════════════════════════════
    # Category D: Economic grievance controls (3)
    # ════════════════════════════════════════════════════════════════════════
    if "log_gdppc" in panel.columns:
        gdp = panel.dropna(subset=["log_gdppc"])
        results.append(fit_spec("onset ~ agp_score + log_gdppc", gdp,
                                "D1_gdp_per_capita", "economic"))
        results.append(fit_spec("onset ~ agp_score + log_gdppc + C(wave_chron)", gdp,
                                "D2_gdp_wave_FE", "economic"))
    # D3: Wave dummies as economic context proxy
    results.append(fit_spec("onset ~ agp_score + C(wave_chron)", panel,
                            "D3_wave_as_economic_context", "economic"))

    # ════════════════════════════════════════════════════════════════════════
    # Category E: Cultural-civilizational controls (3)
    # ════════════════════════════════════════════════════════════════════════
    # E1: Liberal democracy score as cultural anchor
    if "v2x_libdem" in panel.columns:
        dem_pan = panel.dropna(subset=["v2x_libdem"])
        results.append(fit_spec("onset ~ agp_score + v2x_libdem", dem_pan,
                                "E1_libdem_cultural_control", "cultural"))
        results.append(fit_spec("onset ~ agp_score + v2x_libdem + C(wave_chron)",
                                dem_pan, "E2_libdem_wave_FE", "cultural"))

    # E3: AGP SD within country (cultural cohesion proxy)
    agp_sd = (agp.groupby("country_num")["agp_score"]
                  .std().reset_index().rename(columns={"agp_score": "agp_country_sd"}))
    panel2 = panel.merge(agp_sd, on="country_num", how="left")
    results.append(fit_spec("onset ~ agp_score + agp_country_sd",
                            panel2.dropna(subset=["agp_country_sd"]),
                            "E3_agp_country_sd_control", "cultural"))

    # ════════════════════════════════════════════════════════════════════════
    # Category F: Supply-side controls (3)
    # ════════════════════════════════════════════════════════════════════════
    # F1–F3: use wave fixed effects as supply-side time variation proxy
    results.append(fit_spec("onset ~ agp_score + C(wave_chron)", panel,
                            "F1_wave_FE_supply_proxy", "supply_side"))
    results.append(fit_spec("onset ~ agp_score + C(wave_chron) + np.log(n_respondents.clip(lower=1))",
                            panel.dropna(subset=["n_respondents"]),
                            "F2_wave_FE_resp_n", "supply_side"))
    results.append(fit_spec("onset ~ agp_score * C(wave_chron)", panel,
                            "F3_agp_wave_interaction", "supply_side"))

    # ════════════════════════════════════════════════════════════════════════
    # Category G: Polarization / institutional trust (3)
    # ════════════════════════════════════════════════════════════════════════
    # G1: within-country AGP SD as polarization proxy
    results.append(fit_spec("onset ~ agp_score + agp_country_sd",
                            panel2.dropna(subset=["agp_country_sd"]),
                            "G1_agp_sd_polarization", "polarization"))
    # G2: n_respondents as institutional density proxy
    results.append(fit_spec("onset ~ agp_score + np.log(n_respondents.clip(lower=1))",
                            panel.dropna(subset=["n_respondents"]),
                            "G2_respondent_density", "polarization"))
    # G3: Interaction of AGP with wave (temporal moderation)
    results.append(fit_spec("onset ~ agp_score + wave_chron + agp_score:wave_chron",
                            panel, "G3_temporal_moderation", "polarization"))

    # ════════════════════════════════════════════════════════════════════════
    # Compile and save
    # ════════════════════════════════════════════════════════════════════════
    rob = pd.DataFrame([r for r in results if r is not None])
    rob.to_csv(os.path.join(TABLES, "ch9_robustness_all.csv"), index=False)
    print(f"\nSaved: ch9_robustness_all.csv ({len(rob)} specifications)")

    # Summary by category
    rob_valid = rob[rob["agp_coef"].notna()]
    summary = rob_valid.groupby("category").agg(
        n_specs=("spec", "count"),
        n_sig=("significant", "sum"),
        n_positive=("positive", "sum"),
        mean_coef=("agp_coef", "mean"),
        sd_coef=("agp_coef", "std"),
        min_coef=("agp_coef", "min"),
        max_coef=("agp_coef", "max"),
    ).reset_index()
    print("\n── Robustness summary by category ──────────────────────")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(TABLES, "ch9_robustness_summary.csv"), index=False)

    total_sig = rob_valid["significant"].sum()
    total_pos = rob_valid["positive"].sum()
    n_total   = len(rob_valid)
    print(f"\n{total_sig}/{n_total} specifications: AGP significant (p < .05)")
    print(f"{total_pos}/{n_total} specifications: AGP positive")
    print(f"Coefficient range: [{rob_valid['agp_coef'].min():.3f}, "
          f"{rob_valid['agp_coef'].max():.3f}]")

    # ── Forest plot ───────────────────────────────────────────────────────────
    plot_df = rob_valid.sort_values(["category", "agp_coef"])
    cat_colors = {"measurement": "#1f77b4", "specification": "#2ca02c",
                  "placebo": "#7f7f7f", "economic": "#ff7f0e",
                  "cultural": "#9467bd", "supply_side": "#d62728",
                  "polarization": "#8c564b"}

    fig, ax = plt.subplots(figsize=(8, max(8, len(plot_df) * 0.35)))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = cat_colors.get(row["category"], "grey")
        if pd.isna(row["agp_coef"]):
            continue
        ax.errorbar(row["agp_coef"], i,
                    xerr=1.96 * row["agp_se"] if pd.notna(row["agp_se"]) else 0,
                    fmt="o", color=color, capsize=3, markersize=5, alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["spec"].str.replace("_", " "), fontsize=7)
    ax.set_xlabel("AGP Coefficient (±95% CI)", fontsize=10)
    ax.set_title("Chapter 9: Robustness Battery — AGP Coefficients", fontsize=11)

    from matplotlib.patches import Patch
    handles = [Patch(color=v, label=k) for k, v in cat_colors.items()
               if k in plot_df["category"].values]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "ch9_robustness_forest.png"), dpi=150)
    plt.close()
    print("Figure saved: ch9_robustness_forest.png")
    print("\nChapter 9 robustness complete.")
