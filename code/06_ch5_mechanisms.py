"""
06_ch5_mechanisms.py
====================
Chapter 5: Individual-level mechanism models.

Tests H2 (insecurity + institutional trust → AGP),
      H3 (emancipative values → AGP),
      H4 (insecurity × emancipative values interaction).

Uses mixed-effects OLS (country × wave random intercepts) via statsmodels MixedLM.

Produces:
  results/tables/ch5_mechanisms_models.csv     — model coefficients for H2–H4
  results/tables/ch5_mechanisms_summary.csv    — model fit statistics
  results/figures/ch5_mechanism_effects.png    — coefficient plot

Author: Darin R. Molnar
"""

import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC, TABLES, FIGS

# ── Variable construction ──────────────────────────────────────────────────────

def build_individual_data():
    """Load IVS individual file and construct mechanism variables."""
    print("Loading IVS individual file...")
    ivs = pd.read_parquet(os.path.join(PROC, "ivs_individual.parquet"))
    print(f"  Shape: {ivs.shape}")

    MISSING = [-1, -2, -3, -4, -5]
    for col in ivs.select_dtypes(include=[float, int]).columns:
        ivs[col] = ivs[col].where(~ivs[col].isin(MISSING), np.nan)

    # Outcome: agp_mean (already recoded in 02)
    # ── Economic insecurity proxy: income position (X047_WVS / X046)
    income_col = None
    for c in ["X047_WVS", "X047", "X046"]:
        if c in ivs.columns and ivs[c].notna().sum() > 1000:
            income_col = c
            break

    if income_col:
        ivs["insecurity"] = -1 * ivs[income_col]  # low income = high insecurity
        # Standardise within wave × country
        ivs["insecurity"] = ivs.groupby(["country_num", "wave_chron"])["insecurity"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    else:
        ivs["insecurity"] = np.nan

    # ── Institutional trust index: mean of available E069_ items
    trust_cols = [c for c in ["E069_01", "E069_02", "E069_06", "E069_07",
                               "E069_11", "E069_12", "E069_17"]
                  if c in ivs.columns]
    if trust_cols:
        trust_raw = ivs[trust_cols].copy()
        # Recode: 1=A great deal, 4=None at all → reverse so high=trusting
        for col in trust_cols:
            trust_raw[col] = 5 - trust_raw[col]
        ivs["inst_trust"] = trust_raw.mean(axis=1, skipna=True)
        ivs["inst_trust"] = ivs.groupby(["country_num", "wave_chron"])["inst_trust"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    else:
        ivs["inst_trust"] = np.nan

    # ── Emancipative values proxy: Y002 (postmaterialism index) or A165 (social trust)
    # Use Y002 if available; it captures autonomy/self-expression orientation
    ev_col = None
    for c in ["Y002", "Y003"]:
        if c in ivs.columns and ivs[c].notna().sum() > 10000:
            ev_col = c
            break

    if ev_col:
        ivs["emancipative_vals"] = ivs[ev_col]
        ivs["emancipative_vals"] = ivs.groupby(["country_num", "wave_chron"])["emancipative_vals"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    else:
        ivs["emancipative_vals"] = np.nan

    # ── Demographics
    if "X001" in ivs.columns:
        ivs["female"] = (ivs["X001"] == 2).astype(float)
    else:
        ivs["female"] = np.nan

    if "X003" in ivs.columns:
        age = ivs["X003"].copy()
        age = age.where((age >= 15) & (age <= 99), np.nan)
        ivs["age"] = (age - age.mean()) / age.std()
    else:
        ivs["age"] = np.nan

    if "X025" in ivs.columns:
        ivs["edu"] = ivs["X025"]
        ivs["edu"] = ivs.groupby(["country_num", "wave_chron"])["edu"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    else:
        ivs["edu"] = np.nan

    # ── Group identifier for random effects
    ivs["country_wave"] = (
        ivs["country_num"].astype(str) + "_" + ivs["wave_chron"].astype(str)
    )

    # ── Keep analysis sample
    analysis_vars = ["agp_mean", "insecurity", "inst_trust", "emancipative_vals",
                     "female", "age", "edu", "country_wave", "country_num", "wave_chron"]
    sample = ivs[analysis_vars].dropna(subset=["agp_mean"])
    print(f"  Analysis sample: {len(sample):,} individuals, "
          f"{sample['country_wave'].nunique()} country-waves")
    return sample


def run_mechanisms(sample):
    """Run H2, H3, H4 multilevel models."""
    models_out = []

    # ── Model 1: Demographics only (baseline) ─────────────────────────────────
    base_vars = [v for v in ["female", "age", "edu"] if sample[v].notna().mean() > 0.5]
    m1_df = sample[["agp_mean", "country_wave"] + base_vars].dropna()
    if len(m1_df) > 1000:
        formula1 = "agp_mean ~ " + " + ".join(base_vars) if base_vars else "agp_mean ~ 1"
        m1 = smf.mixedlm(formula1, m1_df, groups=m1_df["country_wave"]).fit(reml=False)
        for name, coef, se, pval in zip(
            m1.params.index, m1.params, m1.bse, m1.pvalues
        ):
            models_out.append({"model": "M1_baseline", "predictor": name,
                                "coef": coef, "se": se, "pval": pval,
                                "n": int(m1.nobs), "ngroups": m1_df["country_wave"].nunique()})
        print(f"  M1 (baseline): n={int(m1.nobs):,}, groups={m1_df['country_wave'].nunique()}")

    # ── Model 2: H2 — Insecurity and institutional trust ─────────────────────
    h2_vars = [v for v in ["insecurity", "inst_trust"] + base_vars
               if sample[v].notna().mean() > 0.4]
    m2_df = sample[["agp_mean", "country_wave"] + h2_vars].dropna()
    if len(m2_df) > 1000 and len(h2_vars) > 0:
        formula2 = "agp_mean ~ " + " + ".join(h2_vars)
        m2 = smf.mixedlm(formula2, m2_df, groups=m2_df["country_wave"]).fit(reml=False)
        for name, coef, se, pval in zip(
            m2.params.index, m2.params, m2.bse, m2.pvalues
        ):
            models_out.append({"model": "M2_H2_insecurity_trust", "predictor": name,
                                "coef": coef, "se": se, "pval": pval,
                                "n": int(m2.nobs), "ngroups": m2_df["country_wave"].nunique()})
        ins_coef = m2.params.get("insecurity", np.nan)
        tru_coef = m2.params.get("inst_trust", np.nan)
        print(f"  M2 (H2): insecurity β={ins_coef:.3f}, inst_trust β={tru_coef:.3f}, "
              f"n={int(m2.nobs):,}")

    # ── Model 3: H3 — Emancipative values ─────────────────────────────────────
    h3_vars = [v for v in ["emancipative_vals"] + h2_vars
               if sample[v].notna().mean() > 0.4]
    m3_df = sample[["agp_mean", "country_wave"] + h3_vars].dropna()
    if len(m3_df) > 1000 and "emancipative_vals" in h3_vars:
        formula3 = "agp_mean ~ " + " + ".join(h3_vars)
        m3 = smf.mixedlm(formula3, m3_df, groups=m3_df["country_wave"]).fit(reml=False)
        for name, coef, se, pval in zip(
            m3.params.index, m3.params, m3.bse, m3.pvalues
        ):
            models_out.append({"model": "M3_H3_emancipative", "predictor": name,
                                "coef": coef, "se": se, "pval": pval,
                                "n": int(m3.nobs), "ngroups": m3_df["country_wave"].nunique()})
        ev_coef = m3.params.get("emancipative_vals", np.nan)
        print(f"  M3 (H3): emancipative_vals β={ev_coef:.3f}, n={int(m3.nobs):,}")

    # ── Model 4: H4 — Backlash conditionality (interaction) ───────────────────
    if ("insecurity" in sample.columns and "emancipative_vals" in sample.columns):
        h4_df = sample[["agp_mean", "country_wave", "insecurity",
                         "emancipative_vals"] + base_vars].dropna()
        if len(h4_df) > 1000:
            formula4 = "agp_mean ~ insecurity * emancipative_vals"
            if base_vars:
                formula4 += " + " + " + ".join(base_vars)
            m4 = smf.mixedlm(formula4, h4_df, groups=h4_df["country_wave"]).fit(reml=False)
            for name, coef, se, pval in zip(
                m4.params.index, m4.params, m4.bse, m4.pvalues
            ):
                models_out.append({"model": "M4_H4_interaction", "predictor": name,
                                    "coef": coef, "se": se, "pval": pval,
                                    "n": int(m4.nobs), "ngroups": h4_df["country_wave"].nunique()})
            ix_coef = m4.params.get("insecurity:emancipative_vals", np.nan)
            print(f"  M4 (H4): insecurity×emancipative_vals β={ix_coef:.3f}, n={int(m4.nobs):,}")

    return pd.DataFrame(models_out)


if __name__ == "__main__":
    print("Chapter 5: Individual-level mechanisms")
    print("=" * 55)

    sample = build_individual_data()
    results = run_mechanisms(sample)

    # Save results
    results.to_csv(os.path.join(TABLES, "ch5_mechanisms_models.csv"), index=False)
    print(f"\nSaved: ch5_mechanisms_models.csv ({len(results)} rows)")

    # Summary by model
    summary = results.groupby("model").agg(
        n=("n", "first"),
        ngroups=("ngroups", "first"),
        n_sig=("pval", lambda x: (x < 0.05).sum()),
    ).reset_index()
    summary.to_csv(os.path.join(TABLES, "ch5_mechanisms_summary.csv"), index=False)
    print("Saved: ch5_mechanisms_summary.csv")

    # ── Coefficient plot for key predictors ──────────────────────────────────
    key_preds = ["insecurity", "inst_trust", "emancipative_vals",
                 "insecurity:emancipative_vals"]
    plot_df = results[results["predictor"].isin(key_preds)].copy()

    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        model_colors = {"M2_H2_insecurity_trust": "#1f77b4",
                        "M3_H3_emancipative": "#2ca02c",
                        "M4_H4_interaction": "#d62728"}
        y_pos = 0
        yticks, ylabels = [], []
        for _, row in plot_df.iterrows():
            color = model_colors.get(row["model"], "grey")
            ax.errorbar(row["coef"], y_pos, xerr=1.96 * row["se"],
                        fmt="o", color=color, capsize=4, markersize=7)
            yticks.append(y_pos)
            ylabels.append(f"{row['predictor']}\n({row['model'].split('_')[0]})")
            y_pos += 1
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel("Coefficient estimate (±95% CI)", fontsize=10)
        ax.set_title("Chapter 5: Mechanism Model Coefficients", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "ch5_mechanism_effects.png"), dpi=150)
        plt.close()
        print("Figure saved: ch5_mechanism_effects.png")

    print("\nChapter 5 complete.")
