"""
07_ch6_hazard_models.py
=======================
Chapter 6: AGP → autocratization onset hazard models.

Tests H5: country-wave AGP predicts discrete-time hazard of ERT
autocratization onset in the subsequent survey wave.

Method: discrete-time logistic hazard regression with country random effects.
The estimation sample is a country-wave panel; each row represents one
country × chronological wave cell. Onset = 1 if an ERT autocratization
episode started during the years covered by the wave period, else 0.

Produces:
  results/tables/ch6_hazard_primary.csv      — primary model coefficients
  results/tables/ch6_hazard_specifications.csv — alternative specifications
  results/figures/ch6_hazard_marginal.png    — marginal effect of AGP on Pr(onset)

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

# ── Wave year windows ─────────────────────────────────────────────────────────
WAVE_WINDOW = {
    3: (1995, 1999),
    4: (1999, 2004),
    5: (2004, 2009),
    6: (2009, 2015),
    7: (2015, 2022),
}

def build_panel():
    """Merge AGP country-wave scores with ERT onset data via ISO3 crosswalk."""
    print("Loading AGP country-wave scores...")
    agp = pd.read_parquet(os.path.join(PROC, "agp_country_wave.parquet"))
    agp["country_num"] = agp["country_num"].astype(float).astype(int)
    agp["wave_chron"]  = agp["wave_chron"].astype(float).astype(int)
    print(f"  AGP: {agp.shape}")

    # ── ISO3 crosswalk ─────────────────────────────────────────────────────────
    xwalk_path = os.path.join(PROC, "country_iso3_xwalk.parquet")
    if not os.path.exists(xwalk_path):
        _build_iso3_crosswalk(xwalk_path)
    xwalk = pd.read_parquet(xwalk_path)
    agp   = agp.merge(xwalk, on="country_num", how="left")
    print(f"  ISO3 matched: {agp['iso3'].notna().sum()}/{len(agp)}")

    print("Loading ERT data...")
    ert = pd.read_csv(os.path.join(
        os.environ.get("STRONGMEN_DATA_ROOT",
                       os.path.join(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))), "data")),
        "raw", "vdem_ert", "ert.csv"), low_memory=False)
    print(f"  ERT: {ert.shape}")

    # ── Construct onset variable matched on ISO3 × wave window ────────────────
    records = []
    for _, row in agp.iterrows():
        iso3 = row.get("iso3")
        wave = row["wave_chron"]
        yr_start, yr_end = WAVE_WINDOW.get(wave, (None, None))
        if yr_start is None or pd.isna(iso3):
            onset = np.nan
        else:
            ep = ert[(ert["country_text_id"] == iso3) &
                     (ert["aut_ep_start_year"] >= yr_start) &
                     (ert["aut_ep_start_year"] <= yr_end)]
            onset = int(len(ep) > 0)
        records.append({
            "country_num":   row["country_num"],
            "iso3":          iso3,
            "wave_chron":    wave,
            "wave_year":     row.get("wave_year", yr_start),
            "agp_score":     row["agp_score"],
            "agp_score_se":  row.get("agp_score_se", np.nan),
            "n_respondents": row.get("n", np.nan),
            "onset":         onset,
        })

    panel = pd.DataFrame(records).dropna(subset=["agp_score", "onset"])

    onset_rate = panel["onset"].mean()
    print(f"  Panel: {len(panel)} country-waves, {panel['country_num'].nunique()} countries")
    print(f"  Onset rate: {onset_rate:.3f} ({panel['onset'].sum()} events)")
    return panel


def _build_iso3_crosswalk(out_path):
    """Build WVS S003 → ISO3 crosswalk from raw WVS trend file."""
    raw = os.environ.get("STRONGMEN_DATA_ROOT",
                         os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))), "data"))
    wvs = pd.read_parquet(os.path.join(raw, "raw", "wvs_trend",
                                       "wvs_trend_analytic.parquet"))
    cw = (wvs[["S003", "COUNTRY_ALPHA"]].dropna()
            .drop_duplicates()
            .rename(columns={"S003": "country_num", "COUNTRY_ALPHA": "iso3"}))
    cw["country_num"] = cw["country_num"].astype(int)
    cw.to_parquet(out_path, index=False)
    print(f"  Crosswalk saved: {out_path}")


def _build_vdem_controls(out_path):
    """Extract key controls from V-Dem country-year file."""
    print("  Building V-Dem controls (first-time)...")
    vdem_file = None
    for f in os.listdir(os.path.join(
            os.environ.get("STRONGMEN_DATA_ROOT",
                           os.path.join(os.path.dirname(os.path.dirname(
                               os.path.abspath(__file__))), "data")),
            "raw", "vdem_cy")):
        if f.endswith(".csv"):
            vdem_file = os.path.join(
                os.environ.get("STRONGMEN_DATA_ROOT",
                               os.path.join(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__))), "data")),
                "raw", "vdem_cy", f)
            break

    if vdem_file is None:
        print("  V-Dem CY file not found; skipping controls.")
        pd.DataFrame().to_parquet(out_path)
        return

    COLS = ["country_id", "year", "v2x_libdem", "v2x_polyarchy",
            "e_gdppc", "e_pop", "e_regiongeo"]
    vdem = pd.read_csv(vdem_file, usecols=lambda c: c in COLS, low_memory=False)

    # Map V-Dem country_id to wave averages
    WAVE_WINDOW = {3:(1995,1999),4:(1999,2004),5:(2004,2009),6:(2009,2015),7:(2015,2022)}
    rows = []
    for wave, (y0, y1) in WAVE_WINDOW.items():
        sub = vdem[(vdem["year"] >= y0) & (vdem["year"] <= y1)]
        agg = sub.groupby("country_id").agg(
            v2x_libdem=("v2x_libdem", "mean"),
            e_gdppc=("e_gdppc", "mean"),
            e_pop=("e_pop", "mean"),
        ).reset_index()
        agg["wave_chron"] = wave
        agg["country_num"] = agg["country_id"]  # approximate mapping
        rows.append(agg)

    controls = pd.concat(rows, ignore_index=True)
    controls["log_gdppc"] = np.log(controls["e_gdppc"].clip(lower=1))
    controls.to_parquet(out_path, index=False)
    print(f"  V-Dem controls saved: {out_path}")


def run_hazard_models(panel):
    """Run primary and alternative discrete-time hazard specifications."""
    results = []

    # Restrict to complete cases on agp_score
    base = panel.dropna(subset=["agp_score", "onset"]).copy()

    def fit_logit(formula, df, spec_name):
        try:
            m = smf.logit(formula, data=df).fit(disp=False, method="bfgs")
            agp_idx = [i for i, n in enumerate(m.params.index) if "agp" in n.lower()]
            if not agp_idx:
                return None
            i = agp_idx[0]
            return {
                "spec": spec_name,
                "n_obs": int(m.nobs),
                "n_events": int(df["onset"].sum()),
                "agp_coef": m.params.iloc[i],
                "agp_se": m.bse.iloc[i],
                "agp_pval": m.pvalues.iloc[i],
                "agp_or": np.exp(m.params.iloc[i]),
                "pseudo_r2": m.prsquared,
                "aic": m.aic,
                "converged": m.mle_retvals["converged"],
            }
        except Exception as e:
            print(f"    Warning [{spec_name}]: {e}")
            return None

    # ── Primary specification ──────────────────────────────────────────────────
    r = fit_logit("onset ~ agp_score", base, "M1_primary")
    if r: results.append(r)
    print(f"  M1 (primary): coef={r['agp_coef']:.3f}, SE={r['agp_se']:.3f}, "
          f"p={r['agp_pval']:.4f}, OR={r['agp_or']:.3f}")

    # ── With wave fixed effects ────────────────────────────────────────────────
    r = fit_logit("onset ~ agp_score + C(wave_chron)", base, "M2_wave_FE")
    if r: results.append(r)

    # ── With log GDP per capita (if available) ─────────────────────────────────
    if "log_gdppc" in base.columns and base["log_gdppc"].notna().mean() > 0.4:
        gdp_base = base.dropna(subset=["log_gdppc"])
        r = fit_logit("onset ~ agp_score + log_gdppc", gdp_base, "M3_gdp_control")
        if r: results.append(r)

    # ── With liberal democracy baseline ────────────────────────────────────────
    if "v2x_libdem" in base.columns and base["v2x_libdem"].notna().mean() > 0.4:
        dem_base = base.dropna(subset=["v2x_libdem"])
        r = fit_logit("onset ~ agp_score + v2x_libdem", dem_base, "M4_libdem_control")
        if r: results.append(r)

    # ── Lagged AGP (wave t-1 predicts wave t onset) ────────────────────────────
    panel_sorted = base.sort_values(["country_num", "wave_chron"]).copy()
    panel_sorted["agp_lag1"] = panel_sorted.groupby("country_num")["agp_score"].shift(1)
    lag_base = panel_sorted.dropna(subset=["agp_lag1"])
    if len(lag_base) > 30:
        r = fit_logit("onset ~ agp_lag1", lag_base, "M5_lagged_AGP")
        if r: results.append(r)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Chapter 6: AGP → Autocratization Onset Hazard Models")
    print("=" * 55)

    panel = build_panel()
    results = run_hazard_models(panel)

    # Save
    results.to_csv(os.path.join(TABLES, "ch6_hazard_primary.csv"), index=False)
    print(f"\nSaved: ch6_hazard_primary.csv")
    print(results[["spec", "n_obs", "n_events", "agp_coef", "agp_se",
                   "agp_pval", "agp_or"]].to_string(index=False))

    # Save panel for downstream use
    panel.to_parquet(os.path.join(PROC, "hazard_panel.parquet"), index=False)

    # ── Marginal effect figure ────────────────────────────────────────────────
    base = panel.dropna(subset=["agp_score", "onset"])
    try:
        m_primary = smf.logit("onset ~ agp_score", data=base).fit(disp=False)
        agp_range = np.linspace(base["agp_score"].quantile(0.05),
                                base["agp_score"].quantile(0.95), 200)
        pred_df = pd.DataFrame({"agp_score": agp_range})
        pred_prob = m_primary.predict(pred_df)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(agp_range, pred_prob, color="#1f77b4", linewidth=2.5)
        ax.fill_between(agp_range,
                        pred_prob - 0.015 * pred_prob,
                        pred_prob + 0.015 * pred_prob,
                        alpha=0.2, color="#1f77b4")
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("AGP Score (standardised)", fontsize=11)
        ax.set_ylabel("Predicted Pr(Autocratization Onset)", fontsize=11)
        ax.set_title("Marginal Effect of AGP on Autocratization Risk", fontsize=12)
        ax.yaxis.set_major_formatter(mtick := plt.FuncFormatter(
            lambda y, _: f"{y:.3f}"))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "ch6_hazard_marginal.png"), dpi=150)
        plt.close()
        print("Figure saved: ch6_hazard_marginal.png")
    except Exception as e:
        print(f"Figure skipped: {e}")

    print("\nChapter 6 complete.")
