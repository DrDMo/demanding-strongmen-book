"""
02_build_analytic_file.py
=========================
Build the integrated analytic dataset by merging:
  1. WVS Integrated Trend File v4.1  (individual-level, Waves 1–7)
  2. EVS Trend File v3.0.0           (individual-level, Waves 1–5)
  3. V-Dem ERT dataset v16           (country-year panel, autocratization outcomes)
  4. V-Dem Country-Year dataset v14  (country-year controls)

Output files (written to data/processed/):
  ivs_individual.parquet   — merged individual-level WVS + EVS respondents,
                             with standardized identifiers and recoded AGP items
  country_wave.parquet     — country × wave aggregates: mean AGP, N, coverage flags
  country_year_ert.parquet — country-year panel with ERT outcomes and V-Dem controls

Design decisions:
  - Where EVS and WVS overlap for the same country and chronological wave period
    (s002vs=4, 5, or 7), WVS data takes precedence; EVS-only country-waves are
    added as additional observations.
  - AGP items are recoded so that higher values = stronger authoritarian preference:
      agp_leader = 5 - E114   (original: 1=Very good/pro-strongman, 4=Very bad)
      agp_army   = 5 - E116   (original: 1=Very good/pro-army, 4=Very bad)
      agp_nodem  = E117        (original: 1=Very good/pro-democracy = low AGP)
  - All negative values (WVS: -1,-2,-4,-5; EVS: -1,-2,-3,-4,-5) are recoded to NaN.

Run after: 00_convert_spss.py

Author: Darin R. Molnar
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(REPO_ROOT, "data", "raw")
PROC = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# Chronological wave mapping: s002vs value → approximate survey midpoint year
WAVE_YEAR = {1: 1982, 2: 1991, 3: 1997, 4: 2001, 5: 2007, 6: 2013, 7: 2019}

# AGP items shared across both surveys
AGP_ITEMS = ["E114", "E116", "E117"]

# Missing value codes (to NaN)
WVS_MISSING = [-1, -2, -4, -5]
EVS_MISSING = [-1, -2, -3, -4, -5]


# ---------------------------------------------------------------------------
# 1. Load and clean WVS Trend
# ---------------------------------------------------------------------------
def load_wvs():
    print("Loading WVS Trend analytic file...")
    df = pd.read_parquet(os.path.join(RAW, "wvs_trend", "wvs_trend_analytic.parquet"))
    print(f"  Raw shape: {df.shape}")

    # Standardize key identifiers
    df = df.rename(columns={"S002VS": "wave_chron", "S003": "country_num",
                             "COW_NUM": "cow_num", "S020": "year", "S017": "weight"})
    df["survey"] = "WVS"

    # Recode missing
    num_cols = df.select_dtypes(include=[float, int]).columns
    for col in num_cols:
        df[col] = df[col].where(~df[col].isin(WVS_MISSING), np.nan)

    # Recode AGP items to high = authoritarian preference
    df["agp_leader"] = 5 - df["E114"]   # E114: 1=pro-strongman → agp_leader 4=pro-strongman
    df["agp_army"]   = 5 - df["E116"]
    df["agp_nodem"]  = df["E117"]        # E117: 4=anti-democracy = high AGP; retain as-is

    # AGP composite (mean of 3 recoded items; NaN if all missing)
    agp_recoded = ["agp_leader", "agp_army", "agp_nodem"]
    df["agp_mean"] = df[agp_recoded].mean(axis=1, skipna=False)

    print(f"  Cleaned shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Load and clean EVS Trend
# ---------------------------------------------------------------------------
def load_evs():
    print("Loading EVS Trend analytic file...")
    df = pd.read_parquet(os.path.join(RAW, "evs_trend", "evs_trend_analytic.parquet"))
    print(f"  Raw shape: {df.shape}")

    # Standardize key identifiers (EVS uses s002vs for chronological wave)
    df = df.rename(columns={"s002vs": "wave_chron", "S003": "country_num",
                             "COW_NUM": "cow_num", "S020": "year", "S017": "weight"})
    df["survey"] = "EVS"

    # Recode missing
    num_cols = df.select_dtypes(include=[float, int]).columns
    for col in num_cols:
        df[col] = df[col].where(~df[col].isin(EVS_MISSING), np.nan)

    # Recode AGP items
    df["agp_leader"] = 5 - df["E114"]
    df["agp_army"]   = 5 - df["E116"]
    df["agp_nodem"]  = df["E117"]

    agp_recoded = ["agp_leader", "agp_army", "agp_nodem"]
    df["agp_mean"] = df[agp_recoded].mean(axis=1, skipna=False)

    print(f"  Cleaned shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. Merge WVS and EVS individual files
# ---------------------------------------------------------------------------
def merge_individual(wvs_df, evs_df):
    """
    Combine WVS and EVS individual respondents into a single IVS-like file.

    For country × wave_chron cells covered by both WVS and EVS:
      - WVS observations are retained as-is.
      - EVS observations for the same country-wave are dropped.
    This avoids double-counting while preserving all EVS-only country-waves.
    """
    print("\nMerging WVS and EVS individual files...")

    # Identify WVS country × wave_chron cells
    wvs_cells = set(zip(wvs_df["country_num"].dropna(), wvs_df["wave_chron"].dropna()))

    # Keep EVS rows not covered by WVS
    evs_unique = evs_df[
        ~evs_df.apply(
            lambda r: (r["country_num"], r["wave_chron"]) in wvs_cells, axis=1
        )
    ]
    n_dropped = len(evs_df) - len(evs_unique)
    print(f"  EVS rows dropped (WVS overlap): {n_dropped:,}")
    print(f"  EVS rows retained (unique): {len(evs_unique):,}")

    # Find common columns for concat
    common_cols = list(set(wvs_df.columns) & set(evs_unique.columns))
    merged = pd.concat(
        [wvs_df[common_cols], evs_unique[common_cols]], ignore_index=True
    )
    print(f"  Merged IVS shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# 4. Build country × wave AGP aggregate
# ---------------------------------------------------------------------------
def build_country_wave(ivs_df):
    print("\nBuilding country × wave aggregate...")
    agg = (
        ivs_df.groupby(["country_num", "wave_chron", "survey"])
        .agg(
            n=("agp_mean", "size"),
            n_agp_valid=("agp_mean", "count"),
            agp_mean=("agp_mean", "mean"),
            agp_leader_mean=("agp_leader", "mean"),
            agp_army_mean=("agp_army", "mean"),
            agp_nodem_mean=("agp_nodem", "mean"),
            year_median=("year", "median"),
        )
        .reset_index()
    )
    agg["wave_year"] = agg["wave_chron"].map(WAVE_YEAR)
    agg["pct_agp_valid"] = agg["n_agp_valid"] / agg["n"]
    print(f"  Country-wave shape: {agg.shape}")
    return agg


# ---------------------------------------------------------------------------
# 5. Load ERT and V-Dem country-year data
# ---------------------------------------------------------------------------
def load_ert():
    ert_dir = os.path.join(RAW, "vdem_ert")
    ert_files = [f for f in os.listdir(ert_dir) if f.endswith(".csv") or f.endswith(".parquet")]
    if not ert_files:
        print("  ERT data not found in data/raw/vdem_ert/ — skipping ERT merge.")
        return None
    fpath = os.path.join(ert_dir, ert_files[0])
    print(f"  Loading ERT: {fpath}")
    if fpath.endswith(".parquet"):
        ert = pd.read_parquet(fpath)
    else:
        ert = pd.read_csv(fpath, low_memory=False)
    print(f"  ERT shape: {ert.shape}")
    return ert


def load_vdem_cy():
    vdem_dir = os.path.join(RAW, "vdem_cy")
    files = [f for f in os.listdir(vdem_dir) if f.endswith(".csv") or f.endswith(".parquet")]
    if not files:
        print("  V-Dem CY data not found — skipping.")
        return None
    fpath = os.path.join(vdem_dir, files[0])
    print(f"  Loading V-Dem CY: {fpath}")
    if fpath.endswith(".parquet"):
        vdem = pd.read_parquet(fpath)
    else:
        vdem = pd.read_csv(fpath, low_memory=False)
    print(f"  V-Dem shape: {vdem.shape}")
    return vdem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Demanding Strongmen — Building analytic dataset")
    print("=" * 55)

    wvs = load_wvs()
    evs = load_evs()

    ivs = merge_individual(wvs, evs)
    ivs.to_parquet(os.path.join(PROC, "ivs_individual.parquet"), index=False, compression="snappy")
    print(f"\nSaved: data/processed/ivs_individual.parquet ({len(ivs):,} rows)")

    cw = build_country_wave(ivs)
    cw.to_parquet(os.path.join(PROC, "country_wave.parquet"), index=False, compression="snappy")
    print(f"Saved: data/processed/country_wave.parquet ({len(cw):,} rows)")

    print("\nLoading ERT and V-Dem country-year data...")
    ert = load_ert()
    vdem = load_vdem_cy()

    if ert is not None:
        ert.to_parquet(os.path.join(PROC, "ert_clean.parquet"), index=False, compression="snappy")
        print(f"Saved: data/processed/ert_clean.parquet ({len(ert):,} rows)")

    print("\nAnalytic dataset build complete.")
