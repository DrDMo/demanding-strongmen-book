"""
01_inspect_data.py
==================
Data quality checks and descriptive inspection of the raw converted files.

Produces console output and writes summary tables to data/processed/.

Run after: 00_convert_spss.py

Author: Darin R. Molnar
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW       = os.path.join(REPO_ROOT, "data", "raw")
PROC      = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

AGP_ITEMS = ["E114", "E116", "E117"]

def recode_missing(df, items=None):
    """Set WVS/EVS negative values to NaN across specified columns."""
    cols = items or [c for c in df.columns if df[c].dtype in [float, int]]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].where(df[col] > 0, np.nan)
    return df


def inspect_wvs_trend():
    path = os.path.join(RAW, "wvs_trend", "wvs_trend_analytic.parquet")
    print(f"\n{'='*65}")
    print("WVS Integrated Trend File v4.1")
    print(f"{'='*65}")

    df = pd.read_parquet(path)
    df = recode_missing(df, AGP_ITEMS)
    print(f"Shape: {df.shape}")

    # Wave × country coverage
    wave_tab = (
        df.groupby("S002VS")
        .agg(
            N=("S003", "size"),
            Countries=("S003", "nunique"),
            Year_min=("S020", "min"),
            Year_max=("S020", "max"),
            AGP_complete=(
                "E114",
                lambda x: df.loc[x.index, AGP_ITEMS].notna().all(axis=1).sum(),
            ),
        )
        .reset_index()
    )
    wave_tab["Wave"] = wave_tab["S002VS"].astype(int)
    wave_tab = wave_tab.drop(columns="S002VS")
    print("\nWave × Country Coverage:")
    print(wave_tab.to_string(index=False))

    # AGP item means by wave
    print("\nAGP Item Means by Wave (valid responses only):")
    for wave in sorted(df["S002VS"].dropna().unique()):
        sub = df[df["S002VS"] == wave]
        means = sub[AGP_ITEMS].mean().round(3)
        print(f"  Wave {int(wave)}: E114={means.E114:.3f}  E116={means.E116:.3f}  E117={means.E117:.3f}")

    # Save wave summary
    wave_tab.to_csv(os.path.join(PROC, "wvs_trend_wave_summary.csv"), index=False)
    return df


def inspect_evs_trend():
    path = os.path.join(RAW, "evs_trend", "evs_trend_analytic.parquet")
    print(f"\n{'='*65}")
    print("EVS Trend File v3.0.0 (ZA7503)")
    print(f"{'='*65}")

    df = pd.read_parquet(path)
    df = recode_missing(df, AGP_ITEMS)
    print(f"Shape: {df.shape}")

    # Wave × country coverage
    wave_tab = (
        df.groupby("S002EVS")
        .agg(
            N=("S003", "size"),
            Countries=("S003", "nunique"),
            Year_min=("S020", "min"),
            Year_max=("S020", "max"),
        )
        .reset_index()
    )
    wave_tab["Wave"] = wave_tab["S002EVS"].astype(int)
    wave_tab = wave_tab.drop(columns="S002EVS")
    print("\nWave × Country Coverage:")
    print(wave_tab.to_string(index=False))

    # AGP means by wave
    print("\nAGP Item Means by Wave (valid responses only):")
    for wave in sorted(df["S002EVS"].dropna().unique()):
        sub = df[df["S002EVS"] == wave]
        means = sub[AGP_ITEMS].mean().round(3)
        print(f"  EVS Wave {int(wave)}: E114={means.E114:.3f}  E116={means.E116:.3f}  E117={means.E117:.3f}")

    wave_tab.to_csv(os.path.join(PROC, "evs_trend_wave_summary.csv"), index=False)
    return df


def combined_coverage(wvs_df, evs_df):
    """Report combined country coverage across both survey programs."""
    print(f"\n{'='*65}")
    print("Combined WVS + EVS Coverage (AGP-eligible waves)")
    print(f"{'='*65}")

    wvs_agp = wvs_df[wvs_df["S002VS"] >= 3]["S003"].dropna()
    evs_agp = evs_df[evs_df["S002EVS"] >= 3]["S003"].dropna()
    combined = set(wvs_agp.unique()) | set(evs_agp.unique())

    print(f"WVS countries (Waves 3–7):       {wvs_agp.nunique()}")
    print(f"EVS countries (Waves 3–5):       {evs_agp.nunique()}")
    print(f"Combined unique country codes:   {len(combined)}")
    print(f"WVS AGP respondents:             {(wvs_df[wvs_df.S002VS >= 3][AGP_ITEMS].notna().all(axis=1)).sum():,}")
    print(f"EVS AGP respondents:             {(evs_df[evs_df.S002EVS >= 3][AGP_ITEMS].notna().all(axis=1)).sum():,}")


if __name__ == "__main__":
    wvs = inspect_wvs_trend()
    evs = inspect_evs_trend()
    combined_coverage(wvs, evs)
    print("\nInspection complete. Summary tables saved to data/processed/.")
