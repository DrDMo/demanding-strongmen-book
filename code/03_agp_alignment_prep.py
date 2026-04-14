"""
03_agp_alignment_prep.py
========================
Prepare input files for alignment-based CFA in Mplus.

The alignment method (Asparouhov & Muthén, 2014; Leitgöb et al., 2023)
estimates factor means and variances across country-wave groups while
accommodating partial measurement non-invariance. This script:

  1. Loads the IVS individual file from data/processed/
  2. Selects the three AGP items (E114 recoded, E116 recoded, E117)
  3. Writes one Mplus-format data file per wave (free alignment, ordered-categorical)
  4. Writes the Mplus syntax files to docs/mplus_alignment/

Mplus is proprietary software (https://www.statmodel.com/) and must be
licensed and installed separately. The syntax files produced here are
compatible with Mplus v8.x.

Reference:
  Asparouhov, T., & Muthén, B. (2014). Multiple-group factor analysis alignment.
  Structural Equation Modeling, 21(4), 495–508.

Run after: 02_build_analytic_file.py

Author: Darin R. Molnar
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(REPO_ROOT, "data", "processed")
DOCS  = os.path.join(REPO_ROOT, "docs", "mplus_alignment")
os.makedirs(DOCS, exist_ok=True)

AGP_RECODED = ["agp_leader", "agp_army", "agp_nodem"]
# agp_leader = 5 - E114; agp_army = 5 - E116; agp_nodem = E117
# All on scale 1–4, higher = more pro-authoritarian / anti-democratic


def write_mplus_dat(df_wave, wave_id, out_dir):
    """Write a wave-specific Mplus .dat file (space-delimited, no header)."""
    dat_path = os.path.join(out_dir, f"wave{wave_id}_agp.dat")
    # Select complete cases on AGP items + country group variable
    sub = df_wave[["country_num"] + AGP_RECODED].dropna()
    # Mplus needs integer country codes for group definition
    sub = sub.copy()
    sub["country_num"] = sub["country_num"].astype(int)
    sub[AGP_RECODED] = sub[AGP_RECODED].astype(int)
    sub.to_csv(dat_path, sep=" ", index=False, header=False)
    return dat_path, sub["country_num"].unique().tolist()


def write_mplus_syntax(wave_id, dat_filename, country_codes, out_dir):
    """Write Mplus alignment CFA syntax for a single wave."""
    n_groups = len(country_codes)
    groups_str = " ".join([str(int(c)) for c in sorted(country_codes)])
    syn_path = os.path.join(out_dir, f"wave{wave_id}_alignment.inp")

    syntax = f"""! Alignment CFA — AGP three-item model
! Wave {wave_id} | {n_groups} country groups
! Asparouhov & Muthén (2014) free alignment

TITLE: AGP Alignment CFA Wave {wave_id};

DATA:
  FILE = "{dat_filename}";

VARIABLE:
  NAMES = country agp_leader agp_army agp_nodem;
  USEVARIABLES = agp_leader agp_army agp_nodem;
  CATEGORICAL = agp_leader agp_army agp_nodem;
  GROUPING = country ({groups_str});
  MISSING = ALL (-99);

ANALYSIS:
  TYPE = MIXTURE;
  ESTIMATOR = WLSMV;
  ALIGNMENT = FREE;
  PROCESSORS = 4;

MODEL:
  %OVERALL%
  AGP BY agp_leader* agp_army agp_nodem;
  [AGP@0];
  AGP@1;

OUTPUT:
  ALIGN;
  STDYX;
  TECH1;

SAVEDATA:
  RESULTS = wave{wave_id}_alignment_results.dat;
  FORMAT FREE;
"""
    with open(syn_path, "w") as f:
        f.write(syntax)
    return syn_path


if __name__ == "__main__":
    print("Demanding Strongmen — Preparing alignment CFA inputs")
    print("=" * 55)

    ivs_path = os.path.join(PROC, "ivs_individual.parquet")
    if not os.path.exists(ivs_path):
        raise FileNotFoundError(f"Run 02_build_analytic_file.py first: {ivs_path}")

    df = pd.read_parquet(ivs_path)
    print(f"Loaded IVS: {df.shape}")

    # Process each chronological wave (only waves with AGP data: 3–7)
    for wave in sorted(df["wave_chron"].dropna().unique()):
        if wave < 3:
            continue  # AGP items not available before wave 3
        wave_id = int(wave)
        sub = df[df["wave_chron"] == wave].copy()
        n_countries = sub["country_num"].nunique()
        n_valid = sub[AGP_RECODED].dropna().shape[0]
        print(f"\nWave {wave_id}: {len(sub):,} respondents, {n_countries} countries, "
              f"{n_valid:,} complete AGP cases")

        dat_path, country_codes = write_mplus_dat(sub, wave_id, DOCS)
        syn_path = write_mplus_syntax(
            wave_id, os.path.basename(dat_path), country_codes, DOCS
        )
        print(f"  Data:   {dat_path}")
        print(f"  Syntax: {syn_path}")

    print(f"\nMplus input files written to: {DOCS}")
    print("Run each .inp file in Mplus, then execute 04_agp_scores.py.")
