"""
config.py
=========
Centralised path configuration for the Demanding Strongmen analysis pipeline.

Data location is resolved from (in priority order):
  1. Environment variable  STRONGMEN_DATA_ROOT
  2. Default: <repo_root>/data/

Usage in any script:
  from config import RAW, PROC, RESULTS, FIGS, TABLES

Author: Darin R. Molnar
"""

import os

# Repo root is one level up from this file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow external data root override via environment variable
_data_root = os.environ.get("STRONGMEN_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

RAW     = os.path.join(_data_root, "raw")
PROC    = os.path.join(_data_root, "processed")
RESULTS = os.path.join(REPO_ROOT, "results")
TABLES  = os.path.join(RESULTS, "tables")
FIGS    = os.path.join(RESULTS, "figures")
DOCS    = os.path.join(REPO_ROOT, "docs")
MPLUS   = os.path.join(DOCS, "mplus_alignment")

# Create output directories if they don't exist
for d in [PROC, TABLES, FIGS, MPLUS]:
    os.makedirs(d, exist_ok=True)
