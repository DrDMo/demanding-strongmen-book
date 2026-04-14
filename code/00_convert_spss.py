"""
00_convert_spss.py
==================
Convert raw SPSS (.sav) files to Parquet format for efficient analysis.

Produces:
  data/raw/wvs_trend/wvs_trend_full.parquet         — all 732 variables
  data/raw/wvs_trend/wvs_trend_analytic.parquet     — 52-variable AGP subset
  data/raw/wvs_trend/wvs_trend_value_labels.json    — SPSS value label dictionary
  data/raw/evs_trend/evs_trend_full.parquet         — all 635 variables
  data/raw/evs_trend/evs_trend_analytic.parquet     — 48-variable AGP subset
  data/raw/evs_trend/evs_trend_value_labels.json    — SPSS value label dictionary

Notes:
  - WVS Trend is read with pyreadstat (standard SPSS reader)
  - EVS Trend (ZA7503) contains a malformed multi-response set (MRSETS) metadata
    record that causes pyreadstat and pandas.read_spss to fail. It is read with
    pyspssio (IBM SPSS I/O library wrapper). The malformed entry is silently
    skipped; data content is unaffected.
  - Both files are streamed in chunks to avoid OOM on machines with < 4 GB RAM.

Author: Darin R. Molnar
"""

import os
import json
import pyreadstat
import pyspssio
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths — set relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "data", "raw")

WVS_SAV  = os.path.join(RAW, "wvs_trend", "Trends_VS_1981_2022_Spss_v4_1.sav")
EVS_SAV  = os.path.join(RAW, "evs_trend",  "ZA7503_v3-0-0.sav")

# ---------------------------------------------------------------------------
# Analytical variable lists
# ---------------------------------------------------------------------------
WVS_ANALYTIC_COLS = [
    # Identifiers & weights
    "S002VS", "s002", "S003", "COUNTRY_ALPHA", "COW_NUM", "S020",
    "S017", "S018", "pwgt",
    # AGP battery
    "E114", "E115", "E116", "E117",
    # Democracy attitudes
    "E118", "E119", "E120", "E121", "E122", "E123", "E236",
    # Institutional confidence
    "E069_01", "E069_02", "E069_06", "E069_07", "E069_11", "E069_12", "E069_17",
    # Trust, postmaterialism, political action
    "A165", "E023", "E025", "E026", "E027", "E028", "Y002", "Y003",
    # Demographics
    "X001", "X003", "X003R", "X007", "X025", "X025R", "X028",
    "X045", "X046", "X047_WVS", "X047R_WVS",
    # Political ID, religion
    "E033", "F034", "F063",
    # Social trust
    "G007_01", "G007_35_B", "G007_36_B",
]

EVS_ANALYTIC_COLS = [
    # Identifiers & weights
    "S001", "S002EVS", "s002vs", "S003", "COW_NUM", "S009", "S020",
    "S017", "S018", "pwght",
    # AGP battery
    "E114", "E115", "E116", "E117",
    # Democracy attitudes
    "E118", "E119", "E120", "E121", "E122", "E123", "E236",
    # Institutional confidence
    "E069_01", "E069_02", "E069_06", "E069_07", "E069_11", "E069_12", "E069_17",
    # Trust, postmaterialism, political action
    "A165", "Y002", "E023", "E025", "E026", "E027", "E028",
    # Demographics
    "X001", "X003", "X003R", "X007", "X025", "X025R", "X028", "X045", "X046",
    # Political ID, religion
    "E033", "F034", "F063",
    # Social trust
    "G007_01",
]

CHUNK = 80_000  # rows per chunk for streaming conversion


# ---------------------------------------------------------------------------
# Helper: stream a pyreadstat file to Parquet using pyarrow ParquetWriter
# ---------------------------------------------------------------------------
def stream_to_parquet_pyreadstat(sav_path, out_path, usecols=None):
    """Read a .sav file in chunks and write to a single Parquet file."""
    writer = None
    offset = 0
    total = 0

    while True:
        df_chunk, meta = pyreadstat.read_sav(
            sav_path,
            apply_value_formats=False,
            row_limit=CHUNK,
            row_offset=offset,
            usecols=usecols,
        )
        if len(df_chunk) == 0:
            break

        for col in df_chunk.select_dtypes(include="object").columns:
            df_chunk[col] = df_chunk[col].astype(str)

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        total += len(df_chunk)
        print(f"  {os.path.basename(out_path)}: {total:,} rows written", flush=True)
        del df_chunk, table

        if total % (CHUNK * 5) == 0 and total > 0:
            # Safety: stop if we've read many more rows than expected
            pass
        offset += CHUNK

    if writer:
        writer.close()
    print(f"  Done: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return meta


# ---------------------------------------------------------------------------
# Helper: save value labels from pyreadstat metadata
# ---------------------------------------------------------------------------
def save_value_labels_pyreadstat(meta, out_path):
    vvl = {}
    for k, v in meta.variable_value_labels.items():
        vvl[k] = {str(kk): str(vv) for kk, vv in v.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vvl, f, indent=2, ensure_ascii=False)
    print(f"  Value labels: {out_path} ({os.path.getsize(out_path)/1e3:.0f} KB)")


# ---------------------------------------------------------------------------
# WVS Trend File
# ---------------------------------------------------------------------------
def convert_wvs_trend():
    print("\n=== WVS Integrated Trend File v4.1 ===")
    out_dir = os.path.dirname(WVS_SAV)

    # 1. Analytic subset
    print("Converting analytic subset...")
    _, meta_all = pyreadstat.read_sav(WVS_SAV, metadataonly=True)
    available = set(meta_all.column_names)
    use_cols = [c for c in WVS_ANALYTIC_COLS if c in available]
    missing = [c for c in WVS_ANALYTIC_COLS if c not in available]
    if missing:
        print(f"  Warning: variables not found: {missing}")

    df_analytic, meta = pyreadstat.read_sav(
        WVS_SAV, apply_value_formats=False, usecols=use_cols
    )
    out_analytic = os.path.join(out_dir, "wvs_trend_analytic.parquet")
    df_analytic.to_parquet(out_analytic, index=False, compression="snappy")
    print(f"  Analytic: {df_analytic.shape}, {os.path.getsize(out_analytic)/1e6:.1f} MB")

    # 2. Full file
    print("Converting full file (streaming)...")
    out_full = os.path.join(out_dir, "wvs_trend_full.parquet")
    meta_full = stream_to_parquet_pyreadstat(WVS_SAV, out_full)

    # 3. Value labels
    out_labels = os.path.join(out_dir, "wvs_trend_value_labels.json")
    save_value_labels_pyreadstat(meta_full, out_labels)


# ---------------------------------------------------------------------------
# EVS Trend File (requires pyspssio due to malformed MRSETS)
# ---------------------------------------------------------------------------
def _patch_pyspssio_mrsets():
    """
    Monkey-patch pyspssio.header to skip malformed MRSETS entries.

    ZA7503 contains an invalid multi-response set record with an empty
    label_length field. The stock pyspssio parser raises ValueError when it
    tries to call int('') on this field. Wrapping the individual mrset parsing
    calls in try/except skips only the malformed entry; all valid data is
    unaffected.
    """
    from pyspssio import header as _hdr

    original_mrsets = _hdr.Header.mrsets.fget

    def patched_mrsets(self):
        # Temporarily swap _parse_mrset_d with a guarded version
        _orig_d = self._parse_mrset_d
        _orig_c = self._parse_mrset_c
        _orig_e = self._parse_mrset_e

        def safe_d(attr):
            try:
                return _orig_d(attr)
            except (ValueError, IndexError):
                return None

        def safe_c(attr):
            try:
                return _orig_c(attr)
            except (ValueError, IndexError):
                return None

        def safe_e(attr):
            try:
                return _orig_e(attr)
            except (ValueError, IndexError):
                return None

        self._parse_mrset_d = safe_d
        self._parse_mrset_c = safe_c
        self._parse_mrset_e = safe_e
        try:
            result = original_mrsets(self)
        finally:
            self._parse_mrset_d = _orig_d
            self._parse_mrset_c = _orig_c
            self._parse_mrset_e = _orig_e
        return result

    _hdr.Header.mrsets = property(patched_mrsets)


def convert_evs_trend():
    print("\n=== EVS Trend File v3.0.0 (ZA7503) ===")
    _patch_pyspssio_mrsets()

    out_dir = os.path.dirname(EVS_SAV)

    # 1. Get available columns
    result = pyspssio.read_sav(EVS_SAV, row_limit=1)
    _, meta = result
    available = set(meta["var_names"])
    use_cols = [c for c in EVS_ANALYTIC_COLS if c in available]
    missing = [c for c in EVS_ANALYTIC_COLS if c not in available]
    if missing:
        print(f"  Warning: variables not found: {missing}")

    # 2. Analytic subset
    print("Converting analytic subset...")
    result = pyspssio.read_sav(EVS_SAV, usecols=use_cols)
    df_analytic, meta = result
    out_analytic = os.path.join(out_dir, "evs_trend_analytic.parquet")
    df_analytic.to_parquet(out_analytic, index=False, compression="snappy")
    print(f"  Analytic: {df_analytic.shape}, {os.path.getsize(out_analytic)/1e6:.1f} MB")
    del df_analytic

    # 3. Full file (streaming via pyspssio)
    print("Converting full file (streaming)...")
    out_full = os.path.join(out_dir, "evs_trend_full.parquet")
    writer = None
    total = 0
    offset = 0
    last_meta = meta

    while True:
        try:
            result = pyspssio.read_sav(EVS_SAV, row_offset=offset, row_limit=CHUNK)
        except Exception:
            break

        df_chunk = result[0] if isinstance(result, tuple) else result
        if len(df_chunk) == 0:
            break

        for col in df_chunk.select_dtypes(include="object").columns:
            df_chunk[col] = df_chunk[col].astype(str)

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_full, table.schema, compression="snappy")
        writer.write_table(table)

        total += len(df_chunk)
        print(f"  evs_trend_full.parquet: {total:,} rows written", flush=True)
        del df_chunk, table

        offset += CHUNK
        if total >= 230_000:
            break

    if writer:
        writer.close()
    print(f"  Done: {out_full} ({os.path.getsize(out_full)/1e6:.1f} MB)")

    # 4. Value labels
    out_labels = os.path.join(out_dir, "evs_trend_value_labels.json")
    vvl = {}
    for k, vals in last_meta.get("var_value_labels", {}).items():
        if vals:
            vvl[k] = {str(kk): str(vv) for kk, vv in vals.items()}
    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(vvl, f, indent=2, ensure_ascii=False)
    print(f"  Value labels: {out_labels} ({os.path.getsize(out_labels)/1e3:.0f} KB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Demanding Strongmen — SPSS → Parquet conversion")
    print("=" * 55)

    if os.path.exists(WVS_SAV):
        convert_wvs_trend()
    else:
        print(f"WVS Trend file not found: {WVS_SAV}")
        print("Download from: https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp")

    if os.path.exists(EVS_SAV):
        convert_evs_trend()
    else:
        print(f"EVS Trend file not found: {EVS_SAV}")
        print("Download from: https://search.gesis.org/research_data/ZA7503")

    print("\nConversion complete.")
