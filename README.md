# Demanding Strongmen: Replication Archive

Replication code and processed data for:

> Molnar, D. R. (*forthcoming*). *Demanding Strongmen: Values, Insecurity, and the Global Dynamics of Authoritarianism*. Shoreline Publishing Group.

---

## Overview

This repository contains the data pipeline, measurement, and analysis code supporting all empirical chapters of *Demanding Strongmen*. The book examines cross-national variation in mass preferences for authoritarian governance (AGP) and their relationship to autocratization episodes using survey data from the World Values Survey and European Values Study programs combined with V-Dem's Episodes of Regime Transformation (ERT) dataset.

## Repository Structure

```
demanding-strongmen-book/
├── code/                        # All analysis scripts (numbered by execution order)
│   ├── 00_convert_spss.py       # Convert raw SPSS files to Parquet
│   ├── 01_inspect_data.py       # Data inspection and quality checks
│   ├── 02_build_analytic_file.py  # Merge WVS + EVS + ERT; build analytic dataset
│   ├── 03_agp_alignment_prep.py # Prepare input for alignment CFA (Mplus)
│   ├── 04_agp_scores.py         # Read Mplus output; construct country-wave AGP scores
│   ├── 05_ch4_cross_national.py # Chapter 4: Cross-national AGP distribution
│   ├── 06_ch5_mechanisms.py     # Chapter 5: Individual-level mechanism tests
│   ├── 07_ch6_hazard_models.py  # Chapter 6: AGP → autocratization hazard models
│   ├── 08_ch7_trends.py         # Chapter 7: Longitudinal AGP trend analyses
│   └── 09_ch8_ebb.py            # Chapter 8: Incidence-based ebb measure
├── data/
│   ├── raw/                     # Source data (large files tracked via Git LFS)
│   │   ├── wvs_trend/           # WVS Integrated Trend File v4.1 (1981–2022)
│   │   ├── evs_trend/           # EVS Trend File v3.0.0 (1981–2021)
│   │   ├── wvs_wave7/           # WVS Wave 7 standalone file
│   │   ├── evs_wvs_joint/       # EVS/WVS Joint file v5.0
│   │   ├── vdem_cy/             # V-Dem country-year dataset v14
│   │   ├── vdem_ert/            # V-Dem ERT dataset v16
│   │   └── codebooks/           # Variable codebooks (CSV)
│   └── processed/               # Derived analytic datasets (output of code/02_*)
├── docs/                        # Supplementary documentation
│   ├── data_sources.md          # Dataset citations and download instructions
│   └── mplus_alignment/         # Mplus syntax files for alignment CFA
├── results/
│   ├── tables/                  # Publication-ready tables
│   └── figures/                 # Publication-ready figures
├── requirements.txt             # Python package dependencies
├── CITATION.cff                 # Machine-readable citation metadata
└── LICENSE                      # CC BY 4.0
```

## Data Sources

All source data must be obtained directly from the data archives listed below. Raw SPSS and large Parquet files are tracked via Git LFS.

| Dataset | Version | Source | Citation |
|---------|---------|--------|----------|
| WVS Integrated Trend File | v4.1 | [worldvaluessurvey.org](https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp) | Haerpfer et al. (2022a) |
| WVS Wave 7 | v6.0 | [worldvaluessurvey.org](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) | Haerpfer et al. (2022b) |
| EVS/WVS Joint File | v5.0 | [gesis.org](https://search.gesis.org/research_data/ZA7503) | EVS/WVSA (2022) |
| EVS Trend File | v3.0.0 (ZA7503) | [gesis.org](https://search.gesis.org/research_data/ZA7503) | EVS (2022) |
| V-Dem Country-Year | v14 | [v-dem.net](https://www.v-dem.net/data/the-v-dem-dataset/) | Coppedge et al. (2024) |
| V-Dem ERT | v16 | [v-dem.net](https://www.v-dem.net/data/dataset-archive/) | Maerz et al. (2024) |

## Key Variables

**AGP Battery (WVS/EVS harmonized coding):**
- `E114` — Having a strong leader (no parliament/elections): 1=Very good … 4=Very bad
- `E116` — Having the army rule: 1=Very good … 4=Very bad
- `E117` — Having a democratic political system: 1=Very good … 4=Very bad *(reverse-coded for AGP)*

**Outcome Variable:**
- ERT `v2x_regime_amb_*` onset indicators — autocratization episode onset by country-year

## Reproducing the Analysis

### Requirements

```bash
pip install -r requirements.txt
```

### Execution Order

Run scripts in numerical order. Scripts 00–02 transform raw data into the analytic dataset. Script 03 writes the Mplus alignment syntax to `docs/mplus_alignment/`; run each `.inp` file in Mplus (required separately; not open-source), then script 04 reads the alignment output. Scripts 05–10 produce the chapter results.

```bash
python code/00_convert_spss.py
python code/01_inspect_data.py
python code/02_build_analytic_file.py
python code/03_agp_alignment_prep.py
# run each docs/mplus_alignment/*.inp in Mplus
python code/04_agp_scores.py
python code/05_ch4_cross_national.py
python code/06_ch5_mechanisms.py
python code/07_ch6_hazard_models.py
python code/08_ch7_forecasting.py
python code/09_ch8_ebb.py
python code/10_ch9_robustness.py
```

### Online Appendix

Chapter 9 refers to a consolidated robustness table in the online appendix. It is `results/tables/ch9_robustness_all.csv` (every specification, with AGP coefficient, standard error, p-value, and odds ratio), summarized by category in `results/tables/ch9_robustness_summary.csv`. Both are written by `code/10_ch9_robustness.py`.

## License

Code: [CC BY 4.0](LICENSE) — Darin R. Molnar

Data: Subject to the respective data archive terms of use. See `docs/data_sources.md`.

## Citation

```
@book{molnar_forthcoming,
  author    = {Molnar, Darin R.},
  title     = {Demanding Strongmen: Values, Insecurity, and the Global Dynamics of Authoritarianism},
  year      = {forthcoming},
  publisher = {Shoreline Publishing Group}
}
```
