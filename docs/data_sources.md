# Data Sources and Download Instructions

All raw data files must be obtained from the original archives. Registration
may be required for some sources.

---

## 1. WVS Integrated Trend File v4.1 (1981–2022)

**File:** `Trends_VS_1981_2022_Spss_v4_1.sav` (~509 MB)

**Download:** https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp
- Select "WVS Trend File (1981-2022)" → SPSS format

**Place in:** `data/raw/wvs_trend/`

**Citation:**
Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K.,
Diez-Medrano, J., Lagos, M., Norris, P., Ponarin, E., & Puranen, B. (Eds.).
(2022). World Values Survey Trend File (1981–2022) Cross-National Data-Set.
Version 4.1. JD Systems Institute & WVSA Secretariat.
https://doi.org/10.14281/18241.27

---

## 2. WVS Wave 7 Standalone File v6.0

**File:** `WVS_Cross-National_Wave_7_sav_v6_0.sav`

**Download:** https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp

**Place in:** `data/raw/wvs_wave7/`

**Citation:**
Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K.,
Diez-Medrano, J., Lagos, M., Norris, P., Ponarin, E., & Puranen, B. (Eds.).
(2022). World Values Survey: Round Seven – Country-Pooled Datafile Version 6.0.
JD Systems Institute & WVSA Secretariat.
https://doi.org/10.14281/18241.24

---

## 3. EVS/WVS Joint File v5.0

**Download:** https://search.gesis.org/research_data/ZA7503 (search for Joint file)
  or https://www.worldvaluessurvey.org

**Place in:** `data/raw/evs_wvs_joint/`

**Citation:**
EVS/WVSA (2022). European Values Study and World Values Survey: Joint EVS/WVS
2017-2022 Dataset. GESIS Data Archive & JD Systems Institute.
https://doi.org/10.4232/1.14023

---

## 4. EVS Trend File v3.0.0 (ZA7503)

**File:** `ZA7503_v3-0-0.sav` (~217 MB)

**Download:** https://search.gesis.org/research_data/ZA7503
- Select SPSS format → ZA7503_v3-0-0.sav

**Place in:** `data/raw/evs_trend/`

**Important:** This file contains a malformed multi-response set (MRSETS)
metadata record that causes standard parsers to fail. `00_convert_spss.py`
handles this automatically using pyspssio with a patched MRSETS parser.

**Citation:**
EVS (2022). European Values Study Longitudinal Data File 1981-2021 (EVS Trend).
GESIS Data Archive, Cologne. ZA7503 Data file Version 3.0.0.
https://doi.org/10.4232/1.14021

---

## 5. V-Dem Country-Year Dataset v14

**File:** `V-Dem-CY-Full+Others-v14.csv` (~382 MB)

**Download:** https://www.v-dem.net/data/the-v-dem-dataset/
- Registration required (free academic use)

**Place in:** `data/raw/vdem_cy/`

**Citation:**
Coppedge, M., Gerring, J., Knutsen, C. H., Lindberg, S. I., Teorell, J.,
Altman, D., … Ziblatt, D. (2024). V-Dem Dataset v14.
Varieties of Democracy (V-Dem) Project.
https://doi.org/10.23696/vdemds24

---

## 6. V-Dem Episodes of Regime Transformation (ERT) v16

**File:** `ERT_dataset_v16.csv` (or .xlsx)

**Download:** https://www.v-dem.net/data/dataset-archive/
- Select "Episodes of Regime Transformation" → v16

**Place in:** `data/raw/vdem_ert/`

**Citation:**
Maerz, S. F., Edgell, A. B., Wilson, M. C., Hellmeier, S., & Lindberg, S. I.
(2024). Episodes of regime transformation. Journal of Peace Research, 61(6),
967–984. https://doi.org/10.1177/00223433231168192

---

## Variable Codebooks

Codebook CSVs for the analytic subsets of each survey file are in
`data/raw/codebooks/`:
- `wvs_trend_analytic_codebook.csv`
- `evs_trend_analytic_codebook.csv`

Full codebooks (PDFs) are available from the original download pages and
are referenced here for convenience but not redistributed in this repository.
