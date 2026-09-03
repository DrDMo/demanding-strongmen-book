# Corrections, 3 September 2026

Two bugs in this pipeline produced results that reached the first printing of
*Demanding Strongmen*. Both were silent: nothing errored, nothing looked
obviously wrong, and the affected models fit and reported coefficients as
normal. This file records what happened, what changed, and which published
numbers were affected.

## 1. WVS-only variables were dropped when the files were merged

`02_build_analytic_file.py` concatenated the WVS and EVS frames on the
*intersection* of their columns:

```python
common_cols = list(set(wvs_df.columns) & set(evs_unique.columns))
```

Any variable present in only one source was discarded. `X047_WVS`, the WVS
income scale, is one of those. It never reached
`data/processed/ivs_individual.parquet`.

## 2. The standardiser returned 0 instead of NaN

`06_ch5_mechanisms.py` standardised predictors within country-wave with:

```python
lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
```

When a group had no variance — including when it had no data at all — this
returned a literal `0` rather than `NaN`. Because `0` is not missing,
`dropna()` could not remove those rows and the model fit on a constant.

## What the two bugs did together

With `X047_WVS` gone, the income lookup fell through to `X046`, which has data
in wave 2 only:

| variable | w1 | w2 | w3 | w4 | w5 | w6 | w7 |
|---|---|---|---|---|---|---|---|
| `X046` | 0 | 14,809 | 0 | 0 | 0 | 0 | **0** |
| `X047_WVS` | 5,696 | 22,481 | 67,467 | 53,321 | 76,788 | 86,311 | **94,259** |

So `insecurity` was a constant zero for every Wave 7 respondent. The same
happened to `edu`, which was built from `X025` — also empty in Wave 7, where
the usable item is `X025R` (96,149 respondents).

Chapter 5's economic-insecurity and education results were therefore estimated
on regressors that carried no information. The backlash interaction (H4) is a
product involving insecurity, so it was undefined as well.

A third, separate problem: `insecurity` is defined as `-1 * income`, so the two
are the same variable and cannot both appear as predictors in one model.

## Fixes applied

- `02_build_analytic_file.py` now keeps a named list of analysis variables even
  when they are not common to both sources, and reports their coverage.
- `06_ch5_mechanisms.py` standardiser returns `NaN`, not `0`.
- Education falls back `X025R` → `X025`; income candidates now include `X045`.
- `assert_usable()` raises if any model input is missing, empty, or constant.
  This would have caught both bugs on the first run.

## Corrected results

| file | what changed |
|---|---|
| `ch5_wave7_final.csv` | regenerated on `X047_WVS` and `X025R`; 79,479 respondents, 64 countries |
| `ch6_table61.csv` | new — the four-model ladder the book prints as Table 6.1, which no script previously produced |
| `ch8_transitions.csv` | new — wave-pair transition rates |

Chapter 5, Wave 7, OLS with country fixed effects and SEs clustered by country
(R² = 0.284):

| predictor | β | SE | p |
|---|---|---|---|
| education | −0.083 | 0.007 | <.001 |
| age | −0.068 | 0.012 | <.001 |
| institutional trust | +0.035 | 0.007 | <.001 |
| emancipative values | −0.030 | 0.009 | <.001 |
| male | −0.026 | 0.009 | .003 |
| economic insecurity | −0.006 | 0.006 | .34 |
| insecurity × emancipative | +0.002 | 0.002 | .35 |

## What this changes in the book

- **H2** — the insecurity arm has no individual-level support. It was
  previously reported as supported.
- **H3** — emancipative values remain significant and protective, but are *not*
  the dominant predictor. Education and age both carry larger coefficients.
- **H4** — still not supported, which is what the book already concluded.
- The trust result is unaffected: protective on the democracy item, positively
  associated with strong-leader and army-rule preferences.

## Not affected

The Chapter 6 hazard coefficient is unchanged and reproduces exactly:
`onset ~ agp_score`, n = 325, 48 events, β = 0.6564, SE = 0.1764, p = .0002,
OR = 1.9279. `agp_score` is standardized, so that odds ratio is per standard
deviation.

Chapter 4's country estimates in `ch4_wave7_country_agp.csv` were already
correct; it was the printed Table 4.1 that disagreed with them.
