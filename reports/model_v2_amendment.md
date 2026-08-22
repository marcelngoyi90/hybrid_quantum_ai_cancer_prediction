# Joint Model v2 Development Amendment

Date: 2026-08-23

## Purpose

Improve the leakage-safe SEER lymph-node prediction task while preserving the completed v1 model and all its results.

## v1 development result

For seed 42, the selected v1 checkpoint was epoch 2:

- Camelyon17 OOD validation ROC-AUC: 0.975946
- SEER temporal validation ROC-AUC: 0.782002
- Joint checkpoint score: 0.878974

The image task performed strongly. The SEER task was the main limitation.

## Additional SEER variables

The v2 export adds:

- Derived AJCC T, 7th ed (2010-2015)
- Derived AJCC M, 7th ed (2010-2015)
- Derived SEER Combined T (2016-2017)
- Derived SEER Combined M (2016-2017)
- Derived EOD 2018 T Recode (2018+)
- Derived EOD 2018 M Recode (2018+)

## Harmonized features

The year-appropriate source variables will be combined into:

- `t_stage_harmonized`: T0, Tis, T1, T2, T3, T4, TX, or unknown
- `m_stage_harmonized`: M0, M1, or unknown

Values outside each variable's applicable diagnosis years will not be interpreted as clinical missingness.

## Leakage protection

The following remain forbidden model inputs:

- Every N-stage variable
- Regional nodes positive
- Regional nodes examined
- Stage Group
- Summary Stage
- Treatment and surgery variables
- Patient identifiers
- Split metadata
- Diagnosis year

## Development procedure

1. Preserve all v1 code, metrics, and checkpoints.
2. Build separate v2 cohort, feature, preprocessing, checkpoint, and metrics files.
3. Keep the initial v2 architecture, losses, splits, and training settings unchanged.
4. Change only the addition of the two harmonized T/M features.
5. Train development seed 42.
6. Compare using development validation only.
7. Keep both locked test sets untouched.
8. Run the remaining final seeds only after selecting the final development configuration.
