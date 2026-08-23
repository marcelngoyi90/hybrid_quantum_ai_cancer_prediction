# Joint Model v2 Development Selection

Date: 2026-08-24

## Decision

Joint model v2 is selected as the frozen configuration for the remaining multi-seed development runs.

This is a development-stage selection. It is not a locked-test result and is not yet the final reported model performance.

No architecture, loss, optimizer, scheduler, batching, split, or checkpoint-selection changes will be made before running seeds 123 and 2026.

## Controlled change

The only model-input change from v1 to v2 was the addition of:

- `t_stage_harmonized`
- `m_stage_harmonized`

All original SEER patients, targets, split assignments, baseline features, preprocessing statistics, Camelyon17 data, training settings, and evaluation rules remained unchanged.

## Development seed

The controlled comparison used seed 42 for both v1 and v2.

Both runs:

- Trained for five epochs before early stopping.
- Selected epoch 2.
- Used Camelyon17 OOD validation and SEER temporal validation for checkpoint selection.
- Kept Camelyon17 OOD test and SEER temporal test locked.

## Primary metric comparison

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Camelyon17 OOD validation ROC-AUC | 0.975946 | 0.976476 | +0.000529 |
| SEER temporal validation ROC-AUC | 0.782002 | 0.794125 | +0.012123 |
| Joint checkpoint-selection score | 0.878974 | 0.885301 | +0.006326 |

The joint checkpoint-selection score is the equally weighted mean of the two validation ROC-AUC values. It is not a combined prediction accuracy.

## Camelyon17 OOD validation

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| ROC-AUC | 0.975946 | 0.976476 | +0.000529 |
| Accuracy | 0.911271 | 0.915311 | +0.004040 |
| F1 | 0.904186 | 0.910776 | +0.006590 |
| Sensitivity | 0.837325 | 0.864485 | +0.027160 |
| Specificity | 0.985217 | 0.966136 | -0.019081 |
| Brier score | 0.068985 | 0.066613 | -0.002372 |
| Expected calibration error | 0.071087 | 0.053054 | -0.018032 |

The v2 checkpoint improved image ranking, accuracy, F1, sensitivity, Brier score, and expected calibration error. The fixed 0.5 threshold produced lower specificity and higher sensitivity.

## SEER temporal validation

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| ROC-AUC | 0.782002 | 0.794125 | +0.012123 |
| Average precision | 0.611087 | 0.653834 | +0.042747 |
| Balanced accuracy | 0.669727 | 0.680391 | +0.010664 |
| Accuracy | 0.749126 | 0.759168 | +0.010042 |
| F1 | 0.532167 | 0.548831 | +0.016663 |
| Precision | 0.627172 | 0.650868 | +0.023696 |
| Sensitivity | 0.462159 | 0.474450 | +0.012292 |
| Specificity | 0.877295 | 0.886332 | +0.009037 |
| Brier score | 0.168365 | 0.162496 | -0.005869 |
| Expected calibration error | 0.018833 | 0.029457 | +0.010624 |

The added T and M features improved every listed SEER discrimination and fixed-threshold classification metric. The Brier score improved, while expected calibration error worsened slightly. Calibration will remain a separately reported property and will not be corrected using locked-test data.

## Drift evidence

The development-only drift audit found:

- Maximum overall PSI for internal validation: 0.003368.
- Maximum overall PSI for temporal validation: 0.085539.
- No feature reached the moderate PSI threshold of 0.10.
- Harmonized T and M features had small PSI values.
- Neither feature produced unseen categories in the development splits.
- The locked temporal test was not included.

## Training and overfitting evidence

At the selected v2 checkpoint:

- Image training accuracy: 0.979502.
- Tabular training accuracy: 0.750172.
- Image internal-validation accuracy: 0.986442.
- Tabular internal-validation accuracy: 0.753108.
- SEER temporal-validation accuracy: 0.759168.

The tabular results do not show an obvious training-versus-validation overfitting gap.

The image path showed later-epoch OOD instability even while training loss decreased. Early stopping correctly retained epoch 2 instead of the later epochs.

## Shared-gradient evidence

At the selected checkpoint:

| Gradient metric | v1 | v2 |
|---|---:|---:|
| Mean cosine similarity | 0.030470 | -0.022869 |
| Negative cosine fraction | 0.404255 | 0.574468 |
| Mean image shared-gradient norm | 0.105669 | 0.125454 |
| Mean tabular shared-gradient norm | 0.159005 | 0.173041 |

The v2 tasks produced more conflicting shared gradients. Despite this, both primary validation ROC-AUC values improved at the selected checkpoint.

This result supports retaining gradient monitoring and early stopping. It does not justify changing the optimizer after observing only one development seed.

## Interpretation

V2 is selected because the controlled seed-42 comparison improved:

- Camelyon17 OOD validation ROC-AUC.
- SEER temporal validation ROC-AUC.
- The joint checkpoint-selection score.

The SEER result remains moderate and must not be described as 88.5% accuracy. The value 0.885301 is a joint selection score calculated from two ROC-AUC values.

## Locked-test protection

During v2 development:

- Camelyon17 OOD test was not loaded or evaluated.
- SEER temporal test was not loaded or evaluated.
- Neither test was used for preprocessing.
- Neither test was used for threshold selection.
- Neither test was used for calibration.
- Neither test was used for checkpoint selection.

## Frozen next steps

1. Run v2 with seed 123.
2. Run v2 with seed 2026.
3. Preserve each development history separately.
4. Aggregate development metrics across seeds.
5. Report mean, standard deviation, and individual-seed values.
6. Do not alter the v2 configuration between seeds.
7. Keep both locked tests untouched.
8. Continue to the paired-data phase only after the multi-seed development results are reviewed.