# Final Locked-Test Feature Extraction Protocol

## Status

This protocol is frozen before loading either locked test split.

## Purpose

The purpose of this phase is to generate the final eight-dimensional representations for the two locked test splits:

- Camelyon17-WILDS `ood_test`.
- SEER `temporal_test`.

This protocol covers upstream feature extraction only. Downstream classifier evaluation remains the responsibility of the downstream contributor.

## Frozen upstream models

The following joint-model checkpoints are fixed:

| Seed | Selected epoch | Checkpoint |
|---:|---:|---|
| 42 | 2 | `joint_seed42_best.pt` |
| 123 | 5 | `joint_seed123_best.pt` |
| 2026 | 2 | `joint_seed2026_best.pt` |

No checkpoint may be retrained, replaced or selected again after the locked test phase begins.

## Frozen dimensionality-reduction pipelines

For every seed and task, the existing training-fitted PCA pipeline will be reused:

- `image_32d_to_8d_pca.joblib`
- `tabular_32d_to_8d_pca.joblib`

These pipelines were fitted using training representations only.

They must not be refitted or modified using validation or locked-test data.

## Locked splits

| Task | Locked split | Expected rows |
|---|---|---:|
| Camelyon patch metastasis | `ood_test` | 85,054 |
| SEER N0 versus N+ | `temporal_test` | 111,496 |

All three seeds must be processed. No seed may be excluded after results are observed.

## Extraction procedure

For every task and seed:

1. Load the existing selected joint-model checkpoint.
2. Place the model in evaluation mode.
3. Disable gradient calculation.
4. Preserve the original locked-split row order.
5. Extract the frozen 32-dimensional shared-trunk representation.
6. Load the existing task-specific and seed-specific scaler/PCA state.
7. Transform the 32-dimensional representation to eight dimensions without fitting.
8. Preserve the original binary labels.
9. Store only `features` and `labels` in the private output archive.
10. Do not store patient identifiers or other metadata.

## Expected private outputs

Each seed directory will contain:

```text
image_ood_test_32d.npz
image_ood_test_8d.npz
tabular_temporal_test_32d.npz
tabular_temporal_test_8d.npz
```

The private archives must remain excluded from Git.

## Integrity checks

Before and during extraction, the final script must verify and record:

- SHA-256 hash of every selected joint-model checkpoint.
- SHA-256 hash of every fitted scaler/PCA state.
- Expected locked-split row count.
- Input dimension of 32.
- Output dimension of 8.
- Finite feature values.
- Binary labels.
- Row-count preservation.
- Label-order preservation.
- Feature and label hashes.
- Absence of patient identifiers.
- Absence of any fitting or training operation.

Only aggregate audits and manifests may be committed to Git.

## Prohibited actions

The final locked-test phase must not perform:

- Model training or fine-tuning.
- Checkpoint reselection.
- Feature selection.
- PCA or scaler refitting.
- Threshold selection.
- Calibration fitting.
- Architecture changes.
- Hyperparameter tuning.
- Test-based seed selection.
- Test-based preprocessing changes.
- Removal of difficult test examples.

## Downstream handoff

The eight-dimensional locked representations will be transferred through an encrypted private archive.

The downstream contributor must use the already frozen classical and hybrid model states and the existing deterministic input transformation:

```text
angle = 2 × arctan(feature)
```

The frozen classification threshold is `0.5`.

No downstream model may be retrained or changed after the locked representations are received.

## Reporting

Final results must be reported:

- Separately for Camelyon and SEER.
- Separately for every seed.
- For both frozen downstream classifiers.
- As mean and sample standard deviation across all three seeds.
- Without hiding an unfavourable seed.
- Without changing any decision after viewing the results.

Development and locked-test results must remain clearly separated.

## Post-evaluation rule

After the locked-test results are generated, no model, preprocessing, threshold or evaluation decision may be changed.

Any later analysis must be explicitly labelled as post hoc and must not replace the registered final results.